//! The heap as the body walks (`declaration`, `eval`) see it.
//!
//! Everything a walk does to a heap goes through the methods here:
//!
//! | operation                 | method                 |
//! |---------------------------|------------------------|
//! | read a location           | [`Heap::read`], [`Heap::framing_perm`], [`Heap::covering_holder`] |
//! | `perm(loc)`               | [`Heap::perm`]         |
//! | add permission            | [`Heap::produce`]      |
//! | take permission / frame   | [`Heap::consume`]      |
//! | field write               | [`Heap::write`]        |
//! | branch join, heap sum     | [`Heap::merge`], [`Heap::union`] |
//!
//! Every method here takes the `pc` of the instruction that uses the heap, which
//! inside a block is only the delta past the block's cube. Each one first makes
//! it the **effective** pc, the heap's region and `pc` ([`Heap::effective_pc`]),
//! and everything below works under that: a fact the heap states under the
//! delta alone would hold in every block, also where this heap does not exist
//! (`failing/value_from_chunk_without_permission/summary_cross_branch_leak.vpr`).
//!
//! [`Heap::produce`] and [`Heap::consume`] take an [`Access`]: one location
//! (`acc(e.f, p)`) or a quantified chunk set (`forall xs :: c ==> acc(e.f, p)`).
//! A read's value ([`Heap::read`]) and the permission that frames it
//! ([`Heap::framing_perm`]) are decided by the same holders, and a consume hands
//! back the value it took, so a value always comes from the holders whose
//! permission was checked.
//!
//! How a location is held is the business of the modules behind this one:
//!
//! | module                   | what                                              |
//! |--------------------------|---------------------------------------------------|
//! | `cond`                   | conditions: [`Cube`](super::cond::Cube)s of literals |
//! | `holder`                 | what a heap holds: a body under a presence cube (single chunks, chunk sets, what was taken from a set) |
//! | `view`                   | a heap under a pc: who holds a location (`Shares`), the total, the value, taking from them |
//! | `agree`                  | a location has one value: the agreement between holders (at merge, read, and exhale) |
//! | `algebra`                | amounts (`perm`), the consume ladder, produce, join and sum of heaps, location axioms |
//! | `quantified_permissions` | chunk sets: injectivity, inverses, a set's amount at an address, containment |

use std::rc::Rc;

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{
    Demand, find_chunk_consolidated, heap_union, merge_heaps, perm_value, prove_perm_positive,
    prove_perm_write, subtract_location, union_heaps,
};
use crate::verify::heap::quantified_permissions::{
    covered_by_sum, find_covering_qp, mint_wildcard, quant_chunk_from_caps, take_from_qp,
    take_spread,
};
use crate::verify::heap::view::HolderAt;
use crate::verify::heap::{Chunk, ChunkPerm, Handover, HandoverId, Heap, LocationKind};
use crate::verify::lang::{FuncId, Symbolic};
use crate::verify::quant::ResolvedRead;
use crate::vmir::{self, Polarity, Type, Val};

/// One location: its kind and its address.
#[derive(Clone, Debug)]
pub(crate) struct Loc {
    pub(crate) kind: LocationKind,
    pub(crate) addr: egg::Id,
}

/// A value in the heap together with its certificate provenance: the recipe
/// a resource or function body walk names it by (`None` outside one).
#[derive(Clone, Debug)]
pub(crate) struct Value {
    pub(crate) id: egg::Id,
    pub(crate) recipe: Option<Val>,
}

impl Value {
    /// A value without provenance (a method body, or a fresh value).
    pub(crate) fn plain(id: egg::Id) -> Self {
        Value { id, recipe: None }
    }
}

/// A read's recipe-space context in a certificate walk (see [`Heap::read`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct ReadRecipe {
    /// The address's recipe: a value read from a chunk set is named
    /// `lookup(map, addr)` by it. `None` when the address has none.
    pub(crate) addr: Option<Val>,
}

/// What [`Heap::produce`] adds and [`Heap::consume`] takes.
pub(crate) enum Access {
    /// `acc(loc, perm)`.
    Single { loc: Loc, perm: ChunkPerm },
    /// A quantified permission, its captured temps resolved.
    Set(QuantSpec),
}

/// What a chunk set's body captures from where it was stated: the values of
/// the enclosing temps it mentions (`QuantChunks::free_temps`), and what
/// stands for each enclosing heap read ([`ResolvedRead`]). Fixed when
/// the set is stated, so a later lookup re-evaluates the body exactly as then.
#[derive(Debug, Clone)]
pub(crate) struct Captures {
    pub(crate) temps: Rc<[(usize, egg::Id)]>,
    pub(crate) reads: Rc<[ResolvedRead]>,
    /// A held set whose amount is a `wildcard`: the fresh function `W` over
    /// its locations, minted when it is produced. Its amount at location `l`
    /// is `W(l)`, positive, and the same at every lookup: one unknown amount
    /// per location, not one shared by all. `None` for a set without a
    /// wildcard, and for a demand.
    pub(crate) wildcard: Option<FuncId>,
}

impl FromIterator<(usize, egg::Id)> for Captures {
    /// The captured temps of a body whose heap reads are not resolved (yet).
    fn from_iter<I: IntoIterator<Item = (usize, egg::Id)>>(temps: I) -> Self {
        Captures {
            temps: temps.into_iter().collect(),
            reads: Rc::from([]),
            wildcard: None,
        }
    }
}

/// A quantified permission `forall xs :: c(xs) ==> acc(e(xs).f, p(xs))` ready
/// to be added or taken: the body plus what it captures.
pub(crate) struct QuantSpec {
    pub(crate) q: Rc<vmir::QuantChunks>,
    pub(crate) caps: Captures,
    /// What is still to be proven of it before it is added or taken.
    pub(crate) check: QpCheck,
}

impl QuantSpec {
    pub(crate) fn kind(&self) -> LocationKind {
        LocationKind::from_addr_type(&self.q.loc_ty).expect("QP loc_ty must be Addr")
    }
}

impl ChunkPerm {
    /// Whether `0 < self` holds under `pc`, proven per leaf (a branch-structured
    /// permission is never materialized).
    pub(crate) fn prove_positive(
        &self,
        ctx: &mut VerifyContext<'_>,
        pc: &[(egg::Id, Polarity)],
    ) -> bool {
        prove_perm_positive(ctx, self, pc)
    }
}

/// What is still to be proven of a chunk set's well-formedness before it is
/// added or taken (see `quantified_permissions`: well-definedness of the body,
/// a non-negative amount, receiver injectivity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum QpCheck {
    /// All of it, under the operation's path condition: a set stated in the
    /// program here (an inhale, an exhale, a contract or resource body walked
    /// as code).
    Body,
    /// Injectivity only: a footprint slot *consumed* at a call site. Its
    /// well-definedness and amount were proven for arbitrary arguments when
    /// the resource was checked, under its earlier conjuncts, which the consume
    /// asserts after the slots. Injectivity must still be proven here: an
    /// inverse axiom for a non-injective receiver would make that assertion
    /// vacuous.
    Injectivity,
    /// Nothing: a footprint slot *produced* from a resource, whose certificate
    /// proved all of it for arbitrary arguments under the body's earlier
    /// conjuncts, which the produce assumes. Never valid for a consume (see
    /// [`QpCheck::Injectivity`]).
    Established,
}

/// How [`Heap::consume`] treats what it finds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ConsumeMode {
    /// Take the permission away.
    Take,
    /// Only check that it is held (a heap-dependent function's precondition):
    /// the heap is returned unchanged. For a single location the demanded
    /// permission is a presence indicator: wherever it is positive, the heap
    /// must hold a positive share.
    Frame,
}

impl Heap {
    /// A permission that frames `loc`: what the holders `View::shares` finds
    /// there hold together. Enough to decide a read (its value is
    /// [`Heap::read`], decided by the same holders); for the exact amount see
    /// [`Heap::perm`].
    pub(crate) fn framing_perm(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<ChunkPerm, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        Ok(self.view(pc).shares(ctx, &loc.kind, loc.addr)?.total(ctx))
    }

    /// The value at `loc`, `None` when nothing is known to hold it. Decided by
    /// the same holders as [`Heap::framing_perm`]. `read_recipe` is the read's
    /// recipe-space context in a certificate walk (see [`ReadRecipe`]): it names
    /// a value read from a chunk set, and guards the agreement between holders
    /// that the walk records for its certificate.
    pub(crate) fn read(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
        read_recipe: Option<&ReadRecipe>,
    ) -> Result<Option<Value>, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        let shares = self.view(pc).shares(ctx, &loc.kind, loc.addr)?;
        Ok(shares.value(ctx, read_recipe))
    }

    /// The one holder that frames a read of `loc` on its own: the first of the
    /// [`Heap::read`] holders whose share is positive under `pc`, with its value
    /// there. `Ok(None)` when only several holders together frame it;
    /// `InsufficientPermission` when they do not.
    pub(super) fn covering_holder(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<Option<(HolderAt, egg::Id)>, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        let shares = self.view(pc).all_shares(ctx, &loc.kind, loc.addr)?;
        if let Some(s) = shares
            .iter()
            .find(|s| prove_perm_positive(ctx, &s.perm, pc))
        {
            return Ok(Some((s.at, s.value)));
        }
        match shares.total(ctx).prove_positive(ctx, pc) {
            true => Ok(None),
            false => Err(VerifyError::InsufficientPermission),
        }
    }

    /// `perm(loc)`: exactly the permission this heap holds at `loc`, the sum
    /// over every single chunk and chunk set that may hold it, as a term.
    pub(crate) fn perm(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<egg::Id, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        let total = self
            .view(pc)
            .all_shares(ctx, &loc.kind, loc.addr)?
            .total(ctx);
        Ok(perm_value(ctx, total))
    }

    /// Add `access` holding `value` (a location's value, or a chunk set's value
    /// map) under the path condition `pc`.
    pub(crate) fn produce(
        &self,
        ctx: &mut VerifyContext<'_>,
        access: Access,
        value: Value,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<Heap, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        match access {
            Access::Single { loc, perm } => {
                let chunk = Chunk::new_perm(loc.addr, perm, value.id).with_recipe(value.recipe);
                Ok(heap_union(ctx, self, &loc.kind, chunk, pc))
            }
            Access::Set(mut spec) => {
                let kind = spec.kind();
                if spec.q.body.contains_wc_permission() {
                    spec.caps.wildcard = Some(mint_wildcard(ctx, &spec.q));
                }
                let set = quant_chunk_from_caps(ctx, &spec.q, spec.caps, pc, value.id, spec.check)?;
                let mut set = set;
                set.recipe = value.recipe;
                Ok(self.with_qp_chunk(&kind, set))
            }
        }
    }

    /// Take `access` from this heap under `pc` (or, in [`ConsumeMode::Frame`],
    /// only check that it is held). Returns the heap after and the value that
    /// was held: a location's value, or the covering chunk set's value map.
    pub(crate) fn consume(
        &self,
        ctx: &mut VerifyContext<'_>,
        access: Access,
        mode: ConsumeMode,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<(Heap, Value), VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        match access {
            Access::Single { loc, perm } => match mode {
                ConsumeMode::Take => {
                    // Before a holder gives up its share, its value is tied to
                    // every other holder there (a chunk set among them: two
                    // single chunks already agree where they were merged).
                    if !self.qp_chunks_of(&loc.kind).is_empty() {
                        self.view(pc)
                            .all_shares(ctx, &loc.kind, loc.addr)?
                            .assume_agreement(ctx);
                    }
                    self.take_single(ctx, &loc, perm, pc)
                }
                ConsumeMode::Frame => self.frame_single(ctx, &loc, perm, pc),
            },
            Access::Set(spec) => {
                assert_ne!(
                    spec.check,
                    QpCheck::Established,
                    "a consumed chunk set must have its injectivity checked"
                );
                // A wildcard demand asks for a positive share
                // (`prove_framed_aligned`); a take then debits a fresh amount
                // below what the covering set held (`Captures::wildcard`).
                let wild = spec.q.body.contains_wc_permission();
                let kind = spec.kind();
                // The exhaled set's values are the covering set's (an exhale
                // changes no value), known only once one is found below; until
                // then it carries a placeholder map that is never read.
                let placeholder =
                    ctx.fresh_symbolic_value(Type::ValueMap(Box::new(spec.q.loc_ty.clone())));
                let exhaled =
                    quant_chunk_from_caps(ctx, &spec.q, spec.caps, pc, placeholder, spec.check)?;
                let Some(i) = find_covering_qp(ctx, self, &kind, &exhaled)? else {
                    // No single set covers it: try all holders together. The
                    // exhaled set's values are then not tied to theirs yet (a
                    // fresh map: sound, but its snapshot says nothing).
                    if !covered_by_sum(ctx, self, &kind, &exhaled)? {
                        return Err(VerifyError::InsufficientPermission);
                    }
                    if wild && mode == ConsumeMode::Take {
                        return Err(VerifyError::Unimplemented(
                            "wildcard exhale of a quantified permission across several holders",
                        ));
                    }
                    let heap = match mode {
                        ConsumeMode::Take => {
                            // Every set may give: each keeps its index in the
                            // result, which gets the handovers. Its debits
                            // record the heap before them.
                            let n = self.qp_chunks_of(&kind).len();
                            let out = take_spread(ctx, self, &kind, exhaled)?;
                            self.hand_over(ctx, &kind, 0..n, pc, out)?
                        }
                        ConsumeMode::Frame => self.clone(),
                    };
                    return Ok((heap, Value::plain(placeholder)));
                };
                let held = &self.qp_chunks_of(&kind)[i];
                let value = Value {
                    id: held.value_map,
                    recipe: held.recipe.clone(),
                };
                let heap = match mode {
                    ConsumeMode::Take => {
                        // Before the take: it may drop set `i` and shift the
                        // indices of the others.
                        let before = self.hand_over(ctx, &kind, i..i + 1, pc, self.clone())?;
                        let mut exhaled = exhaled;
                        if wild {
                            // What leaves: `D(l)`, fresh and positive, bounded
                            // by what set `i` held where it is read (`qp_perm_at`).
                            exhaled.caps.wildcard = Some(mint_wildcard(ctx, &exhaled.q));
                        }
                        take_from_qp(ctx, &before, &kind, i, exhaled)
                    }
                    ConsumeMode::Frame => self.clone(),
                };
                Ok((heap, value))
            }
        }
    }

    /// `loc := value`: needs write permission at `loc`, summed over its
    /// holders, and gives every holder the new value (a single chunk that only
    /// may alias `loc` takes it where it does). Permissions are unchanged.
    pub(crate) fn write(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        value: egg::Id,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<Heap, VerifyError> {
        let pc = &self.effective_pc(ctx, pc)[..];
        // The common case: one chunk sits exactly at `loc` with write permission.
        if let Some(mut c) = self.chunk(&loc.kind, loc.addr).cloned() {
            let perm = c.gated_perm(ctx);
            if prove_perm_write(ctx, &perm, &loc.kind.bound, pc) {
                c.set_value(value);
                return Ok(self.with_chunk(&loc.kind, c));
            }
        }
        let holders = self.view(pc).all_shares(ctx, &loc.kind, loc.addr)?;
        let total = holders.total(ctx);
        if !prove_perm_write(ctx, &total, &loc.kind.bound, pc) {
            return Err(VerifyError::InsufficientPermission);
        }
        let mut heap = self.clone();
        for h in holders.iter() {
            match h.at {
                HolderAt::Chunk(at) => {
                    let mut c = self
                        .chunk(&loc.kind, at)
                        .expect("a share's chunk is held")
                        .clone();
                    let new = if ctx.egraph.find(c.addr) == ctx.egraph.find(loc.addr) {
                        value
                    } else {
                        let eq = expr!(ctx, { c.addr } == { loc.addr });
                        ctx.add(Symbolic::Ite([eq, value, c.stored_value()]))
                    };
                    c.set_value(new);
                    heap = heap.with_chunk(&loc.kind, c);
                }
                HolderAt::Set(i) => {
                    // Outside the set's domain its map is never read, so the
                    // store is unconditional.
                    let mut set = heap.qp_chunks_of(&loc.kind)[i].clone();
                    let store = ctx.alloc.value_map(&loc.kind).store;
                    let map = ctx.add(Symbolic::FuncApp(
                        store,
                        Box::new([]),
                        Box::new([set.value_map, loc.addr, value]),
                    ));
                    set.value_map = map;
                    // The old map's recipe no longer names the new map.
                    set.recipe = None;
                    heap = heap.with_qp_chunk_replaced(&loc.kind, i, set);
                }
            }
        }
        Ok(heap)
    }

    /// The heap after a branch: `then` where `cond` holds, `els` elsewhere.
    /// Chunk sets are kept from both arms, each under its arm's literal.
    /// `region` is the join block's cube. The arms are taken as built, not
    /// rehomed: appending each arm's literal is the rehoming.
    pub(crate) fn merge(
        ctx: &mut VerifyContext<'_>,
        cond: egg::Id,
        then: &Heap,
        els: &Heap,
        region: &[(egg::Id, Polarity)],
    ) -> Heap {
        merge_heaps(ctx, cond, then, els, region)
    }

    /// The sum of two heaps held at once.
    ///
    /// A chunk set given back to the set it was taken from cancels that take
    /// (`algebra::combine::give_back`); every other set is kept as it is.
    pub(crate) fn union(
        &self,
        ctx: &mut VerifyContext<'_>,
        other: &Heap,
        pc: &[(egg::Id, Polarity)],
    ) -> Heap {
        let pc = &self.effective_pc(ctx, pc)[..];
        union_heaps(ctx, self, other, pc)
    }

    /// For an exhale of a chunk set that takes from this heap's sets `givers`
    /// (of `kind`) under `pc`: what they hold is tied to every other holder
    /// before any of it is forgotten. Returns `into` with the handovers added
    /// to its sets, which must sit at this heap's indices.
    ///
    /// - Each single chunk of the kind agrees, at its own address, with every
    ///   holder there ([`Shares::assume_agreement`]): ground, stated now.
    /// - Every other set gets a [`Handover`] from each giver: quantified, so
    ///   recorded and assumed per address where the set is read.
    ///
    /// Both use the holders as they are now, before the take, so what any of
    /// them gave up earlier is already accounted for.
    fn hand_over(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        givers: std::ops::Range<usize>,
        pc: &[(egg::Id, Polarity)],
        into: Heap,
    ) -> Result<Heap, VerifyError> {
        for c in self.chunks_of(kind) {
            self.view(pc)
                .all_shares(ctx, kind, c.addr)?
                .assume_agreement(ctx);
        }
        let sets = self.qp_chunks_of(kind).to_vec();
        assert_eq!(
            into.qp_chunks_of(kind).len(),
            sets.len(),
            "handovers are added to a heap whose sets sit at this heap's indices"
        );
        let mut out = into;
        for (j, kept) in sets.iter().enumerate() {
            let new: Vec<Handover> = givers
                .clone()
                .filter(|&g| g != j)
                .map(|g| Handover {
                    id: HandoverId::fresh(ctx),
                    from: sets[g].clone(),
                    kept: kept.clone(),
                    pc: pc.into(),
                })
                .collect();
            if new.is_empty() {
                continue;
            }
            let mut set = out.qp_chunks_of(kind)[j].clone();
            set.handovers = set.handovers.iter().cloned().chain(new).collect();
            out = out.with_qp_chunk_replaced(kind, j, set);
        }
        Ok(out)
    }

    /// [`Heap::consume`] of one location: the permission is taken (from the
    /// single chunks, else from the chunk sets) and the value handed back is
    /// the one the take resolved from the holders it took from.
    fn take_single(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        perm: ChunkPerm,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<(Heap, Value), VerifyError> {
        let demand = if perm.has_wild() {
            Demand::Wildcard
        } else {
            Demand::Concrete
        };
        subtract_location(ctx, self, &loc.kind, loc.addr, perm, pc, demand)
    }

    /// [`Heap::consume`] of one location in [`ConsumeMode::Frame`]: `presence`
    /// says where the location is demanded (`0 < presence`); there the heap must
    /// hold a positive share. Nothing is taken.
    fn frame_single(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        presence: ChunkPerm,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<(Heap, Value), VerifyError> {
        let pp = presence.to_id(ctx);
        let guard = expr!(ctx, (0 / 1) < r { pp });
        let mut under_guard = pc.to_vec();
        under_guard.push((guard, Polarity::Positive));
        // The chunk at the address on ground, when there is one: `guard ⇒ 0 <
        // held`, proven per leaf (the held `Select` is never materialized).
        let (_, existing) = find_chunk_consolidated(ctx, self, &loc.kind, loc.addr, pc, true);
        let framed = match existing {
            Some(c) => {
                let held = c.gated_perm(ctx);
                prove_perm_positive(ctx, &held, &under_guard)
            }
            // Nothing held there on ground: fine where it is not demanded.
            None => {
                let not_guard = expr!(ctx, not { guard });
                ctx.prove_under_pc(not_guard, pc)
            }
        };
        // Last resort: every holder that may sit at the address, chunk sets
        // included. Reached only by a check that would otherwise fail.
        let framed = framed || {
            let holders = self.view(pc).all_shares(ctx, &loc.kind, loc.addr)?;
            let total = holders.total(ctx);
            !holders.is_empty() && prove_perm_positive(ctx, &total, &under_guard)
        };
        if !framed {
            return Err(VerifyError::InsufficientPermission);
        }
        // The value of the location as its holders resolve it, as for a read: a
        // chunk that matched on ground but holds nothing here does not name it.
        let value = self
            .view(pc)
            .shares(ctx, &loc.kind, loc.addr)?
            .resolve(ctx, None);
        Ok((self.clone(), value))
    }
}
