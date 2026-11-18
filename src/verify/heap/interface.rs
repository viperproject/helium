//! The heap as the body walks (`declaration`, `eval`) see it.
//!
//! Everything a walk does to a heap goes through the methods here:
//!
//! | operation                 | method                 |
//! |---------------------------|------------------------|
//! | read a location           | [`Heap::read`], [`Heap::framing_perm`] |
//! | `perm(loc)`               | [`Heap::perm`]         |
//! | add permission            | [`Heap::produce`]      |
//! | take permission / frame   | [`Heap::consume`]      |
//! | field write               | [`Heap::write`]        |
//! | branch join, heap sum     | [`Heap::merge`], [`Heap::union`] |
//!
//! [`Heap::produce`] and [`Heap::consume`] take an [`Access`]: one location
//! (`acc(e.f, p)`) or a quantified chunk set (`forall xs :: c ==> acc(e.f, p)`).
//! A read's value ([`Heap::read`]) and the permission that frames it
//! ([`Heap::framing_perm`]) are decided by the same holders, and a consume hands
//! back the value it took, so a value always comes from the holders whose
//! permission was checked.
//!
//! How a location is held (single chunks, chunk sets, aliasing, the value
//! agreement between holders) is the business of the modules behind this one:
//! `algebra` for single chunks, `quantified_permissions` for chunk sets and
//! `holders` for looking at both together.

use std::rc::Rc;

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{
    Demand, chunk_under_pc, find_chunk_consolidated, heap_union, merge_heaps, perm_value,
    prove_perm_positive, prove_perm_write, summarize_perm_at, union_heaps,
};
use crate::verify::heap::holders::{HolderAt, holders_at, pick_value, subtract_location, sum_perms};
use crate::verify::heap::quantified_permissions::{
    check_qp_nonnegative, covered_by_sum, find_covering_qp, quant_chunk_from_caps,
    quant_chunk_injective, take_spread,
    take_from_qp,
};
use crate::verify::heap::{Chunk, ChunkPerm, Heap, LocationKind, QuantChunk};
use crate::verify::lang::Symbolic;
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
    /// The read's path condition in recipe space, outermost-first. It guards
    /// the agreement facts the read records (see
    /// [`crate::verify::cert::RecipeBuilder::record_side_fact`]). `None` when
    /// no recipe-space pc can be recovered; then nothing is recorded.
    pub(crate) pc: Option<Vec<(Val, Polarity)>>,
}

/// What [`Heap::produce`] adds and [`Heap::consume`] takes.
pub(crate) enum Access {
    /// `acc(loc, perm)`.
    Single { loc: Loc, perm: ChunkPerm },
    /// A quantified permission, its captured temps resolved.
    Set(QuantSpec),
}

/// A quantified permission `forall xs :: c(xs) ==> acc(e(xs).f, p(xs))` ready
/// to be added or taken: the body plus the values of the enclosing temps it
/// captures.
pub(crate) struct QuantSpec {
    pub(crate) q: Rc<vmir::QuantChunks>,
    pub(crate) caps: Rc<[(usize, egg::Id)]>,
    pub(crate) injectivity: Injectivity,
}

impl QuantSpec {
    pub(crate) fn kind(&self) -> LocationKind {
        LocationKind::from_addr_type(&self.q.loc_ty).expect("QP loc_ty must be Addr")
    }

    /// Whether the amount is never negative where the set holds:
    /// `forall xs :: c(xs) ==> not(p(xs) < 0)` under `pc`.
    pub(crate) fn prove_nonnegative(
        &self,
        ctx: &mut VerifyContext<'_>,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<bool, VerifyError> {
        check_qp_nonnegative(ctx, &self.q, &self.caps, pc)
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

/// Whether a chunk set's receiver injectivity still has to be proven.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Injectivity {
    /// Prove it under the operation's path condition.
    Check,
    /// Already proven: a footprint slot *produced* from a resource, whose
    /// certificate proved it for arbitrary arguments under the body's earlier
    /// conjuncts, which the produce assumes. Never valid for a consume: there
    /// the conjuncts are asserted only after the slots, and an inverse axiom for
    /// a non-injective receiver would make that assertion vacuous.
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
    /// A permission that frames `loc`: the matched chunk's when one single
    /// chunk sits there under the pc and no chunk set could also hold it,
    /// otherwise the sum over every holder. Enough to decide a read (its value
    /// is [`Heap::read`], decided by the same holders); for the exact amount see
    /// [`Heap::perm`].
    pub(crate) fn framing_perm(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<ChunkPerm, VerifyError> {
        // fast route
        if self.qp_chunks_of(&loc.kind).is_empty() {
            if let Some(c) = chunk_under_pc(ctx, self.chunks_of(&loc.kind), loc.addr, pc).cloned() {
                return Ok(c.gated_perm(ctx));
            }
            // Nothing matches under the pc: every chunk that may alias it.
            let chunks = self.chunks_of(&loc.kind).to_vec();
            return Ok(summarize_perm_at(ctx, &chunks, loc.addr, pc).0);
        }
        let holders = holders_at(ctx, self, &loc.kind, loc.addr, pc)?;
        Ok(sum_perms(ctx, &holders))
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
        if self.qp_chunks_of(&loc.kind).is_empty() {
            let chunk = chunk_under_pc(ctx, self.chunks_of(&loc.kind), loc.addr, pc);
            return Ok(chunk.map(|c| Value {
                id: c.value,
                recipe: c.recipe.clone(),
            }));
        }
        let holders = holders_at(ctx, self, &loc.kind, loc.addr, pc)?;
        Ok(pick_value(ctx, &loc.kind, &holders, pc, read_recipe))
    }

    /// `perm(loc)`: exactly the permission this heap holds at `loc`, the sum
    /// over every single chunk and chunk set that may hold it, as a term.
    pub(crate) fn perm(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<egg::Id, VerifyError> {
        let holders = holders_at(ctx, self, &loc.kind, loc.addr, pc)?;
        let total = sum_perms(ctx, &holders);
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
        match access {
            Access::Single { loc, perm } => {
                let chunk = Chunk::new_perm(loc.addr, perm, value.id).with_recipe(value.recipe);
                Ok(heap_union(ctx, self, &loc.kind, chunk, pc))
            }
            Access::Set(spec) => {
                let kind = spec.kind();
                let set = match spec.injectivity {
                    Injectivity::Check => {
                        quant_chunk_from_caps(ctx, &spec.q, spec.caps, pc, value.id)?
                    }
                    Injectivity::Established => {
                        quant_chunk_injective(ctx, &spec.q, spec.caps, pc, value.id)?
                    }
                };
                Ok(self.with_qp_chunk(
                    &kind,
                    QuantChunk {
                        recipe: value.recipe,
                        ..set
                    },
                ))
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
        match access {
            Access::Single { loc, perm } => match mode {
                ConsumeMode::Take => self.take_single(ctx, &loc, perm, pc),
                ConsumeMode::Frame => self.frame_single(ctx, &loc, perm, pc),
            },
            Access::Set(spec) => {
                assert_eq!(
                    spec.injectivity,
                    Injectivity::Check,
                    "a consumed chunk set must have its injectivity checked"
                );
                if spec.q.body.contains_wc_permission() {
                    return Err(VerifyError::Unimplemented(
                        "exhaling a wildcard quantified permission",
                    ));
                }
                let kind = spec.kind();
                // The exhaled set's values are the covering set's (an exhale
                // changes no value), known only once one is found below; until
                // then it carries a placeholder map that is never read.
                let placeholder =
                    ctx.fresh_symbolic_value(Type::ValueMap(Box::new(spec.q.loc_ty.clone())));
                let exhaled = quant_chunk_from_caps(ctx, &spec.q, spec.caps, pc, placeholder)?;
                let Some(i) = find_covering_qp(ctx, self, &kind, &exhaled)? else {
                    // No single set covers it: try all holders together. The
                    // exhaled set's values are then not tied to theirs yet (a
                    // fresh map: sound, but its snapshot says nothing).
                    if !covered_by_sum(ctx, self, &kind, &exhaled)? {
                        return Err(VerifyError::InsufficientPermission);
                    }
                    let heap = match mode {
                        ConsumeMode::Take => take_spread(ctx, self, &kind, exhaled)?,
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
                    ConsumeMode::Take => take_from_qp(self, &kind, i, exhaled),
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
        // The common case: one chunk sits exactly at `loc` with write permission.
        if let Some(c) = self.chunk(&loc.kind, loc.addr).cloned() {
            let perm = c.gated_perm(ctx);
            if prove_perm_write(ctx, &perm, &loc.kind.bound, pc) {
                return Ok(self.with_chunk(&loc.kind, Chunk { value, ..c }));
            }
        }
        let holders = holders_at(ctx, self, &loc.kind, loc.addr, pc)?;
        let total = sum_perms(ctx, &holders);
        if !prove_perm_write(ctx, &total, &loc.kind.bound, pc) {
            return Err(VerifyError::InsufficientPermission);
        }
        let mut heap = self.clone();
        for h in &holders {
            match h.at {
                HolderAt::Chunk(i) => {
                    let c = self.chunks_of(&loc.kind)[i].clone();
                    let new = if ctx.egraph.find(c.addr) == ctx.egraph.find(loc.addr) {
                        value
                    } else {
                        let eq = expr!(ctx, { c.addr } == { loc.addr });
                        ctx.add(Symbolic::Ite([eq, value, c.value]))
                    };
                    heap = heap.with_chunk(&loc.kind, Chunk { value: new, ..c });
                }
                HolderAt::Set(i) => {
                    // Outside the set's domain its map is never read, so the
                    // store is unconditional.
                    let set = heap.qp_chunks_of(&loc.kind)[i].clone();
                    let store = ctx.alloc.value_map(&loc.kind).store;
                    let map = ctx.add(Symbolic::FuncApp(
                        store,
                        Box::new([]),
                        Box::new([set.value_map, loc.addr, value]),
                    ));
                    let set = QuantChunk {
                        value_map: map,
                        // The old map's recipe no longer names the new map.
                        recipe: None,
                        ..set
                    };
                    heap = heap.with_qp_chunk_replaced(&loc.kind, i, set);
                }
            }
        }
        Ok(heap)
    }

    /// The heap after a branch: `then` where `cond` holds, `els` elsewhere.
    ///
    /// Chunk sets are not joined yet: a heap holding one panics.
    pub(crate) fn merge(ctx: &mut VerifyContext<'_>, cond: egg::Id, then: &Heap, els: &Heap) -> Heap {
        assert!(
            !then.has_qp_chunks() && !els.has_qp_chunks(),
            "joining heaps that hold quantified permissions is not supported"
        );
        merge_heaps(ctx, cond, then, els)
    }

    /// The sum of two heaps held at once.
    ///
    /// Chunk sets are not summed yet: a heap holding one panics.
    pub(crate) fn union(&self, ctx: &mut VerifyContext<'_>, other: &Heap, pc: &[(egg::Id, Polarity)]) -> Heap {
        assert!(
            !self.has_qp_chunks() && !other.has_qp_chunks(),
            "summing heaps that hold quantified permissions is not supported"
        );
        union_heaps(ctx, self, other, pc)
    }

    /// [`Heap::consume`] of one location: the value is read from the holders
    /// first, then the permission is taken (from the single chunks, else from
    /// one chunk set).
    fn take_single(
        &self,
        ctx: &mut VerifyContext<'_>,
        loc: &Loc,
        perm: ChunkPerm,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<(Heap, Value), VerifyError> {
        // Single chunks only: the chunk at the address (canonically), else a
        // fresh value the consume below binds to whatever it takes. With chunk
        // sets, the value their holders agree on.
        let value = if self.qp_chunks_of(&loc.kind).is_empty() {
            self.chunk_canon(ctx, &loc.kind, loc.addr).map(|c| Value {
                id: c.value,
                recipe: c.recipe.clone(),
            })
        } else {
            self.read(ctx, loc, pc, None)?
        };
        let value = value
            .unwrap_or_else(|| Value::plain(ctx.fresh_symbolic_value(loc.kind.value.clone())));
        let demand = if perm.has_wild() {
            Demand::Wildcard
        } else {
            Demand::Concrete
        };
        let chunk = Chunk::new_perm(loc.addr, perm, value.id).with_recipe(value.recipe.clone());
        let heap = subtract_location(ctx, self, &loc.kind, chunk, pc, demand)?;
        Ok((heap, value))
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
        let mut value = existing.as_ref().map(|c| Value {
            id: c.value,
            recipe: c.recipe.clone(),
        });
        if value.is_none() && !self.qp_chunks_of(&loc.kind).is_empty() {
            value = self.read(ctx, loc, pc, None)?;
        }
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
            let holders = holders_at(ctx, self, &loc.kind, loc.addr, pc)?;
            let total = sum_perms(ctx, &holders);
            !holders.is_empty() && prove_perm_positive(ctx, &total, &under_guard)
        };
        if !framed {
            return Err(VerifyError::InsufficientPermission);
        }
        let value =
            value.unwrap_or_else(|| Value::plain(ctx.fresh_symbolic_value(loc.kind.value.clone())));
        Ok((self.clone(), value))
    }
}
