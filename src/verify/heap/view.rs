//! A heap as seen from one state: [`View`] binds a heap to the path condition
//! of the instruction that reads it, and answers who holds a location there
//! ([`Shares`]) and what follows from that: the total permission, the one value
//! the holders agree on.
//!
//! Every read of a location goes through [`View::shares`], single chunks and
//! chunk sets alike. Each [`Share`] is already gated by everything that
//! conditions it (the holder's presence, the address gate, a set's domain and
//! what was taken from it), so a share's amount can be summed and proven about
//! as it stands.
//!
//! Every take of a single location goes through [`Shares::take`]: the consume
//! ladder only decides *which* shares to take from (the chunk on ground, the
//! chunks the pc makes alias, every chunk that may alias, the chunk sets), and
//! the proof and the debit are the same for all of them.

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::agree::assume_agree;
use crate::verify::heap::algebra::{
    chunk_share_at, chunk_under_pc, gate_amount_by_pc, perm_add, perm_sub, perm_sub_aligned,
    prove_perm_positive, prove_sufficient, prove_sufficient_aligned,
};
use crate::verify::heap::cond::{Cube, Lit};
use crate::verify::heap::interface::{ReadRecipe, Value};
use crate::verify::heap::quantified_permissions::qp_perm_at;
use crate::verify::heap::{Chunk, ChunkPerm, Heap, LocationKind, QuantChunk, Restored, Taken};
use crate::verify::lang::Symbolic;
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, Literal, Polarity, Type, Val};

/// A heap bound to the path condition of the state it is read in. Built by
/// [`Heap::view`]; every question about who holds a location is asked of one,
/// so the pc an answer is valid under travels with the heap it is about.
pub(super) struct View<'a> {
    heap: &'a Heap,
    pc: &'a [Lit],
}

impl Heap {
    /// This heap as read under `pc`.
    pub(super) fn view<'a>(&'a self, pc: &'a [Lit]) -> View<'a> {
        View { heap: self, pc }
    }
}

/// Where a [`Share`]'s holder sits in its heap: a single chunk by its address
/// (its identity in the group), a chunk set by its index in `qp_chunks_of(kind)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HolderAt {
    Chunk(egg::Id),
    Set(usize),
}

/// What one holder holds at an address (see [`View::shares`]).
#[derive(Clone)]
pub(super) struct Share {
    /// Its amount at the address, gated by everything that conditions it: what
    /// it contributes to the location's total.
    pub(super) perm: ChunkPerm,
    /// What it can give up where it sits at the address: its amount as it
    /// counts under the pc, without the address gate (`when` gates the debit
    /// instead).
    hold: ChunkPerm,
    /// The condition under which this holder sits at the address: empty when it
    /// does outright, else an address equality, its presence, or the pc it
    /// aliases under. A debit from it and an agreement with it hold only there.
    when: Cube,
    pub(super) value: egg::Id,
    source: ShareSource,
    pub(super) at: HolderAt,
}

/// The holder behind a [`Share`], with its recipe provenance (certificate walks).
#[derive(Clone)]
enum ShareSource {
    /// A single chunk, as it is debited; the recipe is its value's own.
    Chunk(Chunk),
    /// A quantified chunk set; the recipe is its map's (the value is
    /// `lookup(map, addr)`).
    Set(Option<Val>),
}

/// Everything a heap holds at one address, in heap order (single chunks first).
pub(super) struct Shares {
    kind: LocationKind,
    shares: Vec<Share>,
    /// The shares are exactly one single chunk that *is* the location under the
    /// pc, and no chunk set of its kind is held (other single chunks of the
    /// kind may be: a location that one chunk holds outright is not held by
    /// another, see the non-aliasing axiom). Its value is the location's, with
    /// no agreement to state and no positivity to prove.
    sole: bool,
}

impl<'a> View<'a> {
    /// Who may hold `addr` (of `kind`) here.
    ///
    /// When the kind has no chunk sets and one single chunk sits at the address
    /// under the pc ([`chunk_under_pc`]: on ground, else in a probe that assumes
    /// the pc), that chunk alone: it is the location. Otherwise every single
    /// chunk that may sit there ([`chunk_share_at`], its aliasing gate folded
    /// in) and every chunk set that can name it ([`qp_perm_at`], its presence
    /// folded in, what was exhaled from it subtracted).
    pub(super) fn shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Result<Shares, VerifyError> {
        let heap = self.heap;
        let chunks = heap.chunks_of(kind);
        if heap.qp_chunks_of(kind).is_empty()
            && let Some(c) = chunk_under_pc(ctx, chunks, addr, self.pc)
        {
            // On ground it is the location; otherwise only under the pc.
            let when = if ctx.egraph.find(c.addr) == ctx.egraph.find(addr) {
                Cube::empty()
            } else {
                Cube::from(self.pc)
            };
            let perm = c.gated_perm(ctx);
            let hold = c.perm_under(ctx, self.pc);
            return Ok(Shares {
                kind: kind.clone(),
                shares: vec![chunk_share(c, perm, hold, when)],
                sole: true,
            });
        }
        self.all_shares(ctx, kind, addr)
    }

    /// The **Σ-ite summary** of a location: the permission held at `addr`, summed over
    /// *every* chunk of the group, each gated by whether it sits at that address.
    ///
    /// ```text
    /// perm(addr) = Σ_c  ite(c.addr == addr, guard(c) ? c.perm : 0, 0)
    /// ```
    ///
    /// Two gates compose per chunk, and they are different things:
    /// - the **presence guard** (`guard ? perm : 0`, `Holder::gate`) — the
    ///   guard-hoisted merge stores a conditionally-held chunk as a flat cube over a
    ///   guard-free amount, so reading `Chunk.perm` raw reports a conditional footprint
    ///   as fully held;
    /// - the **address gate** (`c.addr == addr`) — whether this chunk is *at* the
    ///   location at all.
    ///
    /// The address gate is resolved by the engine, not by Rust-side `find` at walk
    /// time. That distinction is the point: an inhaled `a == b` merges the two address
    /// classes only once the `eq-true-union` rewrite fires during saturation, so a
    /// walk-time `find` reads the pre-merge graph and silently drops the aliasing chunk.
    /// Leaving the condition in the term lets whatever saturation the obligation runs
    /// settle it.
    ///
    /// Cheap in the common case, because the gate is decided structurally where it can be:
    /// - **ground-equal address** — ungated, contributing exactly the term the old
    ///   ground-match sum built, so the fast path mints no extra nodes;
    /// - **provably distinct** — skipped outright, minting nothing. This prune is what
    ///   keeps the sum from growing the `ite` tower recorded in
    ///   `project_perm_collapse_root_cause`;
    /// - **equal under the chunk's own presence cube** — ungated too, see below;
    /// - **otherwise** — gated, and the cube it was gated by is returned alongside, so a
    ///   consume can gate its debit by the very same condition.
    ///
    /// # The address gate is decided under the chunk's presence cube
    ///
    /// A chunk's address means nothing where the chunk is not held, so the question is
    /// never "is `c.addr == addr` valid?" but "is it valid **wherever `c` exists**?" —
    /// i.e. under `pc ∧ c.guard`. Deciding it on ground instead is what left the `&mut`
    /// reborrow shape unprovable: the arm mints a fresh reference whose pointee equals
    /// the parameter's, so the two `p_C(..)` addresses coincide — but only through a
    /// chain released under the arm's own reach literal, which is exactly the literal
    /// the chunk's guard carries. Ground `find` cannot see it; a probe under the guard
    /// settles it immediately.
    ///
    /// When the probe succeeds the address gate is dropped entirely and the returned
    /// cube is the chunk's **guard**: the contribution is conditional on presence, not
    /// on an equality, and `held` already encodes that condition. One `prove_under_pc`,
    /// paid only by a chunk that missed on ground and was not disproven — the same
    /// budget the disproof check above already spends.
    ///
    /// The shares stay **structural**, so their total ([`Shares::total`]) is too: a
    /// conditionally-present chunk contributes a branch on its presence literal, and
    /// `prove_sufficient` discharges the sum per leaf with that literal in the pc.
    ///
    /// One share per chunk ([`chunk_share_at`]), in group order; its `when` is
    /// the cube its contribution is conditional on, so a take gates its debit by
    /// the very same condition.
    pub(super) fn point_shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Shares {
        let mut shares = Vec::new();
        for c in self.heap.chunks_of(kind) {
            if let Some((perm, when)) = chunk_share_at(ctx, c, addr, self.pc) {
                let hold = c.perm_under(ctx, self.pc);
                shares.push(chunk_share(c, perm, hold, when));
            }
        }
        Shares {
            kind: kind.clone(),
            shares,
            sole: false,
        }
    }

    /// The chunk set `i` of `kind` alone, as a holder of `addr`.
    pub(super) fn set_shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        i: usize,
        addr: egg::Id,
    ) -> Result<Shares, VerifyError> {
        let set = &self.heap.qp_chunks_of(kind)[i];
        Ok(Shares {
            kind: kind.clone(),
            shares: vec![set_share(ctx, kind, set, i, addr)?],
            sole: false,
        })
    }

    /// Every holder that may hold `addr` (of `kind`), without the sole-chunk
    /// resolution [`View::shares`] tries first.
    pub(super) fn all_shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Result<Shares, VerifyError> {
        let mut shares = self.point_shares(ctx, kind, addr);
        for (i, set) in self.heap.qp_chunks_of(kind).iter().enumerate() {
            shares.shares.push(set_share(ctx, kind, set, i, addr)?);
        }
        Ok(shares)
    }
}

fn chunk_share(c: &Chunk, perm: ChunkPerm, hold: ChunkPerm, when: Cube) -> Share {
    Share {
        perm,
        hold,
        when,
        value: c.value,
        source: ShareSource::Chunk(c.clone()),
        at: HolderAt::Chunk(c.addr),
    }
}

/// What the chunk set `set` (number `i`) holds at `addr`: its amount there and
/// `lookup(map, addr)`. Assumes the definitions of its restored maps there.
fn set_share(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    set: &QuantChunk,
    i: usize,
    addr: egg::Id,
) -> Result<Share, VerifyError> {
    let lookup = ctx.alloc.value_map(kind).lookup;
    let value = ctx.add(Symbolic::FuncApp(
        lookup,
        Box::new([]),
        Box::new([set.value_map, addr]),
    ));
    for r in set.restored.iter() {
        define_restored(ctx, kind, r, addr)?;
    }
    let perm = qp_perm_at(ctx, set, addr)?;
    Ok(Share {
        hold: perm.clone(),
        perm,
        when: Cube::empty(),
        value,
        source: ShareSource::Set(set.recipe.clone()),
        at: HolderAt::Set(i),
    })
}

/// Assume what the restored map `rest.map` reads at `addr` (see [`Restored`]).
fn define_restored(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    rest: &Restored,
    addr: egg::Id,
) -> Result<(), VerifyError> {
    let region = qp_perm_at(ctx, &rest.region, addr)?.to_id(ctx);
    let kept = qp_perm_at(ctx, &rest.kept, addr)?.to_id(ctx);
    let inside = expr!(ctx, (0 / 1) < r { region });
    let lookup = ctx.alloc.value_map(kind).lookup;
    let [read, new, old] = [rest.map, rest.new, rest.old]
        .map(|m| ctx.add(Symbolic::FuncApp(lookup, Box::new([]), Box::new([m, addr]))));
    let value = ctx.add(Symbolic::Ite([inside, new, old]));
    let fact = expr!(ctx, { read } == { value });
    ctx.assume_guarded(fact, std::iter::empty());
    // Where the set given back to held a share too, the values did not change.
    assume_agree(ctx, new, old, &[kept, region], &[]);
    Ok(())
}

impl Shares {
    /// The single chunk `c`, the location itself on ground (normalized for the
    /// consume, see `Chunk::normalized_for`).
    pub(super) fn exact(
        ctx: &VerifyContext<'_>,
        kind: &LocationKind,
        c: &Chunk,
        pc: &[Lit],
    ) -> Shares {
        let perm = c.perm_under(ctx, pc);
        Shares {
            kind: kind.clone(),
            shares: vec![chunk_share(c, perm.clone(), perm, Cube::empty())],
            sole: false,
        }
    }

    /// The chunks `members`, each the location only where `pc` holds (invariant
    /// 7: aliasing a path condition implies). Each counts what it holds under
    /// `pc`; the pc gate on the amount is left to the proof, which runs under it.
    pub(super) fn pc_aliased(
        ctx: &VerifyContext<'_>,
        kind: &LocationKind,
        members: &[Chunk],
        pc: &[Lit],
    ) -> Shares {
        let when = Cube::from(pc);
        let shares = members
            .iter()
            .map(|c| {
                let perm = c.perm_under(ctx, pc);
                chunk_share(c, perm.clone(), perm, when.clone())
            })
            .collect();
        Shares {
            kind: kind.clone(),
            shares,
            sole: false,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.shares.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    /// Take `needed` at `addr` from these holders in `out`, the demand's value
    /// being `value`; `Err(InsufficientPermission)` when they are not shown to
    /// hold enough under `pc`, and then nothing has been changed.
    ///
    /// - **One holder, outright at the address** — `held ≥ needed` is proven
    ///   aligned with both amounts' branch structure, and exactly `needed` is
    ///   taken. A single chunk's remainder is aligned against the demand's arms
    ///   too, so a guarded consume cancels per arm (`c ? 0 : 1/1`) rather than
    ///   leaving `1/1 - (c ? 1/1 : 0/1)` in every leaf. Its value is unioned with
    ///   the demand's outright: the demand's value is minted for this consume (or
    ///   read from these very holders) and carries no prior meaning, so the union
    ///   constrains only it. It is not implied by the sufficiency proof
    ///   (`held ≥ needed` permits `needed == 0`, so a conditional exhale binds the
    ///   value off-path); probed with complementary conditional exhales, both
    ///   correctly rejected.
    /// - **Otherwise** — the total is proven to cover `needed`, per leaf (a holder
    ///   present on one arm only contributes a branch on its literal, and each leaf
    ///   is discharged with it in the pc: flattening to one goal was the whole of
    ///   the `&mut`-reborrow incompleteness). The debit is then **distributed**
    ///   greedily, in order: each holder gives `min(it holds, still needed)` gated
    ///   by its `when`, and the demand is retired by that gated amount, so a
    ///   holder that does not sit at the address gives up nothing and retires
    ///   nothing. No holder is merged — where `when` fails the locations are
    ///   distinct. Distributing (rather than parking the debit on one holder)
    ///   drives every holder to its true remainder, so a later consume needs no
    ///   second alias probe. A single chunk's value agrees with the demand's only
    ///   under its `when` and where its share there is positive.
    ///
    /// A single chunk emptied on every branch (const-fold, not a proof: heap
    /// hygiene must stay O(1), and keeping an empty chunk is sound) is dropped;
    /// otherwise it keeps its presence, the remainder taken from its bare amount.
    /// A chunk set records what was taken at `addr` ([`Taken::Single`]); a
    /// chunk set's value at the address is not bound.
    pub(super) fn take(
        &self,
        ctx: &mut VerifyContext<'_>,
        out: Heap,
        addr: egg::Id,
        needed: &ChunkPerm,
        value: egg::Id,
        pc: &[Lit],
    ) -> Result<Heap, VerifyError> {
        let kind = &self.kind;
        if let [s] = &self.shares[..]
            && s.when.is_empty()
        {
            if !prove_sufficient_aligned(ctx, &s.hold, needed, pc) {
                return Err(VerifyError::InsufficientPermission);
            }
            return Ok(match &s.source {
                ShareSource::Chunk(c) => {
                    ctx.union(c.value, value);
                    let rest = c.map_perm(|held| perm_sub_aligned(ctx, held, needed));
                    with_remainder(ctx, out, kind, rest)
                }
                ShareSource::Set(_) => {
                    let what = Taken::Single {
                        addr,
                        perm: needed.clone(),
                    };
                    take_from_set(ctx, out, kind, s.at, what)
                }
            });
        }

        let needed = needed.to_id(ctx);
        let total = self.total(ctx);
        if !prove_sufficient(ctx, &total, needed, pc) {
            if crate::verify::viz::dump_perm_enabled() {
                let total = total.to_id(ctx);
                eprintln!(
                    "[perm-dump] insufficient in summarized subtract, group {:?}, {} holder(s)\n\
                     total:\n{}needed:\n{}",
                    kind.group,
                    self.shares.len(),
                    crate::verify::viz::dump_term(ctx, total, 64),
                    crate::verify::viz::dump_term(ctx, needed, 64),
                );
            }
            return Err(VerifyError::InsufficientPermission);
        }
        for s in &self.shares {
            if let ShareSource::Chunk(c) = &s.source {
                // Where it sits at the address *and holds a share there*. Sitting
                // there is not enough: a chunk emptied on this path keeps its old
                // value.
                let perm = s.perm.to_id(ctx);
                assume_agree(ctx, c.value, value, &[perm], &s.when);
            }
        }
        let mut out = out;
        let mut remaining = needed;
        for s in &self.shares {
            let hold = s.hold.to_id(ctx);
            // `min(hold, remaining)` — a symbolic hold needs the `ite`; concrete
            // fractions fold it away.
            let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
            let gated = gate_amount_by_pc(ctx, take, &s.when);
            // Retired by what was actually taken, the **gated** amount: a holder
            // whose `when` fails gives up nothing, and must retire nothing.
            remaining = ctx.add(Symbolic::Binary(BinOp::SubR, [remaining, gated]));
            out = match &s.source {
                ShareSource::Chunk(c) => {
                    let rest = c.map_perm(|held| perm_sub(ctx, held, gated));
                    with_remainder(ctx, out, kind, rest)
                }
                ShareSource::Set(_) => {
                    let what = Taken::Single {
                        addr,
                        perm: ChunkPerm::leaf(gated),
                    };
                    take_from_set(ctx, out, kind, s.at, what)
                }
            };
        }
        Ok(out)
    }

    pub(super) fn iter(&self) -> std::slice::Iter<'_, Share> {
        self.shares.iter()
    }

    /// The sum of the shares: everything held at the address.
    pub(super) fn total(&self, ctx: &mut VerifyContext<'_>) -> ChunkPerm {
        let mut total = ChunkPerm::Zero;
        for s in &self.shares {
            total = perm_add(ctx, &total, &s.perm);
        }
        total
    }

    /// Split `demand` greedily over the shares, in order: each gives
    /// `min(what it holds, what is still needed)`, so the takes sum to the
    /// demand wherever the holders hold enough. One take per share.
    pub(super) fn greedy_takes(
        &self,
        ctx: &mut VerifyContext<'_>,
        demand: egg::Id,
    ) -> Vec<egg::Id> {
        let mut remaining = demand;
        self.shares
            .iter()
            .map(|s| {
                let hold = s.perm.to_id(ctx);
                let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
                remaining = ctx.add(Symbolic::Binary(BinOp::SubR, [remaining, take]));
                take
            })
            .collect()
    }

    /// The value at the address, `None` when nothing holds it. A sole chunk's
    /// value is the location's. Otherwise, outside a certificate walk, it is
    /// [`value_at`]; a certificate names one holder instead: the first known to
    /// hold a positive share, with its recipe (see [`holder_recipe`]), and
    /// `None` too when no holder is known positive there.
    ///
    /// Naming one holder drops what the others hold there, and the walk runs in a
    /// scratch graph, so their agreement would not reach a graft site. It is
    /// recorded in the certificate instead (see [`record_agreements`]).
    pub(super) fn value(
        &self,
        ctx: &mut VerifyContext<'_>,
        pc_lits: &[Lit],
        read_recipe: Option<&ReadRecipe>,
    ) -> Option<Value> {
        let holders = &self.shares[..];
        if self.sole {
            let s = &holders[0];
            let ShareSource::Chunk(c) = &s.source else {
                unreachable!("a sole share is a single chunk")
            };
            return Some(Value {
                id: s.value,
                recipe: c.recipe.clone(),
            });
        }
        if holders.is_empty() {
            return None;
        }
        if ctx.recipe.is_none() {
            return Some(Value::plain(value_at(
                ctx,
                holders,
                &self.kind.value,
                pc_lits,
            )));
        }
        let k = holders
            .iter()
            .position(|h| prove_perm_positive(ctx, &h.perm, pc_lits))?;
        let addr_r = read_recipe.and_then(|r| r.addr.as_ref());
        let recipe = holder_recipe(ctx, &self.kind, &holders[k], addr_r);
        record_agreements(
            ctx,
            &self.kind,
            holders,
            k,
            recipe.as_ref(),
            read_recipe,
            pc_lits,
        );
        Some(Value {
            id: holders[k].value,
            recipe,
        })
    }
}

/// `out` with the single chunk `rest` in place of the one at its address, or
/// without it when `rest` holds nothing on any branch.
fn with_remainder(ctx: &VerifyContext<'_>, out: Heap, kind: &LocationKind, rest: Chunk) -> Heap {
    if rest.is_emptied(ctx) {
        out.without_chunk(kind, rest.addr)
    } else {
        out.with_chunk(kind, rest)
    }
}

/// `out` with `what` recorded as taken from its chunk set `at`.
fn take_from_set(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    at: HolderAt,
    what: Taken,
) -> Heap {
    let HolderAt::Set(i) = at else {
        unreachable!("a chunk set's share sits at a set index")
    };
    let set = out.qp_chunks_of(kind)[i].with_taken(ctx, what);
    out.with_qp_chunk_replaced(kind, i, set)
}

/// The value at a location from its `holders`. A location has one value, so
/// any two holders with a positive share there agree; that is assumed here,
/// under the pc. The value is the first holder's whose share is known positive,
/// behind an `ite` on each holder before it that may be positive; fresh when no
/// holder is known positive.
fn value_at(ctx: &mut VerifyContext<'_>, holders: &[Share], ty: &Type, pc_lits: &[Lit]) -> egg::Id {
    let perms: Vec<egg::Id> = holders.iter().map(|h| h.perm.to_id(ctx)).collect();
    for i in 0..holders.len() {
        for j in i + 1..holders.len() {
            let (vi, vj) = (holders[i].value, holders[j].value);
            assume_agree(ctx, vi, vj, &[perms[j], perms[i]], pc_lits);
        }
    }
    let positive: Vec<egg::Id> = perms
        .iter()
        .map(|&p| expr!(ctx, (0 / 1) < r { p }))
        .collect();
    let known = holders
        .iter()
        .position(|h| prove_perm_positive(ctx, &h.perm, pc_lits));
    let (mut value, upto) = match known {
        Some(k) => (holders[k].value, k),
        None => (ctx.fresh_symbolic_value(ty.clone()), holders.len()),
    };
    for i in (0..upto).rev() {
        value = ctx.add(Symbolic::Ite([positive[i], holders[i].value, value]));
    }
    value
}

/// The recipe naming what `h` holds at the address in a certificate walk: a
/// single chunk's own recipe, `lookup(map, addr)` for a chunk set (from its
/// map's recipe and the address recipe `addr_r`). `None` when a part is
/// missing.
fn holder_recipe(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    h: &Share,
    addr_r: Option<&Val>,
) -> Option<Val> {
    match (&h.source, addr_r) {
        (ShareSource::Chunk(c), _) => c.recipe.clone(),
        (ShareSource::Set(Some(map_r)), Some(addr_r)) => {
            let lookup = ctx.alloc.value_map(kind).lookup;
            let rb = ctx
                .recipe
                .as_mut()
                .expect("holder recipe outside a certificate walk");
            Some(rb.emit(AxiomPure::App {
                func: lookup,
                type_args: vec![],
                args: vec![map_r.clone(), addr_r.clone()],
            }))
        }
        (ShareSource::Set(_), _) => None,
    }
}

/// In a certificate walk, the agreement between the named holder `holders[k]`
/// and every later holder also known to hold a positive share: a location has
/// one value. Assumed here, under the pc, and recorded in the certificate as
/// `pc_r ==> v_k == v_j` (the read's recipe-space pc folded in), which each
/// graft site assumes (see `walk_footprint`).
///
/// Guarding by the pc alone is sound because positivity was proven here from
/// what the certificate itself carries: the params, the slot permissions and
/// the body's conjuncts, all reproduced (assumed or proven) at a graft site
/// before the agreement is assumed.
///
/// A pair without a recipe on either side, or a read without a recipe-space
/// pc, records nothing: the graft site then knows less, never too much.
fn record_agreements(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    holders: &[Share],
    k: usize,
    named_r: Option<&Val>,
    read_recipe: Option<&ReadRecipe>,
    pc_lits: &[Lit],
) {
    let addr_r = read_recipe.and_then(|r| r.addr.as_ref());
    let pc_r = read_recipe.and_then(|r| r.pc.as_ref());
    let vk = holders[k].value;
    for h in &holders[k + 1..] {
        if ctx.egraph.find(vk) == ctx.egraph.find(h.value) {
            continue;
        }
        if !prove_perm_positive(ctx, &h.perm, pc_lits) {
            continue;
        }
        // Both are known positive here, so only the pc conditions it.
        assume_agree(ctx, vk, h.value, &[], pc_lits);
        let (Some(named_r), Some(pc_r)) = (named_r, pc_r) else {
            continue;
        };
        let Some(other_r) = holder_recipe(ctx, kind, h, addr_r) else {
            continue;
        };
        let rb = ctx
            .recipe
            .as_mut()
            .expect("agreement outside a certificate walk");
        let mut fact = rb.emit(AxiomPure::Binary(BinOp::Eq, named_r.clone(), other_r));
        let true_ = Val::Literal(Literal::Bool(true));
        // Outermost-first, folded innermost-first (as `implication` does).
        for (g, pol) in pc_r.iter().rev() {
            fact = rb.emit(match pol {
                Polarity::Positive => AxiomPure::Ternary(g.clone(), fact, true_.clone()),
                Polarity::Negative => AxiomPure::Ternary(g.clone(), true_.clone(), fact),
            });
        }
        rb.record_side_fact(fact);
    }
}
