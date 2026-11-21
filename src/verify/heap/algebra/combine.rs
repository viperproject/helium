//! Combining whole heaps: the join after a branch and the sum of two heaps held at once.

use super::*;
use crate::dhash::HashSet;
use crate::verify::heap::holder::Debit;
use crate::verify::heap::quantified_permissions::same_footprint;
use crate::verify::heap::{QuantChunk, Restored};
use crate::verify::{
    context::VerifyContext,
    heap::{Chunk, ChunkPerm, Heap, LocationKind},
};
use crate::vmir::Polarity;
use crate::vmir::Type;
use std::rc::Rc;

/// Structural control-flow merge of two predecessor exit heaps at a binary join
/// (`HeapInst::Merge`). The reachability edge is carried as a **flat per-chunk
/// guard cube** relative to the join heap's `pc`, never a `?:0` `Select` tower —
/// so the amount stays a guard-free `Leaf` and no `0` leaf is ever built (absence
/// is `¬guard`). Per location kind, per canonical address:
/// - **present on one arm only** ⇒ carry the chunk, flat-append the branch
///   literal to its guard (`cond+` for then, `cond−` for els). No `Select`, no
///   `0` leaf, depth stays 0. (`p_Case_j` ends as `guard=[¬c0,¬c1,cⱼ], 1/1`.)
/// - **both arms, equal guard, congruent amount** ⇒ carry with that guard; the
///   branch literal drops (give-back / shared-prefix uplift). Value phi only if
///   the values differ.
/// - **both arms, equal guard, divergent amount** ⇒ a single-level `Select` on
///   `cond` in the amount (genuine perm divergence, per-leaf prove). Guard shared.
/// - **both arms, guards differ** ⇒ genuine `(cond∧g_t)∨(¬cond∧g_e)` disjunction,
///   not a flat cube: fall back to the gated `guard?perm:0` amount encoding
///   (rare — counted via `MergeTrace`).
///
/// The presence guard is reconstructed into the exact `guard?perm:0` obligation
/// **transiently at each consume site** (`Holder::gate`), so merges stay
/// flat while sufficiency/remainder keep the pre-hoist semantics. Runs only when
/// both arms are live, so an absent side is a genuine conditional footprint.
pub(in crate::verify::heap) fn merge_heaps(
    ctx: &mut VerifyContext<'_>,
    cond: egg::Id,
    h_then: &Heap,
    h_els: &Heap,
    region: &[(egg::Id, Polarity)],
) -> Heap {
    let mut out = Heap::empty_in(region);
    let mut kinds: Vec<LocationKind> = h_then.kinds().cloned().collect();
    for k in h_els.kinds() {
        if !kinds.contains(k) {
            kinds.push(k.clone());
        }
    }
    for kind in &kinds {
        let mut addrs: Vec<egg::Id> = Vec::new();
        for c in h_then.chunks_of(kind).iter().chain(h_els.chunks_of(kind)) {
            let canon = ctx.egraph.find(c.addr);
            if !addrs.iter().any(|a| ctx.egraph.find(*a) == canon) {
                addrs.push(canon);
            }
        }
        for addr in addrs {
            let ct = h_then.chunk_canon(ctx, kind, addr).cloned();
            let ce = h_els.chunk_canon(ctx, kind, addr).cloned();
            let chunk = match (ct, ce) {
                (Some(a), Some(b)) => {
                    let value =
                        if ctx.egraph.find(a.stored_value()) == ctx.egraph.find(b.stored_value()) {
                            a.stored_value()
                        } else {
                            expr!(ctx, if {cond} then {a.stored_value()} else {b.stored_value()})
                        };
                    // Same presence on both arms → kept, and the amounts are
                    // bare: congruent ⇒ one, divergent ⇒ a single Select.
                    // Presences differ → a genuine disjunction: each folded
                    // into its amount, under `cond`.
                    if !a.presence().same(ctx, b.presence()) {
                        crate::verify::heap::MergeTrace::bump_zero();
                    }
                    let (mut merged, ()) = Chunk::zip(ctx, &a, &b, |ctx, pa, pb| {
                        (ChunkPerm::select(ctx, cond, pa.clone(), pb.clone()), ())
                    });
                    merged.set_value(value);
                    Some(merged)
                }
                (Some(a), None) => Some(a.under(ctx, (cond, Polarity::Positive))),
                (None, Some(b)) => Some(b.under(ctx, (cond, Polarity::Negative))),
                (None, None) => None,
            };
            if let Some(chunk) = chunk {
                out = out.with_chunk(kind, chunk);
            }
        }
    }
    // Chunk sets: one held on both arms is kept once (see `merge_sets`).
    let kinds = h_then.kinds().chain(h_els.kinds()).collect::<HashSet<_>>();
    for kind in kinds {
        let sets = merge_sets(
            ctx,
            kind,
            cond,
            h_then.qp_chunks_of(kind),
            h_els.qp_chunks_of(kind),
        );
        for set in sets {
            out = out.with_qp_chunk(kind, set);
        }
    }
    assume_location_axioms(ctx, &out);
    out
}

/// The chunk sets of one kind after a join of `then` and `els` on `cond`.
///
/// A set held on both arms is two copies of one inhale: they share their
/// `origin`. Such a pair is kept once ([`merge_set`]). A set held on one arm
/// only counts where that arm was taken: its guard gets the arm's literal.
fn merge_sets(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    cond: egg::Id,
    then: &[QuantChunk],
    els: &[QuantChunk],
) -> Vec<QuantChunk> {
    let mut els_left: Vec<Option<&QuantChunk>> = els.iter().map(Some).collect();
    let mut out = Vec::new();
    for t in then {
        let partner = els_left
            .iter_mut()
            .find(|e| e.is_some_and(|e| e.origin == t.origin))
            .and_then(Option::take);
        out.push(match partner {
            Some(e) => merge_set(ctx, kind, cond, t, e),
            None => t.under(ctx, (cond, Polarity::Positive)),
        });
    }
    for e in els_left.into_iter().flatten() {
        out.push(e.under(ctx, (cond, Polarity::Negative)));
    }
    out
}

/// One set from its copies `t` (then-arm) and `e` (else-arm). `taken` is
/// append-only, so the copies share the exhales recorded before the branch
/// (equal ids); what each arm took after it is kept under that arm's literal. Differing value
/// maps (a write on an arm) join as `join(cond, t's, e's)`, read through the
/// read-over-join axiom (`builtin_axioms::value_map`); the joined map has no
/// recipe.
fn merge_set(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    cond: egg::Id,
    t: &QuantChunk,
    e: &QuantChunk,
) -> QuantChunk {
    // Sets are never dropped from a heap and only a join changes a guard, so
    // copies of one inhale reach a join under the guard they were inhaled with.
    assert!(
        t.presence().same(ctx, e.presence()),
        "two copies of one chunk set reach a join under different guards"
    );
    let shared = t
        .taken
        .iter()
        .zip(e.taken.iter())
        .take_while(|(a, b)| a.id == b.id)
        .count();
    let taken: Rc<[Debit]> = if shared == t.taken.len() && shared == e.taken.len() {
        t.taken.clone()
    } else {
        let then_only = t.taken[shared..]
            .iter()
            .map(|x| x.under(ctx, (cond, Polarity::Positive)));
        let then_only: Vec<Debit> = then_only.collect();
        let els_only: Vec<Debit> = e.taken[shared..]
            .iter()
            .map(|x| x.under(ctx, (cond, Polarity::Negative)))
            .collect();
        t.taken[..shared]
            .iter()
            .cloned()
            .chain(then_only)
            .chain(els_only)
            .collect()
    };
    let (value_map, recipe) = if ctx.egraph.find(t.value_map) == ctx.egraph.find(e.value_map) {
        (t.value_map, t.recipe.clone())
    } else {
        let join = ctx.alloc.value_map(kind).join;
        let args = Box::new([cond, t.value_map, e.value_map]);
        (
            ctx.add(crate::verify::lang::Symbolic::FuncApp(
                join,
                Box::new([]),
                args,
            )),
            None,
        )
    };
    // A joined map reads one arm's map or the other's, so each arm's
    // restored maps still need their definitions.
    let mut restored = t.restored.to_vec();
    for r in e.restored.iter() {
        if !restored
            .iter()
            .any(|x| ctx.egraph.find(x.map) == ctx.egraph.find(r.map))
        {
            restored.push(r.clone());
        }
    }
    let mut out = t.clone();
    out.taken = taken;
    out.value_map = value_map;
    out.recipe = recipe;
    out.restored = restored.into();
    out
}

/// The **sum** of two heaps held simultaneously (`HeapInst::Union`).
///
/// Per location kind, per canonical address:
/// - **present in one heap only** — carried across unchanged.
/// - **present in both, equal guards** — one chunk with `perm = permₐ + perm_b`
///   under that shared guard.
/// - **present in both, guards differ** — the presence condition is a genuine
///   disjunction, which `Chunk.guard` (a flat cube) cannot express. Push both
///   presences into the *amounts* (`Chunk::zip`) and sum those, leaving
///   the merged chunk unguarded — the same fallback `merge_heaps` uses.
///
/// Dropping one side instead would be sound but asymmetric in `a`/`b`: at a loop
/// exit the two sides are the frame and the invariant's footprint, so discarding
/// either loses permission the program genuinely holds. `Heap::with_chunk`
/// replaces at a shared address, so summing here is what preserves the total.
///
/// # Values: only equate what is simultaneously held
///
/// Two chunks of one location agree **only where both are actually held**, i.e.
/// under `permₐ > 0 ∧ perm_b > 0`. Collapsing the values outright (`ctx.union`)
/// whenever the addresses match was unsound: `heap_subtract` keeps a chunk whose
/// remainder it merely *failed to const-fold* to zero (see `perm_all_zero` — a
/// fold, not a prove, for cost reasons), justified by a zero-permission chunk
/// being inert. Reading such a chunk's **value** is exactly what breaks that
/// inertness. With a conditional invariant footprint the frame's residual is
/// `b ? 0 : 1/1`, so the stale pre-loop value survived here and was fused with the
/// loop's havoc'd symbol — erasing the havoc on the branch where the loop ran.
/// See `tests/cases/failing/loops/cond_inv_frame_value.vpr`; Silicon rejects it.
///
/// So the equality is earned, not assumed:
/// - **both fractions provably positive** — `ctx.union`, collapsing the classes.
///   This is the common case (`half_perm_frame`, `peano_frame`), where the
///   agreement is genuinely load-bearing: it carries a value established before a
///   loop across a cut whose invariant never mentions it. [`prove_perm_positive`]
///   decides it per leaf, so a join `Select` never materializes, and its
///   `known_real` fast path settles literal fractions without a prove call.
/// - **otherwise** — [`joint_value`]: `permₐ > 0 ? valueₐ : value_b`, plus the
///   guarded equality, which still fuses the two once saturation establishes both
///   fractions positive.
///
/// The pc-alias consume path states the same rule for the same reason (see
/// [`heap_subtract_pc_aliased`]): unioning outright would claim `x.f == y.f` on
/// the path where `x != y`.
pub(in crate::verify::heap) fn union_heaps(
    ctx: &mut VerifyContext<'_>,
    a: &Heap,
    b: &Heap,
    pc_lits: &[(egg::Id, Polarity)],
) -> Heap {
    let mut out = a.clone();
    for (kind, cb) in b.entries() {
        let existing = a
            .chunks_of(kind)
            .iter()
            .find(|ca| ctx.egraph.find(ca.addr) == ctx.egraph.find(cb.addr))
            .cloned();
        let Some(ca) = existing else {
            out = out.with_chunk(kind, cb.clone());
            continue;
        };
        // Under one shared presence the amounts are bare; otherwise each is gated
        // by its own (see `Chunk::zip`).
        let (mut merged, value) = Chunk::zip(ctx, &ca, cb, |ctx, pa, pb| {
            union_amounts(ctx, pa, pb, ca.stored_value(), cb.stored_value(), pc_lits)
        });
        merged.set_value(value);
        merged.recipe = ca.recipe.clone().or_else(|| cb.recipe.clone());
        out = out.with_chunk(kind, merged);
    }

    // Chunk sets (`out` already holds a's). A set given back to the one it
    // was taken from cancels that take ([`give_back`]): at a loop exit the
    // frame's set took the invariant's at the head, and the invariant hands it
    // back. Every other set is kept as is.
    for kind in b.kinds() {
        for set in b.qp_chunks_of(kind) {
            let mut hit = None;
            for (j, held) in out.qp_chunks_of(kind).iter().enumerate() {
                if let Some(c) = give_back(ctx, kind, set, held, pc_lits)
                    .or_else(|| give_back(ctx, kind, held, set, pc_lits))
                {
                    hit = Some((j, c));
                    break;
                }
            }
            out = match hit {
                Some((j, c)) => out.with_qp_chunk_replaced(kind, j, c),
                None => out.with_qp_chunk(kind, set.clone()),
            };
        }
    }
    ctx.egraph.rebuild();
    out
}

/// The sum of two amounts held at once at one location, `pa` (value `va`) and
/// `pb` (value `vb`), and the location's value. See [`union_heaps`] for the
/// value rule.
fn union_amounts(
    ctx: &mut VerifyContext<'_>,
    pa: &ChunkPerm,
    pb: &ChunkPerm,
    va: egg::Id,
    vb: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> (ChunkPerm, egg::Id) {
    // Decided against the *presence-gated* fractions: a chunk held only under
    // its presence is not held where it fails, however positive its bare amount
    // (`Chunk::zip` hands over bare amounts only under one shared presence).
    // Asked here, unlike an inhale: a loop under a path condition meets this on
    // every exit, and a side known held saves the `ite` (see `joint_value`).
    let known = (
        prove_perm_positive(ctx, pa, pc_lits),
        prove_perm_positive(ctx, pb, pc_lits),
    );
    sum_at_location(ctx, pa, pb, (va, vb), known, pc_lits)
}

/// `x` with `y` given back, or `None` when `y` is not something `x` gave away.
///
/// It is when `x` recorded taking a set with `y`'s footprint (same body and
/// captures, both guards entailed by `pc`) and nothing was taken from `y`
/// since. Then `x` without that take holds what both held. Its values are
/// `y`'s inside the taken region (they may have changed while `x` did not hold
/// it) and `x`'s elsewhere: `y`'s map when the region is all of `x`, `x`'s when
/// the two maps are already one, and otherwise a fresh map defined by a
/// [`Restored`] entry. That one is defined only at addresses read through
/// `x` (`View::shares`), so where it leaves as a snapshot its values are
/// unknown: sound, but incomplete.
fn give_back(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    x: &QuantChunk,
    y: &QuantChunk,
    pc: &[(egg::Id, Polarity)],
) -> Option<QuantChunk> {
    if !y.taken.is_empty() || !y.presence().holds_under(ctx, pc) {
        return None;
    }
    let k = x.taken.iter().position(|t| match t.as_set() {
        Some(s) => s.presence().holds_under(ctx, pc) && same_footprint(ctx, &s, y),
        None => false,
    })?;
    let region = x.taken[k].as_set().expect("matched a taken set above");
    let mut taken = x.taken.to_vec();
    taken.remove(k);
    let mut restored = x.restored.to_vec();
    restored.extend(y.restored.iter().cloned());
    let mut own = x.clone();
    own.taken = Rc::from([]);
    let (value_map, recipe) = if ctx.egraph.find(x.value_map) == ctx.egraph.find(y.value_map) {
        (x.value_map, x.recipe.clone())
    } else if same_footprint(ctx, &own, &region) {
        // `x` gave all it was inhaled with: every value it can name is `y`'s,
        // and outside its domain its map is never read. No map to define,
        // which also keeps the map whole where it leaves as a snapshot.
        (y.value_map, y.recipe.clone())
    } else {
        let addr = Type::Addr {
            group: kind.group,
            value: Box::new(kind.value.clone()),
            bound: kind.bound.clone(),
        };
        let map = ctx.fresh_symbolic_value(Type::ValueMap(Box::new(addr)));
        restored.push(Restored {
            map,
            region,
            kept: x.clone(),
            new: y.value_map,
            old: x.value_map,
        });
        (map, None)
    };
    let mut out = x.clone();
    out.taken = taken.into();
    out.value_map = value_map;
    out.recipe = recipe;
    out.restored = restored.into();
    Some(out)
}
