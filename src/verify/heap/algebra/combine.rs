//! Combining whole heaps: the join after a branch and the sum of two heaps held at once.

use crate::dhash::HashSet;
use super::*;
use crate::verify::{
    context::VerifyContext,
    heap::{Chunk, ChunkPerm, Heap, HeapPc, LocationKind, cube_eq, cube_push},
};
use crate::verify::heap::{QPTaken, QuantChunk};
use std::rc::Rc;
use crate::vmir::Polarity;

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
/// **transiently at each consume site** (`gate_perm_by_guard`), so merges stay
/// flat while sufficiency/remainder keep the pre-hoist semantics. Runs only when
/// both arms are live, so an absent side is a genuine conditional footprint.
pub(in crate::verify::heap) fn merge_heaps(
    ctx: &mut VerifyContext<'_>,
    cond: egg::Id,
    h_then: &Heap,
    h_els: &Heap,
) -> Heap {
    let mut out = Heap::empty();
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
                    let value = if ctx.egraph.find(a.value) == ctx.egraph.find(b.value) {
                        a.value
                    } else {
                        expr!(ctx, if {cond} then {a.value} else {b.value})
                    };
                    if cube_eq(ctx, a.guard(), b.guard()) {
                        // Same presence on both arms → carry with shared guard.
                        // Amount: congruent ⇒ bare; divergent ⇒ single Select.
                        let perm = if ChunkPerm::same(ctx, a.ungated_perm(), b.ungated_perm()) {
                            a.ungated_perm().clone()
                        } else {
                            ChunkPerm::select(
                                ctx,
                                cond,
                                a.ungated_perm().clone(),
                                b.ungated_perm().clone(),
                            )
                        };
                        Some(
                            Chunk::new_perm(a.addr, perm, value)
                                .with_guard(a.guard_pc())
                                .with_recipe(a.recipe.clone()),
                        )
                    } else {
                        // Guards differ: genuine disjunction. Fall back to the
                        // legacy gated encoding — `guard? p : 0` on each side under
                        // `cond` — with an empty residual guard (conditionality
                        // lives in the amount here).
                        crate::verify::heap::MergeTrace::bump_zero();
                        let pa = a.gated_perm(ctx);
                        let pb = b.gated_perm(ctx);
                        let perm = ChunkPerm::select(ctx, cond, pa, pb);
                        Some(Chunk::new_perm(a.addr, perm, value).with_recipe(a.recipe.clone()))
                    }
                }
                (Some(a), None) => {
                    let guard = cube_push(ctx, a.guard(), (cond, Polarity::Positive));
                    Some(a.with_guard(guard))
                }
                (None, Some(b)) => {
                    let guard = cube_push(ctx, b.guard(), (cond, Polarity::Negative));
                    Some(b.with_guard(guard))
                }
                (None, None) => None,
            };
            if let Some(chunk) = chunk {
                out = out.with_chunk(kind, chunk);
            }
        }
    }
    // Chunk sets: one held on both arms is kept once (see `merge_sets`).
    let kinds = h_then.qps.keys().chain(h_els.qps.keys()).collect::<HashSet<_>>();
    for kind in kinds {
        let sets = merge_sets(ctx, kind, cond, h_then.qp_chunks_of(kind), h_els.qp_chunks_of(kind));
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
        cube_eq(ctx, &t.guard, &e.guard),
        "two copies of one chunk set reach a join under different guards"
    );
    let shared = t
        .taken
        .iter()
        .zip(e.taken.iter())
        .take_while(|(a, b)| a.id == b.id)
        .count();
    let taken: Rc<[QPTaken]> = if shared == t.taken.len() && shared == e.taken.len() {
        t.taken.clone()
    } else {
        let then_only = t.taken[shared..].iter().map(|x| x.under(ctx, (cond, Polarity::Positive)));
        let then_only: Vec<QPTaken> = then_only.collect();
        let els_only: Vec<QPTaken> = e.taken[shared..]
            .iter()
            .map(|x| x.under(ctx, (cond, Polarity::Negative)))
            .collect();
        t.taken[..shared].iter().cloned().chain(then_only).chain(els_only).collect()
    };
    let (value_map, recipe) = if ctx.egraph.find(t.value_map) == ctx.egraph.find(e.value_map) {
        (t.value_map, t.recipe.clone())
    } else {
        let join = ctx.alloc.value_map(kind).join;
        let args = Box::new([cond, t.value_map, e.value_map]);
        (ctx.add(crate::verify::lang::Symbolic::FuncApp(join, Box::new([]), args)), None)
    };
    QuantChunk {
        taken,
        value_map,
        recipe,
        ..t.clone()
    }
}

/// The **sum** of two heaps held simultaneously (`HeapInst::Union`).
///
/// Per location kind, per canonical address:
/// - **present in one heap only** — carried across unchanged.
/// - **present in both, equal guards** — one chunk with `perm = permₐ + perm_b`
///   under that shared guard.
/// - **present in both, guards differ** — the presence condition is a genuine
///   disjunction, which `Chunk.guard` (a flat cube) cannot express. Push both
///   guards into the *amounts* via [`gate_perm_by_guard`] and sum those, leaving
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
/// - **otherwise** — [`merge_values`]: `permₐ > 0 ? valueₐ : value_b`, plus the
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
        let (pa, pb, guard): (ChunkPerm, ChunkPerm, HeapPc) =
            if cube_eq(ctx, ca.guard(), cb.guard()) {
                (
                    ca.ungated_perm().clone(),
                    cb.ungated_perm().clone(),
                    ca.guard_pc(),
                )
            } else {
                (
                    ca.gated_perm(ctx),
                    cb.gated_perm(ctx),
                    std::rc::Rc::from(Vec::new()),
                )
            };
        // Decided against the *guard-gated* fractions: a chunk held only under its
        // guard is not held where the guard fails, however positive its bare amount.
        let a_held = prove_perm_positive(ctx, &pa, pc_lits);
        let b_held = prove_perm_positive(ctx, &pb, pc_lits);
        let (ia, ib) = (pa.to_id(ctx), pb.to_id(ctx));
        let value = match (a_held, b_held) {
            // Both definitely held: the values agree outright, so collapse the
            // classes. No ternary, no implication — the cheap common case.
            (true, true) => {
                ctx.union(ca.value, cb.value);
                ca.value
            }
            // One side definitely held: take its value and make the agreement
            // conditional on the *other* fraction. Equivalent to what `merge_values`
            // would build (its ternary reduces once the positive side is known), but
            // without minting the `Ite` — worth it, because a loop under a path
            // condition hits this on every exit.
            (true, false) => {
                assume_values_agree(ctx, ib, ca.value, cb.value, pc_lits);
                ca.value
            }
            (false, true) => {
                assume_values_agree(ctx, ia, cb.value, ca.value, pc_lits);
                cb.value
            }
            (false, false) => merge_values(ctx, ia, ca.value, ib, cb.value, pc_lits),
        };
        // Structural sum, the same dispatch [`merge_chunks`] makes: `perm_add`
        // distributes over the operands' `Select`s, so two arm-wise perms cancel per
        // branch. Flattening first left `ite(c, 1/1, 0) + ite(c, 0, 1/1)` — a full
        // permission the graph cannot see is one, since deciding it needs a case
        // split. That is exactly a loop's frame restore under a path condition: the
        // invariant's footprint on one arm, the frame's residual on the other.
        let sum = if ctx.has_wildcard && (pa.has_wild() || pb.has_wild()) {
            perm_add_wildcard(ctx, &pa, &pb)
        } else {
            perm_add(ctx, &pa, &pb)
        };
        let merged = Chunk::new_perm(ca.addr, sum, value)
            .with_guard(guard)
            .with_recipe(ca.recipe.clone().or_else(|| cb.recipe.clone()));
        out = out.with_chunk(kind, merged);
    }

    // Merge QPs in a naive way (a is already contained in out)
    for (kind, sets) in b.qps.iter() {
        for set in sets.iter() {
            out = out.with_qp_chunk(kind, set.clone());
        }
    }
    ctx.egraph.rebuild();
    out
}
