//! Consuming: subtracting a demanded chunk from a heap, directly, through pc-implied aliases or against a summary of the group.

use super::*;
use crate::verify::{
    context::VerifyContext,
    error::VerifyError,
    heap::{Chunk, ChunkPerm, Cube, Heap, LocationKind, Shape},
    lang::Symbolic,
};
use crate::vmir::{BinOp, Polarity};

/// Heap subtraction for a single location chunk of kind `kind`.
pub(in crate::verify::heap) fn heap_subtract(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    chunk2: Chunk,
    pc_lits: &[(egg::Id, Polarity)],
    demand: Demand,
) -> Result<Heap, VerifyError> {
    heap_subtract_inner(ctx, h1, kind, chunk2, pc_lits, demand, ConsumePass::First)
}

/// Where a [`heap_subtract_inner`] call sits in the ground-miss retry cycle.
///
/// A miss first asks whether the demand is provably zero, then saturates and tries
/// the whole lookup once more. The second pass is what makes a recipe-rebuilt
/// address meet its chunk — but the *saturation* it depends on is the one
/// [`VerifyContext::prove_under_pc`] runs at its own `saturate` tier while failing
/// that zero-demand probe, not the explicit call on the retry line, which is
/// normally a no-op against an already-clean graph.
///
/// So the retry must still happen, while the zero-demand probe must not be re-asked
/// for nothing: an unchanged graph cannot give a different verdict, and a failed
/// probe is **not** memoized ([`VerifyContext::prove_under_pc`] records only
/// successes), so re-asking pays the full ladder — clone and all — a second time.
#[derive(Clone, Copy)]
enum ConsumePass {
    /// First attempt; the saturate-and-retry rung is still available.
    First,
    /// The retry. `zero_demand_settled` says the explicit saturate found the graph
    /// already at a fixpoint, so nothing moved since the first pass asked, and its
    /// `false` still stands. When the saturate *did* work the question is genuinely
    /// open again — a gate collapsing to `0` is exactly what the probe looks for —
    /// and it is re-asked.
    Retry { zero_demand_settled: bool },
}

/// What kind of permission a consume demands, which selects the rule
/// ([`debit_wildcard`] vs [`prove_sufficient`]).
///
/// Carried from the permission's **provenance** rather than recognised in the
/// e-graph: `ChunkPerm::Leaf`'s `wild` flag records that a leaf came from a
/// `wildcard`, set where the permission was evaluated (`eval_perm_structural`,
/// `build_perm`) from the IR's own `PermVal::Wildcard`. Asking the e-graph instead
/// means asking about an **e-class**, which
/// congruence can put other nodes into -- the same lesson the produce side learned
/// when positivity had to be read off operand trees rather than off the fused leaf.
///
/// The produce side keeps [`contains_wildcard`]: there the two summands come out
/// of the heap, and their provenance is genuinely gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::verify::heap) enum Demand {
    /// A fixed amount: prove `held >= needed`.
    Concrete,
    /// Mentions a `wildcard`, bare or under gating/scaling: prove `0 < held` and
    /// hand back a fresh smaller share.
    Wildcard,
}

/// Consume `chunk2` from `h1`. The body is a **ladder, cheapest rung first**, and the
/// order is load-bearing for cost rather than for correctness — every rung below the
/// first is reached only by an obligation that would otherwise fail, so a consume
/// that succeeds outright never pays for the ones under it.
///
/// 1. `find_chunk_consolidated` — ground e-class match on the address.
/// 2. **no chunk**: a provably-zero demand is a no-op (a conditional footprint slot
///    whose guard is false); else one `saturate` retry; else the summarized
///    fallbacks.
/// 3. **wildcard** demand — [`debit_wildcard`], a different rule entirely (hold
///    *some* share rather than `held ≥ needed`).
/// 4. **sufficiency proven** — [`debit_direct`], the hot path.
/// 5. **not proven** — [`heap_subtract_summarized_fallbacks`]: pc-implied aliasing,
///    then the whole-group Σ-ite summary.
///
/// `sat_retry` gates rung 2's re-try under a **full saturation**
/// ([`VerifyContext::saturate`]): the address of a `&mut` reborrow passed to a call
/// meets the held chunk's address only after the full rule set runs, and
/// `find_chunk_consolidated`'s own retry runs the terminating reductions only, which
/// is not enough for it. It sits *after* the provably-zero check, not inside the
/// lookup — retrying at every miss cost **36x** on `structs_enums` (1.48s → 53.5s),
/// since a legitimately absent chunk is common and the zero check closes it without
/// saturation.
fn heap_subtract_inner(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    chunk2: Chunk,
    pc_lits: &[(egg::Id, Polarity)],
    demand: Demand,
    pass: ConsumePass,
) -> Result<Heap, VerifyError> {
    let (out, existing) = find_chunk_consolidated(ctx, h1, kind, chunk2.addr, pc_lits, true);
    // In the form the consume works on: presence kept flat when the pc entails it,
    // folded into the amount otherwise (see `Chunk::normalized_for`).
    let existing = existing.map(|c| c.normalized_for(ctx, pc_lits));
    // The demand is unconditional, so its gated amount is its bare one.
    let needed = chunk2.gated_perm(ctx);
    let chunk2_perm = needed.to_id(ctx);
    let Some(existing) = existing else {
        // No chunk at `addr`. Subtracting a provably-zero permission (e.g. a
        // conditional footprint slot whose guard is false — a nested predicate
        // `b ==> P(..)` with `b` false) is a no-op, so it need not be held. A
        // wildcard is provably positive, so this (correctly) fails — a wildcard
        // cannot be exhaled from an empty location.
        // Ask the pc-alias sum *before* the zero-demand probe. Both rungs answer a
        // ground miss, but their costs have inverted: the partner lookup now reads
        // the block scratch (a `find` per chunk) while the probe assumes the pc on
        // a clone and saturates it — 1.8s of `enum_v8_p1` over 16 calls. A census
        // of every cold consume event in the corpus found the probe closing **0 of
        // 261**, and the alias leg resolving all 398, so this is the likely answer
        // first rather than a new one.
        //
        // Gated on the demand being a **known positive literal**, which is exactly
        // when the zero-demand probe below cannot succeed. Without that guard this
        // reorder is wrong rather than merely eager: for a provably-zero demand the
        // summarized path still rewrites the group, and `borrow_fields`'s
        // `m_pair_guarded_here` then failed a later consume for want of the
        // permission that rewrite had restructured.
        //
        // Only on the **retry** pass. On the first pass the saturate-and-retry
        // below is what finds the chunk for 6 of every 21 misses, and consuming
        // from an alias set before that has run took `m_pair_guarded_here` out.
        //
        // Both rungs stay: an `Err` falls through to the probe below.
        let demand_positive = !matches!(pass, ConsumePass::First)
            && known_real(ctx, chunk2_perm)
                .is_some_and(|r| r > num::BigRational::from(num::BigInt::from(0)));
        if demand_positive
            && let Ok(h) = heap_subtract_summarized_fallbacks(
                ctx,
                h1,
                out.clone(),
                kind,
                None,
                chunk2.clone(),
                chunk2_perm,
                pc_lits,
            )
        {
            return Ok(h);
        }
        // Nothing demanded: not (0 < needed). Skipped on a retry whose saturate
        // was a no-op — see [`ConsumePass::Retry`].
        // Never on the first pass: the saturate-and-retry below is cheap (its
        // `saturate` was measured a no-op in 49 of 49 miss events) and can find the
        // chunk outright, and the alias sum above gets its turn before this probe
        // on the way back — so asking here only pays for an answer the retry
        // usually makes irrelevant. This probe closed 0 of 261 calls corpus-wide.
        if !matches!(pass, ConsumePass::First)
            && !matches!(
                pass,
                ConsumePass::Retry {
                    zero_demand_settled: true
                }
            )
        {
            let nonpos = expr!(ctx, not((0 / 1) < r { chunk2_perm }));
            if ctx.prove_under_pc(nonpos, pc_lits) {
                return Ok(out);
            }
        }
        if matches!(pass, ConsumePass::First) {
            let settled = ctx.is_saturated();
            ctx.saturate();
            return heap_subtract_inner(
                ctx,
                h1,
                kind,
                chunk2,
                pc_lits,
                demand,
                ConsumePass::Retry {
                    zero_demand_settled: settled,
                },
            );
        }
        // Nothing on ground: the demanded address may still match a held chunk under
        // the pc (a `&mut` reborrow inside a branch arm, whose address equality is a
        // pc-guarded fact ground e-class matching cannot see), or under an equality
        // the pc does not mention.
        return heap_subtract_summarized_fallbacks(
            ctx,
            h1,
            out,
            kind,
            None,
            chunk2,
            chunk2_perm,
            pc_lits,
        );
    };

    if demand == Demand::Wildcard {
        return debit_wildcard(ctx, out, kind, &existing, &chunk2, pc_lits);
    }

    // Sufficiency `held ≥ needed`, proven per-leaf over the (possibly
    // branch-structured) held perm — the `Select` never enters the graph.
    let proven =
        prove_sufficient_aligned(ctx, &existing.perm_under(ctx, pc_lits), &needed, pc_lits);
    if !proven {
        // Invariant 7 — consume under **pc-implied aliasing**. `acc(x.f,1/2)` and
        // `acc(y.f,1/2)` are distinct chunks on ground, but under an in-branch
        // `x == y` they are one location holding `1/1`. Sum the demanded chunk with
        // its pc-alias partners and re-prove; the debit stays guarded (below), so
        // ground never consolidates two chunks that are only conditionally equal.
        //
        // Tried only after the plain proof fails: no unaliased consume pays the probe.
        //
        // NOT retried under merge guards here, unlike `heap_subtract_summarized` and
        // the `frame_wildcard_slot` framing check. This is the hot path — every
        // `- acc` reaches it — and measurement says the retry buys nothing on it:
        // where a single chunk's sufficiency is arm-dependent the deciding guard sits
        // below the depth bound, in give-back residue, and a bound wide enough to
        // reach it took `shape_area` from 1s to over ten minutes. The two members
        // that need it are recorded in `expected_failures.txt`.
        let r = heap_subtract_summarized_fallbacks(
            ctx,
            h1,
            out,
            kind,
            Some(&existing),
            chunk2,
            chunk2_perm,
            pc_lits,
        );
        return r;
    }

    Ok(debit_direct(ctx, out, kind, &existing, &chunk2))
}

/// Take `chunk2`'s amount off the single chunk `existing` (normalized for the
/// consume, see `Chunk::normalized_for`) whose sufficiency is already proven. The
/// remainder keeps `existing`'s presence.
///
/// The value union is **unconditional**, unlike the guarded rule [`union_heaps`] and
/// [`merge_chunks`] use. Sound here because the two sides are not symmetric:
/// `chunk2.value` is minted fresh by [`heap_acc`] for *this* consume and carries no
/// prior meaning, so the union constrains only the fresh symbol and cannot corrupt
/// `existing.value`. (The loop frame restore's bug was the symmetric case — two
/// pre-existing values, one of them a havoc symbol something downstream reads.) Note
/// it is NOT implied by the sufficiency proof: `held ≥ needed` permits `needed == 0`,
/// so a conditional exhale does bind the slot value off-path. Probed with
/// complementary conditional exhales and with a call whose `requires` has a
/// conditional footprint; both are correctly rejected.
fn debit_direct(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    existing: &Chunk,
    chunk2: &Chunk,
) -> Heap {
    ctx.union(existing.value, chunk2.value);
    // Remainder stays structural (leaves get `SubR`); never an `ite` in the graph.
    // Aligned against the *demand's* structure too, so a guarded consume cancels per
    // arm (`c ? 0 : 1/1`) rather than leaving `1/1 - (c ? 1/1 : 0/1)` in every leaf --
    // which is what lets `perm_all_zero` below see the emptied branch.
    let needed = chunk2.gated_perm(ctx);
    let rest = existing.map_perm(|held| perm_sub_aligned(ctx, held, &needed));
    // Whether to drop the emptied chunk is a statement about the *heap*, so it has
    // to hold at the heap's scope — **unconditionally**, not under this
    // instruction's `pc`.
    //
    // The two differ because VMIR is linearized: the heap this subtract produces
    // flows on into the sibling branch, where `pc` does not hold. A guarded consume
    // has remainder `1/1 - (c ? 1/1 : 0/1)` — zero under `c`, but a full `1/1`
    // under `!c`, where the permission was never given up, so dropping the chunk
    // would lose it for the off-path arm.
    //
    // Const-fold, not `prove_under_pc`: heap hygiene, not an obligation, so it must
    // stay O(1) — asking the prover runs once per chunk per consume and dominated
    // everything (93s vs 4s). Keeping a chunk we merely failed to prove empty is
    // sound: a zero-permission chunk is inert (`perm > 0` gates every use).
    if rest.is_emptied(ctx) {
        out.without_chunk(kind, existing.addr)
    } else {
        // The value (and so its recipe provenance) is unchanged by a subtract, and
        // so is the presence: the chunk is still held exactly where it was, just
        // for less.
        out.with_chunk(kind, rest)
    }
}

/// Wildcard exhale. Instead of proving `held ≥ needed` (a wildcard has no fixed
/// value), require the location hold *some* permission (`held > 0`) and hand back a
/// **fresh remainder** assumed strictly smaller (`r < held`) under the pc —
/// Silicon's constrainable-ARP rule. The demanded amount never enters the result:
/// what leaves is only known to be positive and smaller, which is all a wildcard
/// ever says.
#[allow(clippy::too_many_arguments)]
fn debit_wildcard(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    existing: &Chunk,
    chunk2: &Chunk,
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Heap, VerifyError> {
    // Walk the DEMAND's branch structure: a gated `exhale b ==> acc(x.f, wildcard)`
    // takes a share only where `b` holds, and states nothing where it does not.
    // Flat, the rule read only `chunk2.value` and constrained the remainder
    // `r < held` on both arms — including the arm where nothing was demanded, which
    // is imprecise rather than unsound but throws away permission the program kept.
    let needed = chunk2.gated_perm(ctx);
    let mut value = existing.value;
    let mut err = None;
    let mut rest = existing.map_perm(|held| {
        let (perm, v, e) =
            debit_wildcard_walk(ctx, held, &needed, existing.value, chunk2.value, pc_lits);
        (value, err) = (v, e);
        perm
    });
    if let Some(e) = err {
        return Err(e);
    }
    rest.value = value;
    Ok(out.with_chunk(kind, rest))
}

/// One arm of a wildcard consume, recursing on the demand.
///
/// At a demand **leaf**:
/// - a provably-zero demand is a no-op — the held permission is returned untouched,
///   with no `held > 0` obligation and no fact about a fresh share. This is the arm
///   a gate switches off;
/// - otherwise the wildcard rule: require the location hold *some* permission
///   (`held > 0`) — a wildcard has no fixed value to compare against — and hand back
///   a **fresh remainder** assumed strictly smaller (`r < held`) under the pc,
///   Silicon's constrainable-ARP rule. The demanded amount never enters the result:
///   what leaves is only known positive and smaller, which is all a wildcard says.
///
/// A fresh `r` rather than a `held − w` node, because with no real-order arithmetic
/// neither `0 < held − w` nor `held − w < held` follows from such a term, so both
/// would have to be assumed about a tree that keeps growing. `r` states them once,
/// and `r < held` is what makes `assert perm(x.f) < 1/2` provable after a wildcard
/// exhale. Positivity is also why there is no drop-if-zero case: the remainder is a
/// wildcard, so the chunk is never emptied.
///
/// The error is returned rather than propagated with `?` so a failing arm does not
/// abandon the walk mid-way through mutating the graph.
fn debit_wildcard_walk(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
    held_value: egg::Id,
    needed_value: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> (ChunkPerm, egg::Id, Option<VerifyError>) {
    match needed.shape() {
        // Nothing demanded on this arm: no obligation, no fact, no debit.
        Shape::Zero => (held.clone(), held_value, None),
        Shape::Leaf { id, .. } => {
            // Nothing demanded on this arm: no obligation, no fact, no debit.
            let nonpos = expr!(ctx, not((0 / 1) < r { id }));
            if ctx.prove_under_pc(nonpos, pc_lits) {
                return (held.clone(), held_value, None);
            }
            let held_id = held.to_id(ctx);
            let held_pos = expr!(ctx, (0 / 1) < r { held_id });
            if !ctx.prove_under_pc(held_pos, pc_lits) {
                return (
                    held.clone(),
                    held_value,
                    Some(VerifyError::InsufficientPermission),
                );
            }
            // Unconditional union is safe on both counts: `held > 0` was just proven
            // and a wildcard `needed` is positive by construction, so this is the
            // both-held case — and `chunk2.value` is fresh besides.
            ctx.union(held_value, needed_value);
            let remainder = ctx.fresh_wildcard();
            let lt = expr!(ctx, { remainder } < r { held_id });
            ctx.assume_all_guarded([lt], pc_lits);
            (ChunkPerm::wild_leaf(remainder), held_value, None)
        }
        Shape::Branch { cond, then, els } => {
            let ht = ChunkPerm::restrict(ctx, cond, held.clone(), true);
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((cond, Polarity::Positive));
            let (t, value, e1) =
                debit_wildcard_walk(ctx, &ht, &then, held_value, needed_value, &pc_t);
            let he = ChunkPerm::restrict(ctx, cond, held.clone(), false);
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((cond, Polarity::Negative));
            let (e, _, e2) = debit_wildcard_walk(ctx, &he, &els, held_value, needed_value, &pc_e);
            (ChunkPerm::select(ctx, cond, t, e), value, e1.or(e2))
        }
    }
}

/// The two **summarized** resolutions of a consume that the direct path could not
/// settle, tried in order. Shared by both callers in [`heap_subtract_inner`] — the
/// no-chunk-at-all case and the chunk-found-but-insufficient case — which ran two
/// near-copies of this ladder before.
///
/// In order, cheapest first:
/// 1. **pc-implied aliasing** ([`VerifyContext::pc_alias_partners`]) — the demanded
///    address coincides with held chunks only under the path condition. Needs a
///    probe, so it is tried only once the direct path has failed.
/// 2. **whole-group Σ-ite summary** ([`summarize_perm_at`]) — a chunk may still sit
///    at this address under an equality the pc does not mention. Strictly the last
///    resort: it walks every chunk of the group.
///
/// `existing` is the chunk that matched on ground, if any, and does double duty:
/// - it **seeds** the pc-alias set, so the ground match's own fraction joins the sum;
/// - it sets the **bar** for the group summary. With nothing matched, any non-empty
///   summary is new information. With a chunk already tried and found insufficient, a
///   one-member summary is that same chunk again — only a genuinely larger set is
///   worth the attempt.
///
/// The pc-alias branch **returns** rather than falling through when it produces a
/// set: its `Err` is the answer. (With `existing = Some`, the set always contains at
/// least that chunk, so this is exactly the pre-refactor control flow of both callers.)
fn heap_subtract_summarized_fallbacks(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    out: Heap,
    kind: &LocationKind,
    existing: Option<&Chunk>,
    chunk2: Chunk,
    chunk2_perm: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Heap, VerifyError> {
    // 1. pc-implied aliasing. The whole partner *set* is collected, not the first
    //    hit: several held chunks can coincide with the demand under the pc, and only
    //    their sum is the permission at that location. A first-hit lookup is
    //    order-dependent — it can land on a chunk an earlier consume already drained
    //    while the full permission sits in another.
    //
    //    Consuming through it goes down the invariant-7 path: sufficiency is proven
    //    under the pc and the debit is **gated** by the pc, so off-path — where the
    //    addresses are unrelated — nothing is taken, and ground never consolidates
    //    two chunks that are only conditionally equal.
    let partners = pc_alias_partners(ctx, h1.chunks_of(kind), chunk2.addr, pc_lits);
    if !partners.is_empty() {
        let (set, total) = pc_alias_set(ctx, h1, kind, existing, &partners, pc_lits);
        if !set.is_empty() {
            let r =
                heap_subtract_summarized(ctx, out, kind, &set, total, chunk2, chunk2_perm, pc_lits);
            return r;
        }
    }
    // 2. Whole-group Σ-ite summary.
    let (total, set) = summarize_perm_at(ctx, h1.chunks_of(kind), chunk2.addr, pc_lits);
    if set.len() > usize::from(existing.is_some())
        && let Ok(h) = heap_subtract_summarized(
            ctx,
            out.clone(),
            kind,
            &set,
            total,
            chunk2.clone(),
            chunk2_perm,
            pc_lits,
        )
    {
        return Ok(h);
    }
    subtract_miss_trace(ctx, h1, kind, existing, &chunk2, chunk2_perm);
    Err(VerifyError::InsufficientPermission)
}

/// Env-gated diagnostics for a consume that resolved nowhere. Two different
/// questions, so two different dumps: with no ground match the useful thing is the
/// demanded address against the held ones (`SILVER_OXIDE_TRACE_MISS`); with a match
/// that proved insufficient it is the two permission terms
/// (`SILVER_OXIDE_DUMP_PERM`).
fn subtract_miss_trace(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    existing: Option<&Chunk>,
    chunk2: &Chunk,
    chunk2_perm: egg::Id,
) {
    match existing {
        Some(existing) => {
            if crate::verify::viz::dump_perm_enabled() {
                let existing_perm = existing.gated_perm(ctx).to_id(ctx);
                eprintln!(
                    "[perm-dump] insufficient at subtract in group {:?}\n\
                     held.perm:\n{}needed.perm:\n{}",
                    kind.group,
                    crate::verify::viz::dump_term(ctx, existing_perm, 64),
                    crate::verify::viz::dump_term(ctx, chunk2_perm, 64),
                );
            }
        }
        None => {
            if std::env::var_os("SILVER_OXIDE_TRACE_MISS").is_some() {
                eprintln!(
                    "[miss] group {:?} demanded addr:\n{}held addrs ({}):",
                    kind.group,
                    crate::verify::viz::dump_term(ctx, chunk2.addr, 40),
                    h1.chunks_of(kind).len(),
                );
                for c in h1.chunks_of(kind).to_vec() {
                    eprintln!("{}", crate::verify::viz::dump_term(ctx, c.addr, 40));
                }
            }
        }
    }
}

/// Build the pc-alias flavour of a [`heap_subtract_summarized`] set: `existing` (when
/// a chunk did match on ground) followed by the pc-alias `partners`, every member
/// gated by the *same* cube — the pc.
///
/// The returned total is not gated by the pc: the sufficiency proof runs under the
/// pc anyway ([`VerifyContext::prove_under_pc`]), so gating each addend would only
/// grow the term. Each addend is gated by its own presence guard, though (see
/// `Chunk::perm_under`).
fn pc_alias_set(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    existing: Option<&Chunk>,
    partners: &[egg::Id],
    pc_lits: &[(egg::Id, Polarity)],
) -> (Vec<(Chunk, Cube)>, ChunkPerm) {
    let members: Vec<Chunk> = existing
        .cloned()
        .into_iter()
        .chain(partners.iter().filter_map(|a| {
            let canon = ctx.egraph.find(*a);
            h1.chunks_of(kind)
                .iter()
                .find(|c| ctx.egraph.find(c.addr) == canon)
                .cloned()
        }))
        .collect();
    // Structural, like the Σ-ite summary's: a member whose perm carries join
    // structure keeps it, so sufficiency is decided per leaf rather than over a
    // flattened `ite`.
    // A member contributes only where it is PRESENT. The pc gate is implicit (the
    // proof runs under the pc), but a member's own presence guard is a different
    // condition — a chunk held on one join arm only — and summing it ungated
    // counted permission on the arm where the chunk does not exist.
    let mut total: Option<ChunkPerm> = None;
    for c in &members {
        let amount = c.perm_under(ctx, pc_lits);
        total = Some(match total {
            None => amount,
            Some(t) => perm_add(ctx, &t, &amount),
        });
    }
    let cube = Cube::from(pc_lits);
    let set = members.into_iter().map(|c| (c, cube.clone())).collect();
    (set, total.unwrap_or(ChunkPerm::Zero))
}

/// Consume `chunk2` against a **summarized** location: a set of chunks that each sit
/// at `chunk2.addr` only *conditionally*, paired with the cube that condition is.
///
/// Two callers supply two different gates, and the algorithm is the same for both:
/// - **pc-alias** (invariant 7 of the two-egraph block model) — cube = the pc.
///   `acc(x.f,1/2)` and `acc(y.f,1/2)` are distinct chunks on ground, but under an
///   in-branch `x == y` they are one location holding `1/1`.
/// - **Σ-ite** — cube = `c.addr == chunk2.addr` itself, the condition the pc gate is
///   only ever a proxy for. Strictly more precise, and it needs no probe.
///
/// - **Sufficiency** is proven against `total`, the sum the caller summarized: at any
///   state where a member's cube holds it is the demanded location, so its fraction
///   joins the sum (`1/2 + 1/2 ≥ 1/1`). Proven **per leaf** of that sum
///   ([`prove_sufficient`]), exactly as the unsummarized path proves a single
///   chunk's: a member present on one arm only contributes a `Select` on its guard
///   literal, and each leaf is discharged with that literal in the pc. Flattening
///   into one goal instead was the whole of the `&mut`-reborrow incompleteness —
///   the arms' contributions are live under complementary literals, so no single
///   monolithic goal holds, while both leaves are trivial. The split set is the
///   guards the chunks already carry, so it is finite and fixed by the CFG, never
///   searched.
/// - **The debit is guarded and _distributed_** across the set, greedily: each chunk
///   gives up `min(it holds, still needed)`, gated by `cube ? take : 0`, and the demand
///   is retired by that **gated** amount. No chunk is merged — where the cube fails the
///   locations are genuinely distinct and nothing there was given up.
///
///   Distribution (rather than parking the whole debit on the demanded chunk as a
///   guarded negative) keeps every *later* operation correct without another alias
///   probe: parking leaves a partner reading `1/2` when the location holds nothing,
///   so `exhale acc(y.f,1/2)` would wrongly succeed without ever reaching this
///   function. Distributing drives every member of the set to its true remainder.
/// - **Value agreement** is likewise assumed only under each member's own cube:
///   unioning the values outright would claim `x.f == y.f` where `x != y`.
#[allow(clippy::too_many_arguments)]
fn heap_subtract_summarized(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    set: &[(Chunk, Cube)],
    total: ChunkPerm,
    chunk2: Chunk,
    chunk2_perm: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Heap, VerifyError> {
    // `needed ≤ total`, per leaf of `total`, each under the pc plus that leaf's
    // branch literals.
    if !prove_sufficient(ctx, &total, chunk2_perm, pc_lits) {
        if crate::verify::viz::dump_perm_enabled() {
            let total = total.to_id(ctx);
            eprintln!(
                "[perm-dump] insufficient in summarized subtract, group {:?}, {} chunk(s)\n\
                 total:\n{}needed:\n{}",
                kind.group,
                set.len(),
                crate::verify::viz::dump_term(ctx, total, 64),
                crate::verify::viz::dump_term(ctx, chunk2_perm, 64),
            );
        }
        return Err(VerifyError::InsufficientPermission);
    }
    // Golden rule, but only where the locations coincide — each member under its own
    // gate, so a chunk that is only conditionally at this address claims value
    // agreement only under that condition.
    for (chunk, cube) in set {
        let agree = expr!(ctx, { chunk.value } == { chunk2.value });
        ctx.assume_guarded(agree, cube.iter().rev().copied());
    }

    // Greedy distribution over the set, demanded chunk first.
    let mut out = out;
    let mut remaining = chunk2_perm;
    for (chunk, cube) in set.iter().cloned() {
        // What the chunk can give up is what it holds where it is present: an
        // absent member (its guard false) must neither be debited nor retire any
        // of the demand, or a present member is left holding permission it had
        // in fact given away.
        let hold = chunk.perm_under(ctx, pc_lits).to_id(ctx);
        // `min(hold, remaining)` — a symbolic hold needs the `ite`; concrete
        // fractions fold it away.
        // `min(hold, remaining)` — a symbolic hold needs the `ite`; concrete
        // fractions fold it away.
        let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
        let gated = gate_amount_by_pc(ctx, take, &cube);
        // Taken from the bare amount: the take is gated by the member's own
        // `cube` already, and the remainder keeps the member's presence.
        let rest = chunk.map_perm(|held| perm_sub(ctx, held, gated));
        // Debit `remaining` by what was actually taken — the **gated** amount, not
        // `take`. With one shared cube (the pc-alias caller) the two coincide on-path
        // and both are `0` off-path, so this is that path unchanged. With per-chunk
        // cubes they diverge, and using `take` would be unsound: a member whose gate
        // is false gives up nothing, yet would still retire part of the demand,
        // leaving the later members to cover less than `needed` and the heap holding
        // permission it had in fact given away.
        remaining = ctx.add(Symbolic::Binary(BinOp::SubR, [remaining, gated]));
        // Same hygiene as the plain path: drop a chunk only when the remainder is
        // *unconditionally* zero (off-path the permission was never given up).
        if rest.is_emptied(ctx) {
            out = out.without_chunk(kind, chunk.addr);
        } else {
            // The member's own presence guard is carried through untouched: unlike
            // the plain path, this one never folded it into the amount — `hold` is
            // the raw perm and the debit is gated by the member's `cube` — so the
            // remainder is still held exactly under that guard. Dropping it here
            // was what destroyed the flat cube on the first consume at a location
            // and left every later one with conditionality it could only see as
            // `ite`s inside the arithmetic.
            out = out.with_chunk(kind, rest);
        }
    }
    Ok(out)
}
