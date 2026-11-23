//! Producing: adding a chunk to a heap, merging it with one already at its address, value agreement and the location axioms.

use super::*;
use crate::verify::heap::agree::joint_value;
use crate::verify::{
    context::VerifyContext,
    heap::{Chunk, ChunkPerm, Cube, Heap, LocationKind, cond::meet},
    lang::Symbolic,
};
use crate::vmir::{Bound, Literal, Polarity};

/// Merge two fractional chunks at the same address. Decouples the operational
/// value pick from the declarative agreement axiom: `perm = p0 + p1`,
/// `value = (p0 > 0) ? v0 : v1` (intentionally asymmetric), and an assumed
/// `(p0 > 0 && p1 > 0) ==> (v0 == v1)`.
///
/// The assume is emitted by unioning the (desugared) implication with `true`, not
/// as an eager `union(v0, v1)`: with both fractions positive, saturation collapses
/// the implication to `v0 == v1` and `eq-true-union` fuses the values, erasing the
/// ternary's asymmetry; with a fraction zero the antecedent is `false` and the
/// asymmetric pick selects the genuinely-held value. `BinOp` has no `>`/`&&`/
/// `==>`, so these desugar to `Lt(0, p)` and `Ite` forms.
///
/// # Presence guards
///
/// Both operands are taken as whole [`Chunk`]s so the two sides' **presence
/// guards** cannot be forgotten — an earlier signature took bare amounts, and
/// both call sites silently summed a guarded chunk's amount as if it were held
/// unconditionally. The bound axiom then folded the over-full total to `false`
/// and made the whole unit inconsistent, so `if (b) { inhale acc(x.f) }` followed
/// by any inhale at `x.f` proved everything. See
/// `tests/cases/failing/guarded_merge_chunks.vpr`.
///
/// The rule is [`union_heaps`]'s, which had it right all along:
/// - **guards equal** — the two are present in exactly the same region, so their
///   sum is held there too: keep the amounts bare and carry the shared guard.
///   Flat, no `ite` minted, and the unconditional hot path (both guards empty)
///   is byte-identical to before.
/// - **guards differ** — no single flat cube describes the sum, so fold each
///   guard into its own amount (`guard ? p : 0`) and hand back an honestly
///   unconditional chunk, with the conditionality inside the total.
///
/// The gated amounts flow into [`joint_value`] too, which is what the fix is
/// worth beyond the bound: its `p0 > 0 && p1 > 0 ==> v0 == v1` reads "both
/// genuinely held" only when the fractions are gated. Ungated, it equated the
/// values of two chunks that are never held at the same time.
///
/// The total of two amounts stays one leaf, never a [`ChunkPerm::Select`]: a
/// `Select`'s arms are *alternatives* (the bound axiom is assumed per leaf,
/// correctly), while these two are *summands* and only their sum is bounded. A
/// concrete leaf is an `AddR` (which const-folds); a wildcard-bearing one is a
/// fresh share carrying facts — see [`perm_add_wildcard`].
fn merge_chunks(
    ctx: &mut VerifyContext<'_>,
    addr: egg::Id,
    a: &Chunk,
    b: &Chunk,
    pc_lits: &[(egg::Id, Polarity)],
) -> Chunk {
    // `Chunk::zip` compares the presences by canonical e-class (`Cube::same`): a
    // false negative is merely conservative — it takes the gated path, still
    // sound — but costs an `ite` per merge.
    let (mut merged, value) = Chunk::zip(ctx, a, b, |ctx, pa, pb| {
        // Neither amount is asked about: an inhale merges at once.
        sum_at_location(
            ctx,
            pa,
            pb,
            (a.stored_value(), b.stored_value()),
            (false, false),
            pc_lits,
        )
    });
    merged.addr = addr;
    merged.set_value(value);
    // A merged value is no single chunk's: it has no recipe.
    merged.recipe = None;
    merged
}

/// `pa + pb` for two amounts held at once at one location (values `va`, `vb`),
/// and the location's value with the agreement between the two assumed
/// ([`joint_value`]; `known` says which amounts are known positive). Shared by
/// the inhale merge ([`merge_chunks`]) and the heap sum ([`union_heaps`]).
pub(super) fn sum_at_location(
    ctx: &mut VerifyContext<'_>,
    pa: &ChunkPerm,
    pb: &ChunkPerm,
    (va, vb): (egg::Id, egg::Id),
    known: (bool, bool),
    pc_lits: &[(egg::Id, Polarity)],
) -> (ChunkPerm, egg::Id) {
    // Structural sum: `perm_add` distributes over the gates' `Select`s, so the
    // total keeps one concrete amount per branch (`Select(b, 1/1+1/2, 0+1/2)`)
    // rather than collapsing to an opaque `AddR` whose positivity nothing can
    // decide. Safe only because the bound axiom gates each leaf by the arm that
    // reaches it (`for_each_leaf_under`) — otherwise an over-bound arm would fold
    // the bound ungated and make the unit inconsistent.
    let (p0, p1) = (pa.to_id(ctx), pb.to_id(ctx));
    // A sum with a wildcard in it is an opaque `AddR` leaf nothing can decide, so
    // it is not built: [`perm_add_wildcard`] replaces the wildcard-bearing leaves
    // with fresh shares carrying the facts instead. Gated on `has_wildcard` first,
    // so a wildcard-free program builds byte-identical terms for one `bool` test.
    let perm = if ctx.has_wildcard && (pa.has_wild() || pb.has_wild()) {
        perm_add_wildcard(ctx, pa, pb)
    } else {
        perm_add(ctx, pa, pb)
    };
    let value = joint_value(ctx, (p0, va), (p1, vb), known, pc_lits);
    (perm, value)
}

/// A location chunk extracted from a heap for the location axioms: its
/// permission, the location group tag, its canonical address, its bound, and the
/// **presence guard** the first four are only meaningful under.
///
/// # The one place that deliberately holds an ungated amount
///
/// [`Chunk`] keeps its amount and presence private precisely so the two cannot drift apart
/// ([`Chunk::gated_perm`]). This struct is the deliberate exception: both axioms
/// need the amount *structural* — the bound is assumed per `Leaf` and
/// non-aliasing sums two bare leaves — so folding the guard into the amount here
/// would defeat them. The gating instead happens on the **fact**, in
/// [`assume_location_axioms`], via the `cubes` meet computed there.
///
/// That is a coupling across ~130 lines, and it is load-bearing: skipping it made
/// `if (b) { inhale acc(x.f) } else { inhale acc(y.f) }` prove `x != y`
/// (`tests/cases/failing/guarded_nonaliasing_two_arms.vpr`). Anything reading
/// `LocationChunk::perm` must pair it with the corresponding entry of `cubes`.
struct LocationChunk {
    /// Kept **structural** — a bounded location's axiom is assumed per leaf, and
    /// non-aliasing only fires for bare (`Leaf`) perms, so a join `Select` never
    /// materializes as an `ite` in the graph.
    perm: ChunkPerm,
    group: lasso::Spur,
    /// The canonical address e-class. Non-aliasing is stated over *locations*,
    /// never over decomposed address arguments — see [`assume_location_axioms`].
    addr: egg::Id,
    bound: Bound,
    /// The chunk's presence verbatim: the flat cube under which this chunk is present
    /// at all. `perm` is the **ungated** amount, so every axiom stated about it
    /// has to be gated by this cube — see [`assume_location_axioms`].
    guard: Cube,
}

/// Extract the location chunks of `h`: for each chunk whose canonical address has
/// an `Addr{group,bound,..}` type (recovered by `infer_type`, so **computed**
/// addresses count too), record its perm, group, bound, and — for a direct
/// `@addr` application — its value-arg e-classes (used by the non-aliasing axiom).
fn location_chunks(ctx: &VerifyContext<'_>, h: &Heap) -> Vec<LocationChunk> {
    let mut out = Vec::new();
    for (kind, chunk) in h.entries() {
        // group/bound come straight from the chunk's location kind (VMIR-sourced) —
        // no inference. The address is the chunk's own identity, so unlike the
        // old argument decomposition there is nothing to recover and nothing that
        // can be missing.
        let (perm, presence) = chunk.amount_and_presence();
        out.push(LocationChunk {
            perm: perm.clone(),
            group: kind.group,
            bound: kind.bound.clone(),
            addr: ctx.egraph.find(chunk.addr),
            guard: presence.clone(),
        });
    }
    out
}

/// `fact ∧ cube`, as a nested `ite` over `false` — the boolean twin of
/// [`VerifyContext::gate_amount_by_pc`]. With an empty cube this is `fact`
/// unchanged (no node).
fn gate_bool_by_cube(
    ctx: &mut VerifyContext<'_>,
    fact: egg::Id,
    cube: &[(egg::Id, Polarity)],
) -> egg::Id {
    let false_ = expr!(ctx, false);
    cube.iter().fold(fact, |acc, (lit, pol)| {
        let arms = if matches!(pol, Polarity::Positive) {
            [*lit, acc, false_]
        } else {
            [*lit, false_, acc]
        };
        ctx.add(Symbolic::Ite(arms))
    })
}

/// Emit the location axioms over `h` after a consolidation. Both are e-graph
/// facts the engine *resolves itself* (no Rust-side const-fold queries):
/// - **bound:** a `Bounded(b)` cell holds `perm ≤ b` — `union((b < perm) ?
///   false : true, true)`; a permission folding to `> b` unions `false == true`,
///   making the unit inconsistent so any goal is dischargeable.
/// - **non-aliasing:** two chunks of the *same* bounded location satisfy
///   `(permᵢ + permⱼ > b) ⟹ ¬(args equal)`, encoded as
///   `union(conj, (b < sum) ? false : conj)` where `conj = a0==b0 && a1==b1 …`.
///   When `b < sum` folds true the `ite` collapses `conj` to `false`. For arity
///   1 this is the single-`Eq` collapse (drives `a0 != a1`); for higher arity it
///   sets the whole conjunction false.
///
/// Unbounded locations (predicates) never participate.
///
/// # Both axioms are stated under the chunks' presence guards
///
/// `Chunk.perm` is the **ungated** amount: since the guard-hoisted merge a
/// conditionally-held chunk keeps a bare `Leaf` amount and carries its
/// conditionality in its presence, so reading the amount raw reports a
/// conditional footprint as fully held (the same hazard `View::point_shares`
/// documents). Neither axiom is valid off-guard:
/// - the bound would constrain an amount at a location that is not held there,
///   and an off-guard residual folding above `b` makes the whole unit
///   inconsistent — i.e. proves everything;
/// - non-aliasing would derive `addrᵢ ≠ addrⱼ` for two chunks that are never
///   held at once. Two complementary one-sided chunks of a single join
///   (`[c+]`/`[c−]`, `1/1` each) sum above the bound on paper while the program
///   holds at most one of them, so `if (b) { inhale acc(x.f) } else { inhale
///   acc(y.f) }` proved `x != y` — unsound, and it verified from `b186cbb`
///   (the guard hoist) until this gating, because the pre-hoist `?:0` encoding
///   left such a chunk's perm a `Select`, which the `as_leaf()` filter below
///   skipped for cost reasons. See `tests/cases/failing/guarded_nonaliasing.vpr`.
///
/// So the guard is folded into the fact rather than assumed around it: the bound
/// becomes `guard ⇒ perm ≤ b` and non-aliasing fires on `gt ∧ guardᵢ ∧ guardⱼ`.
/// Both stay valid on *every* path, so the unconditional `union` — and with it
/// the collapse mechanics both axioms rely on — is unchanged, and no
/// [`VerifyContext::assume_guarded`] implication chain is built. With empty guards this is byte-identical to the pre-gating
/// encoding, so the unconditional hot path pays nothing.
pub(super) fn assume_location_axioms(ctx: &mut VerifyContext<'_>, h: &Heap) {
    let chunks = location_chunks(ctx, h);
    if chunks.is_empty() {
        return;
    }
    let true_ = expr!(ctx, true);

    // The cube each chunk is actually present under: the heap's region (it only
    // exists there) meet the chunk's own presence. Not the walker's ambient cube,
    // which is empty in a join phase, where a merge and a loop head's invariant
    // inhale build their heaps. A contradiction means the chunk is unreachable
    // here, so it states nothing.
    let cubes: Vec<Option<Vec<(egg::Id, Polarity)>>> =
        chunks.iter().map(|c| c.guard.within(ctx, &h.region)).collect();

    // Bound: perm ≤ b at each bounded location — assumed **per leaf** of the
    // (possibly branch-structured) perm, so a join `Select` never materializes.
    // A literal leaf (`1/1`, `0`) folds the axiom to a tautology (no node kept).
    for (c, cube) in chunks.iter().zip(&cubes) {
        let Bound::Bounded(b) = &c.bound else {
            continue;
        };
        let Some(cube) = cube else { continue };
        let b = ctx.add(Symbolic::Lit(Literal::Real(b.clone())));
        // Each leaf carries the `Select` conditions that reach it: a leaf states the
        // permission only *under its own arm*, so the fact is gated by the chunk's
        // presence cube AND that arm cube. Gating by the chunk cube alone is sound
        // only while every leaf independently respects the bound — an invariant the
        // sum sites (`merge_chunks`, `union_heaps`) preserve by flattening a total
        // into one opaque `AddR`, and which arm-gating here is what would let them
        // stop doing.
        let mut leaves: Vec<(egg::Id, Vec<(egg::Id, Polarity)>)> = Vec::new();
        c.perm
            .for_each_leaf_under(&mut |l, arm| leaves.push((l, arm.to_vec())));
        for (leaf, arm) in leaves {
            let le = expr!(ctx, not({ b } < r { leaf }));
            // `cube ∧ arm ⇒ leaf ≤ b`; with both empty `implication` returns `le`
            // itself, so the unconditional hot path is the old `union(le, true_)`.
            let Some(gate) = meet(ctx, cube, &arm) else {
                // Arm contradicts the chunk's presence cube — that leaf describes no
                // reachable state and states nothing.
                continue;
            };
            let fact = ctx.implication(le, gate.iter().rev().copied());
            ctx.union(fact, true_);
        }
    }

    // Non-aliasing, stated over **locations**: two chunks of the same bounded
    // group whose perms sum above the bound must sit at different addresses.
    //
    //     permᵢ + permⱼ > b  ⟹  addrᵢ ≠ addrⱼ
    //
    // encoded as `union(eq, ite(gt, false, eq))` — when `gt` folds true the `ite`
    // collapses `eq` to `false`.
    //
    // Stated over the address (the chunk's own identity) rather than over the
    // decomposed `@addr` arguments: those args are recoverable only for a *direct*
    // application, and a missing-args conjunction is empty, i.e. `true` — which
    // made the axiom read `true == false` as soon as two `1/1` field chunks summed
    // above the bound, rendering the whole unit inconsistent.
    //
    // Where the two addresses are already the same e-class this does derive
    // `false` — correctly: `1/1` of `x.f` plus `1/1` of `y.f` with `x == y` is
    // `2/1` at one location, so that state is unreachable.
    //
    // Only fires for bare (`Leaf`) perms — a branch-structured perm would need
    // the sum materialized; skipping only loses a disequality, and the merged heap
    // holds one leaf-perm chunk per location. Purely a cost filter: conditional
    // presence is handled by the guard meet below, not by this skip.
    for i in 0..chunks.len() {
        for j in (i + 1)..chunks.len() {
            if chunks[i].group != chunks[j].group {
                continue;
            }
            let Bound::Bounded(b) = &chunks[i].bound else {
                continue;
            };
            let (Some(pi), Some(pj)) = (chunks[i].perm.as_leaf(), chunks[j].perm.as_leaf()) else {
                continue;
            };
            // The two amounts are held simultaneously only where BOTH presence
            // cubes hold. Disjoint cubes (the complementary one-sided chunks of one
            // join) mean the pair never coexists, so their sum describes no
            // reachable state and no disequality may be drawn from it at all.
            let (Some(ci), Some(cj)) = (&cubes[i], &cubes[j]) else {
                continue;
            };
            let Some(guard) = meet(ctx, ci, cj) else {
                continue;
            };
            let b = ctx.add(Symbolic::Lit(Literal::Real(b.clone())));
            let gt = expr!(ctx, { b } < r({ pi } + r { pj }));
            // Fold the joint guard into the trigger, preserving the
            // `union(eq, ite(gt, false, eq))` collapse shape — the equation stays
            // valid on every path, so the union may stay unconditional.
            let gt = gate_bool_by_cube(ctx, gt, &guard);
            // Both operand orders — the `!=` goal's `Eq` order is source-dependent.
            for (x, y) in [
                (chunks[i].addr, chunks[j].addr),
                (chunks[j].addr, chunks[i].addr),
            ] {
                let eq = expr!(ctx, { x } == { y });
                // `gt ==> addr_x != addr_y`: when `gt` folds true the `ite`
                // collapses `eq` to `false`.
                let imp = expr!(ctx, if {gt} then false else {eq});
                ctx.union(eq, imp);
            }
        }
    }
    ctx.egraph.rebuild();
}

/// The consolidating chunk lookup shared by [`heap_union`] and
/// [`heap_subtract`]: find the chunk at `addr` (canonical-address match) in
/// `kind`'s group, first re-merging any stored chunks whose addresses have
/// *become* one e-class since insertion (a union learned later — e.g. an
/// aliasing fact — splits the held permission invisibly across chunks; the
/// permission checks only ever see one chunk, so the split loses permission).
/// On a lookup miss, normalize once (`reduce`) and retry: a recipe-rebuilt
/// snapshot address spine may only meet the held chunk's class after the
/// terminating reductions collapse the snapshot towers.
///
/// Returns the (possibly consolidated) heap and the chunk found at `addr`.
///
/// `retry_on_miss` gates the normalize-and-retry: a subtract miss is an error
/// about to be reported (rare — always worth one `reduce`), while a union miss
/// is the ordinary "first chunk at this location" case, where a per-inhale
/// `reduce` is pure overhead.
pub(in crate::verify::heap) fn find_chunk_consolidated(
    ctx: &mut VerifyContext<'_>,
    h: &Heap,
    kind: &LocationKind,
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
    retry_on_miss: bool,
) -> (Heap, Option<Chunk>) {
    // Address matching is ground e-class equality. Chunk canons are snapshotted up
    // front because the retry below mutates the graph, and canons from before and
    // after a `reduce` are not comparable.
    let chunks_vec: Vec<Chunk> = h.chunks_of(kind).to_vec();
    let canon = ctx.egraph.find(addr);
    let chunk_canons: Vec<egg::Id> = chunks_vec.iter().map(|c| ctx.egraph.find(c.addr)).collect();
    let collect_found = |canon: egg::Id, chunk_canons: &[egg::Id]| -> Vec<Chunk> {
        chunks_vec
            .iter()
            .zip(chunk_canons)
            .filter(|(_, cc)| **cc == canon)
            .map(|(c, _)| c.clone())
            .collect()
    };
    let mut found = collect_found(canon, &chunk_canons);
    // Retry-on-miss: the subtracted address may be a recipe-rebuilt snapshot spine
    // (`@addr(cons.0(f(.., Some(unwrap(proj_0(s))))))`) that only meets the held
    // chunk's address after the terminating reductions collapse the snapshot towers.
    if found.is_empty() && retry_on_miss {
        ctx.reduce();
        let canon = ctx.egraph.find(addr);
        let chunk_canons: Vec<egg::Id> =
            chunks_vec.iter().map(|c| ctx.egraph.find(c.addr)).collect();
        found = collect_found(canon, &chunk_canons);
    }
    let Some(first) = found.first().cloned() else {
        return (h.clone(), None);
    };
    if found.len() == 1 {
        return (h.clone(), Some(first));
    }
    // Several stored chunks collapsed into one address class: fold them into
    // one chunk (sound — they genuinely alias, so their fractions add and the
    // golden-rule value agreement applies) and rewrite the group.
    let mut out = h.clone();
    let mut acc = first;
    for next in &found[1..] {
        out = out.without_chunk(kind, next.addr);
        // Keep `acc`'s stored address as the key; `merge_chunks` folds in both
        // presence guards.
        acc = merge_chunks(ctx, acc.addr, &acc, next, pc_lits);
    }
    out = out.with_chunk(kind, acc.clone());
    (out, Some(acc))
}

/// Heap addition for a single location chunk of kind `kind`.
pub(in crate::verify::heap) fn heap_union(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    chunk2: Chunk,
    pc_lits: &[(egg::Id, Polarity)],
) -> Heap {
    let (mut out, existing) = find_chunk_consolidated(ctx, h1, kind, chunk2.addr, pc_lits, false);
    if let Some(existing) = existing {
        // Replace the existing chunk in place (keep its stored address key).
        // `merge_chunks` reconciles the two presence guards — the held chunk may
        // be conditional (a one-sided join arm) while an inhaled `chunk2` is not.
        let merged = merge_chunks(ctx, existing.addr, &existing, &chunk2, pc_lits);
        out = out.with_chunk(kind, merged);
    } else {
        out = out.with_chunk(kind, chunk2);
    }
    assume_location_axioms(ctx, &out);
    out
}
