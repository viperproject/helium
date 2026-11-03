//! Heap algebra: the operations over [`Heap`]/[`Chunk`]/[`ChunkPerm`] that add,
//! subtract, join and summarize permission at a location, plus the permission
//! proofs and location axioms they rest on.
//!
//! Split out of `verify::declaration`, which had grown to hold the whole verifier
//! in one file. The boundary is *state*: nothing here sees an `EvalState` or a
//! VMIR `Inst` — these functions take a heap, a location, a permission and a path
//! condition, and hand back a heap. The instruction walk that drives them stays in
//! `declaration`.
//!
//! The three entry points are [`heap_union`] (produce), [`heap_subtract`]
//! (consume) and [`merge_heaps`]/[`union_heaps`] (control-flow join and loop frame
//! restore). Everything else supports those.

use crate::verify::{
    context::VerifyContext,
    error::VerifyError,
    heap::{Chunk, ChunkPerm, Heap, HeapPc, LocationKind, cube_eq, cube_meet, cube_push},
    lang::Symbolic,
};
use crate::vmir::{BinOp, Bound, Literal, Polarity};

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
/// The gated amounts flow into [`merge_values`] too, which is what the fix is
/// worth beyond the bound: its `p0 > 0 && p1 > 0 ==> v0 == v1` reads "both
/// genuinely held" only when the fractions are gated. Ungated, it equated the
/// values of two chunks that are never held at the same time.
///
/// The total of two amounts stays one leaf, never a [`ChunkPerm::Select`]: a
/// `Select`'s arms are *alternatives* (the bound axiom is assumed per leaf,
/// correctly), while these two are *summands* and only their sum is bounded. A
/// concrete leaf is an `AddR` (which const-folds); a wildcard-bearing one is a
/// fresh share carrying facts — see [`perm_add_wildcard`].
pub(crate) fn merge_chunks(
    ctx: &mut VerifyContext<'_>,
    addr: egg::Id,
    a: &Chunk,
    b: &Chunk,
    pc_lits: &[(egg::Id, Polarity)],
) -> Chunk {
    // `cube_eq` (canonical e-class comparison), not the `Rc<[..]>` structural
    // equality `union_heaps` uses: a false negative is merely conservative — it
    // takes the gated path, still sound — but costs an `ite` per merge.
    let (pa, pb, guard): (ChunkPerm, ChunkPerm, crate::verify::heap::HeapPc) =
        if cube_eq(ctx, a.guard(), b.guard()) {
            // Shared cube: amounts stay bare and the guard is re-attached below.
            let g: HeapPc = a.guard_pc();
            (a.ungated_perm().clone(), b.ungated_perm().clone(), g)
        } else {
            (
                a.gated_perm(ctx),
                b.gated_perm(ctx),
                std::rc::Rc::from(Vec::new()),
            )
        };
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
        perm_add_wildcard(ctx, &pa, &pb)
    } else {
        perm_add(ctx, &pa, &pb)
    };
    let value = merge_values(ctx, p0, a.value, p1, b.value, pc_lits);
    Chunk::new_perm(addr, perm, value).with_guard(guard)
}

/// [`perm_add`], but a leaf whose sum mentions a `wildcard` becomes a **fresh
/// share carrying facts** instead of an `AddR` node.
///
/// The e-graph has no real-order arithmetic, so an `AddR` over a wildcard is an
/// opaque leaf: `0 < 1/2 + w` is underivable even though `0 < w` was assumed at
/// mint, and so is `1/2 < 1/2 + w`. Building the node and then bolting facts onto
/// it means restating those facts over a tree that keeps getting deeper. Building
/// a fresh `s` instead states the same thing once, at any depth:
///
/// - `0 < s` — free, from [`VerifyContext::fresh_wildcard`]'s mint-time assumption;
/// - `x < s` for each summand `x` whose **other** summand is positive.
///
/// That second fact is what the `AddR` never gave: it makes `perm(x.f) > 1/2`
/// provable after inhaling a wildcard onto `1/2`, and — meeting the bound axiom's
/// `not (b <r leaf)` in [`assume_location_axioms`] — makes `inhale write; inhale
/// wildcard` inconsistent, which is Silicon's verdict on that state.
///
/// Silicon materializes the same sums we do and leans on Z3's linear real
/// arithmetic (`PermPlus` → `mkAdd`); it has no wildcard case for `+` at all. What
/// it *does* do is throw a wildcard's magnitude away under `*`
/// (`WildcardSimplifyingPermTimes`: `w * q` with `q` a positive literal collapses
/// to `w`), which is the same observation from the other side — a wildcard carries
/// no magnitude, only a sign.
///
/// # Per leaf, not per tree
///
/// Both tests are made on the **leaf**, after [`perm_add`]'s distribution over the
/// operands' `Select`s has already split the guard arms apart:
///
/// - **`contains_wildcard`** — a concrete leaf keeps its `AddR`, which const-folds
///   (`1/2 + 0` is `1/2`, not an opaque symbol). Replacing those would *lose*
///   information, so the arm where a gated wildcard chunk is absent — a literal
///   `0` leaf — is untouched.
/// - **[`perm_known_positive`]** — a fresh share may only be minted where the sum
///   really is positive. A conditionally-present wildcard (`ite(g, w, 0)`) is not
///   positive, so a leaf with no positive summand keeps its `AddR` and states
///   nothing, which is the truth.
///
/// Both facts are **ungated**: [`perm_known_positive`] is a structural,
/// path-independent judgement, so `x < x + y` with `y` positive holds on every
/// path. The arm-dependence lives entirely in *which leaf* — which the per-leaf
/// positivity test already decides — not in whether the leaf's fact holds. This is
/// why no pc/guard cube is threaded here, unlike the bound axiom.
fn perm_add_wildcard(ctx: &mut VerifyContext<'_>, a: &ChunkPerm, b: &ChunkPerm) -> ChunkPerm {
    match (a, b) {
        (ChunkPerm::Leaf { id: x, wild: wx }, ChunkPerm::Leaf { id: y, wild: wy }) => {
            let (x, y, wild) = (*x, *y, *wx || *wy);
            let (xpos, ypos) = (perm_known_positive(ctx, x), perm_known_positive(ctx, y));
            if !wild || !(xpos || ypos) {
                return ChunkPerm::Leaf {
                    id: ctx.add(Symbolic::Binary(BinOp::AddR, [x, y])),
                    wild,
                };
            }
            // `0 < s` comes with the mint.
            let s = ctx.fresh_wildcard();
            // `x < x + y` needs *y* strictly positive, and vice versa. The other
            // summand's non-negativity is the chunk-permission invariant, not a
            // proof: every permission reaching a chunk has passed `Combine`'s
            // `perm ≥ 0`, or — for the wildcard-bearing perms that skip it
            // (a [`crate::vmir::PermVal`]) — is non-negative by construction.
            let mut facts: Vec<egg::Id> = Vec::new();
            if ypos {
                facts.push(expr!(ctx, { x } < r { s }));
            }
            if xpos {
                facts.push(expr!(ctx, { y } < r { s }));
            }
            ctx.assume_all_guarded(facts, &[]);
            ChunkPerm::wild_leaf(s)
        }
        // Descend on `a` (this arm also covers `Select`/`Select`).
        (ChunkPerm::Select { cond, then, els }, other) => {
            let ot = ChunkPerm::restrict(ctx, *cond, other.clone(), true);
            let oe = ChunkPerm::restrict(ctx, *cond, other.clone(), false);
            let t = perm_add_wildcard(ctx, then, &ot);
            let e = perm_add_wildcard(ctx, els, &oe);
            ChunkPerm::select(ctx, *cond, t, e)
        }
        (other, ChunkPerm::Select { cond, then, els }) => {
            let ot = ChunkPerm::restrict(ctx, *cond, other.clone(), true);
            let oe = ChunkPerm::restrict(ctx, *cond, other.clone(), false);
            let t = perm_add_wildcard(ctx, &ot, then);
            let e = perm_add_wildcard(ctx, &oe, els);
            ChunkPerm::select(ctx, *cond, t, e)
        }
    }
}

/// Syntactic `0 < t`, decided by structure alone — no prover call, no saturation.
///
/// A wildcard is positive by its mint-time assumption; a real literal by its
/// value; a sum by one positive and one non-negative summand; a product by two
/// positive factors; an `ite` when **both** arms are. `SubR` is deliberately
/// absent: a remainder `held − w` is positive only under the pc that assumed it,
/// which is a fact about a path, not about the term.
///
/// The visited set only breaks cycles — ids are removed on the way out, so a
/// shared subterm is not poisoned by an in-progress ancestor.
fn perm_known_positive(ctx: &VerifyContext<'_>, id: egg::Id) -> bool {
    perm_sign(ctx, id, true, &mut crate::dhash::HashSet::default())
}

/// Whether `0 < id` is already a **proven** fact in the graph, by pure lookup: the
/// `<r` node must exist and its class be the `true` class. Both are true for a
/// freshly minted share, whose positivity is unioned into `true` at the mint.
///
/// Deliberately not a `prove_under_pc`: this runs inside [`perm_sign`], a structural
/// path-independent judgement on the *term*, and a probe there would put a prover
/// call under every leaf of every merge.
fn positivity_known(ctx: &VerifyContext<'_>, id: egg::Id) -> bool {
    let zero = num::BigRational::from(num::BigInt::from(0));
    let Some(z) = ctx.egraph.lookup(Symbolic::Lit(Literal::Real(zero))) else {
        return false;
    };
    let Some(lt) = ctx
        .egraph
        .lookup(Symbolic::Binary(BinOp::LtR, [z, ctx.egraph.find(id)]))
    else {
        return false;
    };
    matches!(
        ctx.egraph[ctx.egraph.find(lt)].data.known(),
        Some(Literal::Bool(true))
    )
}

/// `strict`: `0 < t`. Otherwise `0 ≤ t`, which additionally admits `0` itself and
/// a sum/product/`ite` of non-negatives.
fn perm_sign(
    ctx: &VerifyContext<'_>,
    id: egg::Id,
    strict: bool,
    seen: &mut crate::dhash::HashSet<(egg::Id, bool)>,
) -> bool {
    let id = ctx.egraph.find(id);
    if !seen.insert((id, strict)) {
        return false;
    }
    // A known literal settles the class outright, whatever nodes it holds.
    let out = match ctx.egraph[id].data.known() {
        Some(Literal::Real(r)) => {
            let zero = num::BigRational::from(num::BigInt::from(0));
            if strict { *r > zero } else { *r >= zero }
        }
        // A standing positivity fact settles it too, without inspecting nodes: a
        // wildcard is minted with `0 < w` unioned into `true`, and so is every fresh
        // share `perm_add_wildcard` hands back. Reading the fact rather than
        // recognising the `Wildcard` node keeps this true of any leaf the graph
        // happens to know is positive, and is what lets the node itself go away.
        // A pure `lookup` — no node is added, so asking never grows the graph.
        _ if strict && positivity_known(ctx, id) => true,
        // Any node witnessing the sign settles it: all nodes of a class are equal.
        _ => ctx.egraph[id].nodes.iter().any(|n| match n {
            Symbolic::Ite([_, t, e]) => {
                perm_sign(ctx, *t, strict, seen) && perm_sign(ctx, *e, strict, seen)
            }
            Symbolic::Binary(BinOp::AddR, [x, y]) => {
                if strict {
                    (perm_sign(ctx, *x, true, seen) && perm_sign(ctx, *y, false, seen))
                        || (perm_sign(ctx, *y, true, seen) && perm_sign(ctx, *x, false, seen))
                } else {
                    perm_sign(ctx, *x, false, seen) && perm_sign(ctx, *y, false, seen)
                }
            }
            Symbolic::Binary(BinOp::MulR, [x, y]) => {
                perm_sign(ctx, *x, strict, seen) && perm_sign(ctx, *y, strict, seen)
            }
            _ => false,
        }),
    };
    seen.remove(&(id, strict));
    out
}

/// The value of a location covered by two chunks, when it is **not** already
/// known that both are held.
///
/// Yields `p0 > 0 ? v0 : v1` and assumes `(PC ∧ p0 > 0 ∧ p1 > 0) ==> v0 == v1`.
/// Shared by the inhale path ([`merge_chunks`]) and the loop frame restore
/// ([`union_heaps`]); the two used to disagree, and the latter's unconditional
/// `union(v0, v1)` was unsound — see the note there.
///
/// Silicon's `combineSnapshots` splits the same three ways and, when neither
/// fraction is definitely positive, mints a *fresh* snapshot constrained by both
/// implications, with the comment "it is not sound to use t1 or t2 and constrain
/// it". The asymmetric ternary is the same statement made total: the value is
/// case-analysed on `p0 > 0` rather than left unconstrained, which is strictly
/// more precise and needs no fresh symbol.
/// Assume `(PC ∧ p > 0) ==> v == other`, where `v` is the value of a chunk already
/// known to be held and `p` is the *other* chunk's fraction.
///
/// The one-sided half of [`merge_values`], for when one side's positivity is
/// settled: the ternary would reduce to `v` anyway, so only the conditional
/// agreement is left to state. Silicon's `combineSnapshots` cases `(True, b2)` and
/// `(b1, True)` are exactly this.
pub(crate) fn assume_values_agree(
    ctx: &mut VerifyContext<'_>,
    p: egg::Id,
    v: egg::Id,
    other: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) {
    let p_pos = expr!(ctx, (0 / 1) < r { p });
    let eq = expr!(ctx, { v } == { other });
    let antecedents = [(p_pos, Polarity::Positive)]
        .into_iter()
        .chain(pc_lits.iter().rev().copied());
    let imp = ctx.implication(eq, antecedents);
    let true_ = expr!(ctx, true);
    ctx.union(imp, true_);
}

pub(crate) fn merge_values(
    ctx: &mut VerifyContext<'_>,
    p0: egg::Id,
    v0: egg::Id,
    p1: egg::Id,
    v1: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> egg::Id {
    let zero = ctx.add(Symbolic::Lit(Literal::Real(num::BigRational::from(
        num::BigInt::from(0),
    ))));
    let p0_pos = expr!(ctx, { zero } < r { p0 });
    let p1_pos = expr!(ctx, { zero } < r { p1 });

    let value = expr!(ctx, if {p0_pos} then {v0} else {v1});

    // `(PC ∧ p0 > 0 ∧ p1 > 0) ==> (v0 == v1)` as the golden-rule ITE chain.
    // Fold innermost-first: p1_pos, p0_pos, then PC literals in reverse.
    let true_ = expr!(ctx, true);
    let eq = expr!(ctx, { v0 } == { v1 });
    let antecedents = [(p1_pos, Polarity::Positive), (p0_pos, Polarity::Positive)]
        .into_iter()
        .chain(pc_lits.iter().rev().copied());
    let imp = ctx.implication(eq, antecedents);
    ctx.union(imp, true_);

    value
}

/// A location chunk extracted from a heap for the location axioms: its
/// permission, the location group tag, its canonical address, its bound, and the
/// **presence guard** the first four are only meaningful under.
///
/// # The one place that deliberately holds an ungated amount
///
/// [`Chunk`] keeps `perm`/`guard` private precisely so the two cannot drift apart
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
pub(crate) struct LocationChunk {
    /// Kept **structural** — a bounded location's axiom is assumed per leaf, and
    /// non-aliasing only fires for bare (`Leaf`) perms, so a join `Select` never
    /// materializes as an `ite` in the graph.
    perm: ChunkPerm,
    group: lasso::Spur,
    /// The canonical address e-class. Non-aliasing is stated over *locations*,
    /// never over decomposed address arguments — see [`assume_location_axioms`].
    addr: egg::Id,
    bound: Bound,
    /// [`Chunk::guard`] verbatim: the flat cube under which this chunk is present
    /// at all. `perm` is the **ungated** amount, so every axiom stated about it
    /// has to be gated by this cube — see [`assume_location_axioms`].
    guard: crate::verify::heap::HeapPc,
}

/// Extract the location chunks of `h`: for each chunk whose canonical address has
/// an `Addr{group,bound,..}` type (recovered by `infer_type`, so **computed**
/// addresses count too), record its perm, group, bound, and — for a direct
/// `@addr` application — its value-arg e-classes (used by the non-aliasing axiom).
pub(crate) fn location_chunks(ctx: &VerifyContext<'_>, h: &Heap) -> Vec<LocationChunk> {
    let mut out = Vec::new();
    for (kind, chunk) in h.entries() {
        // group/bound come straight from the chunk's location kind (VMIR-sourced) —
        // no inference. The address is the chunk's own identity, so unlike the
        // old argument decomposition there is nothing to recover and nothing that
        // can be missing.
        out.push(LocationChunk {
            perm: chunk.ungated_perm().clone(),
            group: kind.group,
            bound: kind.bound.clone(),
            addr: ctx.egraph.find(chunk.addr),
            guard: chunk.guard_pc(),
        });
    }
    out
}

/// `fact ∧ cube`, as a nested `ite` over `false` — the boolean twin of
/// [`VerifyContext::gate_amount_by_pc`]. With an empty cube this is `fact`
/// unchanged (no node).
pub(crate) fn gate_bool_by_cube(
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
/// conditionality in [`Chunk::guard`], so reading `perm` raw reports a
/// conditional footprint as fully held (the same hazard [`summarize_perm_at`]
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
/// [`VerifyContext::assume_guarded`] (whose scratch half is unguarded, invariant
/// 4) is involved. With empty guards this is byte-identical to the pre-gating
/// encoding, so the unconditional hot path pays nothing.
pub(crate) fn assume_location_axioms(ctx: &mut VerifyContext<'_>, h: &Heap) {
    let chunks = location_chunks(ctx, h);
    if chunks.is_empty() {
        return;
    }
    let true_ = expr!(ctx, true);

    // The cube each chunk is actually present under: the block's control cube
    // (this heap only exists on that path — `heap_union` runs per `inhale`, inside
    // a block) meet the chunk's own residual merge guard. A contradiction means the
    // chunk is unreachable here, so it states nothing.
    let block: Vec<(egg::Id, Polarity)> = ctx.current_cube().to_vec();
    let cubes: Vec<Option<Vec<(egg::Id, Polarity)>>> = chunks
        .iter()
        .map(|c| cube_meet(ctx, &block, &c.guard))
        .collect();

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
            let Some(gate) = cube_meet(ctx, cube, &arm) else {
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
            let Some(guard) = cube_meet(ctx, ci, cj) else {
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
pub(crate) fn find_chunk_consolidated(
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
pub(crate) fn heap_union(
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

/// Prove a per-leaf permission predicate over a (possibly branch-structured)
/// [`ChunkPerm`] **WITHOUT materializing the `Select` into the ground graph**.
/// A `Select` pushes its condition into the pc and requires the predicate on
/// both arms; a `Leaf` discharges it under the accumulated pc. This is the CFG
/// case split done through the pc — no `ite` node, no `lt-ite`, no goal-directed split, and
/// (crucially) no perm-tower bloat in the live graph. The predicate is built
/// from a leaf amount id by `mk`.
pub(crate) fn prove_perm_leaves(
    ctx: &mut VerifyContext<'_>,
    perm: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
    mk: &dyn Fn(&mut VerifyContext<'_>, egg::Id, &[(egg::Id, Polarity)]) -> bool,
) -> bool {
    match perm {
        ChunkPerm::Leaf { id, .. } => mk(ctx, *id, pc_lits),
        ChunkPerm::Select { cond, then, els } => {
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((*cond, Polarity::Positive));
            if !prove_perm_leaves(ctx, then, &pc_t, mk) {
                return false;
            }
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((*cond, Polarity::Negative));
            prove_perm_leaves(ctx, els, &pc_e, mk)
        }
    }
}

/// The folded rational value of an e-class, if known — used by the per-leaf
/// fast paths to decide a comparison against a literal WITHOUT a prove_under_pc.
pub(crate) fn known_real(ctx: &VerifyContext<'_>, id: egg::Id) -> Option<num::BigRational> {
    match ctx.egraph[ctx.egraph.find(id)].data.known() {
        Some(Literal::Real(r)) => Some(r.clone()),
        _ => None,
    }
}

/// `held ≥ needed` over a structured `held`, per leaf (no `Select` in the graph).
pub(crate) fn prove_sufficient(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    prove_perm_leaves(ctx, held, pc_lits, &move |ctx, h, pc| {
        sufficient_leaf(ctx, h, needed, pc, &mut Vec::new())
    })
}

/// `held ≥ needed` descending the **demand's** branch structure as well as the
/// held one — the dual of [`prove_perm_leaves`], which splits only what the heap
/// holds.
///
/// A gated `acc` is why this exists. `predicate q(b, x) { b ==> acc(x.f) }` demands
/// `b ? 1/1 : 0` of a chunk holding a flat `1/1`, and proving
/// `not (1/1 < ite(b, 1/1, 0))` outright needs `b` decided up front.
/// [`sufficient_leaf`] has a case-split fallback for exactly that shape, but it
/// splits the *held* amount, and here the held side is the flat one. Splitting the
/// demand instead asks `1/1 ≥ 1/1` under `b` and `1/1 ≥ 0` under `¬b`, both trivial.
/// For an enum predicate — one gated slot per variant — each slot is discharged
/// under its own discriminant test.
///
/// Zero arms cost nothing: a demand leaf that const-folds to `0` is discharged
/// without a probe, so an N-variant enum pays for the one live arm rather than two
/// probes per dead one. That is what keeps pushing demand literals onto the pc from
/// multiplying with the arms it introduces.
pub(crate) fn prove_sufficient_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    match needed {
        ChunkPerm::Leaf { id: n, .. } => {
            // Nothing demanded on this arm: every held amount is ≥ 0.
            if let Some(r) = known_real(ctx, *n) {
                if r <= num::BigRational::from(num::BigInt::from(0)) {
                    return true;
                }
            }
            prove_sufficient(ctx, held, *n, pc_lits)
        }
        ChunkPerm::Select { cond, then, els } => {
            // The demand's own branch, taken both ways: `held ≥ needed` follows by
            // case analysis. `restrict` aligns the held side — it takes the matching
            // arm when held branches on the same condition class and carries the
            // whole tree in when it does not, so an unrelated held structure degrades
            // to today's behaviour rather than being mishandled.
            let ht = ChunkPerm::restrict(ctx, *cond, held.clone(), true);
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((*cond, Polarity::Positive));
            if !prove_sufficient_aligned(ctx, &ht, then, &pc_t) {
                return false;
            }
            let he = ChunkPerm::restrict(ctx, *cond, held.clone(), false);
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((*cond, Polarity::Negative));
            prove_sufficient_aligned(ctx, &he, els, &pc_e)
        }
    }
}

/// `held ≥ needed` for one `ChunkPerm` leaf, falling back to a **case split on an
/// `ite`-shaped amount** when the flat prove fails.
///
/// [`prove_perm_leaves`] already splits the `Select` structure a *join* builds,
/// but an amount minted by unfolding a **conditional footprint** carries its
/// branch inside the graph instead: `p_Shape`'s body gives `p_Shape_1_owned`
/// the amount `ite(discr == cons(1), 1/1, 0)`. Asking `¬(that < 1/1)` outright
/// needs the condition decided up front. Splitting asks it per arm instead —
/// `1/1 ≥ 1/1` under the condition, and `0 ≥ 1/1` under its negation, the
/// second discharged exactly when assuming the negation **refutes itself**.
///
/// That is the same step [`VerifyContext::prove_by_ite_decomposition`] makes for
/// boolean goals, and the same one a branch-splitting verifier gets for free by
/// evaluating the ternary per branch. It is a fallback, not the first move: the
/// flat prove closes the overwhelming majority of leaves, and only the ones it
/// cannot are worth two extra probes.
///
/// `seen` is the classes already split on this path — an `ite` arm can be its own
/// class in a cyclic e-graph, and re-splitting it would not terminate.
pub(crate) fn sufficient_leaf(
    ctx: &mut VerifyContext<'_>,
    h: egg::Id,
    needed: egg::Id,
    pc: &[(egg::Id, Polarity)],
    seen: &mut Vec<egg::Id>,
) -> bool {
    // Fast path: two known literals that already satisfy `h ≥ needed` need
    // no prove (the give-back `1/1 ≥ 1/1` case — the vast majority of leaves).
    // A literal that FAILS may still be a dead branch (`0 ≥ 1` on an
    // infeasible arm), so it falls through to the pc-aware prove.
    if let (Some(hr), Some(nr)) = (known_real(ctx, h), known_real(ctx, needed)) {
        if hr >= nr {
            return true;
        }
    }
    // `held >= needed`, i.e. not (held < needed).
    let goal = expr!(ctx, not({ h } < r { needed }));
    if ctx.prove_under_pc(goal, pc) {
        return true;
    }
    let cls = ctx.egraph.find(h);
    if seen.contains(&cls) {
        return false;
    }
    // Only a **gate** is worth splitting: an `ite` whose two arms are constant
    // amounts, which is the shape a conditional footprint mints (`c ? 1/1 : 0`).
    // Every other `ite` in an amount's class is give-back residue — a perm tower
    // over addresses and merge conditions — where the split has nothing to decide
    // and each arm drags in another two probes. Restricting to the gate is what
    // keeps this a fallback rather than a search.
    let ites: Vec<[egg::Id; 3]> = ctx.egraph[cls]
        .nodes
        .iter()
        .filter_map(|n| match n {
            Symbolic::Ite(arms) => Some(*arms),
            _ => None,
        })
        .filter(|[_, t, e]| known_real(ctx, *t).is_some() && known_real(ctx, *e).is_some())
        .collect();
    seen.push(cls);
    // A class routinely holds several `ite` nodes (every join that produced this
    // amount left one); any of them is a valid split, so try them in turn.
    let proved = ites.into_iter().any(|[c, t, e]| {
        let mut pc_t = pc.to_vec();
        pc_t.push((c, Polarity::Positive));
        let mut pc_e = pc.to_vec();
        pc_e.push((c, Polarity::Negative));
        sufficient_leaf(ctx, t, needed, &pc_t, seen) && sufficient_leaf(ctx, e, needed, &pc_e, seen)
    });
    seen.pop();
    proved
}

/// `0 < held` over a structured `held`, per leaf.
pub(crate) fn prove_perm_positive(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    prove_perm_leaves(ctx, held, pc_lits, &|ctx, h, pc| {
        if let Some(hr) = known_real(ctx, h) {
            if hr > num::BigRational::from(num::BigInt::from(0)) {
                return true;
            }
        }
        let zero = expr!(ctx, 0 / 1);
        let goal = expr!(ctx, { zero } < r { h });
        ctx.prove_under_pc(goal, pc)
    })
}

/// `¬(held < cap)` (full/write permission) over a structured `held`, per leaf.
/// The threshold is the location's own permission cap, not a hardcoded `1/1`:
/// writing needs *all* the permission a cell can carry. A field's cap is `1/1`,
/// so this is the usual obligation; an `Unbounded` location (a predicate) has no
/// full amount to hold, so the write is never provable.
pub(crate) fn prove_perm_write(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    bound: &Bound,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    let Bound::Bounded(cap) = bound else {
        return false;
    };
    let cap = cap.clone();
    prove_perm_leaves(ctx, held, pc_lits, &|ctx, h, pc| {
        if let Some(hr) = known_real(ctx, h) {
            if hr >= cap {
                return true;
            }
        }
        let write = ctx.add(Symbolic::Lit(Literal::Real(cap.clone())));
        // Write permission at this leaf: not (held < cap).
        let goal = expr!(ctx, not({ h } < r { write }));
        ctx.prove_under_pc(goal, pc)
    })
}

/// The structural **sum** of two permission trees, distributing `AddR` over
/// `Select` so the branch structure survives into the result instead of being
/// flattened into an `ite` in the graph.
///
/// That survival is the whole point: a summarized location's total is what
/// [`prove_sufficient`] then discharges **per leaf**, with each branch literal
/// pushed into the pc. Flattening first is what forced the old summarized path to
/// prove one monolithic goal over a sum whose addends are live on different arms —
/// a goal that needs a case split nothing was left to perform.
///
/// Descending into one side restricts the other to the same arm
/// ([`ChunkPerm::restrict`], `ite`-idempotence), so two trees branching on one
/// condition class yield a single `Select` layer rather than two. The result's
/// conditions are exactly the union of the operands' — no condition is invented,
/// so no split is either: the leaf count is fixed by how many distinct branch
/// literals the participating chunks' guards mention, never by a search bound.
pub(crate) fn perm_add(ctx: &mut VerifyContext<'_>, a: &ChunkPerm, b: &ChunkPerm) -> ChunkPerm {
    match (a, b) {
        (ChunkPerm::Leaf { id: x, wild: wx }, ChunkPerm::Leaf { id: y, wild: wy }) => {
            ChunkPerm::Leaf {
                id: ctx.add(Symbolic::Binary(BinOp::AddR, [*x, *y])),
                wild: *wx || *wy,
            }
        }
        // Descend on `a` (this arm also covers `Select`/`Select`).
        (ChunkPerm::Select { cond, then, els }, other) => {
            let ot = ChunkPerm::restrict(ctx, *cond, other.clone(), true);
            let oe = ChunkPerm::restrict(ctx, *cond, other.clone(), false);
            let t = perm_add(ctx, then, &ot);
            let e = perm_add(ctx, els, &oe);
            ChunkPerm::select(ctx, *cond, t, e)
        }
        (other, ChunkPerm::Select { cond, then, els }) => {
            let ot = ChunkPerm::restrict(ctx, *cond, other.clone(), true);
            let oe = ChunkPerm::restrict(ctx, *cond, other.clone(), false);
            let t = perm_add(ctx, &ot, then);
            let e = perm_add(ctx, &oe, els);
            ChunkPerm::select(ctx, *cond, t, e)
        }
    }
}

/// `held − needed`, kept structural (leaves get `SubR`, the tree stays a
/// `Select` via the smart constructor so it never materializes as an `ite`).
pub(crate) fn perm_sub(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: egg::Id,
) -> ChunkPerm {
    match held {
        // The demand is concrete on this path (a wildcard demand goes to
        // `debit_wildcard`), so the remainder's provenance is the held side's.
        ChunkPerm::Leaf { id, wild } => ChunkPerm::Leaf {
            id: ctx.add(Symbolic::Binary(BinOp::SubR, [*id, needed])),
            wild: *wild,
        },
        ChunkPerm::Select { cond, then, els } => {
            let t = perm_sub(ctx, then, needed);
            let e = perm_sub(ctx, els, needed);
            ChunkPerm::select(ctx, *cond, t, e)
        }
    }
}

/// [`perm_sub`], but descending the **demand's** branch structure as well as the
/// held one, so arms that agree on a condition cancel per branch instead of leaving
/// the whole demand inside every leaf.
///
/// A guarded consume is the case that matters. [`perm_sub`] pushes the flat demand
/// into each held leaf, so exhaling `c ? 1/1 : 0` from `1/1` gives
/// `1/1 - (c ? 1/1 : 0/1)` — correct, but opaque: nothing const-folds, and
/// [`perm_all_zero`] cannot see that the chunk is emptied under `c`. Aligning the
/// arms gives `c ? (1/1 - 1/1) : (1/1 - 0/1)`, whose leaves fold to `c ? 0 : 1/1`.
///
/// Alignment is [`ChunkPerm::restrict`]'s job, not ours: it takes the matching arm
/// when the held perm branches on the same condition and carries the whole term in
/// when it does not. So a held perm on an *unrelated* condition degrades to exactly
/// [`perm_sub`]'s behaviour rather than being mishandled — the encoding stays correct
/// and only the precision varies. This is the dual of [`perm_add_wildcard`]'s
/// `Select` arms on the add side.
pub(crate) fn perm_sub_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
) -> ChunkPerm {
    match needed {
        // Demand is flat: nothing to align against, so this is `perm_sub`.
        ChunkPerm::Leaf { id: n, .. } => perm_sub(ctx, held, *n),
        ChunkPerm::Select { cond, then, els } => {
            let ht = ChunkPerm::restrict(ctx, *cond, held.clone(), true);
            let he = ChunkPerm::restrict(ctx, *cond, held.clone(), false);
            let t = perm_sub_aligned(ctx, &ht, then);
            let e = perm_sub_aligned(ctx, &he, els);
            ChunkPerm::select(ctx, *cond, t, e)
        }
    }
}

/// Whether every leaf of `perm` const-folds to `0` (the chunk is emptied on
/// every branch, so it can be dropped). Const-fold only, no `Select` in the graph.
pub(crate) fn perm_all_zero(ctx: &VerifyContext<'_>, perm: &ChunkPerm) -> bool {
    match perm {
        ChunkPerm::Leaf { id, .. } => matches!(
            ctx.egraph[ctx.egraph.find(*id)].data.known(),
            Some(Literal::Real(r)) if *r == num::BigRational::from(num::BigInt::from(0))
        ),
        ChunkPerm::Select { then, els, .. } => perm_all_zero(ctx, then) && perm_all_zero(ctx, els),
    }
}

/// Heap subtraction for a single location chunk of kind `kind`.
pub(crate) fn heap_subtract(
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
pub(crate) enum ConsumePass {
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
pub(crate) enum Demand {
    /// A fixed amount: prove `held >= needed`.
    Concrete,
    /// Mentions a `wildcard`, bare or under gating/scaling: prove `0 < held` and
    /// hand back a fresh smaller share.
    Wildcard,
}

/// Put the found chunk into the form the consume works on, and report the guard the
/// remainder should be stored back under.
///
/// A stored chunk keeps a FLAT presence guard and a guard-free amount, so there are
/// two ways to make it consumable, and which one applies is decided by the pc:
///
/// - **pc does not entail the guard** — fold `guard ? perm : 0` into the amount and
///   hand back a chunk that is unconditional by construction. Sufficiency and the
///   remainder are then proven against a perm that is present only where the guard
///   holds. The residual guard is empty: the conditionality now lives in the amount.
/// - **pc already entails the guard** — the consume happens inside the very region
///   the chunk is present in, so the gate is a no-op (the pc decides the `ite` arms
///   anyway) and folding it in is actively *harmful*. The remainder is stored as a
///   bare amount, so folding would **destroy** the flat cube on the first consume at
///   the location, leaving the chunk's conditionality visible only as `ite`s buried
///   in its arithmetic. A later join-level consume then has no structure to
///   case-split on — precisely how the `&mut` reborrow shape lost its proof. Keep
///   the amount bare and carry the cube through untouched.
///
/// Returns the chunk to consume against and the guard to re-attach to the remainder
/// (empty in the first case, the chunk's own cube in the second).
pub(crate) fn normalize_guard_for_consume(
    ctx: &mut VerifyContext<'_>,
    existing: Option<Chunk>,
    pc_lits: &[(egg::Id, Polarity)],
) -> (Option<Chunk>, crate::verify::heap::HeapPc) {
    let empty: crate::verify::heap::HeapPc = std::rc::Rc::from(Vec::new());
    let Some(c) = existing else {
        return (None, empty);
    };
    if c.guard().is_empty() {
        return (Some(c), empty);
    }
    if c.pc_entails_guard(ctx, pc_lits) {
        let kept = c.guard_pc();
        (Some(c), kept)
    } else {
        let gated = c.gated_perm(ctx);
        (Some(c.with_perm(gated).with_guard(empty.clone())), empty)
    }
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
pub(crate) fn heap_subtract_inner(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    chunk2: Chunk,
    pc_lits: &[(egg::Id, Polarity)],
    demand: Demand,
    pass: ConsumePass,
) -> Result<Heap, VerifyError> {
    let (out, existing) = find_chunk_consolidated(ctx, h1, kind, chunk2.addr, pc_lits, true);
    let (existing, kept) = normalize_guard_for_consume(ctx, existing, pc_lits);
    let chunk2_perm = chunk2.ungated_perm().to_id(ctx);
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
        return debit_wildcard(ctx, out, kind, &existing, &chunk2, pc_lits, kept);
    }

    // Sufficiency `held ≥ needed`, proven per-leaf over the (possibly
    // branch-structured) held perm — the `Select` never enters the graph.
    let proven =
        prove_sufficient_aligned(ctx, existing.ungated_perm(), chunk2.ungated_perm(), pc_lits);
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
        // the `SlotPerm::Presence` framing check. This is the hot path — every
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

    Ok(debit_direct(ctx, out, kind, &existing, &chunk2, kept))
}

/// Take `chunk2_perm` off a single chunk whose sufficiency is already proven, and
/// store the remainder back under `kept`.
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
pub(crate) fn debit_direct(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    existing: &Chunk,
    chunk2: &Chunk,
    kept: crate::verify::heap::HeapPc,
) -> Heap {
    ctx.union(existing.value, chunk2.value);
    // Remainder stays structural (leaves get `SubR`); never an `ite` in the graph.
    // Aligned against the *demand's* structure too, so a guarded consume cancels per
    // arm (`c ? 0 : 1/1`) rather than leaving `1/1 - (c ? 1/1 : 0/1)` in every leaf --
    // which is what lets `perm_all_zero` below see the emptied branch.
    let remainder = perm_sub_aligned(ctx, existing.ungated_perm(), chunk2.ungated_perm());
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
    if perm_all_zero(ctx, &remainder) {
        out.without_chunk(kind, existing.addr)
    } else {
        // The value (and so its recipe provenance) is unchanged by a subtract, and
        // so is the presence guard when the pc entailed it: the chunk is still held
        // exactly where it was, just for less.
        out.with_chunk(
            kind,
            Chunk::new_perm(existing.addr, remainder, existing.value)
                .with_recipe(existing.recipe.clone())
                .with_guard(kept),
        )
    }
}

/// Wildcard exhale. Instead of proving `held ≥ needed` (a wildcard has no fixed
/// value), require the location hold *some* permission (`held > 0`) and hand back a
/// **fresh remainder** assumed strictly smaller (`r < held`) under the pc —
/// Silicon's constrainable-ARP rule. The demanded amount never enters the result:
/// what leaves is only known to be positive and smaller, which is all a wildcard
/// ever says.
#[allow(clippy::too_many_arguments)]
pub(crate) fn debit_wildcard(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    existing: &Chunk,
    chunk2: &Chunk,
    pc_lits: &[(egg::Id, Polarity)],
    kept: crate::verify::heap::HeapPc,
) -> Result<Heap, VerifyError> {
    // Walk the DEMAND's branch structure: a gated `exhale b ==> acc(x.f, wildcard)`
    // takes a share only where `b` holds, and states nothing where it does not.
    // Flat, the rule read only `chunk2.value` and constrained the remainder
    // `r < held` on both arms — including the arm where nothing was demanded, which
    // is imprecise rather than unsound but throws away permission the program kept.
    let (perm, value, err) = debit_wildcard_walk(
        ctx,
        existing.ungated_perm(),
        chunk2.ungated_perm(),
        existing.value,
        chunk2.value,
        pc_lits,
    );
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out.with_chunk(
        kind,
        Chunk::new_perm(existing.addr, perm, value)
            .with_recipe(existing.recipe.clone())
            .with_guard(kept),
    ))
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
    match needed {
        ChunkPerm::Leaf { id, .. } => {
            // Nothing demanded on this arm: no obligation, no fact, no debit.
            let nonpos = expr!(ctx, not ((0/1) <r {*id}));
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
        ChunkPerm::Select { cond, then, els } => {
            let ht = ChunkPerm::restrict(ctx, *cond, held.clone(), true);
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((*cond, Polarity::Positive));
            let (t, value, e1) =
                debit_wildcard_walk(ctx, &ht, then, held_value, needed_value, &pc_t);
            let he = ChunkPerm::restrict(ctx, *cond, held.clone(), false);
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((*cond, Polarity::Negative));
            let (e, _, e2) = debit_wildcard_walk(ctx, &he, els, held_value, needed_value, &pc_e);
            (ChunkPerm::select(ctx, *cond, t, e), value, e1.or(e2))
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
pub(crate) fn heap_subtract_summarized_fallbacks(
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

/// The `fail` trace for a consume that resolved nowhere. Two different questions,
/// so two different events: with no ground match the useful thing is the demanded
/// address against the held ones (`miss`); with a match that proved insufficient
/// it is the two permission terms (`insufficient`).
pub(crate) fn subtract_miss_trace(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    existing: Option<&Chunk>,
    chunk2: &Chunk,
    chunk2_perm: egg::Id,
) {
    match existing {
        Some(existing) => {
            if crate::trace::enabled(crate::trace::Category::Fail) {
                let existing_perm = existing.ungated_perm().to_id(ctx);
                crate::trace::trace_event!(
                    Fail,
                    "insufficient",
                    group = ctx.groups.resolve(&kind.group).to_string(),
                    held = crate::verify::viz::dump_term(ctx, existing_perm, 64),
                    needed = crate::verify::viz::dump_term(ctx, chunk2_perm, 64),
                );
            }
        }
        None => {
            crate::trace::trace_event!(
                Fail,
                "miss",
                group = ctx.groups.resolve(&kind.group).to_string(),
                demanded = crate::verify::viz::dump_term(ctx, chunk2.addr, 40),
                held = crate::json::Json::Arr(
                    h1.chunks_of(kind)
                        .iter()
                        .map(|c| crate::verify::viz::dump_term(ctx, c.addr, 40).into())
                        .collect(),
                ),
            );
        }
    }
}

/// A chunk's amount as it counts under `pc`: bare when the pc already entails the
/// chunk's presence guard (the common case, and it keeps the term small), gated
/// `guard ? perm : 0` otherwise.
fn present_amount(
    ctx: &mut VerifyContext<'_>,
    c: &Chunk,
    pc_lits: &[(egg::Id, Polarity)],
) -> ChunkPerm {
    if c.guard().is_empty() || c.pc_entails_guard(ctx, pc_lits) {
        c.ungated_perm().clone()
    } else {
        c.gated_perm(ctx)
    }
}

/// Build the pc-alias flavour of a [`heap_subtract_summarized`] set: `existing` (when
/// a chunk did match on ground) followed by the pc-alias `partners`, every member
/// gated by the *same* cube — the pc.
///
/// The returned total is not gated by the pc: the sufficiency proof runs under the
/// pc anyway ([`VerifyContext::prove_under_pc`]), so gating each addend would only
/// grow the term. Each addend is gated by its own presence guard, though (see
/// [`present_amount`]).
pub(crate) fn pc_alias_set(
    ctx: &mut VerifyContext<'_>,
    h1: &Heap,
    kind: &LocationKind,
    existing: Option<&Chunk>,
    partners: &[egg::Id],
    pc_lits: &[(egg::Id, Polarity)],
) -> (Vec<(Chunk, crate::verify::heap::HeapPc)>, ChunkPerm) {
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
        let amount = present_amount(ctx, c, pc_lits);
        total = Some(match total {
            None => amount,
            Some(t) => perm_add(ctx, &t, &amount),
        });
    }
    let cube: crate::verify::heap::HeapPc = std::rc::Rc::from(pc_lits.to_vec());
    let set = members.into_iter().map(|c| (c, cube.clone())).collect();
    (
        set,
        total.unwrap_or_else(|| ChunkPerm::leaf(expr!(ctx, 0 / 1))),
    )
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
pub(crate) fn heap_subtract_summarized(
    ctx: &mut VerifyContext<'_>,
    out: Heap,
    kind: &LocationKind,
    set: &[(Chunk, crate::verify::heap::HeapPc)],
    total: ChunkPerm,
    chunk2: Chunk,
    chunk2_perm: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Heap, VerifyError> {
    // `needed ≤ total`, per leaf of `total`, each under the pc plus that leaf's
    // branch literals.
    if !prove_sufficient(ctx, &total, chunk2_perm, pc_lits) {
        if crate::trace::enabled(crate::trace::Category::Fail) {
            let total = total.to_id(ctx);
            crate::trace::trace_event!(
                Fail,
                "insufficient_summarized",
                group = ctx.groups.resolve(&kind.group).to_string(),
                chunks = set.len(),
                held = crate::verify::viz::dump_term(ctx, total, 64),
                needed = crate::verify::viz::dump_term(ctx, chunk2_perm, 64),
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
        let hold = present_amount(ctx, &chunk, pc_lits).to_id(ctx);
        // `min(hold, remaining)` — a symbolic hold needs the `ite`; concrete
        // fractions fold it away.
        // `min(hold, remaining)` — a symbolic hold needs the `ite`; concrete
        // fractions fold it away.
        let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
        let gated = gate_amount_by_pc(ctx, take, &cube);
        let rest = perm_sub(ctx, chunk.ungated_perm(), gated);
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
        if perm_all_zero(ctx, &rest) {
            out = out.without_chunk(kind, chunk.addr);
        } else {
            // The member's own presence guard is carried through untouched: unlike
            // the plain path, this one never folded it into the amount — `hold` is
            // the raw perm and the debit is gated by the member's `cube` — so the
            // remainder is still held exactly under that guard. Dropping it here
            // was what destroyed the flat cube on the first consume at a location
            // and left every later one with conditionality it could only see as
            // `ite`s inside the arithmetic.
            out = out.with_chunk(
                kind,
                Chunk::new_perm(chunk.addr, rest, chunk.value)
                    .with_recipe(chunk.recipe.clone())
                    .with_guard(chunk.guard_pc()),
            );
        }
    }
    Ok(out)
}

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
///   (rare — counted by the `heap` trace).
///
/// The presence guard is reconstructed into the exact `guard?perm:0` obligation
/// **transiently at each consume site** (`gate_perm_by_guard`), so merges stay
/// flat while sufficiency/remainder keep the pre-hoist semantics. Runs only when
/// both arms are live, so an absent side is a genuine conditional footprint.
pub(crate) fn merge_heaps(
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
                        crate::trace::trace_tally!(Heap, "zero_leaf", [], built = 1);
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
    assume_location_axioms(ctx, &out);
    out
}

/// The **Σ-ite summary** of a location: the permission held at `addr`, summed over
/// *every* chunk of the group, each gated by whether it sits at that address.
///
/// ```text
/// perm(addr) = Σ_c  ite(c.addr == addr, guard(c) ? c.perm : 0, 0)
/// ```
///
/// Two gates compose per chunk, and they are different things:
/// - the **presence guard** (`guard ? perm : 0`, [`gate_perm_by_guard`]) — the
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
/// The total is returned **structural** ([`perm_add`]), not flattened: a
/// conditionally-present chunk contributes a `Select` on its guard literal, and
/// [`prove_sufficient`] discharges the sum per leaf with that literal in the pc.
pub(crate) fn summarize_perm_at(
    ctx: &mut VerifyContext<'_>,
    chunks: &[Chunk],
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> (ChunkPerm, Vec<(Chunk, crate::verify::heap::HeapPc)>) {
    let canon = ctx.egraph.find(addr);
    let mut total: Option<ChunkPerm> = None;
    let mut set: Vec<(Chunk, crate::verify::heap::HeapPc)> = Vec::new();
    for c in chunks {
        let held = c.gated_perm(ctx);
        // Ground match: no address gate at all (and no `Eq` node minted).
        let (amount, cube): (ChunkPerm, crate::verify::heap::HeapPc) =
            if ctx.egraph.find(c.addr) == canon {
                (held, std::rc::Rc::from(Vec::new()))
            } else {
                let eq = expr!(ctx, { c.addr } == { addr });
                // Disproven aliasing contributes nothing — skip before minting the gate.
                if matches!(
                    ctx.egraph[ctx.egraph.find(eq)].data.known(),
                    Some(Literal::Bool(false))
                ) {
                    continue;
                }
                // Equal wherever this chunk is held: no address gate, and the
                // condition on the contribution is the guard `held` already carries.
                let presence = match c.presence_cube(ctx, pc_lits) {
                    // Guard contradicts the pc — the chunk is unreachable here.
                    None => continue,
                    Some(cube) => cube,
                };
                if !c.guard().is_empty() && ctx.prove_under_pc(eq, &presence) {
                    (held, c.guard_pc())
                } else {
                    // Structural, not folded into the term. Gating the amount as
                    // `ite(eq, held, 0)` collapses the whole summary to one flat
                    // leaf, and `prove_perm_leaves` then has nothing to split on:
                    // sufficiency has to be shown against
                    // `ite(a==c, p, 0) + ite(b==c, q, 0)` with every alias gate
                    // still open. Kept as a `Select`, `perm_add` distributes it,
                    // and each leaf is proven under its own gate polarities.
                    let cube = vec![(eq, Polarity::Positive)];
                    let zero = ChunkPerm::leaf(expr!(ctx, 0 / 1));
                    (
                        ChunkPerm::select(ctx, eq, held, zero),
                        std::rc::Rc::from(cube),
                    )
                }
            };
        set.push((c.clone(), cube));
        total = Some(match total {
            None => amount,
            Some(t) => perm_add(ctx, &t, &amount),
        });
    }
    (
        total.unwrap_or_else(|| ChunkPerm::leaf(expr!(ctx, 0 / 1))),
        set,
    )
}

/// The permission held at `addr`, as a term — what `perm(loc)` evaluates to.
///
/// This is the one place the Σ-ite summary is built **eagerly**, because it is the
/// one place the user asked for the permission *value* rather than for a consume to
/// succeed. Heap operations reach for it only as a fallback
/// ([`heap_subtract_inner`]), matching Silicon's split between greedy chunk matching
/// and `--exhaleMode=1`.
pub(crate) fn perm_held_at(
    ctx: &mut VerifyContext<'_>,
    chunks: &[Chunk],
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> egg::Id {
    // `perm(loc)` is asked for as a *value*, so the branch structure has to be
    // materialized here — the one place it is.
    let total = summarize_perm_at(ctx, chunks, addr, pc_lits).0;
    let id = total.to_id(ctx);
    // Flattening loses what the tree made obvious. `perm(x.f)` after a gated
    // wildcard exhale is `ite(c, r, 1/1)`: positive on both arms, but deciding that
    // from the flat term needs a case split the prover will not make for a `<r`
    // goal. [`perm_known_positive`] reads it off the structure (its `Ite` arm
    // requires both), so state the fact here, once, where the value is built.
    if perm_known_positive(ctx, id) {
        let pos = expr!(ctx, (0 / 1) < r { id });
        let t = expr!(ctx, true);
        ctx.union(pos, t);
    }
    id
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
pub(crate) fn union_heaps(
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
    ctx.egraph.rebuild();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::test_support::{fresh_ctx, real, test_kind};
    use crate::vmir::Type;

    // A multi-arg bounded location (not Viper-reachable; only via direct VMIR):
    // two chunks of the same location whose perms sum > bound, with all args
    // forced equal, must contradict (the conjunction of arg-equalities collapses
    // to false against the assumed-true equalities).
    #[test]
    fn multiarg_location_nonaliasing_all_args_equal_is_inconsistent() {
        let interner = lasso::Rodeo::new();
        let decls =
            typed_index_collections::TiVec::<crate::vmir::MemberId, crate::vmir::Declaration>::new(
            );
        let mut groups = lasso::Rodeo::<lasso::Spur>::new();
        let g = groups.get_or_intern("g");
        let mut alloc = crate::verify::func_registry::FuncRegistry::empty();
        let mut ctx = VerifyContext::new(&interner, &decls, &groups, &mut alloc);

        // A 2-arg bounded address group `g` of held type `Int`, cap `1/1`. The
        // address is an ordinary `FuncApp` to the group's address function
        // (`FuncId(0)` here); its `Addr{..}` return type is recorded in
        // `func_ret_types` (recovered by `location_chunks`).
        let addr_ty = Type::addr(
            g,
            Type::Int,
            Bound::Bounded(num::BigRational::from(num::BigInt::from(1))),
        );
        let addr_fn = crate::verify::lang::FuncId(0);
        let (x0, y0) = (ctx.add(Symbolic::Fresh(0)), ctx.add(Symbolic::Fresh(1)));
        let (x1, y1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));
        let a0 = ctx.add_func_app_id(addr_fn, Box::new([]), addr_ty.clone(), Box::new([x0, y0]));
        let a1 = ctx.add_func_app_id(addr_fn, Box::new([]), addr_ty.clone(), Box::new([x1, y1]));
        let k = LocationKind::from_addr_type(&addr_ty).unwrap();
        let (v0, v1) = (ctx.add(Symbolic::Fresh(4)), ctx.add(Symbolic::Fresh(5)));
        let (p0, p1) = (real(&mut ctx, 3, 4), real(&mut ctx, 1, 2)); // sum 5/4 > 1
        let heap = Heap::empty()
            .with_chunk(&k, Chunk::new(a0, p0, v0))
            .with_chunk(&k, Chunk::new(a1, p1, v1));

        // Assume both argument pairs are equal: x0==x1, y0==y1.
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        for (a, b) in [(x0, x1), (y0, y1)] {
            let eq = ctx.add(Symbolic::Binary(BinOp::Eq, [a, b]));
            ctx.union(eq, true_);
        }
        ctx.egraph.rebuild();

        assume_location_axioms(&mut ctx, &heap);
        ctx.saturate();
        assert!(
            ctx.is_inconsistent(),
            "holding 5/4 across a 2-arg location with all args equal must contradict"
        );
    }

    #[test]
    fn heap_union_merges_egg_equivalent_addresses() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(b, p2, v2), &[]);

        let canon = ctx.egraph.find(a);
        let chunk = merged
            .chunk(&test_kind(), canon)
            .expect("merged chunk missing");

        let expected_perm = ctx.add(Symbolic::Binary(BinOp::AddR, [p1, p2]));
        ctx.saturate();
        assert_eq!(
            ctx.egraph.find(chunk.perm_repr_id()),
            ctx.egraph.find(expected_perm)
        );
        // Both fractions positive (1, 2) → agreement axiom fuses the values.
        assert_eq!(ctx.egraph.find(v1), ctx.egraph.find(v2));
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v1));
        assert_eq!(merged.entries().count(), 1);
    }

    // Two chunks inserted at *distinct* addresses whose classes collapse
    // later (an aliasing union learned after insertion): the consolidating
    // lookup must re-merge them so a subtract sees the full held amount —
    // first-match lookup only ever saw one 1/2 fragment and failed.
    #[test]
    fn subtract_consolidates_post_hoc_aliased_chunks() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let half = real(&mut ctx, 1, 2);
        let (v0, v1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));
        let h = Heap::empty()
            .with_chunk(&test_kind(), Chunk::new(a, half, v0))
            .with_chunk(&test_kind(), Chunk::new(b, half, v1));
        assert_eq!(h.entries().count(), 2);

        // The aliasing fact arrives after both chunks are in the heap.
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let one = real(&mut ctx, 1, 1);
        let out = heap_subtract(
            &mut ctx,
            &h,
            &test_kind(),
            Chunk::new(a, one, v0),
            &[],
            Demand::Concrete,
        )
        .expect("full permission is held across the two aliased fragments");
        // 1/2 + 1/2 − 1/1 = 0 const-folds → the emptied chunk is dropped.
        assert_eq!(out.entries().count(), 0);
    }

    // The borrow / give-back cycle: a chunk whose permission was reduced to
    // `1 − p` gets `p` unioned back in. The merged permission `(1 − p) + p`
    // must land back in `1/1`'s class (the cancellation rewrite), so the next
    // full-permission check is O(1) — the "regain full perm after a match arm"
    // shape, with `p = ite(c, 1, 0)` a branch-scaled borrow.
    #[test]
    fn give_back_restores_full_permission() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let c = ctx.add(Symbolic::Fresh(1));
        let one = real(&mut ctx, 1, 1);
        let zero = real(&mut ctx, 0, 1);
        let p = ctx.add(Symbolic::Ite([c, one, zero]));
        let rest = ctx.add(Symbolic::Binary(BinOp::SubR, [one, p]));
        let (v0, v1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));

        let h = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, rest, v0));
        let out = heap_union(&mut ctx, &h, &test_kind(), Chunk::new(a, p, v1), &[]);
        let chunk = out
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();
        assert_eq!(
            ctx.egraph.find(chunk.perm_repr_id()),
            ctx.egraph.find(one),
            "(1 - p) + p must collapse back to the full permission"
        );
    }

    #[test]
    fn merge_zero_fraction_picks_active_value() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(a, p1, v1), &[]);
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // p0 = 0 → asymmetric ternary picks the active half v1; no fusion.
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v1));
        assert_ne!(ctx.egraph.find(v0), ctx.egraph.find(v1));
    }

    #[test]
    fn merge_both_active_fuses_values() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(a, p1, v1), &[]);
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // Both fractions positive → agreement axiom fuses the symbolic values.
        assert_eq!(ctx.egraph.find(v0), ctx.egraph.find(v1));
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v0));
    }

    #[test]
    fn merge_under_false_pc_blocks_fusion() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let false_lit = ctx.add(Symbolic::Lit(Literal::Bool(false)));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[(false_lit, Polarity::Positive)],
        );
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // PC literal is `false` → implication collapses to its `true` fallback;
        // values must NOT fuse even though both fractions are positive.
        assert_ne!(ctx.egraph.find(v0), ctx.egraph.find(v1));
        // Value pick is independent of the PC gate.
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v0));
    }

    #[test]
    fn merge_under_true_pc_allows_fusion() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let true_lit = ctx.add(Symbolic::Lit(Literal::Bool(true)));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[(true_lit, Polarity::Positive)],
        );
        let _chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // PC literal is `true` + both fractions positive → agreement fires.
        assert_eq!(ctx.egraph.find(v0), ctx.egraph.find(v1));
    }

    #[test]
    fn subtract_symbolic_perm_fails_without_proof() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p_have = ctx.add(Symbolic::Fresh(1));
        let p_take = ctx.add(Symbolic::Fresh(2));
        let v1 = ctx.add(Symbolic::Fresh(3));
        let v2 = ctx.add(Symbolic::Fresh(4));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p_have, v1));
        // Symbolic perms → `have >= take` not provable by equality saturation.
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p_take, v2),
            &[],
            Demand::Concrete,
        )
        .expect_err("symbolic-perm exhale must fail without a proof");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }

    #[test]
    fn heap_subtract_canonical_match() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p2, v1));
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let result = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(b, p1, v2),
            &[],
            Demand::Concrete,
        )
        .expect("subtract should succeed");

        let canon = ctx.egraph.find(a);
        let chunk = result
            .chunk(&test_kind(), canon)
            .expect("result chunk missing");
        let expected_perm = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        ctx.egraph.rebuild();
        assert_eq!(
            ctx.egraph.find(chunk.perm_repr_id()),
            ctx.egraph.find(expected_perm)
        );
        assert_eq!(ctx.egraph.find(v1), ctx.egraph.find(v2));
    }

    #[test]
    fn heap_subtract_exact_match_drops_chunk() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        let result = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v2),
            &[],
            Demand::Concrete,
        )
        .expect("subtract should succeed");

        let canon = ctx.egraph.find(a);
        assert!(
            result.chunk(&test_kind(), canon).is_none(),
            "zero-perm chunk must be dropped"
        );
    }

    #[test]
    fn heap_subtract_over_consume_fails() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p2, v2),
            &[],
            Demand::Concrete,
        )
        .expect_err("over-consumption must fail");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }

    #[test]
    fn heap_subtract_missing_addr_fails() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let _b = ctx.add(Symbolic::Fresh(1));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty();
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[],
            Demand::Concrete,
        )
        .expect_err("subtract from empty must fail");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }
}

// ---- pc-sensitive location lookups (moved off VerifyContext) ----

/// Resolve which of `chunks` sits at address `addr`, consulting aliasing
/// that may only hold under the path condition `pc_lits`.
///
/// Fast path (what normal framing hits): a canonical match in the **live**
/// graph — the address `add`ed for the read is congruent to a held chunk's
/// address. Zero extra cost, no clone.
///
/// Slow path (a miss, and only then): clone, assume the path condition, and
/// saturate. An assumed `x == y` fires `eq-true-union`, congruence then merges
/// `f(x)` and `f(y)`, so the chunk `acc(x.f)` answers a read of `y.f` — which is
/// what lets a predicate body like `acc(x.f) && x == y && y.f == 10` frame its
/// `y.f` deref.
///
/// The returned chunk's `perm`/`value` ids are live-graph ids (the probe never
/// touches live state), so they are valid to discharge obligations over.
pub(crate) fn chunk_under_pc<'c>(
    ctx: &mut VerifyContext<'_>,
    chunks: &'c [Chunk],
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Option<&'c Chunk> {
    let canon = ctx.egraph.find(addr);
    if let Some(c) = chunks.iter().find(|c| ctx.egraph.find(c.addr) == canon) {
        return Some(c);
    }
    // No unconditional match. Aliasing under the path condition can only help
    // if there is one; a truly-unheld location stays a miss.
    if pc_lits.is_empty() {
        return None;
    }
    let mut probe = ctx.egraph.clone();
    for (id, pol) in pc_lits {
        let want_true = matches!(pol, Polarity::Positive);
        // An unsatisfiable path condition makes every read vacuous — leave
        // the resolution to the (vacuous-pc) obligation check, don't invent
        // a chunk here.
        if matches!(probe[*id].data.known(), Some(Literal::Bool(b)) if *b != want_true) {
            return None;
        }
        let lit = probe.add(Symbolic::Lit(Literal::Bool(want_true)));
        probe.union(*id, lit);
    }
    probe.rebuild();
    let probe = ctx.run_probe(probe);
    let canon = probe.find(addr);
    chunks.iter().find(move |c| probe.find(c.addr) == canon)
}

/// The chunks that alias `addr` **only under `pc_lits`** — pc-equal but not
/// ground-equal. These are the partners a consume may draw on (invariant 7): at
/// a state where the pc holds they are the *same* location as `addr`, so their
/// fractions add, while on ground they stay distinct and must not be merged.
/// Returns their addresses (stable keys into the heap group). Same probe shape as
/// [`Self::chunk_under_pc`], but collects every match: sufficiency needs the sum.
pub(crate) fn pc_alias_partners(
    ctx: &mut VerifyContext<'_>,
    chunks: &[Chunk],
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Vec<egg::Id> {
    if pc_lits.is_empty() {
        return Vec::new();
    }
    // Inside a method block the answer is already sitting in the block scratch: it is
    // ground with this block's cube assumed and saturated, warm from the sufficiency
    // proof that got us here. Reading the coincidence off it costs a `find` per
    // chunk, where the clone+saturate below is a full probe — half of every probe
    // saturation on the enum grid was this one lookup (40 of `enum_v8_p1`'s 80,
    // 16.8s of its 70.5s). Measured to return the identical partner set on every
    // call it was asked (55/55 across `structs_enums` and `enum_v5_p1`).
    //
    // Only when the instruction's pc *is* the block cube: a wider pc would ask the
    // scratch about literals it never assumed.
    if ctx.cube_matches(pc_lits)
        && let Some(partners) =
            ctx.scratch_alias_partners(addr, &chunks.iter().map(|c| c.addr).collect::<Vec<_>>())
    {
        return partners;
    }
    let ground = ctx.egraph.find(addr);
    let mut probe = ctx.egraph.clone();
    for (id, pol) in pc_lits {
        let want_true = matches!(pol, Polarity::Positive);
        // Unsatisfiable pc: the consume is vacuous, so inventing partners would
        // only mask that — leave it to the (vacuous-pc) obligation check.
        if matches!(probe[*id].data.known(), Some(Literal::Bool(b)) if *b != want_true) {
            return Vec::new();
        }
        let lit = probe.add(Symbolic::Lit(Literal::Bool(want_true)));
        probe.union(*id, lit);
    }
    probe.rebuild();
    let probe = ctx.run_probe(probe);
    let canon = probe.find(addr);
    chunks
        .iter()
        .filter(|c| probe.find(c.addr) == canon && ctx.egraph.find(c.addr) != ground)
        .map(|c| c.addr)
        .collect()
}

/// Gate a permission amount by a path condition: `pc ? amount : 0`. The ground
/// heap records a pc-alias consume as a **guarded** debit (invariant 7) — the
/// full amount comes off the demanded chunk where the pc holds, and nothing comes
/// off where it does not (there the chunks are distinct and nothing was given up).
pub(crate) fn gate_amount_by_pc(
    ctx: &mut VerifyContext<'_>,
    amount: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> egg::Id {
    let zero = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())));
    pc_lits.iter().fold(amount, |acc, (lit, pol)| {
        let arms = if matches!(pol, Polarity::Positive) {
            [*lit, acc, zero]
        } else {
            [*lit, zero, acc]
        };
        ctx.add(Symbolic::Ite(arms))
    })
}
