//! Permission amounts: arithmetic over [`ChunkPerm`] and the proofs about it (sufficient, positive, write).

use crate::verify::{
    context::VerifyContext,
    heap::{ChunkPerm, Shape},
    lang::Symbolic,
};
use crate::vmir::{BinOp, Bound, Literal, Polarity};

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
pub(super) fn perm_add_wildcard(
    ctx: &mut VerifyContext<'_>,
    a: &ChunkPerm,
    b: &ChunkPerm,
) -> ChunkPerm {
    match (a.shape(), b.shape()) {
        // Nothing to add: the other summand is the sum.
        (Shape::Zero, _) => b.clone(),
        (_, Shape::Zero) => a.clone(),
        (Shape::Leaf { id: x, wild: wx }, Shape::Leaf { id: y, wild: wy }) => {
            let wild = wx || wy;
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
        // Descend on `a` (this arm also covers `Branch`/`Branch`).
        (Shape::Branch { cond, then, els }, _) => {
            let ot = ChunkPerm::restrict(ctx, cond, b.clone(), true);
            let oe = ChunkPerm::restrict(ctx, cond, b.clone(), false);
            let t = perm_add_wildcard(ctx, &then, &ot);
            let e = perm_add_wildcard(ctx, &els, &oe);
            ChunkPerm::select(ctx, cond, t, e)
        }
        (_, Shape::Branch { cond, then, els }) => {
            let ot = ChunkPerm::restrict(ctx, cond, a.clone(), true);
            let oe = ChunkPerm::restrict(ctx, cond, a.clone(), false);
            let t = perm_add_wildcard(ctx, &ot, &then);
            let e = perm_add_wildcard(ctx, &oe, &els);
            ChunkPerm::select(ctx, cond, t, e)
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

/// Prove a per-leaf permission predicate over a (possibly branch-structured)
/// [`ChunkPerm`] **WITHOUT materializing the `Select` into the ground graph**.
/// A `Select` pushes its condition into the pc and requires the predicate on
/// both arms; a `Leaf` discharges it under the accumulated pc. This is the CFG
/// case split done through the pc — no `ite` node, no `lt-ite`, no goal-directed split, and
/// (crucially) no perm-tower bloat in the live graph. The predicate is built
/// from a leaf amount id by `mk`.
fn prove_perm_leaves(
    ctx: &mut VerifyContext<'_>,
    perm: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
    mk: &dyn Fn(&mut VerifyContext<'_>, egg::Id, &[(egg::Id, Polarity)]) -> bool,
) -> bool {
    match perm.shape() {
        Shape::Zero => {
            let zero = expr!(ctx, 0 / 1);
            mk(ctx, zero, pc_lits)
        }
        Shape::Leaf { id, .. } => mk(ctx, id, pc_lits),
        Shape::Branch { cond, then, els } => {
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((cond, Polarity::Positive));
            if !prove_perm_leaves(ctx, &then, &pc_t, mk) {
                return false;
            }
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((cond, Polarity::Negative));
            prove_perm_leaves(ctx, &els, &pc_e, mk)
        }
    }
}

/// The folded rational value of an e-class, if known — used by the per-leaf
/// fast paths to decide a comparison against a literal WITHOUT a prove_under_pc.
pub(super) fn known_real(ctx: &VerifyContext<'_>, id: egg::Id) -> Option<num::BigRational> {
    match ctx.egraph[ctx.egraph.find(id)].data.known() {
        Some(Literal::Real(r)) => Some(r.clone()),
        _ => None,
    }
}

/// `held ≥ needed` over a structured `held`, per leaf (no `Select` in the graph).
pub(in crate::verify::heap) fn prove_sufficient(
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
pub(in crate::verify::heap) fn prove_sufficient_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    prove_aligned(ctx, held, needed, pc_lits, sufficient_amount)
}

/// [`prove_sufficient_aligned`] for what a frame demands: a `wildcard` leaf
/// asks for *some* share, so what is held there must be positive. Nothing is
/// taken, so the demanded amount needs no bound; a take uses
/// [`prove_sufficient_aligned`] (or `consume::debit_wildcard`) instead.
pub(in crate::verify::heap) fn prove_framed_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    prove_aligned(ctx, held, needed, pc_lits, |ctx, held, n, wild, pc| {
        if wild {
            prove_perm_positive(ctx, held, pc)
        } else {
            sufficient_amount(ctx, held, n, wild, pc)
        }
    })
}

/// When one demand leaf `n` (a `wildcard` when `wild`) is met by `held`.
type LeafRule =
    fn(&mut VerifyContext<'_>, &ChunkPerm, egg::Id, bool, &[(egg::Id, Polarity)]) -> bool;

/// `held ≥ n` for one demand leaf, whatever its origin.
fn sufficient_amount(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    n: egg::Id,
    _wild: bool,
    pc_lits: &[(egg::Id, Polarity)],
) -> bool {
    if let Some(r) = known_real(ctx, n) {
        if r <= num::BigRational::from(num::BigInt::from(0)) {
            return true;
        }
    }
    prove_sufficient(ctx, held, n, pc_lits)
}

/// Descend the demand's branches, meeting each leaf by `leaf`.
fn prove_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
    pc_lits: &[(egg::Id, Polarity)],
    leaf: LeafRule,
) -> bool {
    match needed.shape() {
        // Nothing demanded on this arm: every held amount is ≥ 0.
        Shape::Zero => true,
        Shape::Leaf { id, wild } => leaf(ctx, held, id, wild, pc_lits),
        Shape::Branch { cond, then, els } => {
            // The demand's own branch, taken both ways: `held ≥ needed` follows by
            // case analysis. `restrict` aligns the held side — it takes the matching
            // arm when held branches on the same condition class and carries the
            // whole tree in when it does not, so an unrelated held structure degrades
            // to today's behaviour rather than being mishandled.
            let ht = ChunkPerm::restrict(ctx, cond, held.clone(), true);
            let mut pc_t = pc_lits.to_vec();
            pc_t.push((cond, Polarity::Positive));
            if !prove_aligned(ctx, &ht, &then, &pc_t, leaf) {
                return false;
            }
            let he = ChunkPerm::restrict(ctx, cond, held.clone(), false);
            let mut pc_e = pc_lits.to_vec();
            pc_e.push((cond, Polarity::Negative));
            prove_aligned(ctx, &he, &els, &pc_e, leaf)
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
fn sufficient_leaf(
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
pub(in crate::verify::heap) fn prove_perm_positive(
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
pub(in crate::verify::heap) fn prove_perm_write(
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
pub(in crate::verify::heap) fn perm_add(
    ctx: &mut VerifyContext<'_>,
    a: &ChunkPerm,
    b: &ChunkPerm,
) -> ChunkPerm {
    match (a.shape(), b.shape()) {
        // Nothing to add: the other summand is the sum.
        (Shape::Zero, _) => b.clone(),
        (_, Shape::Zero) => a.clone(),
        (Shape::Leaf { id: x, wild: wx }, Shape::Leaf { id: y, wild: wy }) => ChunkPerm::Leaf {
            id: ctx.add(Symbolic::Binary(BinOp::AddR, [x, y])),
            wild: wx || wy,
        },
        // Descend on `a` (this arm also covers `Branch`/`Branch`).
        (Shape::Branch { cond, then, els }, _) => {
            let ot = ChunkPerm::restrict(ctx, cond, b.clone(), true);
            let oe = ChunkPerm::restrict(ctx, cond, b.clone(), false);
            let t = perm_add(ctx, &then, &ot);
            let e = perm_add(ctx, &els, &oe);
            ChunkPerm::select(ctx, cond, t, e)
        }
        (_, Shape::Branch { cond, then, els }) => {
            let ot = ChunkPerm::restrict(ctx, cond, a.clone(), true);
            let oe = ChunkPerm::restrict(ctx, cond, a.clone(), false);
            let t = perm_add(ctx, &ot, &then);
            let e = perm_add(ctx, &oe, &els);
            ChunkPerm::select(ctx, cond, t, e)
        }
    }
}

/// `held − needed`, kept structural (leaves get `SubR`, the tree stays a
/// `Select` via the smart constructor so it never materializes as an `ite`).
pub(in crate::verify::heap) fn perm_sub(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: egg::Id,
) -> ChunkPerm {
    match held.shape() {
        // Taken from nothing: what is left is `0 - needed`, which is `0` exactly
        // where nothing was needed.
        Shape::Zero => {
            let zero = expr!(ctx, 0 / 1);
            ChunkPerm::leaf(ctx.add(Symbolic::Binary(BinOp::SubR, [zero, needed])))
        }
        // The demand is concrete on this path (a wildcard demand goes to
        // `debit_wildcard`), so the remainder's provenance is the held side's.
        Shape::Leaf { id, wild } => ChunkPerm::Leaf {
            id: ctx.add(Symbolic::Binary(BinOp::SubR, [id, needed])),
            wild,
        },
        Shape::Branch { cond, then, els } => {
            let t = perm_sub(ctx, &then, needed);
            let e = perm_sub(ctx, &els, needed);
            ChunkPerm::select(ctx, cond, t, e)
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
pub(in crate::verify::heap) fn perm_sub_aligned(
    ctx: &mut VerifyContext<'_>,
    held: &ChunkPerm,
    needed: &ChunkPerm,
) -> ChunkPerm {
    match needed.shape() {
        // Nothing demanded on this arm: the held amount is untouched.
        Shape::Zero => held.clone(),
        // Demand is flat: nothing to align against, so this is `perm_sub`.
        Shape::Leaf { id: n, .. } => perm_sub(ctx, held, n),
        Shape::Branch { cond, then, els } => {
            let ht = ChunkPerm::restrict(ctx, cond, held.clone(), true);
            let he = ChunkPerm::restrict(ctx, cond, held.clone(), false);
            let t = perm_sub_aligned(ctx, &ht, &then);
            let e = perm_sub_aligned(ctx, &he, &els);
            ChunkPerm::select(ctx, cond, t, e)
        }
    }
}

/// Whether every leaf of `perm` const-folds to `0` (the chunk is emptied on
/// every branch, so it can be dropped). Const-fold only, no `Select` in the graph.
pub(in crate::verify::heap) fn perm_all_zero(ctx: &VerifyContext<'_>, perm: &ChunkPerm) -> bool {
    match perm {
        ChunkPerm::Zero | ChunkPerm::Leaf { .. } => perm.is_zero(ctx),
        // A gate is zero off its cube, so it is all zero exactly when its body is.
        ChunkPerm::Gate { then, .. } => perm_all_zero(ctx, then),
        ChunkPerm::Select { then, els, .. } => perm_all_zero(ctx, then) && perm_all_zero(ctx, els),
    }
}

/// A permission held at some address (the caller's summary, see
/// `perm_total_at` in `declaration.rs`), as a term — what `perm(loc)` evaluates
/// to.
///
/// This is the one place the Σ-ite summary is materialized **eagerly**, because it
/// is the one place the user asked for the permission *value* rather than for a
/// consume to succeed. Heap operations reach for the summary only as a fallback
/// ([`heap_subtract_inner`]), matching Silicon's split between greedy chunk
/// matching and `--exhaleMode=1`.
pub(in crate::verify::heap) fn perm_value(
    ctx: &mut VerifyContext<'_>,
    total: ChunkPerm,
) -> egg::Id {
    // `perm(loc)` is asked for as a *value*, so the branch structure has to be
    // materialized here — the one place it is.
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
