//! Who sits at an address: a chunk's share there, the summary over a group, and the chunks a path condition makes alias it.

use super::*;
use crate::verify::{
    context::VerifyContext,
    heap::{Chunk, ChunkPerm, Cube},
    lang::Symbolic,
};
use crate::vmir::{Literal, Polarity};

/// What the single chunk `c` contributes at `addr`: its gated amount there and
/// the cube that contribution is conditional on (empty on a ground match, the
/// chunk's guard when the equality holds wherever it is held, else the address
/// equality itself). `None` when it provably contributes nothing (aliasing
/// disproven, or its guard contradicts the pc). One summand of
/// [`summarize_perm_at`].
pub(in crate::verify::heap) fn chunk_share_at(
    ctx: &mut VerifyContext<'_>,
    c: &Chunk,
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Option<(ChunkPerm, Cube)> {
    let canon = ctx.egraph.find(addr);
    let held = c.gated_perm(ctx);
    // Ground match: no address gate at all (and no `Eq` node minted).
    let share: (ChunkPerm, Cube) = if ctx.egraph.find(c.addr) == canon {
        (held, Cube::empty())
    } else {
        let eq = expr!(ctx, { c.addr } == { addr });
        // Disproven aliasing contributes nothing — skip before minting the gate.
        if matches!(
            ctx.egraph[ctx.egraph.find(eq)].data.known(),
            Some(Literal::Bool(false))
        ) {
            return None;
        }
        // Equal wherever this chunk is held: no address gate, and the
        // condition on the contribution is the guard `held` already carries.
        let presence = c.presence_within(ctx, pc_lits)?;
        if !c.presence().is_empty() && ctx.prove_under_pc(eq, &presence) {
            (held, c.presence().clone())
        } else {
            // Structural, not folded into the term. Gating the amount as
            // `ite(eq, held, 0)` collapses the whole summary to one flat
            // leaf, and `prove_perm_leaves` then has nothing to split on:
            // sufficiency has to be shown against
            // `ite(a==c, p, 0) + ite(b==c, q, 0)` with every alias gate
            // still open. Kept as a `Select`, `perm_add` distributes it,
            // and each leaf is proven under its own gate polarities.
            let cube = vec![(eq, Polarity::Positive)];
            (
                ChunkPerm::select(ctx, eq, held, ChunkPerm::Zero),
                Cube::from(cube),
            )
        }
    };
    Some(share)
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
pub(in crate::verify::heap) fn summarize_perm_at(
    ctx: &mut VerifyContext<'_>,
    chunks: &[Chunk],
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> (ChunkPerm, Vec<(Chunk, Cube)>) {
    let mut total: Option<ChunkPerm> = None;
    let mut set: Vec<(Chunk, Cube)> = Vec::new();
    for c in chunks {
        let Some((amount, cube)) = chunk_share_at(ctx, c, addr, pc_lits) else {
            continue;
        };
        set.push((c.clone(), cube));
        total = Some(match total {
            None => amount,
            Some(t) => perm_add(ctx, &t, &amount),
        });
    }
    (total.unwrap_or(ChunkPerm::Zero), set)
}

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
pub(in crate::verify::heap) fn chunk_under_pc<'c>(
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
pub(super) fn pc_alias_partners(
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
pub(super) fn gate_amount_by_pc(
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
