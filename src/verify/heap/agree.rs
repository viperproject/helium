//! A location has one value: wherever two holders both hold a positive share of
//! it, their values agree. Every place the heap states that is here.
//!
//! The fact is [`assume_agree`]. Which amounts it is conditional on is the
//! caller's: an amount already **known** positive (proven under the pc) is left
//! out of the antecedent, so the agreement fires as soon as the others are, and
//! two known amounts need no antecedent at all.
//!
//! Silicon's `combineSnapshots` splits the same ways. When neither amount is
//! known positive it mints a fresh snapshot constrained by both implications
//! ("it is not sound to use t1 or t2 and constrain it"); [`joint_value`] makes
//! the same statement total instead, case-analysing the value on the first
//! amount, which is strictly more precise and needs no fresh symbol.

use crate::verify::context::VerifyContext;
use crate::verify::heap::cond::Lit;
use crate::vmir::Polarity;

/// Assume `a == b` wherever every amount in `unknown` is positive and `guard`
/// holds: `(guard ∧ ⋀ 0 < p) ⇒ a == b`. `unknown` is innermost first (the order
/// the implication chain folds), `guard` outermost first (a pc). Nothing when
/// the two are one e-class already.
pub(super) fn assume_agree(
    ctx: &mut VerifyContext<'_>,
    a: egg::Id,
    b: egg::Id,
    unknown: &[egg::Id],
    guard: &[Lit],
) {
    if ctx.egraph.find(a) == ctx.egraph.find(b) {
        return;
    }
    let eq = expr!(ctx, { a } == { b });
    let positive: Vec<Lit> = unknown
        .iter()
        .map(|&p| (expr!(ctx, (0 / 1) < r { p }), Polarity::Positive))
        .collect();
    ctx.assume_guarded(eq, positive.into_iter().chain(guard.iter().rev().copied()));
}

/// The value of a location two chunks hold at once — amount `pa` with value
/// `va`, amount `pb` with value `vb` — under the path condition `pc`, with the
/// agreement between them assumed. `known` says which amounts are already
/// known positive there:
///
/// - **both** — the values are one where `pc` holds: the classes are collapsed
///   outright when `pc` is empty, the cheap common case (a loop frame restore
///   carrying a value across a cut), and agree under `pc` otherwise.
/// - **one** — its value, agreeing with the other's wherever that one is held.
/// - **neither** — `0 < pa ? va : vb` (intentionally asymmetric: where `pa` is
///   empty the value is `vb`'s), agreeing wherever both are held.
pub(super) fn joint_value(
    ctx: &mut VerifyContext<'_>,
    (pa, va): (egg::Id, egg::Id),
    (pb, vb): (egg::Id, egg::Id),
    known: (bool, bool),
    pc: &[Lit],
) -> egg::Id {
    match known {
        // Collapsed outright only when that is true on every path: the amounts
        // were shown positive under `pc`, and on a path outside it these two
        // chunks may not exist. There the agreement is guarded like any other.
        (true, true) if pc.is_empty() => {
            ctx.union(va, vb);
            va
        }
        (true, true) => {
            assume_agree(ctx, va, vb, &[], pc);
            va
        }
        (true, false) => {
            assume_agree(ctx, va, vb, &[pb], pc);
            va
        }
        (false, true) => {
            assume_agree(ctx, vb, va, &[pa], pc);
            vb
        }
        (false, false) => {
            assume_agree(ctx, va, vb, &[pb, pa], pc);
            let a_held = expr!(ctx, (0 / 1) < r { pa });
            expr!(ctx, if {a_held} then {va} else {vb})
        }
    }
}
