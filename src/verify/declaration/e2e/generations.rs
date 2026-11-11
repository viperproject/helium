//! Quantifier instance generations: a matching loop stops at the generation
//! bound, at a fixpoint, with every instance within the bound built.

use super::*;
use crate::verify::rewrite::forall::MAX_GENERATION;

fn verify_all(input: &str) -> (Vec<crate::verify::VerifyResult>, crate::verify::VerifyStats) {
    let analyzed = crate::vmir::analyze(lower(input)).expect("acyclic");
    let (results, _, stats) = crate::verify::verify_with_stats(&analyzed);
    (results, stats)
}

fn outcome<'r>(
    results: &'r [crate::verify::VerifyResult],
    name: &str,
) -> &'r Result<(), VerifyError> {
    &results
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no result for {name}"))
        .1
}

/// Each instance builds the next trigger term. The loop ends at the bound, as a
/// fixpoint rather than at the iteration limit, and the facts within the bound
/// are derived.
#[test]
fn matching_loop_saturates_at_the_generation_bound() {
    let input = r#"
domain Chain {
    function f(x: Int): Int
    function g(x: Int): Int
    axiom grows { forall x: Int :: { f(x) } f(x) == f(g(x)) }
}

method near(a: Int) {
    assert f(a) == f(g(g(g(a))))
}
"#;
    let (results, stats) = verify_all(input);
    assert!(outcome(&results, "near").is_ok());
    assert_eq!(stats.sat_bound_stops, 0, "the loop must end at a fixpoint");
    assert_eq!(stats.inst_generation_peak, u64::from(MAX_GENERATION));
    assert!(stats.inst_deferred > 0);
}

/// `f(i) == f(i + 1)` from `f(0)`: reaching `f(d)` takes the instance at
/// `i = d - 1`, whose binding is the literal a generation-`(d - 1)` instance
/// folded. Depth `MAX_GENERATION` is derived, one more is not: the e-graph
/// leaves that one to Z3, which proves it only when retried with the axiom.
#[test]
fn derivations_reach_exactly_the_generation_bound() {
    let program = |d: u32| {
        format!(
            r#"
domain Succ {{
    function f(i: Int): Bool
    axiom step {{ forall i: Int :: {{ f(i) }} f(i) == f(i + 1) }}
}}
method m()
    requires f(0)
{{
    assert f({d})
}}
"#
        )
    };
    let (results, stats) = verify_all(&program(MAX_GENERATION));
    assert!(outcome(&results, "m").is_ok());
    assert_eq!(stats.z3_timing.0.retries, 0, "the e-graph derives it");
    assert_eq!(stats.sat_bound_stops, 0);
    let (results, stats) = verify_all(&program(MAX_GENERATION + 1));
    assert!(outcome(&results, "m").is_ok());
    assert_eq!(stats.z3_timing.0.retries, 1, "only Z3, with the axiom");
    assert_eq!(stats.sat_bound_stops, 0);
}

/// An instance deferred at the bound is not forgotten: once a union makes its
/// binding shallow, it is built. `P(nx(a))` is the program's; the loop then
/// builds `nx^k(a)` at generation `k - 1`, so the instance at `nx^(B+1)(a)`
/// (`B = MAX_GENERATION`) is deferred, and naming that term with a program
/// variable `b` makes it generation 0.
#[test]
fn deferred_instance_is_built_once_its_binding_becomes_shallow() {
    let nx = |k: u32| (0..k).fold("a".to_string(), |t, _| format!("nx({t})"));
    let program = |last: &str| {
        format!(
            r#"
domain Reach {{
    function nx(x: Ref): Ref
    function P(x: Ref): Bool
    axiom step {{ forall x: Ref :: {{ P(x) }} P(x) ==> P(nx(x)) }}
}}

method m(a: Ref, b: Ref)
    requires P(a)
{{
    assert P(nx(a))
    {last}
}}
"#
        )
    };
    let frontier = nx(MAX_GENERATION + 1);
    // One step past the bound the instance is deferred, and only Z3, retried
    // with the axiom, proves the goal.
    let (results, stats) = verify_all(&program(&format!("assert P(nx({frontier}))")));
    assert!(outcome(&results, "m").is_ok());
    assert!(stats.inst_deferred > 0);
    assert_eq!(stats.z3_timing.0.retries, 1, "the e-graph alone stops");
    let named = format!("assume b == {frontier}\n    assert P(nx(b))");
    let (results, stats) = verify_all(&program(&named));
    assert_eq!(stats.z3_timing.0.retries, 0, "the e-graph builds it");
    assert!(
        outcome(&results, "m").is_ok(),
        "{:?}",
        outcome(&results, "m")
    );
}
