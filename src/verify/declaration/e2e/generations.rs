//! Quantifier instance generations: a matching loop stops at the generation
//! bound, at a fixpoint, with every instance within the bound built; a unit that
//! pass cannot decide is verified again without the bound, keeping its proofs.

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
    assert_eq!(stats.unit_reruns, 0);
}

/// `f(i) == f(i + 1)` from `f(0)`, with `body` in a method that requires `f(0)`.
/// Reaching `f(d)` takes the instance at `i = d - 1`, whose binding is the
/// literal a generation-`(d - 1)` instance folded.
fn succ(body: &str) -> String {
    format!(
        r#"
domain Succ {{
    function f(i: Int): Bool
    axiom step {{ forall i: Int :: {{ f(i) }} f(i) == f(i + 1) }}
}}

method m()
    requires f(0)
{{
    {body}
}}
"#
    )
}

/// Depth `MAX_GENERATION` is derived by the first pass. One more is past the
/// bound there, and the unit is verified again without it: the bound decides how
/// deep the first pass goes, not the verdict.
#[test]
fn a_chain_past_the_bound_is_verified_by_the_second_pass() {
    let (results, stats) = verify_all(&succ(&format!("assert f({MAX_GENERATION})")));
    assert!(outcome(&results, "m").is_ok());
    assert_eq!(stats.unit_reruns, 0);
    let (results, stats) = verify_all(&succ(&format!("assert f({})", MAX_GENERATION + 1)));
    assert!(outcome(&results, "m").is_ok());
    assert_eq!((stats.unit_reruns, stats.unit_reruns_verified), (1, 1));
}

/// A goal that fails on a divergent axiom is where a stop is unavoidable: the
/// second pass ends its runs at the iteration limit, and the goal fails. The
/// unit is verified twice, however many of its obligations fail.
#[test]
fn a_goal_failing_on_a_divergent_axiom_is_decided_by_one_second_pass() {
    let goal = format!("assert !f({})", MAX_GENERATION + 1);
    let (results, stats) = verify_all(&succ(&format!("{goal}\n    {goal}")));
    assert!(outcome(&results, "m").is_err());
    assert_eq!((stats.unit_reruns, stats.unit_reruns_verified), (1, 0));
    assert!(stats.sat_bound_stops > 0);
}

/// A `refute` passes when its proof fails, which the bound can cause: a unit
/// whose first pass passed a refute after deferring is decided by the second.
#[test]
fn a_refute_passed_past_the_bound_is_decided_without_it() {
    let (results, stats) = verify_all(&succ(&format!("refute f({})", MAX_GENERATION + 1)));
    assert!(matches!(
        outcome(&results, "m").as_ref().map_err(|e| e.root_cause()),
        Err(VerifyError::RefuteFailed)
    ));
    assert_eq!(stats.unit_reruns, 1);
}

/// A `refute` whose expression the first pass proved fails without a second
/// pass, although that pass deferred instances: the proof stands, and a pass
/// whose runs a termination bound stops sooner could only fail to repeat it.
#[test]
fn a_refute_proven_after_deferring_fails_without_a_second_pass() {
    let input = r#"
domain Chain {
    function f(x: Int): Int
    function g(x: Int): Int
    axiom grows { forall x: Int :: { f(x) } f(x) == f(g(x)) }
}

method m(a: Int) {
    refute f(a) == f(g(g(a)))
}
"#;
    let (results, stats) = verify_all(input);
    assert!(matches!(
        outcome(&results, "m").as_ref().map_err(|e| e.root_cause()),
        Err(VerifyError::RefuteFailed)
    ));
    assert!(stats.inst_deferred > 0);
    assert_eq!(stats.unit_reruns, 0);
}

/// A `refute` the first pass proved after assuming a failed obligation is where
/// it stopped: the second pass checks the earlier obligations, and fails the
/// refute there without proving it again, which a termination bound could
/// stop. If an earlier obligation fails in the second pass, that is the error.
#[test]
fn a_refute_proven_after_a_deferred_failure_fails_in_the_second_pass() {
    let deep = format!("assert f({})", MAX_GENERATION + 1);
    let (results, stats) = verify_all(&succ(&format!("{deep}\n    refute f(1)")));
    assert!(matches!(
        outcome(&results, "m").as_ref().map_err(|e| e.root_cause()),
        Err(VerifyError::RefuteFailed)
    ));
    assert_eq!(stats.unit_reruns, 1);
    assert_eq!(stats.prove_failed_first_pass, 1);
    assert_eq!(stats.prove_first_pass, 1, "the refute is taken as proven");
    let (results, stats) = verify_all(&succ(&format!(
        "assert !f({})\n    refute f(1)",
        MAX_GENERATION + 1
    )));
    assert!(matches!(
        outcome(&results, "m").as_ref().map_err(|e| e.root_cause()),
        Err(VerifyError::AssertionFailed)
    ));
    assert_eq!((stats.unit_reruns, stats.prove_first_pass), (1, 0));
}

/// The second pass takes what the first proved as proven, before or after an
/// obligation the first failed: `y + z > 7` needs Z3, which the first pass ran
/// and the second does not. `f(2B)` is past the bound (and past what Z3
/// instantiates), so the first pass fails it, assumes it and goes on, and the
/// second proves it.
#[test]
fn the_second_pass_takes_the_first_pass_proofs_as_proven() {
    let deep = format!("assert f({})", 2 * MAX_GENERATION);
    let z3 = "assert y + z > 7";
    for body in [format!("{z3}\n    {deep}"), format!("{deep}\n    {z3}")] {
        let input = format!(
            r#"
domain Succ {{
    function f(i: Int): Bool
    axiom step {{ forall i: Int :: {{ f(i) }} f(i) == f(i + 1) }}
}}

method m(y: Int, z: Int)
    requires f(0) && y > 3 && z > 3
{{
    {body}
}}
"#
        );
        let (results, stats) = verify_all(&input);
        assert!(outcome(&results, "m").is_ok(), "{body}");
        assert_eq!((stats.unit_reruns, stats.unit_reruns_verified), (1, 1));
        assert_eq!(stats.prove_failed_first_pass, 1);
        assert_eq!(stats.prove_first_pass, 1);
        // The first pass asks Z3 for both asserts, the second for neither.
        assert_eq!(stats.z3_timing.0.calls, 2);
    }
}

/// A `refute` the first pass passed rests on a failed proof, so the second
/// pass checks it again rather than taking it as passed, even where it comes
/// before the first pass's failure.
#[test]
fn a_refute_the_first_pass_passed_is_checked_again() {
    let body = format!(
        "refute f({})\n    assert f({})",
        MAX_GENERATION + 1,
        MAX_GENERATION + 2
    );
    let (results, stats) = verify_all(&succ(&body));
    assert!(matches!(
        outcome(&results, "m").as_ref().map_err(|e| e.root_cause()),
        Err(VerifyError::RefuteFailed)
    ));
    assert_eq!(stats.unit_reruns, 1);
}

/// A unit whose first pass deferred nothing ran the unbounded computation, so
/// its failure stands without a second pass.
#[test]
fn a_failure_without_deferral_is_not_rerun() {
    let input = r#"
domain Once {
    function f(i: Int): Bool
    axiom one { forall i: Int :: { f(i) } f(i) ==> i >= 0 }
}

method m(i: Int)
    requires f(i)
{
    assert i > 0
}
"#;
    let (results, stats) = verify_all(input);
    assert!(outcome(&results, "m").is_err());
    assert_eq!(stats.inst_deferred, 0);
    assert_eq!(stats.unit_reruns, 0);
}

/// An instance deferred at the bound is not forgotten: once a union makes its
/// binding shallow, the first pass builds it. `P(nx(a))` is the program's; the
/// loop then builds `nx^k(a)` at generation `k - 1`, so the instance at
/// `nx^(B+1)(a)` (`B = MAX_GENERATION`) is deferred, and naming that term with a
/// program variable `b` makes it generation 0.
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
    let (results, stats) = verify_all(&program(&format!("assert P(nx({frontier}))")));
    assert!(outcome(&results, "m").is_ok());
    assert_eq!(stats.unit_reruns, 1, "one step past the bound is deferred");
    let named = format!("assume b == {frontier}\n    assert P(nx(b))");
    let (results, stats) = verify_all(&program(&named));
    assert!(
        outcome(&results, "m").is_ok(),
        "{:?}",
        outcome(&results, "m")
    );
    assert_eq!(stats.unit_reruns, 0, "built by the first pass");
}
