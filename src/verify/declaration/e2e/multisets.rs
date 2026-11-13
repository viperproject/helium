//! Multisets through the whole pipeline: `x in m` as a count, quantifiers
//! triggered on it, and the multiset axioms (`verify::coll_axioms::multiset`).
//! The corpus (`tests/cases/{passing,failing}/multiset`) has the broad
//! coverage; these pin what the axioms give and what they must not.

use super::*;

fn assert_fails(program: &vmir::Program, method: &str) {
    let result = verify_named_method(program, method);
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "{method}: expected AssertionFailed, got {result:?}"
    );
}

fn assert_verifies(program: &vmir::Program, methods: &[&str]) {
    for method in methods {
        let result = verify_named_method(program, method);
        assert!(result.is_ok(), "{method}: expected Ok, got {result:?}");
    }
}

#[test]
fn count_trigger_fires_at_a_mentioned_element() {
    let input = r#"
method m(s: Multiset[Int], k: Int)
    requires forall x: Int :: { x in s } (x in s) > 0 ==> x > 0
    requires (k in s) > 0
{
    assert k > 0
}
"#;
    assert_verifies(&lower(input), &["m"]);
}

/// A count trigger is a multiset term: neither a set membership of the same
/// shape nor a count in another multiset matches it.
#[test]
fn count_trigger_does_not_fire_on_another_multiset_or_a_set() {
    let input = r#"
method other_multiset(s: Multiset[Int], t: Multiset[Int], k: Int)
    requires forall x: Int :: { x in s } (x in s) > 0 ==> x > 0
    requires (k in t) > 0
{
    assert k > 0
}

method set(s: Multiset[Int], q: Set[Int], k: Int)
    requires forall x: Int :: { x in s } (x in s) > 0 ==> x > 0
    requires k in q
{
    assert k > 0
}
"#;
    let program = lower(input);
    assert_fails(&program, "other_multiset");
    assert_fails(&program, "set");
}

#[test]
fn axioms_give_counts() {
    let input = r#"
method literals(x: Int, y: Int)
    requires x != y
{
    assert (x in Multiset(x, y, x)) == 2
    assert (y in Multiset(x, y, x)) == 1
    assert (x in Multiset[Int]()) == 0
}

method operations(a: Multiset[Int], b: Multiset[Int], x: Int)
    requires (x in a) == 2 && (x in b) == 1
{
    assert (x in a union b) == 3
    assert (x in a intersection b) == 1
    assert (x in a setminus b) == 1
    assert (x in b setminus a) == 0
}
"#;
    assert_verifies(&lower(input), &["literals", "operations"]);
}

#[test]
fn axioms_give_cardinalities() {
    let input = r#"
method literals(x: Int) {
    assert |Multiset[Int]()| == 0
    assert |Multiset(x)| == 1
    assert |Multiset(x, x)| == 2
}

method operations(a: Multiset[Int], b: Multiset[Int]) {
    assert |a union b| == |a| + |b|
    assert |a setminus b| == |a| - |a intersection b|
    assert 0 <= |a|
}

method nonempty(a: Multiset[Int], x: Int)
    requires (x in a) > 0
{
    assert |a| > 0
}
"#;
    assert_verifies(&lower(input), &["literals", "operations", "nonempty"]);
}

#[test]
fn axioms_give_subset_and_extensionality() {
    let input = r#"
method subset(a: Multiset[Int], b: Multiset[Int], c: Multiset[Int])
    requires a subset b && b subset c
{
    assert a subset c
    assert a subset (a union b)
    assert (a intersection b) subset a
}

method extensional(a: Multiset[Int], b: Multiset[Int])
    requires a subset b && b subset a
{
    assert a == b
}

method literals() {
    assert Multiset(1, 2, 1) == Multiset(2, 1, 1)
    assert Multiset(1, 1) != Multiset(1)
}
"#;
    assert_verifies(&lower(input), &["subset", "extensional", "literals"]);
}

/// The facts a multiset differs from a set in: duplicates count, and union
/// adds rather than joins. Each must not hold, and the axioms must stay
/// consistent with a satisfiable precondition.
#[test]
fn axioms_do_not_prove_what_does_not_hold() {
    let input = r#"
method duplicates_collapse(x: Int) {
    assert Multiset(x, x) == Multiset(x)
}

method union_is_idempotent(a: Multiset[Int]) {
    assert a union a == a
}

method equal_cards(a: Multiset[Int], b: Multiset[Int])
    requires |a| == |b|
{
    assert a == b
}

method converse_subset(a: Multiset[Int], b: Multiset[Int])
    requires a subset b
{
    assert b subset a
}

method negative_difference(a: Multiset[Int], b: Multiset[Int], x: Int)
    requires (x in a) == 1 && (x in b) == 3
{
    assert (x in a setminus b) < 0
}

method consistent(a: Multiset[Int], b: Multiset[Int], x: Int)
    requires a subset b && (x in a) == 2 && |b| == 3
{
    assert false
}
"#;
    let program = lower(input);
    for method in [
        "duplicates_collapse",
        "union_is_idempotent",
        "equal_cards",
        "converse_subset",
        "negative_difference",
        "consistent",
    ] {
        assert_fails(&program, method);
    }
}

/// With Z3 off, the facts one axiom instance gives are still proved: the
/// e-graph instantiates the multiset recipes itself, so Z3 cannot mask a
/// recipe that never fires.
#[test]
fn egraph_instantiates_the_multiset_axioms_without_z3() {
    let input = r#"
method singleton(x: Int) {
    assert (x in Multiset(x)) == 1
    assert |Multiset(x)| == 1
}

method empty(x: Int) {
    assert (x in Multiset[Int]()) == 0
    assert |Multiset[Int]()| == 0
}

method union_count(a: Multiset[Int], b: Multiset[Int], x: Int)
    requires (x in a) == 1 && (x in b) == 2
{
    assert (x in a union b) == 3
}

method subset_count(a: Multiset[Int], b: Multiset[Int], x: Int)
    requires a subset b && (x in a) == 2
{
    assert (x in b) >= 2
}
"#;
    let program = lower(input);
    crate::verify::without_z3(|| {
        assert_verifies(
            &program,
            &["singleton", "empty", "union_count", "subset_count"],
        )
    });
}
