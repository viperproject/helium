//! Sets through the whole pipeline: quantifiers triggered on set operations,
//! and the set axioms (`verify::coll_axioms::set`).

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
fn membership_trigger_fires_at_a_mentioned_member() {
    let input = r#"
method m(s: Set[Int], k: Int)
    requires forall x: Int :: { x in s } x in s ==> x > 0
    requires k in s
{
    assert k > 0
}
"#;
    assert_verifies(&lower(input), &["m"]);
}

#[test]
fn membership_trigger_does_not_fire_on_another_set_or_a_sequence() {
    let input = r#"
method other_set(s: Set[Int], t: Set[Int], k: Int)
    requires forall x: Int :: { x in s } x in s ==> x > 0
    requires k in t
{
    assert k > 0
}

method sequence(s: Set[Int], q: Seq[Int], k: Int)
    requires forall x: Int :: { x in s } x in s ==> x > 0
    requires k in q
{
    assert k > 0
}
"#;
    let program = lower(input);
    assert_fails(&program, "other_set");
    assert_fails(&program, "sequence");
}

#[test]
fn set_literal_trigger_matches_its_element_type() {
    let input = r#"
domain Fns {
    function f(s: Set[Int]): Bool
    function p(s: Set[Bool]): Bool
}

method same(k: Int)
    requires forall x: Int :: { Set(x) } f(Set(x))
{
    assert f(Set(k))
}

method other_element_type(b: Bool)
    requires forall x: Int :: { Set(x) } f(Set(x))
{
    assert p(Set(b))
}
"#;
    let program = lower(input);
    assert_verifies(&program, &["same"]);
    assert_fails(&program, "other_element_type");
}

#[test]
fn axioms_give_membership() {
    let input = r#"
method literals(x: Int, y: Int) {
    assert x in Set(x, y) && y in Set(x, y)
    assert !(x in Set[Int]())
    assert !(3 in Set(1, 2))
}

method operations(a: Set[Int], b: Set[Int], x: Int)
    requires x in a && !(x in b)
{
    assert x in a union b
    assert !(x in a intersection b)
    assert x in a setminus b
    assert !(x in b setminus a)
}
"#;
    assert_verifies(&lower(input), &["literals", "operations"]);
}

#[test]
fn axioms_give_cardinalities() {
    let input = r#"
method literals(x: Int, y: Int)
    requires x != y
{
    assert |Set[Int]()| == 0
    assert |Set(x)| == 1
    assert |Set(x, x)| == 1
    assert |Set(x, y)| == 2
}

method operations(a: Set[Int], b: Set[Int]) {
    assert |a union b| + |a intersection b| == |a| + |b|
    assert |a setminus b| == |a| - |a intersection b|
    assert 0 <= |a|
}

method nonempty(a: Set[Int], x: Int)
    requires x in a
{
    assert |a| > 0
}
"#;
    assert_verifies(&lower(input), &["literals", "operations", "nonempty"]);
}

#[test]
fn axioms_give_subset_and_extensionality() {
    let input = r#"
method subset(a: Set[Int], b: Set[Int], c: Set[Int])
    requires a subset b && b subset c
{
    assert a subset c
    assert a subset (a union b)
    assert (a intersection b) subset a
}

method extensional(a: Set[Int], b: Set[Int])
    requires a subset b && b subset a
{
    assert a == b
}

method literals() {
    assert Set(1, 2) == Set(2, 1, 1)
    assert Set(1) != Set(2)
}
"#;
    assert_verifies(&lower(input), &["subset", "extensional", "literals"]);
}

#[test]
fn axioms_do_not_prove_what_does_not_hold() {
    let input = r#"
method equal_cards(a: Set[Int], b: Set[Int])
    requires |a| == |b|
{
    assert a == b
}

method union_card(a: Set[Int], b: Set[Int]) {
    assert |a union b| == |a| + |b|
}

method converse_subset(a: Set[Int], b: Set[Int])
    requires a subset b
{
    assert b subset a
}

method consistent(a: Set[Int], b: Set[Int])
    requires a subset b && |a| == |b| && 1 in a
{
    assert false
}
"#;
    let program = lower(input);
    for method in ["equal_cards", "union_card", "converse_subset", "consistent"] {
        assert_fails(&program, method);
    }
}

/// With Z3 off, the facts one axiom instance gives are still proved: the
/// e-graph instantiates the set recipes itself, so Z3 cannot mask a recipe
/// that never fires.
#[test]
fn egraph_instantiates_the_set_axioms_without_z3() {
    let input = r#"
method singleton(x: Int) {
    assert x in Set(x)
    assert |Set(x)| == 1
}

method empty(x: Int) {
    assert !(x in Set[Int]())
    assert |Set[Int]()| == 0
}

method union_member(a: Set[Int], b: Set[Int], x: Int)
    requires x in a
{
    assert x in a union b
}

method subset_member(a: Set[Int], b: Set[Int], x: Int)
    requires a subset b && x in a
{
    assert x in b
}
"#;
    let program = lower(input);
    crate::verify::without_z3(|| {
        assert_verifies(
            &program,
            &["singleton", "empty", "union_member", "subset_member"],
        )
    });
}
