//! Quantifiers triggered on sequence operations. Each assertion follows only from
//! instantiating the hypothesis, so these cover trigger matching and not the
//! sequence rewrite rules.

use super::*;

#[test]
fn index_trigger_fires_at_a_mentioned_index() {
    let input = r#"
method m(s: Seq[Int], k: Int)
    requires forall i: Int :: { s[i] } 0 <= i && i < |s| ==> s[i] > 0
    requires 0 <= k && k < |s|
{
    assert s[k] > 0
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn index_trigger_does_not_fire_on_another_sequence() {
    let input = r#"
method m(s: Seq[Int], t: Seq[Int], k: Int)
    requires forall i: Int :: { s[i] } 0 <= i && i < |s| ==> s[i] > 0
    requires 0 <= k && k < |t|
{
    assert t[k] > 0
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn take_trigger_does_not_fire_on_a_drop() {
    let input = r#"
domain Fns { function h(n: Int): Int }

method m(s: Seq[Int], k: Int)
    requires forall n: Int :: { s[..n] } h(|s[..n]|) == n
{
    assert h(|s[k..]|) == k
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn singleton_trigger_matches_its_element_type() {
    let input = r#"
domain Fns {
    function f(s: Seq[Int]): Bool
    function p(s: Seq[Bool]): Bool
}

method same(k: Int)
    requires forall x: Int :: { Seq(x) } f(Seq(x))
{
    assert f(Seq(k))
}

method other_element_type(b: Bool)
    requires forall x: Int :: { Seq(x) } f(Seq(x))
{
    assert p(Seq(b))
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "same");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let result = verify_named_method(&program, "other_element_type");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn singleton_trigger_fires_on_each_element_of_a_longer_literal() {
    // `Seq(j, k)` is `Seq(j) ++ Seq(k)`, so it mentions both singletons.
    let input = r#"
domain Fns { function g(x: Int): Bool }

method both_elements(j: Int, k: Int)
    requires forall x: Int :: { Seq(x) } g(x)
{
    var s: Seq[Int] := Seq(j, k)
    assert g(j)
    assert g(k)
}

method not_a_longer_instance(j: Int, k: Int)
    requires forall x: Int :: { Seq(x) } g(x)
{
    var s: Seq[Int] := Seq(j)
    assert g(k)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "both_elements");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let result = verify_named_method(&program, "not_a_longer_instance");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn longer_literal_trigger_matches_the_concatenation() {
    let input = r#"
domain Fns { function f(s: Seq[Int]): Bool }

method same_length(j: Int, k: Int)
    requires forall x: Int, y: Int :: { Seq(x, y) } f(Seq(x, y))
{
    assert f(Seq(j, k))
}

method written_as_a_concatenation(j: Int, k: Int)
    requires forall x: Int, y: Int :: { Seq(x, y) } f(Seq(x, y))
{
    assert f(Seq(j) ++ Seq(k))
}

method shorter_literal(k: Int)
    requires forall x: Int, y: Int :: { Seq(x, y) } f(Seq(x, y))
{
    assert f(Seq(k))
}
"#;
    let program = lower(input);
    for method in ["same_length", "written_as_a_concatenation"] {
        let result = verify_named_method(&program, method);
        assert!(result.is_ok(), "{method}: expected Ok, got {result:?}");
    }
    let result = verify_named_method(&program, "shorter_literal");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn literal_is_the_concatenation_of_its_singletons() {
    // Holds with no sequence rule: both sides lower to the same term.
    let input = r#"
method m(j: Int, k: Int) {
    assert Seq(j) ++ Seq(k) == Seq(j, k)
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn binder_fixed_by_another_term_of_the_group_is_accepted() {
    let input = r#"
domain Fns { function f(s: Seq[Int]): Bool }

method m(t: Seq[Int], k: Int)
    requires forall s: Seq[Int], i: Int :: { f(s), s[i] } f(s) && 0 <= i && i < |s| ==> s[i] == 0
    requires f(t)
    requires 0 <= k && k < |t|
{
    assert t[k] == 0
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn binder_under_a_sequence_head_fires_at_its_element_type() {
    let input = r#"
method m(t: Seq[Int], k: Int)
    requires forall s: Seq[Int], i: Int :: { s[i] } 0 <= i && i < |s| ==> s[i] > 0
    requires 0 <= k && k < |t|
{
    assert t[k] > 0
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn binder_under_a_sequence_head_ignores_other_element_types() {
    // Not vacuous: firing at `t` would prove the assertion.
    let input = r#"
method m(t: Seq[Int])
    requires forall s: Seq[Bool] :: { |s| } |s| == 3
{
    assert |t| == 3
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn wrong_element_type_instance_does_not_reach_the_ite() {
    // Instantiating at `t` would make `3` the condition of an `ite`, which
    // `ConstFold` treats as a broken invariant.
    let input = r#"
method m(t: Seq[Int])
    requires |t| > 0
    requires t[0] == 3
    requires forall s: Seq[Bool], i: Int :: { s[i] } 0 <= i && i < |s| ==> (s[i] ==> s[i])
{
    assert t[0] == 3
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn binders_linked_to_each_other_fire_at_their_element_type() {
    let input = r#"
domain Fns { function f(s: Seq[Int]): Bool }

method m(a: Seq[Int], b: Seq[Int], y: Int)
    requires forall s: Seq[Int], t: Seq[Int] :: { s ++ t } f(s ++ t)
    requires forall s: Seq[Int], x: Int :: { x in s } x in s ==> x > 0
    requires y in a
{
    assert f(a ++ b)
    assert y > 0
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
