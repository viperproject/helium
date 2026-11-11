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
    // Not vacuous: firing at `t` would prove the assertion. The hypothesis must
    // hold of every `Seq[Bool]` (`|s| == 3` would contradict the sequence axioms'
    // `|Seq[Bool]()| == 0` and prove anything).
    let input = r#"
domain Fns { function p(n: Int): Bool }

method m(t: Seq[Int])
    requires forall s: Seq[Bool] :: { |s| } p(|s|)
{
    assert p(|t|)
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

// The sequence axioms (`verify::seq_axioms`), through the whole pipeline.

#[test]
fn axioms_give_lengths_indices_and_slices() {
    let input = r#"
method lengths(s: Seq[Int], t: Seq[Int]) {
    assert |Seq(1, 2) ++ Seq(3)| == 3
    assert |s ++ t| == |s| + |t|
    assert 0 <= |s|
    assert |Seq[Int]()| == 0
}

method indices(x: Int, y: Int) {
    assert Seq(x, y)[1] == y
    assert [2..5)[1] == 3
    assert |[2..5)| == 3
    assert 3 in [2..5)
}

method slices(s: Seq[Int]) {
    assert s[..0] == Seq[Int]()
    assert s[0..] == s
    assert s[..|s|] == s
}
"#;
    let program = lower(input);
    for method in ["lengths", "indices", "slices"] {
        let result = verify_named_method(&program, method);
        assert!(result.is_ok(), "{method}: expected Ok, got {result:?}");
    }
}

#[test]
fn axioms_give_containment_and_extensionality() {
    let input = r#"
method contains(s: Seq[Int], k: Int)
    requires 0 <= k && k < |s|
{
    assert s[k] in s
}

method update(s: Seq[Int], i: Int, v: Int)
    requires 0 <= i && i < |s|
{
    assert s[i := v][i] == v
    assert |s[i := v]| == |s|
}

method extensional(s: Seq[Int], t: Seq[Int])
    requires |s| == |t|
    requires forall i: Int :: { s[i] } { t[i] } 0 <= i && i < |s| ==> s[i] == t[i]
{
    assert s == t
}
"#;
    let program = lower(input);
    for method in ["contains", "update", "extensional"] {
        let result = verify_named_method(&program, method);
        assert!(result.is_ok(), "{method}: expected Ok, got {result:?}");
    }
}

#[test]
fn axioms_do_not_prove_what_does_not_hold() {
    let input = r#"
method m(s: Seq[Int], t: Seq[Int])
    requires |s| == |t|
{
    assert s == t
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
#[ignore = "needs the sequence axioms in Z3, which are not ported (e-graph only)"]
fn sequences_of_different_lengths_are_unequal() {
    // Needs the lengths, which the program never writes: only the Z3 axioms
    // build them. They used to reach Z3 only on a *second* sync, after the
    // first had made the `Seq` sorts, so a goal decided by one call missed them.
    let input = r#"
method m() {
    assert Seq(1, 2) != Seq(1, 2, 3)
}
"#;
    let result = verify_named_method(&lower(input), "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
#[ignore = "needs the sequence axioms in Z3, which are not ported (e-graph only)"]
fn a_call_result_has_the_callee_result_type() {
    // `r := m(..)` used to havoc `r` as an `Int`, so Z3 declared it at the
    // wrong sort and dropped every quantifier mentioning it, `m`'s
    // postconditions among them.
    let input = r#"
method reverse(s: Seq[Int]) returns (r: Seq[Int])
  ensures |r| == |s|
  ensures forall i: Int :: { r[i] } 0 <= i && i < |s| ==> r[i] == s[|s| - 1 - i]

method twice(s: Seq[Int]) {
  var r: Seq[Int]
  var rr: Seq[Int]
  r := reverse(s)
  rr := reverse(r)
  assert rr == s
}
"#;
    let result = verify_named_method(&lower(input), "twice");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
#[ignore = "needs the sequence axioms in Z3, which are not ported (e-graph only)"]
fn shifted_indices_do_not_loop_through_drop() {
    // `drop-index2` reads `s[i]` as `s[n..][i - n]`, and `drop-index1` reads
    // that back as `s[(i - n) + n]`. Without the round-trip equality the
    // second is a new index term, and `sorted(s)` (triggered on pairs of
    // them) instantiates without end. (`tests/cases/passing/seq/insert_sorted.vpr`.)
    let input = r#"
function sorted(s: Seq[Int]): Bool
{
  forall i: Int, j: Int :: { s[i], s[j] } 0 <= i && i < j && j < |s| ==> s[i] <= s[j]
}

method insert_sorted(s: Seq[Int], x: Int) returns (r: Seq[Int])
  requires sorted(s)
  ensures sorted(r)
{
  var i: Int := 0
  while (i < |s| && s[i] < x)
    invariant 0 <= i && i <= |s|
    invariant forall j: Int :: { s[j] } 0 <= j && j < i ==> s[j] < x
  {
    i := i + 1
  }
  r := s[..i] ++ Seq(x) ++ s[i..]
  assert r[i] == x
  assert forall k: Int :: { s[k] } 0 <= k && k < i ==> r[k] == s[k]
  assert forall k: Int :: { s[k] } i <= k && k < |s| ==> r[k + 1] == s[k]
}
"#;
    let result = verify_named_method(&lower(input), "insert_sorted");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
