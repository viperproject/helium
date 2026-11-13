//! Maps through the whole pipeline: the lookup side condition, `k in m` as
//! `k in domain(m)`, and the map axioms (`verify::coll_axioms::map`).
//! The corpus (`tests/cases/{passing,failing}/map`) has the broad coverage;
//! these pin what the e-graph tiers decide on their own.

use super::*;

fn assert_fails_with(program: &vmir::Program, method: &str, want: fn(&VerifyError) -> bool) {
    let result = crate::verify::z3::without_z3(|| verify_named_method(program, method));
    assert!(
        matches!(result, Err(ref err) if want(err.root_cause())),
        "{method}: expected the given failure, got {result:?}"
    );
}

fn assert_verifies(program: &vmir::Program, methods: &[&str]) {
    for method in methods {
        let result = crate::verify::z3::without_z3(|| verify_named_method(program, method));
        assert!(result.is_ok(), "{method}: expected Ok, got {result:?}");
    }
}

/// A lookup needs its key in the map, and an update or a guard supplies it.
#[test]
fn lookup_needs_its_key() {
    let input = r#"
method unguarded(m: Map[Int, Int], k: Int) {
    var v: Int := m[k]
}
method after_update(m: Map[Int, Int], k: Int) {
    var v: Int := m[k := 3][k]
    assert v == 3
}
method guarded(m: Map[Int, Int], k: Int)
    requires k in m
{
    var v: Int := m[k]
}
"#;
    let program = lower(input);
    assert_fails_with(&program, "unguarded", |e| {
        matches!(e, VerifyError::SideCondition("map may not contain the key"))
    });
    assert_verifies(&program, &["after_update", "guarded"]);
}

/// `k in m` and `k in domain(m)` are one term: a quantifier triggered on
/// either form fires on the other.
#[test]
fn key_membership_is_domain_membership() {
    let input = r#"
method on_domain(m: Map[Int, Int], k: Int)
    requires forall j: Int :: { j in domain(m) } j in domain(m) ==> j > 0
    requires k in m
{
    assert k > 0
}
method on_map(m: Map[Int, Int], k: Int)
    requires forall j: Int :: { j in m } j in m ==> j > 0
    requires k in domain(m)
{
    assert k > 0
}
"#;
    assert_verifies(&lower(input), &["on_domain", "on_map"]);
}

/// The update axioms, at a key that is the updated one and at one that is
/// not, and the cardinality of an update.
#[test]
fn update_axioms() {
    let input = r#"
method same(m: Map[Int, Bool], k: Int) {
    assert k in m[k := true]
    assert m[k := true][k]
}
method other(m: Map[Int, Bool], j: Int, k: Int)
    requires j != k && j in m
{
    assert j in m[k := true]
    assert m[k := true][j] == m[j]
}
method card_new(m: Map[Int, Bool], k: Int)
    requires !(k in m)
{
    assert |m[k := true]| == |m| + 1
}
method leaks(m: Map[Int, Bool], j: Int, k: Int)
    requires j != k && j in m
{
    assert m[k := true][j]
}
"#;
    let program = lower(input);
    assert_verifies(&program, &["same", "other", "card_new"]);
    assert_fails_with(&program, "leaks", |e| {
        matches!(e, VerifyError::AssertionFailed)
    });
}

/// A map with a key is nonempty, as is its domain (a set of the size of the
/// map), and that key's value is in its range.
#[test]
fn domain_and_range_are_sets() {
    let input = r#"
method dom(m: Map[Int, Int], k: Int)
    requires k in m
{
    assert |m| != 0
    assert |domain(m)| != 0
}
method ran(m: Map[Int, Int], k: Int)
    requires k in m
{
    assert m[k] in range(m)
    assert |range(m)| > 0
}
"#;
    assert_verifies(&lower(input), &["dom", "ran"]);
}
