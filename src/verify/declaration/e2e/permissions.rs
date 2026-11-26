//! Permission-accounting focused tests: fractional merges, aliasing via
//! over-permissioning, chained exhales, conditional permission amounts.

use super::*;

#[test]
fn fractional_perm_merge_reads_back_summed() {
    // Two disjoint-fraction inhales of the same chunk must merge additively.
    let input = r#"
field f: Int

method client(x: Ref)
{
    inhale acc(x.f, 1/4)
    inhale acc(x.f, 1/2)

    assert perm(x.f) == 3/4
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn chained_partial_exhales_exceeding_held_fails() {
    // Two sequential partial exhales that together exceed the held
    // permission must be rejected on the second, not silently clamped to 0.
    let input = r#"
field f: Int

method client(x: Ref)
{
    inhale acc(x.f, 3/4)

    exhale acc(x.f, 1/2)
    exhale acc(x.f, 1/2)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn conditional_perm_amount_exhaled_by_same_condition() {
    // A conditionally-inhaled permission amount can be exhaled back by the
    // same condition, regardless of which branch `b` takes.
    let input = r#"
field f: Int

method client(x: Ref, b: Bool)
{
    inhale b ? acc(x.f, 1/1) : acc(x.f, 1/2)

    exhale b ? acc(x.f, 1/1) : acc(x.f, 1/2)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn conditional_perm_amount_wrong_branch_exhale_fails() {
    // Exhaling the amount for the OTHER branch unconditionally must fail:
    // only 1/2 was inhaled, but a full 1/1 is demanded unconditionally.
    let input = r#"
field f: Int

method client(x: Ref)
{
    inhale acc(x.f, 1/2)

    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn heap_dep_call_with_fractional_share_suffices() {
    // Function preconditions are lowered to **wildcards** (a function only needs
    // *some* positive share to read), so a heap-dependent call requires only a
    // positive fraction at the call site — not the written amount. Holding 1/2
    // when the callee's `requires acc(x.f)` names the full amount now suffices
    // (the wildcard exhale assumes `w < 1/2`). Contrast
    // `heap_dep_call_without_permission_fails`: holding *no* permission still
    // fails, since a wildcard cannot be taken from an empty location.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f)
{ x.f }

method m(y: Ref)
    requires acc(y.f, 1/2)
{
    var a: Int := get(y)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "a positive fractional share satisfies a wildcard precondition, got {result:?}"
    );
}

#[test]
fn conditionally_held_perm_reads_back_gated() {
    // A guard-hoisted merge stores a conditionally-held chunk as a flat guard
    // cube over a guard-free `1/1` amount. A `perm()` read must reconstruct that
    // gating (`perm_value`), or it reports the chunk as fully held on the arm
    // that gave it away — and `assert acc(x.f, write)` then succeeds on a state
    // where `exhale acc(x.f)` fails.
    let input = r#"
field f: Int

method client(x: Ref, b: Bool)
    requires acc(x.f)
{
    if (b) { } else { exhale acc(x.f) }

    assert acc(x.f, write)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        result.is_err(),
        "permission is only held on the `b` arm, got {result:?}"
    );
}

#[test]
fn conditionally_held_perm_expression_is_not_write() {
    // Same defect through the `perm()` expression itself: the amount is
    // `b ? 1/1 : 0`, so neither `== write` nor `== none` is provable.
    let input = r#"
field f: Int

method client(x: Ref, b: Bool)
    requires acc(x.f)
{
    if (b) { } else { exhale acc(x.f) }

    assert perm(x.f) == write
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        result.is_err(),
        "perm(x.f) is `b ? 1/1 : 0`, got {result:?}"
    );
}

#[test]
fn unconditionally_held_perm_still_reads_back_full() {
    // Guard-gating must not cost the ordinary case: with both arms keeping the
    // permission the chunk carries no residual guard and reads back as `write`.
    let input = r#"
field f: Int

method client(x: Ref, b: Bool)
    requires acc(x.f)
{
    if (b) { } else { }

    assert acc(x.f, write)
    assert perm(x.f) == write
    exhale acc(x.f, write)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
