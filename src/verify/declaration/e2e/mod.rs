//! End-to-end verification tests: lower a Silver source string through the
//! full pipeline (parse → … → translate) and verify, asserting pass/fail and
//! specific `VerifyError` variants. The e-graph unit tests stay in `super`'s
//! `mod tests`.
//!
//! New topic-focused tests go in the submodules below, which share this file's
//! `lower()`/`verify_named_*` helpers via `use super::*;`.

mod branching;
mod functions;
mod interplay;
mod permissions;
mod predicates;
mod wildcards;

use std::sync::Arc;

use super::*;
use crate::translate;
use crate::viper::{
    GlobalsCollector, IdentCollector, disambiguate, inline_macros, typecheck_program, viper_parser,
    walk::AstWalkable,
};

fn lower(input: &str) -> vmir::Program {
    let mut program = viper_parser::vpr_program(input).expect("parse");
    let mut ic = IdentCollector::default();
    program.walk_mut(&mut ic);
    let interner = ic.finalize();
    let mut gc = GlobalsCollector::new(&interner);
    program.walk(&mut gc);
    let globals = gc.finalize().expect("globals");
    disambiguate(&mut program, &interner, &globals).expect("disambiguation");
    inline_macros(&mut program, &interner).expect("macros");
    let typed = typecheck_program(&mut program, interner, &globals).expect("typecheck");
    // `Option` is a verifier builtin (the mono allocator registers it), not a
    // program declaration — nothing to inject here.
    translate::translate(&typed).expect("translate")
}

#[test]
fn double_consume_predicate_should_fail() {
    let input = r#"
predicate number(this: Ref)

method consume(this: Ref)
    requires number(this)

method caller(this: Ref)
    requires number(this)
{
    consume(this)
    consume(this)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "caller");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn ensures_does_not_double_count_carried_permission() {
    // `read` requires AND ensures `number(this)`. The ensures delta must be
    // produced-only (perm 1), not accumulated onto the requires delta
    // (which would yield perm 2). Likewise `add` carries number(this)/
    // number(other) through and adds number(res); every chunk stays at 1.
    let input = r#"
predicate number(this: Ref)

method assign(this: Ref, value: Int)
    ensures number(this)

method read(this: Ref) returns (val: Int)
    requires number(this)
    ensures number(this)

method add(this: Ref, other: Ref) returns (res: Ref)
    requires number(this) && number(other)
    ensures number(this) && number(other) && number(res)
{
    var a: Int := read(this)
    var b: Int := read(other)
    var sum: Int := a + b
    assign(res, sum)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "add");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn exhale_exceeding_held_permission_fails() {
    // `client` holds only `1/2` of `acc(x.f)` but calls `needs_full`, whose
    // precondition exhales the full `1/1`. The exhale would drive the
    // permission to `-1/2`, so verification must fail rather than allow a
    // negative permission.
    let input = r#"
field f: Int

method needs_full(x: Ref)
    requires acc(x.f, 1/1)

method client(x: Ref)
    requires acc(x.f, 1/2)
{
    needs_full(x)
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
fn branch_preserves_held_permission_through_both_arms() {
    // `m` holds `acc(x.f)` and neither arm of the `if` touches it, so the
    // permission must still be held at the merge to discharge the `ensures`.
    // Exercises CFG linearization: per-block path conditions and the single
    // linear heap threaded across both arms.
    let input = r#"
field f: Int

method m(c: Bool, x: Ref)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1)
{
    if (c) { } else { }
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn branch_establishing_resource_on_one_arm_only_fails() {
    // `give` establishes `number(this)`, but it is only called on the `c` arm.
    // On the `!c` arm the predicate is never produced, so the `ensures` cannot
    // hold unconditionally — the per-arm permission gating must surface this as
    // insufficient permission rather than (unsoundly) verifying.
    let input = r#"
predicate number(this: Ref)

method give(this: Ref)
    ensures number(this)

method m(c: Bool, this: Ref)
    ensures number(this)
{
    if (c) { give(this) } else { }
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn goto_three_way_join_is_not_a_diamond() {
    // A merge reached three ways (`goto M` from each `if`, plus fall-through) is
    // only constructible with `goto` — it is not a clean `c ∨ !c` diamond, so it
    // exercises the materialized-OR reach fallback (`pc = <a ∨ (!a∧b) ∨ (!a∧!b)>`).
    // No arm touches `x.f`, so the permission survives and `ensures` holds; the
    // verifier assumes the (tautological) reach literal to discharge it.
    let input = r#"
field f: Int

method m(a: Bool, b: Bool, x: Ref)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1)
{
    if (a) { goto done }
    if (b) { goto done }
    label done
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn forward_goto_skips_dead_block() {
    // The `exhale` between `goto skip` and `label skip` is unreachable, so the
    // linearizer drops it (reachability filter) and the permission is retained.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1)
{
    goto skip
    exhale acc(x.f, 1/1)
    label skip
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn crossing_non_planar_control_flow() {
    // `X` and `Y` both branch on `b` but to *swapped* targets `P`/`Q` — a
    // crossing (non-planar) CFG, only constructible with `goto`. The four edge
    // conditions `a∧b`, `a∧!b`, `!a∧b`, `!a∧!b` are pairwise exclusive, so each
    // join's two in-edges stay mutually exclusive (phi exhaustive) and the final
    // merge's reach is the tautology of all four. Planarity is irrelevant: the
    // linearizer only uses topological order and per-edge reach conditions.
    let input = r#"
field f: Int

method m(a: Bool, b: Bool, x: Ref)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1)
{
    if (a) { goto x_blk } else { goto y_blk }
    label x_blk
    if (b) { goto p_blk } else { goto q_blk }
    label y_blk
    if (b) { goto q_blk } else { goto p_blk }
    label p_blk
    goto end_blk
    label q_blk
    goto end_blk
    label end_blk
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn value_postcondition_reflexive_and_copied() {
    // `ensures r == a` after `r := a` reduces to `a == a` — discharged by the
    // `eq-refl` rule (also covers a copy chain `t := a; r := t` via congruence).
    for body in ["r := a", "var t: Int := a  r := t"] {
        let input = format!("method m(a: Int) returns (r: Int) ensures r == a {{ {body} }}");
        let program = lower(&input);
        let result = verify_named_method(&program, "m");
        assert!(result.is_ok(), "body `{body}`: expected Ok, got {result:?}");
    }
}

#[test]
fn ensures_resource_uses_precondition_facts() {
    // The `#ensures` resource body divides by `x`, which is only well-formed
    // because the precondition `x != 0` is grafted into the ctx slot and assumed
    // when the resource is verified self-contained. Without the `requires`, the
    // same division must be rejected.
    let with_req = r#"
method m(x: Int) returns (r: Int)
    requires x != 0
    ensures r == 100 / x
{ r := 100 / x }
"#;
    let program = lower(with_req);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "division in ensures should be safe given `requires x != 0`"
    );

    let without_req = r#"
method m(x: Int) returns (r: Int)
    ensures r == 100 / x
{ r := 100 / x }
"#;
    let program = lower(without_req);
    // Without a precondition the `#ensures` resource is self-framed and its
    // division has no nonzero witness — it fails as a resource.
    assert!(
        matches!(
            verify_named_resource(&program, "m#ensures"),
            Err(ref err) if matches!(err.root_cause(), VerifyError::SideCondition(_))
        ),
        "division in ensures must fail without a precondition framing the divisor"
    );
}

#[test]
fn old_in_ensures_reads_pre_state() {
    // `old(x.f)` reads the method pre-state. Untouched field: `x.f == old(x.f)`
    // holds. Mutated field: it must not.
    let unchanged = r#"
field f: Int
method m(x: Ref) returns (r: Int)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1) && x.f == old(x.f)
{ r := x.f }
"#;
    let program = lower(unchanged);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "untouched field equals its old value"
    );

    let mutated = r#"
field f: Int
method m(x: Ref)
    requires acc(x.f, 1/1)
    ensures acc(x.f, 1/1) && x.f == old(x.f)
{ x.f := 7 }
"#;
    let program = lower(mutated);
    assert!(
        verify_named_method(&program, "m").is_err(),
        "mutated field must not equal its old value"
    );
}

#[test]
fn old_over_heap_dependent_function_binds_pre_state() {
    // `old(get(this))` applies a heap-dependent function under `old`: the
    // ensures body reads the pre-state via `Snap` on the a bound `inhale`-widened
    // snapshot parameter, which must congruence-collapse to the caller's real
    // pre-state values at the exhale graft.
    let unchanged = r#"
field v: Int

predicate number(this: Ref) { acc(this.v) }

function get(this: Ref): Int
    requires number(this)
{ unfolding number(this) in this.v }

method keep(this: Ref)
    requires number(this)
    ensures number(this) && old(get(this)) == get(this)
{ }
"#;
    let program = lower(unchanged);
    assert!(
        verify_named_method(&program, "keep").is_ok(),
        "old(get(this)) must equal get(this) for an untouched predicate"
    );

    // Mutating the value under the predicate must break the equality.
    let mutated = r#"
field v: Int

predicate number(this: Ref) { acc(this.v) }

function get(this: Ref): Int
    requires number(this)
{ unfolding number(this) in this.v }

method bump(this: Ref)
    requires number(this)
    ensures number(this) && old(get(this)) == get(this)
{
    unfold number(this)
    this.v := this.v + 1
    fold number(this)
}
"#;
    let program = lower(mutated);
    assert!(
        verify_named_method(&program, "bump").is_err(),
        "old(get(this)) must not equal get(this) after mutating this.v"
    );
}

/// Verify the resource interned under `name`, panicking if it is missing or
/// is not a `Resource`.
fn verify_named_resource(program: &vmir::Program, name: &str) -> Result<(), VerifyError> {
    let id = program
        .id(name)
        .unwrap_or_else(|| panic!("missing resource {name}"));
    let vmir::Declaration::Resource(r) = &program.decls[id] else {
        panic!("{name} must be a Resource");
    };
    let mut alloc = crate::verify::func_registry::FuncRegistry::new(program);
    // Build certificates for the *other* resources (dependency order ≈ decl
    // order for these small fixtures), tolerating failures, so a body that
    // unfolds another predicate can graft its certificate. The target itself is
    // skipped so a deliberately-failing target still returns `Err`.
    let fn_certs = build_fn_certs(program, &mut alloc);
    let mut certs = HashMap::default();
    for (cid, decl) in program.decls.iter_enumerated() {
        if cid == id {
            continue;
        }
        if let vmir::Declaration::Resource(cr) = decl {
            let cname = program.name(cid).to_string();
            if let Ok(cert) = verify_resource(program, &cname, cr, &certs, &fn_certs, &mut alloc) {
                certs.insert(cid, cert);
            }
        }
    }
    verify_resource(program, name, r, &certs, &fn_certs, &mut alloc).map(|_| ())
}

/// Verify every function in `program` except `skip` (test helper), caching
/// certificates. Iterates to a fixpoint so callees are certified before callers
/// (a function whose ensures/callees aren't yet grafted fails and is retried on a
/// later pass) — the helper's stand-in for the driver's topological order.
/// Functions are heap-free and never call resources, so an empty resource-cert
/// map suffices.
fn build_fn_certs_except(
    program: &vmir::Program,
    skip: Option<MemberId>,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> HashMap<MemberId, Arc<FunctionDefinition>> {
    let no_certs = HashMap::default();
    let mut fn_certs = HashMap::default();
    loop {
        let mut progress = false;
        for (id, decl) in program.decls.iter_enumerated() {
            if Some(id) == skip || fn_certs.contains_key(&id) {
                continue;
            }
            if let vmir::Declaration::Function(f) = decl {
                let name = program.name(id).to_string();
                if let Ok(Some(cert)) =
                    verify_function(program, &name, id, f, &no_certs, &fn_certs, None, alloc)
                {
                    fn_certs.insert(id, cert);
                    progress = true;
                }
            }
        }
        if !progress {
            break;
        }
    }
    fn_certs
}

/// Verify every function in `program` (test helper). See [`build_fn_certs_except`].
fn build_fn_certs(
    program: &vmir::Program,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> HashMap<MemberId, Arc<FunctionDefinition>> {
    build_fn_certs_except(program, None, alloc)
}

/// Build certificates for every resource in `program` (test helper). Shares the
/// `alloc` so certificate ids match the method's later use.
fn build_certs(
    program: &vmir::Program,
    fn_certs: &HashMap<MemberId, Arc<FunctionDefinition>>,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> HashMap<MemberId, ResourceDefinition> {
    let mut certs = HashMap::default();
    for (id, decl) in program.decls.iter_enumerated() {
        if let vmir::Declaration::Resource(r) = decl {
            let name = program.name(id).to_string();
            let cert = verify_resource(program, &name, r, &certs, fn_certs, alloc)
                .expect("resource verifies");
            certs.insert(id, cert);
        }
    }
    certs
}

#[test]
fn resource_negative_permission_rejected() {
    // `acc(x.f, 1/1 - 2/1)` folds to permission -1 → side condition fails.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1 - 2/1)
"#;
    let program = lower(input);
    let result = verify_named_resource(&program, "m#requires");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::SideCondition(_))),
        "expected SideCondition, got {result:?}"
    );
}

#[test]
fn resource_positive_permission_ok() {
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
"#;
    let program = lower(input);
    let result = verify_named_resource(&program, "m#requires");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn resource_div_by_zero_rejected() {
    // `x / 0` in the precondition → divisor side condition fails.
    let input = r#"
method m(x: Int)
    requires x / 0 == x
"#;
    let program = lower(input);
    let result = verify_named_resource(&program, "m#requires");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::SideCondition(_))),
        "expected SideCondition, got {result:?}"
    );
}

/// Jointly build resource and function certificates to a fixpoint (test
/// helper) — the stand-in for the driver's topological order when the two
/// kinds depend on each other: a heap-dependent function's body needs its
/// `f#requires` **Resource** cert (a bound `inhale`), while a resource body may call
/// functions. Failures are tolerated (retried until no progress) so a
/// deliberately-failing member simply ends up without a cert.
#[allow(clippy::type_complexity)]
fn build_all_certs(
    program: &vmir::Program,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> (
    HashMap<MemberId, ResourceDefinition>,
    HashMap<MemberId, Arc<FunctionDefinition>>,
) {
    let mut certs = HashMap::default();
    let mut fn_certs = HashMap::default();
    loop {
        let mut progress = false;
        for (id, decl) in program.decls.iter_enumerated() {
            match decl {
                vmir::Declaration::Resource(r) if !certs.contains_key(&id) => {
                    let name = program.name(id).to_string();
                    if let Ok(cert) = verify_resource(program, &name, r, &certs, &fn_certs, alloc) {
                        certs.insert(id, cert);
                        progress = true;
                    }
                }
                vmir::Declaration::Function(f) if !fn_certs.contains_key(&id) => {
                    let name = program.name(id).to_string();
                    if let Ok(Some(cert)) =
                        verify_function(program, &name, id, f, &certs, &fn_certs, None, alloc)
                    {
                        fn_certs.insert(id, cert);
                        progress = true;
                    }
                }
                _ => {}
            }
        }
        if !progress {
            break;
        }
    }
    (certs, fn_certs)
}

/// Verify the method `name`, panicking if missing or not a `Method`.
fn verify_named_method(program: &vmir::Program, name: &str) -> Result<(), VerifyError> {
    let id = program
        .id(name)
        .unwrap_or_else(|| panic!("missing method {name}"));
    let vmir::Declaration::Method(m) = &program.decls[id] else {
        panic!("{name} must be a Method");
    };
    let mut alloc = crate::verify::func_registry::FuncRegistry::new(program);
    let (certs, fn_certs) = build_all_certs(program, &mut alloc);
    verify_method(program, name, m, &certs, &fn_certs, &mut alloc)
}

/// Verify the function `name` with all other members' certs built (test
/// helper for heap-dependent functions, which need their `#requires` Resource
/// cert).
fn verify_named_function(program: &vmir::Program, name: &str) -> Result<(), VerifyError> {
    let id = program
        .id(name)
        .unwrap_or_else(|| panic!("missing function {name}"));
    let vmir::Declaration::Function(f) = &program.decls[id] else {
        panic!("{name} must be a Function");
    };
    let mut alloc = crate::verify::func_registry::FuncRegistry::new(program);
    let (certs, mut fn_certs) = build_all_certs(program, &mut alloc);
    // Re-verify the target itself so a failing target returns its `Err` (the
    // fixpoint helper swallowed it).
    fn_certs.remove(&id);
    verify_function(program, name, id, f, &certs, &fn_certs, None, &mut alloc).map(|_| ())
}

#[test]
fn adt_discriminator_on_known_constructor() {
    // `one()` is a known constructor, so `tag(one()) ⇒ 0`; the `istwo`
    // discriminator desugars to `tag(x) == 1`, which folds to `false`.
    let input = r#"
adt MyAdt { one() two() }
method m()
{
    var x: MyAdt := one()
    assert !x.istwo
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "!one().istwo should verify"
    );
}

#[test]
fn adt_discriminator_wrong_variant_fails() {
    // `one().istwo` is `false`, so asserting it must fail.
    let input = r#"
adt MyAdt { one() two() }
method m()
{
    var x: MyAdt := one()
    assert x.istwo
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)
        ),
        "asserting one().istwo should fail"
    );
}

#[test]
fn adt_destructor_projects_constructor_field() {
    // `mk(3,4).fst` projects to `3` via the projection reduction.
    let input = r#"
adt Pair { mk(fst: Int, snd: Int) }
method m()
{
    var p: Pair := mk(3, 4)
    assert p.fst == 3
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "mk(3,4).fst == 3 should verify"
    );
}

#[test]
fn adt_destructor_wrong_field_value_fails() {
    let input = r#"
adt Pair { mk(fst: Int, snd: Int) }
method m()
{
    var p: Pair := mk(3, 4)
    assert p.fst == 4
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)
        ),
        "mk(3,4).fst == 4 should fail"
    );
}

#[test]
fn generic_adt_two_monomorphizations() {
    // A user-written generic ADT used at two element types. The e-graph is
    // polymorphic: `Box[Int]` and `Box[Bool]` share one `mk` constructor id but
    // carry distinct `Ty` type-arg children, so congruence keeps the two
    // projections disjoint (`bi.v == 5`, `bb.v == true` never merge).
    let input = r#"
adt Box[T] { mk(v: T) }
method m()
{
    var bi: Box[Int] := mk(5)
    var bb: Box[Bool] := mk(true)
    assert bi.v == 5
    assert bb.v == true
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "generic Box at Int and Bool should verify"
    );
}

#[test]
fn generic_adt_wrong_field_value_fails() {
    let input = r#"
adt Box[T] { mk(v: T) }
method m()
{
    var bi: Box[Int] := mk(5)
    assert bi.v == 6
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)
        ),
        "mk(5).v == 6 should fail"
    );
}

#[test]
fn domain_function_call_is_pure_and_congruent() {
    // A domain function is uninterpreted but a *function*: two syntactically
    // identical calls land in one e-class by congruence, so `f(3) == f(3)`
    // verifies. (Exercises the pure `DomainFunctionCall` lowering path — a
    // domain call must reach the pure node, not the heap `FunctionCall`.)
    let input = r#"
domain D { function f(x: Int): Int }
method m()
{
    assert f(3) == f(3)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "f(3) == f(3) should verify by congruence"
    );
}

#[test]
fn domain_function_distinct_args_do_not_merge() {
    // Uninterpreted: `f(3)` and `f(4)` have distinct argument enodes, so they
    // are not provably equal — asserting their equality must fail.
    let input = r#"
domain D { function f(x: Int): Int }
method m()
{
    assert f(3) == f(4)
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)
        ),
        "f(3) == f(4) should fail"
    );
}

#[test]
fn fold_unfold_roundtrip_preserves_field() {
    // `fold` then `unfold` recovers the exact field value via the snapshot.
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref)
  requires acc(x.f, write) && x.f == 5
{
  fold acc(Cell(x), write)
  unfold acc(Cell(x), write)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "fold/unfold round-trip should preserve x.f == 5"
    );
}

#[test]
fn unfolding_expression_reads_field() {
    // `unfolding acc(Cell(x), write) in x.f` reads the field through a scoped
    // unfold without a preceding statement `unfold` — the predicate stays
    // folded afterwards (the unfolded heap is discarded).
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref) returns (v: Int)
  requires acc(x.f, write) && x.f == 5
{
  fold acc(Cell(x), write)
  v := unfolding acc(Cell(x), write) in x.f
  assert v == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "unfolding expression should read x.f == 5 through a scoped unfold"
    );
}

#[test]
fn unfolding_in_predicate_body_unfolds_nested() {
    // A predicate body holds a nested predicate and reads through it via
    // `unfolding`, verified by the shared resource-body unfold path (the cert
    // of the nested `Inner` is grafted).
    let input = r#"
field f: Int
predicate Inner(x: Ref) { acc(x.f, write) }
predicate Outer(x: Ref) {
  acc(Inner(x), write) && (unfolding acc(Inner(x), write) in x.f) == 0
}
"#;
    let program = lower(input);
    assert!(
        verify_named_resource(&program, "Outer").is_ok(),
        "Outer well-formedness should verify via nested unfold"
    );
}

#[test]
fn unfolding_in_predicate_body_without_holding_fails() {
    // `unfolding Inner(x)` without holding `acc(Inner(x))` lacks permission.
    let input = r#"
field f: Int
predicate Inner(x: Ref) { acc(x.f, write) }
predicate Bad(x: Ref) { (unfolding acc(Inner(x), write) in x.f) == 0 }
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_resource(&program, "Bad"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::InsufficientPermission)
        ),
        "unfolding a predicate that is not held should lack permission"
    );
}

#[test]
fn unfolding_recursive_predicate_in_resource_body() {
    // A predicate body unfolds a *recursive* predicate one level by grafting
    // the unfolded predicate's certificate (`List` is verified first).
    let input = r#"
field val: Int
field next: Ref
predicate List(this: Ref) {
  acc(this.val, write) && acc(this.next, write) &&
  (this.next != null ==> List(this.next))
}
predicate Head(this: Ref) {
  acc(List(this), write) && (unfolding acc(List(this), write) in this.val) == 0
}
"#;
    let program = lower(input);
    assert!(
        verify_named_resource(&program, "Head").is_ok(),
        "Head should verify by inlining List one level (cert-free)"
    );
}

#[test]
fn fold_consumes_field_permission() {
    // After `fold`, the field permission has moved into the predicate, so a
    // direct read of `x.f` no longer has permission.
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref)
  requires acc(x.f, write) && x.f == 5
{
  fold acc(Cell(x), write)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::InsufficientPermission)
        ),
        "reading x.f after fold should lack permission"
    );
}

#[test]
fn fold_unfold_two_field_predicate() {
    // A two-field predicate round-trips both fields (values set by
    // assignment to avoid the conjunction-assume gap on `&&` of facts).
    let input = r#"
field f: Int
field g: Int
predicate Pair(x: Ref) { acc(x.f, write) && acc(x.g, write) }
method m(x: Ref)
  requires acc(x.f, write) && acc(x.g, write)
{
  x.f := 1
  x.g := 2
  fold acc(Pair(x), write)
  unfold acc(Pair(x), write)
  assert x.f == 1
  assert x.g == 2
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "two-field fold/unfold round-trip should preserve both fields"
    );
}

#[test]
fn fold_unfold_conditional_true_branch() {
    // A predicate with a conditional acc (`b ==> acc(x.f)`): when the guard
    // is true the field is captured (member `Some(v)`), so the round-trip
    // recovers it. Exercises conditional folding + the optional discriminant
    // `0 < (b ? p : 0)` collapsing to `b`.
    let input = r#"
field f: Int
predicate Maybe(x: Ref, b: Bool) { b ==> acc(x.f, write) }
method m(x: Ref)
  requires acc(x.f, write) && x.f == 5
{
  fold acc(Maybe(x, true), write)
  unfold acc(Maybe(x, true), write)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "conditional fold/unfold (guard true) should preserve x.f == 5"
    );
}

#[test]
fn fold_unfold_conditional_false_keeps_field() {
    // With the guard false the predicate captures nothing (member `None`),
    // so the field permission is retained and `x.f` is still readable.
    let input = r#"
field f: Int
predicate Maybe(x: Ref, b: Bool) { b ==> acc(x.f, write) }
method m(x: Ref)
  requires acc(x.f, write) && x.f == 5
{
  fold acc(Maybe(x, false), write)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "conditional fold (guard false) should retain the field permission"
    );
}

#[test]
fn fold_unfold_mixed_element_types() {
    // A predicate over an Int and a Bool field: each field's snapshot member
    // monomorphises a distinct `Option` instance (`Some@Int` vs `Some@Bool`,
    // distinct member ids), and both round-trip independently.
    let input = r#"
field f: Int
field g: Bool
predicate Both(x: Ref) { acc(x.f, write) && acc(x.g, write) }
method m(x: Ref)
  requires acc(x.f, write) && acc(x.g, write)
{
  x.f := 7
  x.g := true
  fold acc(Both(x), write)
  unfold acc(Both(x), write)
  assert x.f == 7
  assert x.g == true
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "mixed Int/Bool fold/unfold round-trip should preserve both fields"
    );
}

#[test]
fn fold_unfold_aliased_footprint() {
    // Two `acc` on the *same* location: the footprint has two (unmerged)
    // slots, so the snapshot keeps two members, even though the merged
    // accounting view holds a single `x.f` chunk (perm 1/2 + 1/2 = write).
    let input = r#"
field f: Int
predicate dup(x: Ref) { acc(x.f, 1/2) && acc(x.f, 1/2) }
method m(x: Ref)
  requires acc(x.f, write) && x.f == 5
{
  fold dup(x)
  unfold dup(x)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "aliased-footprint fold/unfold should preserve x.f == 5"
    );
}

#[test]
fn fold_unfold_fractional_with_pure_fact() {
    // Bare `fold`/`unfold P(x)` syntax, a predicate carrying a pure fact,
    // unfolding an opaque (requires-held) predicate, and a fractional
    // exhale/unfold round-trip preserving the snapshot value. (cases/folds.vpr)
    let input = r#"
field f: Int
predicate pos(x: Ref) { acc(x.f) && x.f > 0 }
method m(x: Ref)
    requires pos(x)
{
    unfold pos(x)
    assert x.f > 0
    x.f := 10
    fold pos(x)
    exhale acc(pos(x), 1/2)
    unfold acc(pos(x), 1/2)
    assert x.f > 0
    assert x.f == 10
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "folds.vpr scenario should verify"
    );
}

#[test]
fn predicate_body_derefs_aliased_location_under_branch() {
    // `acc(x.f) && x == y && y.f == 10`: the `y.f` read is framed only by the
    // `acc(x.f)` chunk, reachable because the preceding `x == y` conjunct puts
    // the deref on a branch where `x` and `y` alias. The frame lookup must
    // resolve the address *under that path condition* — assuming `x == y` merges
    // `x`/`y` (via `eq-true-union`) and, by congruence, `f(x)`/`f(y)`. Without
    // it the `y.f` deref reports insufficient permission. (cases/pred_merge.vpr)
    let input = r#"
field f: Int
predicate merge(x: Ref, y: Ref) { acc(x.f) && x == y && y.f == 10 }
"#;
    let program = lower(input);
    assert!(
        verify_named_resource(&program, "merge").is_ok(),
        "pred_merge.vpr: y.f should frame against acc(x.f) under x == y"
    );
}

#[test]
fn inline_inhale_then_exhale_roundtrips() {
    // Inhale a field + a fact about it (read against the growing heap), then
    // exhale the fact (read against the pre-exhale heap) and the permission.
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/1) && x.f == 5
    exhale x.f == 5
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "inhale/exhale roundtrip should verify"
    );
}

#[test]
fn inline_exhale_without_permission_fails() {
    // Exhaling `1/1` while only `1/2` was inhaled drives permission negative.
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/2)
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn inline_exhale_unproven_fact_fails() {
    // The exhaled boolean `x.f == 5` is not known (nothing assumed it).
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/1)
    exhale x.f == 5
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn new_single_field_grants_full_permission() {
    let input = r#"
field f: Int

method m()
{
    var x: Ref := new(f)
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "new(f) should grant full permission to x.f"
    );
}

#[test]
fn new_permission_is_exactly_full() {
    // `new(f)` grants exactly `1/1`; exhaling it twice over-consumes.
    let input = r#"
field f: Int

method m()
{
    var x: Ref := new(f)
    exhale acc(x.f, 1/1)
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn perm_in_exhale_sees_removed_permission() {
    // `acc(x.f)` is exhaled first, so `perm(x.f)` then reads `none` (0).
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    exhale acc(x.f, 1/1) && perm(x.f) == none
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "perm() in exhale must see the post-removal heap"
    );
}

#[test]
fn perm_before_acc_in_exhale_is_full() {
    // `perm(x.f)` is read before its `acc` is subtracted, so it is `write` (1).
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    exhale perm(x.f) == write && acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "perm() read before its acc must be full"
    );
}

#[test]
fn perm_in_inhale_sees_added_permission() {
    // Inhale tracks the growing heap: after `acc(x.f)`, `perm(x.f) == write`.
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/1) && perm(x.f) == write
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "perm() in inhale must see the added permission"
    );
}

#[test]
fn exhale_value_read_uses_pre_exhale_heap() {
    // The value `x.f` is read in the same exhale that gives up `acc(x.f)`;
    // value reads resolve against the fixed pre-exhale heap, so it works.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    inhale x.f == 5
    exhale acc(x.f, 1/1) && x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "value read in exhale must use the pre-exhale heap"
    );
}

#[test]
fn assert_held_permission_ok() {
    // `assert acc(x.f)` becomes `perm(x.f) >= write`; held in full → ok, and
    // it is non-destructive, so the permission is still exhalable after.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    assert acc(x.f, 1/1)
    exhale acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "assert acc must hold and not consume the permission"
    );
}

#[test]
fn assert_unheld_permission_fails() {
    // `perm(x.f) = 0 >= write` is false.
    let input = r#"
field f: Int

method m(x: Ref)
{
    assert acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn deref_without_permission_fails() {
    let input = r#"
field f: Int

method m(x: Ref)
{
    assert x.f == 5
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn assert_pure_unproven_fails() {
    let input = r#"
method m(x: Int)
{
    assert x == 5
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn assume_then_assert_pure() {
    let input = r#"
method m(x: Int)
{
    assume x == 5
    assert x == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "assumed fact must be assertable"
    );
}

#[test]
fn assume_then_assert_acc() {
    // `assume acc(x.f)` records the fact `perm(x.f) >= write` (it adds no
    // chunk — unlike `inhale`); asserting the same fact then holds. The
    // permission is genuinely held here so the assumed fact is consistent.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    assume acc(x.f, 1/1)
    assert acc(x.f, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "assumed perm fact must be assertable"
    );
}

#[test]
fn new_multiple_fields() {
    let input = r#"
field f: Int
field g: Int

method m()
{
    var x: Ref := new(f, g)
    exhale acc(x.f, 1/1) && acc(x.g, 1/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "new(f, g) should grant full permission to both fields"
    );
}

#[test]
fn concrete_predicate_body_verifies() {
    // A predicate with a concrete body lowers to a resource and is verified
    // well-formed (reading `this.f` needs the `acc(this.f)` it just granted).
    let input = r#"
field f: Int

predicate number(this: Ref) {
    acc(this.f, 1/1) && this.f == 0
}
"#;
    let program = lower(input);
    assert!(
        verify_named_resource(&program, "number").is_ok(),
        "concrete predicate should verify well-formed"
    );
}

#[test]
fn ensures_equality_is_reusable_at_call_site() {
    // `seteq`'s postcondition establishes `x.f == y.f`. The caller rebuilds the
    // ensures recipe, assumes that boolean, and can then discharge the same
    // equality.
    let input = r#"
field f: Int

method seteq(x: Ref, y: Ref)
    requires acc(x.f, 1/1) && acc(y.f, 1/1)
    ensures acc(x.f, 1/1) && acc(y.f, 1/1) && x.f == y.f

method m(x: Ref, y: Ref)
    requires acc(x.f, 1/1) && acc(y.f, 1/1)
{
    seteq(x, y)
    assert x.f == y.f
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "the grafted ensures equality should be reusable"
    );
}

#[test]
fn literal_division_folds_to_real() {
    // `4/2` is const-folded at translation to the Real literal `2/1`, so it
    // matches an explicit `2/1` on exhale.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 4/2)
{
    exhale acc(x.f, 2/1)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "4/2 should fold to 2/1 and match"
    );
}

#[test]
fn under_pc_verifies() {
    // `assume (b && true) ==> x.f == 10` then `assert (b && b) ==> x.f == 10`:
    // both antecedents collapse to `b`, so the implications are congruent.
    let input = r#"
field f: Int

method under_pc(x: Ref, b: Bool)
{
    inhale acc(x.f, 1/1)
    assume (b && true) ==> x.f == 10
    assert (b && b) ==> x.f == 10
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "under_pc").is_ok(),
        "under_pc should verify with the and-true / and-self rewrites"
    );
}

#[test]
fn unlabeled_old_reads_post_requires_heap() {
    // Unlabeled `old(...)` reads the post-requires-inhale heap. The
    // precondition holds `acc(x.f, 1/2)`; after inhaling another `1/2` the
    // current permission is `1/1`, but `old(perm(x.f))` must still see the
    // `1/2` held right after the precondition was inhaled.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/2)
{
    inhale acc(x.f, 1/2)
    assert perm(x.f) == 1/1
    assert old(perm(x.f)) == 1/2
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "old(perm(x.f)) should see the 1/2 permission held after the precondition"
    );
}

#[test]
fn labeled_old_reads_label_heap_permission() {
    // `label L` captures the heap holding `acc(x.f, 1/2)`. After inhaling
    // another `1/2`, the current permission is `1/1`, but `old[L](perm(x.f))`
    // must still see the `1/2` held at `L` — proving `old[L]` reaches the
    // captured heap, not the current one.
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/2)
{
    label L
    inhale acc(x.f, 1/2)
    assert perm(x.f) == 1/1
    assert old[L](perm(x.f)) == 1/2
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "old[L](perm(x.f)) should see the 1/2 permission held at L"
    );
}

#[test]
fn old_before_its_label_is_a_translation_error() {
    // Straight-line lowering only knows labels it has already passed. An
    // `old[L]` used before `label L` cannot find the captured heap and is a
    // clean translation error (not a panic).
    let input = r#"
field f: Int

method m(x: Ref)
    requires acc(x.f, 1/1)
{
    assert old[L](x.f) == x.f
    label L
}
"#;
    let mut program = viper_parser::vpr_program(input).expect("parse");
    let mut ic = IdentCollector::default();
    program.walk_mut(&mut ic);
    let interner = ic.finalize();
    let mut gc = GlobalsCollector::new(&interner);
    program.walk(&mut gc);
    let globals = gc.finalize().expect("globals");
    disambiguate(&mut program, &interner, &globals).expect("disambiguation");
    inline_macros(&mut program, &interner).expect("macros");
    let typed = typecheck_program(&mut program, interner, &globals).expect("typecheck");
    assert!(
        translate::translate(&typed).is_err(),
        "old[L] before label L must fail translation"
    );
}

// A conditional spatial assertion `b ? A : A'` lowers to one additive heap
// timeline with the branch folded into the permission fractions (no heap
// ternary). The verifier recovers each branch by assuming the condition.
const COND_INHALE: &str = r#"
field f: Int

method m(x: Ref, b: Bool)
{
    inhale b ? (acc(x.f, 1/2) && x.f == 0) : (acc(x.f, 1/1) && x.f == 1)
"#;

#[test]
fn conditional_inhale_true_branch_verifies() {
    let input =
        format!("{COND_INHALE}    assume b\n    assert perm(x.f) == 1/2\n    assert x.f == 0\n}}");
    let program = lower(&input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "under `b`, the held permission is 1/2 and x.f == 0"
    );
}

#[test]
fn conditional_inhale_false_branch_verifies() {
    // `b == false` (not `!b`): the e-graph propagates equality with a literal
    // via `eq-true-union`, whereas a `!b` ternary's negation isn't pushed
    // back onto `b` — a separate backend gap, not the branch lowering.
    let input = format!(
        "{COND_INHALE}    assume b == false\n    assert perm(x.f) == 1/1\n    assert x.f == 1\n}}"
    );
    let program = lower(&input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "under `!b`, the held permission is 1/1 and x.f == 1"
    );
}

#[test]
fn conditional_inhale_does_not_leak_other_branch() {
    // Under the true branch, the false branch's value (`x.f == 1`) must NOT
    // be derivable — the agreement axiom keeps the branch values isolated.
    let input = format!("{COND_INHALE}    assume b\n    assert x.f == 1\n}}");
    let program = lower(&input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)
        ),
        "true branch must not leak the false branch's value"
    );
}

#[test]
fn implies_spatial_verifies() {
    // `b ==> (acc(x.f) && x.f == 7)` gives full permission and the value
    // only under `b`; assuming `b`, both are recoverable.
    let input = r#"
field f: Int

method m(x: Ref, b: Bool)
{
    inhale b ==> (acc(x.f, 1/1) && x.f == 7)
    assume b
    assert perm(x.f) == 1/1
    assert x.f == 7
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "under `b`, the implication grants full permission and x.f == 7"
    );
}

#[test]
fn field_assign_updates_value() {
    // With write permission, `x.f := 10` mutates the heap value so a later
    // `assert x.f == 10` discharges.
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/1)
    x.f := 10
    assert x.f == 10
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "field assignment under write permission should verify"
    );
}

#[test]
fn field_assign_without_write_permission_fails() {
    // Only 1/2 held after the exhale: a field write needs full permission.
    let input = r#"
field f: Int

method m(x: Ref)
{
    inhale acc(x.f, 1/1)
    exhale acc(x.f, 1/2)
    x.f := 20
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::InsufficientPermission)
        ),
        "field write without full permission must fail"
    );
}

#[test]
fn field_assign_with_no_permission_fails() {
    let input = r#"
field f: Int

method m(x: Ref)
{
    x.f := 1
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::InsufficientPermission)
        ),
        "field write with no permission held must fail"
    );
}

// ============================================================================
// Nested & recursive predicates (fold & unfold)
//
// A predicate whose body holds another predicate instance — nested
// (`Outer{ Inner(x) }`) or recursive (`List{ .. List(this.next) }`) — is folded
// by consuming the already-held inner chunk as one opaque footprint slot (its
// value = the inner predicate's snapshot). Folding is NOT recursive: the inner
// instance must already be folded. A nested-predicate slot is therefore
// structurally identical to a field slot, and recursion works for free.
// ============================================================================

/// A recursive linked-list predicate lowers through the whole pipeline without
/// error.
#[test]
fn recursive_predicate_lowers() {
    let input = r#"
field val: Int
field next: Ref
predicate List(this: Ref) {
  acc(this.val, write) && acc(this.next, write) &&
  (this.next != null ==> List(this.next))
}
"#;
    // Must not panic / error during parse → typecheck → translate.
    let _ = lower(input);
}

/// Base case: a recursive `List` whose tail is `null` has its inner-list slot
/// absent (`None`), so fold/unfold round-trips the two fields.
#[test]
fn recursive_predicate_base_case_roundtrip() {
    let input = r#"
field val: Int
field next: Ref
predicate List(this: Ref) {
  acc(this.val, write) && acc(this.next, write) &&
  (this.next != null ==> List(this.next))
}
method m(this: Ref)
  requires acc(this.val, write) && acc(this.next, write) && this.next == null
{
  this.val := 5
  fold acc(List(this), write)
  unfold acc(List(this), write)
  assert this.val == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "base-case List fold/unfold round-trip should preserve this.val"
    );
}

/// Unfolding a held recursive `List(this)` opens it back into its footprint
/// (two fields + the conditional inner list), so the fields become readable.
#[test]
fn recursive_predicate_unfold_exposes_fields() {
    let input = r#"
field val: Int
field next: Ref
predicate List(this: Ref) {
  acc(this.val, write) && acc(this.next, write) &&
  (this.next != null ==> List(this.next))
}
method m(this: Ref)
  requires acc(List(this), write)
{
  unfold acc(List(this), write)
  this.val := 7
  assert this.val == 7
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "unfolding List(this) should expose its fields for read/write"
    );
}

/// Recursive one level: holding the fields plus the inner `List(this.next)`
/// (with `next != null`) lets `List(this)` fold — the inner list chunk moves
/// into the outer predicate.
#[test]
fn recursive_predicate_one_level_fold() {
    let input = r#"
field val: Int
field next: Ref
predicate List(this: Ref) {
  acc(this.val, write) && acc(this.next, write) &&
  (this.next != null ==> List(this.next))
}
method m(this: Ref)
  requires acc(this.val, write) && acc(this.next, write) &&
           this.next != null && acc(List(this.next), write)
{
  fold acc(List(this), write)
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "folding List(this) with a held inner List(this.next) should succeed"
    );
}

/// Nested non-recursive: `Outer{ Inner(x) }` round-trips through the held
/// `Inner(x)` chunk and recovers the inner field after both unfolds.
#[test]
fn nested_predicate_fold_unfold_roundtrip() {
    let input = r#"
field f: Int
predicate Inner(x: Ref) { acc(x.f, write) }
predicate Outer(x: Ref) { Inner(x) }
method m(x: Ref)
  requires acc(x.f, write)
{
  x.f := 5
  fold acc(Inner(x), write)
  fold acc(Outer(x), write)
  unfold acc(Outer(x), write)
  unfold acc(Inner(x), write)
  assert x.f == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "Outer Inner(x) fold/unfold round-trip should recover x.f == 5"
    );
}

/// Folding `Outer(x)` consumes the inner `Inner(x)` chunk: unfolding `Inner(x)`
/// directly afterwards must fail for lack of permission (it moved into `Outer`).
#[test]
fn nested_predicate_fold_consumes_inner() {
    let input = r#"
field f: Int
predicate Inner(x: Ref) { acc(x.f, write) }
predicate Outer(x: Ref) { Inner(x) }
method m(x: Ref)
  requires acc(Inner(x), write)
{
  fold acc(Outer(x), write)
  unfold acc(Inner(x), write)
}
"#;
    let program = lower(input);
    assert!(
        matches!(
            verify_named_method(&program, "m"),
            Err(ref e) if matches!(e.root_cause(), VerifyError::InsufficientPermission)
        ),
        "unfolding Inner(x) after it was folded into Outer(x) must lack permission"
    );
}

#[test]
fn mutually_recursive_snapshots_verify() {
    // `A` and `B` reference each other only through a predicate *location*
    // (`acc(B(..))` / `acc(A(..))`), not fold/unfold — so neither is a
    // verification dependency of the other. Their snapshots are mutually
    // recursive (nominal, by id), which is allowed: both verify.
    let input = r#"
field f: Int
field nxt: Ref
predicate A(x: Ref) { acc(x.f, write) && acc(x.nxt, write) && (x.nxt != null ==> B(x.nxt)) }
predicate B(x: Ref) { acc(x.f, write) && acc(x.nxt, write) && (x.nxt != null ==> A(x.nxt)) }
"#;
    let program = lower(input);
    assert!(verify_named_resource(&program, "A").is_ok());
    assert!(verify_named_resource(&program, "B").is_ok());
}

#[test]
fn cyclic_unfolding_predicates_rejected() {
    // `P` unfolds `Q` and `Q` unfolds `P`: each appears in the other's body in
    // an *unfolding* context, so each is a verification dependency of the other
    // → a cycle, rejected by `analyze` (unlike the mutual-snapshot case above).
    let input = r#"
predicate P(x: Ref) { acc(Q(x), write) && (unfolding acc(Q(x), write) in true) }
predicate Q(x: Ref) { acc(P(x), write) && (unfolding acc(P(x), write) in true) }
"#;
    let program = lower(input);
    assert!(
        matches!(
            crate::vmir::analyze(program),
            Err(crate::vmir::AnalysisError::CircularDependency(_))
        ),
        "mutually-unfolding predicates should be a circular dependency"
    );
}

#[test]
fn function_body_inlined_when_spec_insufficient() {
    // The spec (`result >= 0`) alone can't prove `five() == 5`; only the grafted
    // body definition (`five() == 5`) discharges the assert.
    let input = r#"
function five(): Int
    ensures result >= 0
{ 5 }

method m()
{
    assert five() == 5
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "grafted function body should discharge `five() == 5`"
    );
}

#[test]
fn chained_function_bodies_inlined() {
    // `six` calls `five`; proving `six() == 6` needs both bodies inlined.
    let input = r#"
function five(): Int { 5 }
function six(): Int { five() + 1 }

method m()
{
    assert six() == 6
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "chained function bodies should inline to prove `six() == 6`"
    );
}

#[test]
fn function_postcondition_discharged_by_body() {
    // The exit `assert inc#ensures(x, result)` is discharged via the grafted
    // `inc#ensures` definition against the body result `x + 1`.
    let input = r#"
function inc(x: Int): Int
    ensures result == x + 1
{ x + 1 }
"#;
    let program = lower(input);
    assert!(
        verify_named_function(&program, "inc").is_ok(),
        "function whose ensures follows from its body should verify"
    );
}

#[test]
fn recursive_function_accepted_as_scc() {
    // A self-recursive function is a self-looping singleton SCC → `analyze` now
    // accepts it (limited-function encoding) rather than rejecting the cycle, and
    // marks it recursive so its calls route through the limited twin.
    let input = r#"
function loop(x: Int): Int { loop(x) }
"#;
    let program = lower(input);
    let loop_id = program.id("loop").expect("loop function");
    let analyzed = crate::vmir::analyze(program).expect("function recursion is accepted");
    assert_eq!(
        analyzed.recursive_scc(loop_id),
        Some(crate::dhash::HashSet::from_iter([loop_id])),
        "self-recursive function should be its own recursive SCC"
    );
}

#[test]
fn heap_dep_function_verifies_and_defines_at_call_site() {
    // The full snapshot-passing pipeline: the call site's `Snap` checks the
    // precondition (footprint + bool) against the caller heap, the callee's
    // cert (verified against `H(s)`) grafts as `get(y, s) == unwrap(proj_0(s))`,
    // and `s = cons(Some(y.f))` collapses it to the caller's chunk value — so
    // `a == y.f` proves without any postcondition.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f) && x.f > 0
    ensures result == x.f
{ x.f }

method m(y: Ref)
    requires acc(y.f) && y.f > 0
{
    var a: Int := get(y)
    assert a > 0
    assert a == y.f
}
"#;
    let program = lower(input);
    assert!(
        verify_named_function(&program, "get").is_ok(),
        "heap-dep function must verify (body framed by H(s), ensures from body)"
    );
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn heap_dep_call_without_permission_fails() {
    // `m` holds no permission to `y.f`: the call-site `Snap`'s footprint
    // sufficiency check fails.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f)
{ x.f }

method m(y: Ref)
{
    var a: Int := get(y)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn heap_dep_call_precondition_bool_fails() {
    // `m` holds the footprint but cannot prove the precondition's pure fact
    // (`y.f > 0`): the `Snap`'s implicit bool assert fails.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f) && x.f > 0
{ x.f }

method m(y: Ref)
    requires acc(y.f)
{
    var a: Int := get(y)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn heap_dep_body_read_outside_footprint_fails() {
    // The body reads `x.g` but the precondition only grants `x.f`: the deref
    // is not framed by the reconstructed `H(s)`.
    let input = r#"
field f: Int
field g: Int

function get(x: Ref): Int
    requires acc(x.f)
{ x.g }
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "get");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

#[test]
fn heap_dep_calls_frame_across_unrelated_write() {
    // Two calls on either side of a write to an *unrelated* field: the `f`
    // chunk value is unchanged, so both `Snap`s build the same `cons` and the
    // two applications are congruent — snapshot-passing gives heap framing for
    // free.
    let input = r#"
field f: Int
field g: Int

function get(x: Ref): Int
    requires acc(x.f)
{ x.f }

method m(y: Ref)
    requires acc(y.f) && acc(y.g)
{
    var a: Int := get(y)
    y.g := 5
    var b: Int := get(y)
    assert a == b
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn heap_dep_heap_reading_ensures_consumed_at_call_site() {
    // An **abstract** heap-dependent function: its heap-reading postcondition is
    // the call site's only source of information about the result. Delivered by
    // the synthesized post axiom (`contract_post_definition`), guarded by the
    // pre-token `get#requires#pre(y, s)` — which the call site's `Snap` released
    // when it proved the precondition. So the call site learns `a == y.f`.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f)
    ensures result == x.f

method m(y: Ref)
    requires acc(y.f)
{
    var a: Int := get(y)
    assert a == y.f
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// ---- Domain axioms ---------------------------------------------------------

#[test]
fn ground_axiom_discharges_assert() {
    // `size()` is uninterpreted; only the axiom pins its value.
    let input = r#"
domain D {
    function size(): Int
    axiom sz { size() == 0 }
}
method client() {
    assert size() == 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn ground_axiom_over_abstract_silver_function() {
    // `f` is an abstract (bodyless, contractless) Silver function — the axiom
    // is the only source of `f() == 42`.
    let input = r#"
function f(): Int
domain D {
    axiom a { f() == 42 }
}
method client() {
    assert f() == 42
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn unbacked_assert_still_fails_with_axioms_present() {
    // Negative control: the axiom pins `f`, not `g` — asserting about `g`
    // must still fail.
    let input = r#"
function f(): Int
function g(): Int
domain D {
    axiom a { f() == 42 }
}
method client() {
    assert g() == 42
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn axiom_body_is_never_verified() {
    // The axiom divides by zero; axioms are trusted (no well-definedness
    // obligations), so an unrelated method still verifies.
    let input = r#"
domain D {
    function w(): Int
    axiom bad { w() == 1 / 0 }
}
method client() {
    assert true
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn ground_axiom_available_in_function_bodies() {
    // Axioms are assumed in every unit, not just methods: the function body's
    // exit `assert f#ensures` needs the axiom.
    let input = r#"
domain D {
    function size(): Int
    axiom sz { size() == 0 }
}
function probe(): Int
    ensures result == 0
{
    size()
}
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "probe");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// ---- Pure `forall` quantifiers (v1: domain axioms, no Tier-4) --------------

#[test]
fn quantifier_basic() {
    // A bare `forall` axiom: the occurrence is unioned `true` directly, so the
    // trigger application `foo(7)` releases `foo(7) == true`.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    axiom basic { forall i: Int :: {foo(i)} foo(i) }
}
method m() {
    assert foo(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn quantifier_multivar() {
    // Two binders, positional σ read off the two-argument trigger.
    let input = r#"
domain D {
    function bar(i: Int, j: Int): Bool
    axiom mv { forall i: Int, j: Int :: {bar(i, j)} bar(i, j) }
}
method m() {
    assert bar(3, 4)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn quantifier_guarded_concrete() {
    // Guarded `forall`: with `b()` concretely true, const-fold collapses the
    // guard `Ite(b(), Q, true)` to `Q = true`, releasing the instance. No
    // Tier-4 case-split needed.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function b(): Bool
    axiom g { b() ? (forall i: Int :: {foo(i)} foo(i)) : true }
}
method m() {
    inhale b()
    assert foo(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn quantifier_guard_stuck_fails() {
    // Guarded `forall` with `b()` unknown: the guard never collapses, so the
    // instance stays gated and `foo(0)` is unprovable. (Soundness: the guard
    // must not leak.)
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function b(): Bool
    axiom g { b() ? (forall i: Int :: {foo(i)} foo(i)) : true }
}
method m() {
    assert foo(0)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn quantifier_wrong_instance_fails() {
    // The body is `i > 0 ==> foo(i)`. At the instance i := 0 it is
    // `Ite(false, foo(0), true) = true` — vacuously true, yielding nothing
    // about `foo(0)`. (Soundness: σ must be exact.)
    let input = r#"
domain D {
    function foo(i: Int): Bool
    axiom w { forall i: Int :: {foo(i)} i > 0 ==> foo(i) }
}
method m() {
    assert foo(0)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn quantifier_guarded_implication_telescopes() {
    // Proving `b() ==> foo(7)` means collapsing the guard `Ite(b(), Q, true)`
    // and then `Ite(Q, foo(7), true)`. Neither collapses on its own (nothing
    // concrete to fold), but both are `true`-constant-arm shapes, so the
    // `ite_decompose` tier telescopes them one assumed condition at a time — no
    // fork, and no case split needed.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function b(): Bool
    axiom g { b() ? (forall i: Int :: {foo(i)} foo(i)) : true }
}
method m() {
    assert b() ==> foo(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "expected Ok via the ite_decompose tier, got {result:?}"
    );
}

#[test]
fn nested_quantifier_cascade() {
    // Nested `forall`: mentioning `f(1)` instantiates the outer quantifier at
    // i := 1, which materializes the inner occurrence `Q_inner(1)` (and merges
    // it `true` via the collapsed outer guard); the ground `g(1, 2)` then
    // matches the inner trigger `{g(i, j)}` — capture position 0 equals the
    // occurrence's capture 1 — releasing `g(1, 2) == true`.
    let input = r#"
domain D {
    function f(i: Int): Bool
    function g(i: Int, j: Int): Bool
    axiom nest { forall i: Int :: {f(i)} (forall j: Int :: {g(i, j)} g(i, j)) }
}
method m() {
    inhale f(1)
    assert g(1, 2)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn nested_quantifier_outer_untriggered_fails() {
    // Without any `f(..)` application the outer quantifier never instantiates,
    // so the inner occurrence is never materialized — `g(1, 2)` stays unknown
    // even though its own trigger is ground. (Soundness: no instantiation
    // without an occurrence.)
    let input = r#"
domain D {
    function f(i: Int): Bool
    function g(i: Int, j: Int): Bool
    axiom nest { forall i: Int :: {f(i)} (forall j: Int :: {g(i, j)} g(i, j)) }
}
method m() {
    assert g(1, 2)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn nested_quantifier_capture_mismatch_fails() {
    // The only materialized inner occurrence is `Q_inner(1)` (from `f(1)`), but
    // `g(2, 3)`'s capture position carries 2 ≠ 1 — the pair must be skipped, so
    // nothing is learned about `g(2, 3)`. (Soundness: capture positions must
    // e-match the occurrence's capture args.)
    let input = r#"
domain D {
    function f(i: Int): Bool
    function g(i: Int, j: Int): Bool
    axiom nest { forall i: Int :: {f(i)} (forall j: Int :: {g(i, j)} g(i, j)) }
}
method m() {
    inhale f(1)
    assert g(2, 3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

// ---- Pure `forall` quantifiers (v3: method bodies) --------------------------

#[test]
fn method_inhale_forall_instantiates() {
    // A `forall` inhaled in a method body: the occurrence is assumed true, so
    // the ground `foo(7)` triggers an instance and the assert discharges.
    let input = r#"
domain D { function foo(i: Int): Bool }
method client() {
    inhale forall i: Int :: {foo(i)} foo(i)
    assert foo(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn function_unfold_exposes_trigger_for_quantifier() {
    // `wrap`'s body is a bare `foo(x)` call — its certificate is captured raw
    // (unsaturated) and consumed lazily by its own `function_rule`, so
    // unfolding `wrap(7)` must expose a *literal* `foo(7)` occurrence for the
    // quantifier rule (also chained into the same saturation) to key off of,
    // in the same pass, not something a pre-emptive simplification erased.
    let input = r#"
domain D { function foo(i: Int): Bool }
function wrap(x: Int): Bool
{
    foo(x)
}
method client() {
    inhale forall i: Int :: {foo(i)} foo(i)
    assert wrap(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn method_inhale_forall_captures_local() {
    // The quantifier captures a method local; instantiation must match the
    // occurrence's capture argument.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
method client(x: Int) {
    var l: Int := x
    inhale forall i: Int :: {g(l, i)} g(l, i)
    assert g(x, 3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn method_inhale_forall_capture_mismatch_fails() {
    // Only `Q(x)` is inhaled — `g(y, 3)` has capture y ≠ x, so nothing is
    // learned about it. (Soundness: captures gate instantiation.)
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
method client(x: Int, y: Int) {
    inhale forall i: Int :: {g(x, i)} g(x, i)
    assert g(y, 3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn method_inhale_forall_under_conjunction() {
    // The occurrence sits under a spatial `&&`: inhale-truth must decompose
    // down to the occurrence (and the pure left conjunct).
    let input = r#"
domain D { function foo(i: Int): Bool }
method client(x: Int) {
    inhale x > 0 && (forall i: Int :: {foo(i)} foo(i))
    assert foo(2) && x > 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

/// Translate `input`, expecting it to be *rejected* by translation.
fn lower_err(input: &str) -> Vec<crate::translate::TranslationError> {
    let mut program = viper_parser::vpr_program(input).expect("parse");
    let mut ic = IdentCollector::default();
    program.walk_mut(&mut ic);
    let interner = ic.finalize();
    let mut gc = GlobalsCollector::new(&interner);
    program.walk(&mut gc);
    let globals = gc.finalize().expect("globals");
    disambiguate(&mut program, &interner, &globals).expect("disambiguation");
    inline_macros(&mut program, &interner).expect("macros");
    let typed = typecheck_program(&mut program, interner, &globals).expect("typecheck");
    translate::translate(&typed).expect_err("expected translation to reject this program")
}

#[test]
fn forall_body_division_guarded_by_the_binder_is_well_defined() {
    // WD of a quantifier body, checked once at the point it is stated, against
    // *fresh* binders in a scratch clone of the live graph. The body's own guard
    // travels with it (short-circuit lowering emits the path condition), so the
    // division is discharged under `<i != 0>`.
    let input = r#"
domain D { function foo(i: Int): Bool }
method m() {
    inhale forall i: Int :: {foo(i)} i != 0 ==> foo(10 / i)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn forall_body_unguarded_division_is_not_well_defined() {
    // The same body without the guard: for an arbitrary binding the divisor may be
    // zero, so the quantifier is ill-defined and the unit is rejected. (Under the
    // old encoding quantifier bodies were trusted — this obligation did not exist.)
    let input = r#"
domain D { function foo(i: Int): Bool }
method m() {
    inhale forall i: Int :: {foo(i)} foo(10 / i)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition(_))),
        "expected a division side condition, got {result:?}"
    );
}

#[test]
fn forall_under_implication_wd_uses_the_host_path_condition() {
    // A quantifier's WD is checked against fresh binders at the point it is
    // stated, so it must be discharged under the path condition *reaching* that
    // point — not just the body's own guards. Here the divisor is framed only by
    // `b`, and the `forall` sits under `b ==>`, so the division is well-defined.
    //
    // The `forall` step used to be emitted with an empty `pc` (its emitter takes a
    // pre-allocated temp and hardcoded `PathConds::default()`), which discharged
    // this side condition unconditionally and failed spuriously.
    let input = r#"
domain D { function g(x: Int): Int }
method m(n: Int, b: Bool)
    requires b ==> n != 0
{
    inhale b ==> (forall i: Int :: {g(i)} g(i) == 10 / n)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn forall_in_a_branch_wd_uses_the_block_cube() {
    // The same, reaching the quantifier as a block *cube* literal rather than a
    // `Branch` one — the fork model lowers a block's reaching condition under
    // `PcKind::Cube`. Both kinds are in `Sink::guard`, so both must arrive.
    let input = r#"
domain D { function g(x: Int): Int }
method m(n: Int, b: Bool)
    requires b ==> n != 0
{
    if (b) {
        inhale forall i: Int :: {g(i)} g(i) == 10 / n
    }
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn forall_under_implication_wd_still_needs_a_framing_divisor() {
    // Non-vacuity for the two tests above: carrying the path condition must not
    // make the obligation vanish. Nothing here establishes `n != 0` on any path,
    // so the quantifier stays ill-defined.
    let input = r#"
domain D { function g(x: Int): Int }
method m(n: Int, b: Bool)
{
    inhale b ==> (forall i: Int :: {g(i)} g(i) == 10 / n)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition(_))),
        "expected a division side condition, got {result:?}"
    );
}

#[test]
fn forall_body_callee_precondition_must_hold_for_every_binding() {
    // The other half of WD: a call inside the body stitches `assert f#requires(..)`,
    // which must hold for an arbitrary binding. The binder guard establishes it.
    let input = r#"
domain D { function foo(i: Int): Bool }
function pos(i: Int): Bool
    requires i > 0
{ true }

method ok() {
    inhale forall i: Int :: {foo(i)} i > 0 ==> pos(i)
}
"#;
    let program = lower(input);
    assert!(verify_named_method(&program, "ok").is_ok());
}

#[test]
fn forall_body_unguarded_callee_precondition_fails() {
    // Without the guard the callee's precondition is unprovable for a fresh binder.
    let input = r#"
domain D { function foo(i: Int): Bool }
function pos(i: Int): Bool
    requires i > 0
{ true }

method bad() {
    inhale forall i: Int :: {foo(i)} pos(i)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "bad");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed (callee precondition), got {result:?}"
    );
}

#[test]
fn nested_forall_body_wd_is_checked_under_the_outer_binder() {
    // A nested quantifier is WD-checked recursively, with its encloser's binders
    // already fresh: dividing by the *outer* binder is ill-defined unless the outer
    // body guards it.
    let input = r#"
domain D {
    function f(i: Int): Bool
    function g(i: Int, j: Int): Bool
}
method m() {
    inhale forall i: Int :: {f(i)} (forall j: Int :: {g(i, j)} g(i, 10 / i))
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition(_))),
        "expected a division side condition from the inner body, got {result:?}"
    );
}

#[test]
fn forall_in_a_function_body_instantiates_at_a_call_site() {
    // A quantifier inside a *function* body reaches a caller through the function's
    // certificate: the purified recipe carries a `Forall` step, so unfolding `q(3)`
    // at the call site rebuilds the quantifier e-node with the caller's argument as
    // its capture — and the single generic rule instantiates it in that unit, mid-run.
    // Impossible while quantifiers were rewrite rules: a rule cannot be injected into
    // a running egg `Runner`.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }

function q(x: Int): Bool
{ forall i: Int :: {g(x, i)} g(x, i) }

method m() {
    inhale q(3)
    assert g(3, 7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn heap_dependent_call_in_a_forall_body_is_rejected() {
    // A quantifier body must stay pure and heap-free: its compiled recipe is
    // rebuilt inside a rewrite rule, which can neither read the symbolic heap nor
    // discharge the `Snap`'s implicit precondition check. (A binder-dependent
    // footprint would need quantified permissions, which are unsupported.)
    let input = r#"
field f: Int
domain D { function t(i: Int): Bool }

function get(x: Ref): Int
    requires acc(x.f)
{ x.f }

method m(y: Ref)
    requires acc(y.f)
{
    inhale forall i: Int :: {t(i)} get(y) == get(y)
}
"#;
    let errs = lower_err(input);
    assert!(
        errs.iter().any(|e| matches!(
            e,
            crate::translate::TranslationError::HeapDepFunctionInQuantifier(_)
        )),
        "expected HeapDepFunctionInQuantifier, got {errs:?}"
    );
}

#[test]
fn a_forall_is_compiled_only_when_the_walk_reaches_it() {
    // Recipes are interned on the eval walk, not up front. A quantifier this unit
    // never reaches costs it nothing — previously every `forall` in the program was
    // compiled before verification started, and the instantiation rule's searcher
    // enumerated all of them on every search.
    let input = r#"
domain D { function foo(i: Int): Bool }
method plain() {
    assert true
}
method quantified() {
    inhale forall i: Int :: {foo(i)} foo(i)
    assert foo(3)
}
"#;
    let program = lower(input);
    let method = |name: &str| {
        let id = program.id(name).expect("missing method");
        let vmir::Declaration::Method(m) = &program.decls[id] else {
            panic!("{name} must be a Method");
        };
        m
    };
    let mut alloc = crate::verify::func_registry::FuncRegistry::new(&program);
    let recipes = |alloc: &crate::verify::func_registry::FuncRegistry| {
        alloc.quant_table().read().expect("recipe table lock").len()
    };
    assert_eq!(
        recipes(&alloc),
        0,
        "nothing compiled by registry construction"
    );

    let (certs, fn_certs) = build_all_certs(&program, &mut alloc);
    verify_method(
        &program,
        "plain",
        method("plain"),
        &certs,
        &fn_certs,
        &mut alloc,
    )
    .expect("plain verifies");
    assert_eq!(
        recipes(&alloc),
        0,
        "another method's quantifier is not this unit's cost"
    );

    verify_method(
        &program,
        "quantified",
        method("quantified"),
        &certs,
        &fn_certs,
        &mut alloc,
    )
    .expect("quantified verifies");
    assert_eq!(recipes(&alloc), 1, "reached, hence compiled");
}

#[test]
fn statements_after_a_forall_shadow_its_body_temps() {
    // The quantifier's frame is not reserved: the body is numbered from the
    // `forall` step's own temp and the enclosing stream resumes one past it, so a
    // later statement reuses the very temps the body used: `x + 1` here lands on
    // the temp the body's first step had. The two scopes never overlap in time, so
    // both walks resolve their own `Temp(k)` — the body against the frame, the
    // method against its own table — and the quantifier still instantiates at the
    // shadowed term.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function bar(i: Int): Bool
}
method m(x: Int) {
    inhale forall i: Int :: {foo(i)} foo(i)
    assert foo(3)
    inhale bar(x + 1)
    assert bar(x + 1)
    assert foo(x + 1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn identical_bodies_with_different_triggers_do_not_pool_them() {
    // Two quantifiers with the same body but different patterns are separate
    // recipes: `{foo(i)}` must not start firing on `bar(i)` because someone else
    // wrote that trigger over the same proposition. Only `bar` is ever applied
    // here, so the `{foo(i)}` occurrence stays uninstantiated and the goal fails.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function bar(i: Int): Bool
    function p(i: Int): Bool
}
method m() {
    inhale forall i: Int :: {foo(i)} p(i)
    inhale bar(3)
    assert p(3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed (no trigger match), got {result:?}"
    );

    // The same proposition stated with the trigger that *is* matched does fire —
    // so the failure above is the trigger, not the encoding.
    let input = r#"
domain D {
    function foo(i: Int): Bool
    function bar(i: Int): Bool
    function p(i: Int): Bool
}
method m() {
    inhale forall i: Int :: {bar(i)} p(i)
    inhale bar(3)
    assert p(3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn method_assert_forall_fails_gracefully() {
    // Proving a `forall` goal is out of scope: the occurrence never merges
    // `true`, so the assert fails cleanly (no crash, no unsound success).
    let input = r#"
domain D { function foo(i: Int): Bool }
method client() {
    inhale foo(1)
    assert forall i: Int :: {foo(i)} foo(i)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn method_inhale_forall_two_instantiations() {
    // One inhaled quantifier feeds two distinct ground instances in the same
    // unit.
    let input = r#"
domain D { function foo(i: Int): Bool }
method client() {
    inhale forall i: Int :: {foo(i)} foo(i)
    assert foo(1) && foo(2)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// ---- Pure `forall` quantifiers (v3: contracts + predicate bodies) ----------

#[test]
fn method_requires_forall_usable_in_body() {
    // The entry inhale of `m#requires` assumes the resource bool — the
    // occurrence — so the body can instantiate it.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
method m(x: Int)
    requires forall i: Int :: {g(x, i)} g(x, i)
{
    assert g(x, 42)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn callee_ensures_forall_inhaled_at_call_site() {
    // The call inhales `producer#ensures`, assuming its occurrence; the caller
    // then instantiates it.
    let input = r#"
domain D { function foo(i: Int): Bool }
method producer()
    ensures forall i: Int :: {foo(i)} foo(i)
method client() {
    producer()
    assert foo(5)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn method_ensures_forall_discharged_by_an_identical_requires() {
    // A `forall` is an e-node whose identity is its compiled body + captures, so
    // the same syntactic quantifier in `requires` and `ensures` hash-conses to ONE
    // e-class: assuming it at entry discharges the exit exhale. (Under the old
    // twin-declaration encoding these were two unrelated occurrence functions and
    // this could not be proven.) Proving a forall *outright* is still out of
    // scope — see `method_assert_forall_fails_gracefully`.
    let input = r#"
domain D { function foo(i: Int): Bool }
method m()
    requires forall i: Int :: {foo(i)} foo(i)
    ensures forall i: Int :: {foo(i)} foo(i)
{
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn predicate_body_forall_released_by_unfold() {
    // Unfolding the predicate assumes its body bool — the conjunction of the
    // guard and the occurrence — releasing both.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
predicate P(i: Int) { i != 0 && (forall x: Int :: {g(i, x)} g(i, x)) }
method m(i: Int)
    requires P(i)
{
    unfold P(i)
    assert g(i, 3) && i != 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn function_requires_forall_usable_in_body() {
    // Heap-free function: the body assumes `f#requires(params)`; the contract
    // function's grafted definition equates that with the occurrence, so the
    // body can instantiate the quantifier to discharge the ensures.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
function f(x: Int): Bool
    requires forall i: Int :: {g(x, i)} g(x, i)
    ensures result
{
    g(x, 1)
}
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "f");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn function_requires_forall_discharged_from_an_identical_inhale() {
    // The caller's inhaled `forall` and the callee's `#requires` quantifier are
    // the same e-node (same body, same capture `x`), so the call-site
    // `assert f#requires(x)` discharges. This is the twin-decl limitation
    // dissolving: quantifier identity is now structural, not per-declaration.
    let input = r#"
domain D { function g(a: Int, i: Int): Bool }
function f(x: Int): Bool
    requires forall i: Int :: {g(x, i)} g(x, i)
{
    g(x, 1)
}
method m(x: Int) {
    inhale forall i: Int :: {g(x, i)} g(x, i)
    inhale f(x)
    assert g(x, 1)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn function_ensures_forall_assumed_at_call_site() {
    // `f` is abstract (no body): its postcondition arrives via the synthesized
    // post axiom (`f#ensures(f())` — no requires, so unguarded), whose unfold
    // exposes the quantified fact; the trigger `foo(9)` then instantiates it.
    let input = r#"
domain D { function foo(i: Int): Bool }
function f(): Int
    ensures forall i: Int :: {foo(i)} foo(i)
method m() {
    var r: Int := f()
    assert foo(9)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "abstract function's post axiom should deliver the forall; got {result:?}"
    );
}

#[test]
fn division_by_zero_in_function_body_fails() {
    // Instruction side conditions (`inst_obligations`) must be discharged in a
    // *function* body, not just a resource body: a literal zero divisor is a
    // verification failure, not a silently-accepted term.
    let input = r#"
function fdiv(a: Int): Int
{ a / 0 }
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "fdiv");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition("divisor may be zero"))),
        "expected SideCondition(divisor), got {result:?}"
    );
}

#[test]
fn division_by_zero_in_method_body_fails() {
    // Same obligation in a method body.
    let input = r#"
method mdiv(a: Int) {
    var x: Int := a / 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "mdiv");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition("divisor may be zero"))),
        "expected SideCondition(divisor), got {result:?}"
    );
}

#[test]
fn division_by_provably_nonzero_divisor_verifies() {
    // The divisor obligation is discharged from the precondition, and the `1/0`
    // on the dead ternary arm is discharged by its (false) path condition —
    // together these pin that the new check is not vacuously failing.
    let input = r#"
method mok(a: Int)
    requires a != 0
{
    var x: Int := 10 / a
    var y: Int := (true ? 1 : 1 / 0)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "mok");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn modulo_by_zero_in_method_body_fails() {
    let input = r#"
method mmod(a: Int) {
    var x: Int := a % 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "mmod");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::SideCondition("divisor may be zero"))),
        "expected SideCondition(divisor), got {result:?}"
    );
}

#[test]
fn modulo_by_provably_nonzero_divisor_verifies() {
    let input = r#"
method mok(a: Int)
    requires a != 0
{
    var x: Int := 10 % a
    var y: Int := (true ? 1 : 1 % 0)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "mok");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn conditional_inhale_permission_is_nonnegative() {
    // CFG linearization encodes the guarded `inhale` as a scaled permission
    // `c ? 1/1 : 0/1`, so the permission ≥ 0 obligation — now also checked in
    // method bodies — is a `<` over an `ite`. `lt-ite` distributes it.
    let input = r#"
predicate number(this: Ref)

method give(this: Ref)
    ensures number(this)

method m(c: Bool, this: Ref)
{
    if (c) { give(this) } else { }
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// --- Phase 1: assumes are guarded by the path condition (Finding A) ----------

#[test]
fn conditional_inhale_bool_does_not_leak_past_its_branch() {
    // `give`'s `ensures x > 0` is inhaled only on the `c` arm, so `x > 0` must
    // NOT hold at the unconditional `assert` after the `if`. Before guarding the
    // inhaled bool (by `0 < perm`), the bool was unioned with `true`
    // unconditionally and this verified unsoundly.
    let input = r#"
method give(x: Int) ensures x > 0
method m(c: Bool, x: Int) {
    if (c) { give(x) }
    assert x > 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed (inhaled bool must not leak past its branch), got {result:?}"
    );
}

#[test]
fn conditional_assume_does_not_leak_past_its_branch() {
    // The same, one level down: a bare `assume` inside a branch holds only on
    // that branch. `InstKind::Assume` used to ignore `inst.pc`.
    let input = r#"
method n(c: Bool, x: Int) {
    if (c) { assume x > 0 }
    assert x > 0
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "n");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed (assume must not leak past its branch), got {result:?}"
    );
}

#[test]
fn unconditional_inhale_bool_is_still_assumed() {
    // Guarding must not break the common case: a straight-line inhale (perm
    // `1/1`, guard `0 < 1` folds to `true`) still assumes its bool.
    let input = r#"
method give(x: Int) ensures x > 0
method p1(x: Int) { give(x)  assert x > 0 }
"#;
    let program = lower(input);
    assert!(verify_named_method(&program, "p1").is_ok());
}

#[test]
fn assert_under_the_same_guard_that_assumed_it_holds() {
    // `assume` and `assert` under the same branch guard: the implication
    // `c ⇒ x>0` discharges the goal `x>0` under pc `c` (both share the literal).
    let input = r#"
method p3(c: Bool, x: Int) { if (c) { assume x > 0  assert x > 0 } }
"#;
    let program = lower(input);
    assert!(verify_named_method(&program, "p3").is_ok());
}

#[test]
#[ignore = "needs the removed nested same-condition ite shapes; see the header"]
fn both_arms_establishing_a_fact_verifies_via_structural_join() {
    // IGNORED — the rules this depended on were removed deliberately.
    //
    // Both arms inhale `x > 0`, so it genuinely holds at the merge: the structural
    // block join recombines `c ⇒ x>0` and `¬c ⇒ x>0` into `x > 0` with no case
    // split. Silicon proves it too.
    //
    // It was discharged by the two nested same-condition `ite` shapes in
    // `IteReduceApplier`: `ite(c, true, x>0)` lands in the `true` class by
    // congruence, which puts an inner ite on the same `c` inside the outer ite's
    // else-arm, and the mirror shape collapses it. That scan is over a whole branch
    // class, so it carried a `NESTED_SCAN_BOUND = 64` cutoff — and the class it has
    // to search is the `true` class, the one that accretes past 64 immediately.
    // Measured: this shape verifies with 0 or 20 unrelated facts in scope and
    // **fails at 60**, so the rules had been inert on every non-trivial program
    // since the bound landed (2026-07-28, `9cf90c4`). This test passed only because
    // its graph is 16 nodes.
    //
    // Nothing else needed them: the full Prusti corpus (4494 members) and the
    // panic-freedom suite are byte-identical without them, and the permission-side
    // analogue of the identity is applied structurally at construction by
    // `ChunkPerm::collapse_same_cond` — no scan, no budget.
    //
    // Re-enable if a case-split rule `(c ⇒ f) ∧ (¬c ⇒ f) ⊢ f` is ever added: record
    // each polarity on union into `true` and fire when both appear, which is O(1)
    // and states the inference directly instead of pattern-matching its residue.
    let input = r#"
method give(x: Int) ensures x > 0
method p2(c: Bool, x: Int) { if (c) { give(x) } else { give(x) }  assert x > 0 }
"#;
    let program = lower(input);
    assert!(verify_named_method(&program, "p2").is_ok());
}

// --- Phase 3: function definitions are purified recipes, not e-graph grafts ---

#[test]
fn function_definition_does_not_leak_precondition_into_call_site() {
    // `f`'s body has a divisor obligation (`g(x)/g(x)`) discharged from
    // `requires g(x) == 5`, so verifying `f` saturates `g(x) ≡ 5` into its e-graph.
    // A certificate that cloned that e-graph would install `g(3) ≡ 5`
    // unconditionally when `f(3)` unfolds, passing the empty-pc `assert g(3) == 5`
    // that `m` never establishes. A purified recipe imports no e-classes.
    let input = r#"
function g(x: Int): Int
function f(x: Int): Int requires g(x) == 5 { g(x) / g(x) }
method m(b: Bool) {
  if (b) { assume g(3) == 5  var y: Int := f(3) }
  assert g(3) == 5
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed (precondition must not leak from the function \
         definition); got {result:?}"
    );
}

#[test]
fn purified_function_definition_still_defines_the_body() {
    // The recipe must still install `f(a) == body`: `f(x) { x + 1 }` unfolds so
    // that `assert f(2) == 3` holds. (Guards nothing here — total function.)
    let input = r#"
function f(x: Int): Int { x + 1 }
method m() { assert f(2) == 3 }
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "m").is_ok(),
        "purified function definition should still discharge f(2) == 3"
    );
}

#[test]
fn heap_dependent_function_purifies_to_snapshot_projection() {
    // A heap-dependent body (a bound `inhale`; `Deref`) purifies to `unwrap(proj_0(s))`.
    // The function verifies (frames its precondition footprint) end to end.
    let input = r#"
field f: Int
function get(x: Ref): Int requires acc(x.f) { x.f }
"#;
    let program = lower(input);
    assert!(
        verify_named_function(&program, "get").is_ok(),
        "heap-dependent function should verify with the purified recipe"
    );
}

// --- Function contracts as guarded rewrites (posts delivered transitively) ---

#[test]
fn abstract_function_post_available_at_call_site() {
    // `foo` is abstract: nothing about it used to survive outside the
    // (removed) immediate call-site assume. Its post now arrives via the
    // synthesized guarded axiom `foo#requires(x) ⟹ foo#ensures(x, foo(x))`,
    // whose guard the call-site `assert foo#requires(x)` establishes.
    let input = r#"
function foo(x: Int): Int
    requires x != 0
    ensures result == 10

method m(x: Int)
    requires x != 0
{
    assert foo(x) == 10
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "abstract post should deliver; got {result:?}"
    );
}

#[test]
fn function_post_propagates_transitively() {
    // `m` never calls `mk`/`foo` directly — their applications only appear by
    // unfolding `g`'s body. `mk`'s (abstract, unguarded) post gives
    // `ok(mk(x))`; `foo`'s guarded post then fires (its defined pre-token
    // `foo#requires(mk(x))` unfolds to `ok(mk(x))`), yielding
    // `foo(mk(x)) == 10` and so `g(x) == 11`. Pure occurrence-keyed rewrites,
    // no call-site stitching anywhere in `m`.
    let input = r#"
domain D { function ok(y: Int): Bool }

function mk(x: Int): Int
    ensures ok(result)

function foo(y: Int): Int
    requires ok(y)
    ensures result == 10

function g(x: Int): Int { foo(mk(x)) + 1 }

method m(x: Int) {
    assert g(x) == 11
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "posts should propagate to the transitive call site; got {result:?}"
    );
}

#[test]
fn postcondition_wd_under_precondition() {
    // Post WD depends on pre truth: the divisor obligation inside
    // `safediv#ensures`'s body is only provable under the entry
    // `assume safediv#requires(x, y)` (mirroring the heap-dependent
    // a bound `inhale` entry).
    let input = r#"
function safediv(x: Int, y: Int): Int
    requires y != 0
    ensures result == x / y
{ x / y }
"#;
    let program = lower(input);
    assert!(
        verify_named_function(&program, "safediv#ensures").is_ok(),
        "post WD must hold under the assumed precondition"
    );
    assert!(
        verify_named_function(&program, "safediv").is_ok(),
        "body + exit post check must verify"
    );
}

#[test]
fn recursive_function_postcondition_by_induction() {
    // Induction via the limited encoding: while checking `f`'s body, `f`'s own
    // spec-derived post rule is installed (Silicon's phase-1 `post` axiom), so
    // the recursive call's post (`f(next(x)) == 0`, guarded by
    // `ok(next(x))` from the axiom) discharges the exit assert. At the outer
    // call site the post rides the certificate's post fact.
    let input = r#"
domain D {
    function ok(x: Int): Bool
    function next(x: Int): Int
    axiom { forall x: Int :: {next(x)} ok(next(x)) }
}

function f(x: Int): Int
    requires ok(x)
    ensures result == 0
{ x == 0 ? 0 : f(next(x)) }

method m(x: Int)
    requires ok(x)
{
    assert f(x) == 0
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("recursive function SCC accepted");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn failed_function_exports_no_facts() {
    // Success gating: `g` cannot prove `foo`'s precondition, so `g` itself
    // fails — and none of its axioms (definition, facts) may install. `m`
    // then knows nothing about `g` and fails too, instead of receiving facts
    // proven from a refuted premise.
    let input = r#"
domain D { function ok(y: Int): Bool }

function foo(y: Int): Int
    requires ok(y)
    ensures result == 10

function g(x: Int): Int { foo(x) + 1 }

method m(x: Int) {
    assert g(x) == 11
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("acyclic");
    let results = crate::verify::verify(&analyzed);
    let get = |n: &str| {
        results
            .iter()
            .find(|(name, _)| name == n)
            .unwrap_or_else(|| panic!("no result for {n}"))
    };
    assert!(get("g").1.is_err(), "g cannot prove foo's precondition");
    assert!(
        get("m").1.is_err(),
        "a failed g must not export its definition or facts to m"
    );
}

#[test]
fn heap_dep_post_fires_only_where_the_precondition_bool_holds() {
    // The pre-token stamps `get#requires`'s *whole* precondition — footprint and
    // bool. `m` establishes `y.f > 0` before the call, so the `Snap` check
    // passes, the token is released, and the guarded post (`result == x.f`)
    // fires; `n` cannot show `y.f > 0`, so the call's own check fails and the
    // token is never stamped for it.
    let input = r#"
field f: Int

function get(x: Ref): Int
    requires acc(x.f) && x.f > 0
    ensures result == x.f

method m(y: Ref)
    requires acc(y.f) && y.f > 0
{
    var a: Int := get(y)
    assert a == y.f
}

method n(y: Ref)
    requires acc(y.f)
{
    var a: Int := get(y)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let result = verify_named_method(&program, "n");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "the precondition's bool is unproven, so the call must fail: {result:?}"
    );
}

#[test]
fn heap_dep_post_propagates_through_a_transitive_call() {
    // `outer`'s body calls the abstract heap-dep `inner`. At `m`'s call site the
    // unfold rule rebuilds `outer`'s recipe, which materializes `inner(y, s')` —
    // but nothing re-runs `inner`'s precondition check there. The `Snap` arm of
    // `purify_function` exported `outer#pre ⟹ inner#pre(x, s')` for exactly this
    // (Silicon's `bodyPreconditionPropagationAxiom`), so `inner`'s post fact
    // fires on the materialized occurrence and `m` learns `outer(y) == y.f + 1`.
    let input = r#"
field f: Int

function inner(x: Ref): Int
    requires acc(x.f)
    ensures result == x.f

function outer(x: Ref): Int
    requires acc(x.f)
{ inner(x) + 1 }

method m(y: Ref)
    requires acc(y.f)
{
    assert outer(y) == y.f + 1
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("acyclic");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn heap_dep_exit_postcondition_check_bites() {
    // The heap-dependent exit check (`assert bad#ensures(x, result, s)`) is now
    // emitted, so a body that does not establish its own postcondition fails —
    // and therefore exports no facts.
    let input = r#"
field f: Int

function bad(x: Ref): Int
    requires acc(x.f)
    ensures result == x.f
{ x.f + 1 }
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "bad");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed on the exit post check, got {result:?}"
    );
}

#[test]
fn branchless_recursive_heap_dep_function_verifies_by_induction() {
    // Recursion over a snapshot works — it is the *branch*, not the recursion, that
    // the pre-token cannot cross on its own (that branching case is the
    // characterized incompleteness in
    // `tests/cases/known_limitations/goal_needs_case_split_heap_dep.vpr`). With
    // the recursive `Snap` at
    // empty pc the token is released unconditionally, so the in-batch post rule (the
    // induction hypothesis) gives `rf(x, next(n), s') == x.f`, discharging the exit
    // assert. The call site in `m` gets the post from the certificate's post fact.
    let input = r#"
domain D {
    function next(n: Int): Int
}

field f: Int

function rf(x: Ref, n: Int): Int
    requires acc(x.f)
    ensures result == x.f
{ rf(x, next(n)) }

method m(y: Ref)
    requires acc(y.f)
{
    assert rf(y, 3) == y.f
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("recursive function SCC accepted");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn mutually_recursive_abstract_functions_deliver_each_others_posts() {
    // `gen`/`con` are an abstract inverse pair, so the two sit in one recursion
    // SCC — and an SCC member's calls are lowered to the limited twin `f'` in
    // every sibling's recipe. An abstract member must therefore *also* carry a
    // limited twin, so its unfold rule frames `f(x) == f'(x)`; without the frame
    // `gen`'s post fact speaks about `con'(gen(x))` while the goal holds
    // `con(gen(x))`, and the two never meet. (Prusti's `make_generic_*` /
    // `make_concrete_*` pairs have exactly this shape.)
    let input = r#"
domain D {
    function tag(x: Int): Int
}

function gen(x: Int): Int
    ensures tag(result) == 7
    ensures con(result) == x

function con(y: Int): Int
    ensures gen(result) == y

method round_trip(a: Int, b: Int)
{
    assert tag(gen(a)) == 7
    assert con(gen(a)) == a
    assert gen(con(b)) == b
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("abstract function SCC accepted");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn adt_constructor_is_injective() {
    // Two applications of the same constructor sharing an e-class must union
    // their arguments pairwise. Here `f(p)`'s body and `g(p)`'s body both build a
    // `Pair`, and `p` equates them — but the goal mentions no projection, so the
    // component equality is reachable only through the injectivity rule
    // (congruence runs forward only; `proj_rule` needs a `projᵢ` node to fire).
    let input = r#"
adt Pair {
    mk(fst: Int, snd: Int)
}

function f(p: Pair): Pair
    ensures result == mk(1, 2)

function g(p: Pair): Pair
    ensures result == mk(1, p.snd)

method components(p: Pair)
{
    assume f(p) == g(p)
    assert p.snd == 2
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn distinct_adt_constructors_are_disequal() {
    // Constructor distinctness with no `tag` term anywhere: the `ConstFold`
    // lattice reads the constructor identity straight off the operand's e-class,
    // so `Nil() == Cons(..)` folds to `false`. In an SMT encoding this needs an
    // O(variants) family of `tag` axioms plus a term to trigger them.
    let input = r#"
adt List {
    Nil()
    Cons(head: Int, tail: List)
}

method disequal(x: Int, l: List)
{
    assert (Nil() == Cons(x, l)) == false
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn merging_distinct_constructors_is_a_contradiction() {
    // The dual: *assuming* two distinct constructors equal makes the e-class
    // contradictory (`Data::Inconsistent`), which `prove_under_pc` reports at
    // tier 0 — so anything is provable from it. This is the free-constructor
    // property, and it is what makes the `assert false` below go through.
    let input = r#"
adt List {
    Nil()
    Cons(head: Int, tail: List)
}

method absurd(x: Int, l: List)
{
    assume Nil() == Cons(x, l)
    assert false
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

// ---- Trigger shapes: nesting, multi-term groups, alternatives --------------

#[test]
fn nested_trigger_instantiates_on_the_nested_application() {
    // The trigger is `f(g(i))`: the binder sits under a nested call, so σ(i) is
    // read off the *inner* application's argument. Matching `f(g(7))` in the goal
    // gives σ = {i ↦ 7}.
    let input = r#"
domain D {
    function g(i: Int): Int
    function f(i: Int): Bool
    axiom nested { forall i: Int :: {f(g(i))} f(g(i)) }
}
method m() {
    assert f(g(7))
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn nested_trigger_does_not_fire_on_the_outer_head_alone() {
    // `f(3)` is an application of the trigger's *root* function, but its argument
    // is not a `g` application — the nested pattern does not match, so nothing is
    // instantiated and the goal stands unproven.
    let input = r#"
domain D {
    function g(i: Int): Int
    function f(i: Int): Bool
    axiom nested { forall i: Int :: {f(g(i))} f(g(i)) }
}
method m() {
    assert f(3)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "expected AssertionFailed, got {result:?}"
    );
}

#[test]
fn multi_term_trigger_needs_every_term_present() {
    // Group `{p(i), q(i)}` is conjunctive: `p(7)` alone must not instantiate the
    // quantifier, so the goal `p(7) ==> q(7)`'s body stays unproven.
    let input = r#"
domain D {
    function p(i: Int): Bool
    function q(i: Int): Bool
    function r(i: Int): Bool
    axiom both { forall i: Int :: {p(i), q(i)} r(i) }
}
method m() {
    assert p(7) == p(7) && r(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "only `p(7)` is present, so the group must not fire; got {result:?}"
    );
}

#[test]
fn multi_term_trigger_fires_when_all_terms_present() {
    // Same quantifier, now with both `p(7)` and `q(7)` in the e-graph: the group
    // matches at σ = {i ↦ 7} and releases `r(7)`.
    let input = r#"
domain D {
    function p(i: Int): Bool
    function q(i: Int): Bool
    function r(i: Int): Bool
    axiom both { forall i: Int :: {p(i), q(i)} r(i) }
}
method m() {
    assume p(7)
    assume q(7)
    assert r(7)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn alternative_trigger_groups_each_fire() {
    // `{p(i)}{q(i)}` are alternatives — either application alone instantiates the
    // quantifier. Each method exercises one group.
    let input = r#"
domain D {
    function p(i: Int): Bool
    function q(i: Int): Bool
    function r(i: Int): Bool
    axiom alt { forall i: Int :: {p(i)}{q(i)} r(i) }
}
method via_p() {
    assume p(7)
    assert r(7)
}
method via_q() {
    assume q(8)
    assert r(8)
}
"#;
    let program = lower(input);
    for m in ["via_p", "via_q"] {
        let result = verify_named_method(&program, m);
        assert!(result.is_ok(), "{m}: expected Ok, got {result:?}");
    }
}

#[test]
fn literal_trigger_argument_matches_only_that_literal() {
    // The trigger `f(i, 0)` pins its second argument: `f(7, 0)` instantiates the
    // quantifier, `f(7, 1)` does not.
    let input = r#"
domain D {
    function f(i: Int, j: Int): Bool
    function r(i: Int): Bool
    axiom lit { forall i: Int :: {f(i, 0)} r(i) }
}
method hit() {
    assume f(7, 0)
    assert r(7)
}
method miss() {
    assume f(7, 1)
    assert r(7)
}
"#;
    let program = lower(input);
    let hit = verify_named_method(&program, "hit");
    assert!(hit.is_ok(), "expected Ok, got {hit:?}");
    let miss = verify_named_method(&program, "miss");
    assert!(
        matches!(miss, Err(ref e) if matches!(e.root_cause(), VerifyError::AssertionFailed)),
        "`f(7, 1)` must not match the literal trigger; got {miss:?}"
    );
}

#[test]
fn adt_constructor_trigger_matches_a_construction() {
    // A trigger may be headed by an ADT constructor: `{Cons(x, l)}` fires wherever
    // a `Cons` application exists.
    let input = r#"
adt List {
    Nil()
    Cons(head: Int, tail: List)
}
domain D {
    function ok(l: List): Bool
}
domain A {
    axiom cons_ok { forall x: Int, l: List :: {Cons(x, l)} ok(Cons(x, l)) }
}
method m(l: List) {
    assert ok(Cons(3, l))
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn enum_exhaustiveness_through_domain_boxed_discriminator() {
    // The structs_enums.vpr fallthrough shape: the discriminator is boxed in a
    // *domain* (`cons`/`value` are axiom-defined uninterpreted functions, not
    // ADT ctor/proj — the projection reduction cannot see through), the switch
    // compares the *unboxed* value, and the flow goes through a heap
    // predicate. Needs the unary push-down in the disproven-eq applier:
    // `value(ite(c, cons(1), cons(0)))` commutes into the branches, where the
    // domain axiom decides each arm.
    let input = r#"
domain s_Int_isize {
    function s_Int_isize_cons(arg0: Int): s_Int_isize
    function s_Int_isize_value(arg0: s_Int_isize): Int
    axiom ax_cons {
        forall s: s_Int_isize :: { s_Int_isize_value(s) }
            s_Int_isize_cons(s_Int_isize_value(s)) == s
    }
    axiom ax_value {
        forall value: Int :: { s_Int_isize_cons(value) }
            s_Int_isize_value(s_Int_isize_cons(value)) == value
    }
}

adt s_MaybeInt {
    s_MaybeInt_0_cons()
    s_MaybeInt_1_cons(f: Int)
}

field p_Int_isize_val: s_Int_isize

predicate p_Int_isize(self: Ref) {
    acc(self.p_Int_isize_val, write)
}

function p_Int_isize_snap(self: Ref): s_Int_isize
    requires acc(p_Int_isize(self), write)
{
    (unfolding acc(p_Int_isize(self), write) in self.p_Int_isize_val)
}

method p_Int_isize_assign(self: Ref, value: s_Int_isize)
    ensures acc(p_Int_isize(self), write)
    ensures p_Int_isize_snap(self) == value

function s_MaybeInt_discr(self: s_MaybeInt): s_Int_isize
{
    (self.iss_MaybeInt_1_cons ? s_Int_isize_cons(1) : s_Int_isize_cons(0))
}

method exhaustive(v: s_MaybeInt, _2p: Ref)
{
    p_Int_isize_assign(_2p, s_MaybeInt_discr(v))
    var _tmp0: s_Int_isize := p_Int_isize_snap(_2p)
    exhale acc(p_Int_isize(_2p), write)
    if (s_Int_isize_value(_tmp0) == 0) {
    } elseif (s_Int_isize_value(_tmp0) == 1) {
    } else {
        assert false
    }
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn old_address_exhale_after_dead_branch_normalizes() {
    // Minimized from structs_enums.vpr's m_point_step: a guarded generic/
    // concrete predicate conversion cycle at a heap-read (&mut pointee)
    // address, an unreachable switch arm, and an exit ensures at the OLD
    // snapshot's address. The dead arm's instructions pre-create the ensures
    // recipe's snapshot-spine e-nodes, so the rebuild-time conditional reduce
    // (which fires only when the rebuild adds a node) skips — and the exhale's
    // rebuilt address `@addr(cons.0(f(.., Some(unwrap(proj_0(s))))))` never
    // normalizes onto the held chunk's. Pinned by the subtract-time
    // reduce-and-retry in `heap_subtract`.
    let input = r#"
domain Type {
    function s_Point_type(): Type
}

domain s_Param {}

domain s_Int_isize {
    function s_Int_isize_cons(arg0: Int): s_Int_isize
    function s_Int_isize_value(arg0: s_Int_isize): Int
    axiom ax_cons {
        forall s: s_Int_isize :: { s_Int_isize_value(s) }
            s_Int_isize_cons(s_Int_isize_value(s)) == s
    }
    axiom ax_value {
        forall value: Int :: { s_Int_isize_cons(value) }
            s_Int_isize_value(s_Int_isize_cons(value)) == value
    }
}

adt s_Ref_mutable {
    s_Ref_mutable_cons(s_Ref_mutable_0: Ref, s_Ref_mutable_1: s_Param)
}

adt s_Point {
    s_Point_cons(x: Int)
}

function make_generic_s_Point(self: s_Point): s_Param
    ensures make_concrete_s_Point(result) == self

function make_concrete_s_Point(snap: s_Param): s_Point
    ensures make_generic_s_Point(result) == snap

field p_Ref_mutable_val: s_Ref_mutable
field p_Point_val: s_Point
field p_Int_isize_val: s_Int_isize

predicate p_Param(self: Ref, T$0: Type)

predicate p_Ref_mutable(self: Ref, U$1: Type) {
    acc(self.p_Ref_mutable_val, write)
}

predicate p_Point(self: Ref) {
    acc(self.p_Point_val, write)
}

predicate p_Int_isize(self: Ref) {
    acc(self.p_Int_isize_val, write)
}

function p_Dir_field_discr(self: Ref): Ref
    ensures (self == null) == (result == null)

predicate p_Dir(self: Ref) {
    acc(p_Int_isize(p_Dir_field_discr(self)), write) &&
    (p_Int_isize_snap(p_Dir_field_discr(self)) == s_Int_isize_cons(0) ||
     p_Int_isize_snap(p_Dir_field_discr(self)) == s_Int_isize_cons(1))
}

function p_Int_isize_snap(self: Ref): s_Int_isize
    requires acc(p_Int_isize(self), write)
{
    (unfolding acc(p_Int_isize(self), write) in self.p_Int_isize_val)
}

function p_Ref_mutable_snap(self: Ref, U$1: Type): s_Ref_mutable
    requires acc(p_Ref_mutable(self, U$1), write)
{
    (unfolding acc(p_Ref_mutable(self, U$1), write) in self.p_Ref_mutable_val)
}

function p_Point_snap(self: Ref): s_Point
    requires acc(p_Point(self), write)
{
    (unfolding acc(p_Point(self), write) in self.p_Point_val)
}

function p_Param_snap(self: Ref, T$0: Type): s_Param
    requires acc(p_Param(self, T$0), write)

method p_Int_isize_assign(self: Ref, value: s_Int_isize)
    ensures acc(p_Int_isize(self), write)
    ensures p_Int_isize_snap(self) == value

method make_generic_Point(self: Ref)
    requires acc(p_Point(self), write)
    ensures acc(p_Param(self, s_Point_type()), write)
    ensures old(make_generic_s_Point(p_Point_snap(self))) ==
        p_Param_snap(self, s_Point_type())

method make_concrete_Point(self: Ref)
    requires acc(p_Param(self, s_Point_type()), write)
    ensures acc(p_Point(self), write)
    ensures old(p_Param_snap(self, s_Point_type())) ==
        make_generic_s_Point(p_Point_snap(self))

method step(_1p: Ref, _2p: Ref, _3p: Ref)
    requires acc(p_Ref_mutable(_1p, s_Point_type()), write)
    requires acc(p_Dir(_2p), write)
    requires acc(p_Param(p_Ref_mutable_snap(_1p, s_Point_type()).s_Ref_mutable_0,
        s_Point_type()), write)
    ensures acc(p_Param(old(p_Ref_mutable_snap(_1p, s_Point_type())).s_Ref_mutable_0,
        s_Point_type()), write)
{
    var _from_bb0_to_bb2: Bool
    var _from_bb0_to_bb3: Bool
    var _from_bb0_to_bb1: Bool
    var _tmp0: s_Int_isize
    label start
    _from_bb0_to_bb2 := false
    _from_bb0_to_bb3 := false
    _from_bb0_to_bb1 := false
    goto bb_0
    label bb_0
    p_Int_isize_assign(_3p, (unfolding acc(p_Dir(_2p)) in
        p_Int_isize_snap(p_Dir_field_discr(_2p))))
    _tmp0 := p_Int_isize_snap(_3p)
    exhale acc(p_Int_isize(_3p), write)
    if (s_Int_isize_value(_tmp0) == 0) {
        _from_bb0_to_bb2 := true
        goto bb_2
    } elseif (s_Int_isize_value(_tmp0) == 1) {
        _from_bb0_to_bb3 := true
        goto bb_3
    } else {
        _from_bb0_to_bb1 := true
        goto bb_1
    }
    label bb_2
    if (_from_bb0_to_bb2) {
        make_concrete_Point(p_Ref_mutable_snap(_1p, s_Point_type()).s_Ref_mutable_0)
    }
    if (_from_bb0_to_bb2) {
        make_generic_Point(p_Ref_mutable_snap(_1p, s_Point_type()).s_Ref_mutable_0)
    }
    goto bb_join
    label bb_3
    goto bb_join
    label bb_1
    exhale false
    inhale false
    goto end
    label bb_join
    goto end
    label end
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn branch_join_exhale() {
    // A CFG join leaves the ensured chunk's permission as a sum of
    // branch-scaled ites (`ite(c,1,0) + ite(c,0,ite(c2,1,0))`); the exit
    // exhale needs it under the disjunction of the branch flags, with the
    // third arm excluded by exhaustiveness. No single saturation proves the
    // sufficiency; the goal is discharged by the `ite_decompose` tier walking
    // the branch flags, one assumption per iteration.
    let input = r#"
domain s_Int_isize {
    function s_Int_isize_cons(arg0: Int): s_Int_isize
    function s_Int_isize_value(arg0: s_Int_isize): Int
    axiom ax_value {
        forall value: Int :: { s_Int_isize_cons(value) }
            s_Int_isize_value(s_Int_isize_cons(value)) == value
    }
}

adt s_MaybeInt {
    s_MaybeInt_0_cons()
    s_MaybeInt_1_cons(f: Int)
}

field p_Bool_val: Bool

predicate p_Bool(self: Ref) {
    acc(self.p_Bool_val, write)
}

method p_Bool_assign(self: Ref, value: Bool)
    ensures acc(p_Bool(self), write)

function s_MaybeInt_discr(self: s_MaybeInt): s_Int_isize
{
    (self.iss_MaybeInt_1_cons ? s_Int_isize_cons(1) : s_Int_isize_cons(0))
}

method exhaustive(v: s_MaybeInt, _0p: Ref)
    ensures acc(p_Bool(_0p), write)
{
    var t: s_Int_isize := s_MaybeInt_discr(v)
    if (s_Int_isize_value(t) == 0) {
        p_Bool_assign(_0p, false)
    } elseif (s_Int_isize_value(t) == 1) {
        p_Bool_assign(_0p, true)
    } else {
        assert false
        inhale false
    }
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn disequality_holds_in_both_argument_orders() {
    // Prusti writes div preconditions constant-first (`requires 0 != value(b)`)
    // while the div obligation builds `Eq(b, 0)`. `Binary(Eq, ..)` is not
    // commutative as an e-node; the disproven-eq mirror in the `eq-false-then/else` applier
    // lands the flipped node in the same (false) class.
    let input = r#"
function d_flip(a: Int, b: Int): Int
    requires 0 != b
{ a \ b }
"#;
    let program = lower(input);
    let result = verify_named_function(&program, "d_flip");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn enum_exhaustiveness_two_variants_via_boxed_discriminator() {
    // Prusti's enum-match exhaustiveness shape: an opaque discriminator value
    // (`box(0)`/`box(1)` — *not* e-graph literals) selected by an `ite` over
    // the variant test. Excluding both tags must derive `false`. Needs the
    // `eq-false-then/else` unit propagation: `d != box(0)` with `d = isOne ? box(0) : box(1)`
    // pins `isOne = false`, then `d != box(1)` pins `isOne = true` —
    // inconsistent, so `assert false` discharges.
    let input = r#"
domain BoxedInt {
    function box(arg: Int): BoxedInt
    function unbox(arg: BoxedInt): Int
    axiom ax_box_unbox { forall s: BoxedInt :: { unbox(s) } box(unbox(s)) == s }
    axiom ax_unbox_box { forall v: Int :: { box(v) } unbox(box(v)) == v }
}

adt TwoCase {
    One()
    Two()
}

function discrTwoCase(v: TwoCase): BoxedInt {
    v.isOne ? box(0) : box(1)
}

method exhaustive(v: TwoCase) {
    var d: BoxedInt := discrTwoCase(v)
    assume d != box(0)
    assume d != box(1)
    assert false
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn enum_exhaustiveness_three_variants_via_boxed_discriminator() {
    // Nested-`ite` discriminator (3+ variants): the unit propagation must chain —
    // pinning the outer condition lets `ite-reduce` collapse `d` onto the inner
    // `ite`, whose node then sits in `d`'s e-class for the next exclusion.
    let input = r#"
domain BoxedInt {
    function box(arg: Int): BoxedInt
    function unbox(arg: BoxedInt): Int
    axiom ax_box_unbox { forall s: BoxedInt :: { unbox(s) } box(unbox(s)) == s }
    axiom ax_unbox_box { forall v: Int :: { box(v) } unbox(box(v)) == v }
}

adt ThreeCase {
    One()
    Two()
    Three()
}

function discrThreeCase(v: ThreeCase): BoxedInt {
    v.isOne ? box(0) : (v.isTwo ? box(1) : box(2))
}

method exhaustive(v: ThreeCase) {
    var d: BoxedInt := discrThreeCase(v)
    assume d != box(0)
    assume d != box(1)
    assume d != box(2)
    assert false
}
"#;
    let program = lower(input);
    let analyzed = crate::vmir::analyze(program).expect("analyze");
    let results = crate::verify::verify(&analyzed);
    for (name, r) in &results {
        assert!(r.is_ok(), "{name} should verify; got {r:?}");
    }
}

#[test]
fn fold_with_unprovable_perm_sign_fails() {
    // A fold's multiplier must be provably non-negative: a negative scale
    // would flip the footprint subtraction into permission fabrication.
    // `p` is unconstrained, so the sign side condition must fail.
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref, p: Perm)
  requires acc(x.f, write)
{
  fold acc(Cell(x), p)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::SideCondition(_))),
        "expected the perm-sign side condition to fail, got {result:?}"
    );
}

#[test]
fn unfold_with_unprovable_perm_sign_fails() {
    // Same side condition on unfold: consuming the predicate chunk at an
    // unconstrained (possibly negative) multiplier must be rejected.
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref, p: Perm)
  requires acc(x.f, write)
{
  fold acc(Cell(x), write)
  unfold acc(Cell(x), p)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::SideCondition(_))),
        "expected the perm-sign side condition to fail, got {result:?}"
    );
}

#[test]
fn fold_with_constrained_nonneg_perm_verifies() {
    // With the multiplier's sign pinned by the precondition, the side
    // condition discharges and the fractional fold verifies.
    let input = r#"
field f: Int
predicate Cell(x: Ref) { acc(x.f, write) }
method m(x: Ref, p: Perm)
  requires acc(x.f, write) && p == 1/2
{
  fold acc(Cell(x), p)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "m");
    assert!(
        result.is_ok(),
        "constrained non-negative fold should verify, got {result:?}"
    );
}
