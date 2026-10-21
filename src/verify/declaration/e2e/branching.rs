//! Branching-focused tests: fold/exhale established on one arm vs both,
//! interacting with the join.

use super::*;

#[test]
fn fold_on_both_arms_available_past_join() {
    let input = r#"
field f: Int

predicate cell(r: Ref)
{ acc(r.f) }

method client(r: Ref, c: Bool)
{
    inhale acc(r.f)

    if (c) {
        fold cell(r)
    } else {
        fold cell(r)
    }

    unfold cell(r)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn fold_on_one_arm_only_unavailable_past_join() {
    // Fold-flavored sibling of `branch_establishing_resource_on_one_arm_only_fails`
    // (which uses raw `acc`) -- the fold itself must be branch-scoped too.
    let input = r#"
field f: Int

predicate cell(r: Ref)
{ acc(r.f) }

method client(r: Ref, c: Bool)
{
    inhale acc(r.f)

    if (c) {
        fold cell(r)
    }

    unfold cell(r)
}
"#;
    let program = lower(input);
    let result = verify_named_method(&program, "client");
    assert!(
        matches!(result, Err(ref err) if matches!(err.root_cause(), VerifyError::InsufficientPermission)),
        "expected InsufficientPermission, got {result:?}"
    );
}

// ---- `f%pre` token release under the call's path condition -----------------
//
// A call site mints the callee's `f%pre(args)` token: its *presence* lets the
// unfold rule materialize the callee's body here, its *truth* releases the
// resulting axioms (`tok ==> f == body`, `tok ==> each exported fact`). The truth
// is assumed under the call's path condition, so a call under a ternary or an
// implication must not release the callee's facts onto a sibling intra-block
// path. CFG-level branches are separately isolated by the fork/block model; these
// tests cover the intra-block `PcKind`s, which are the ones the token carries.

#[test]
fn ternary_call_does_not_leak_callee_ensures() {
    // `g` has no precondition, so the callee-side guard chain (its `#requires`
    // application, then its own internal pc) is vacuous — the caller's token is
    // the only thing that can confine `ensures false` to the `b` arm. A failing
    // assertion means *every* tier rejected, so this covers the ground graph, the
    // working graph and its probes in one test.
    let input = r#"
function g(a: Int): Bool
  ensures false

method client(v: Int, b: Bool)
{
    var r: Bool := b ? g(v) : true
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "`ensures false` from a call under `b` must not reach the `!b` path"
    );
}

#[test]
fn implication_guarded_call_does_not_leak_callee_ensures() {
    // The `PcKind::Fact` analogue of the test above: an implication, not a
    // ternary. Both intra-block kinds must confine the release.
    let input = r#"
function g(a: Int): Bool
  ensures false

method client(v: Int, b: Bool)
{
    var r: Bool := b ==> g(v)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "`ensures false` from a call under `b ==> _` must not reach the `!b` path"
    );
}

#[test]
fn ternary_call_ensures_available_on_its_own_branch() {
    // Non-vacuity for the tests above: the guard must *collapse* where the call
    // does occur, which happens in the probe clone that assumes the goal's pc.
    // Without that, gating would trade a leak for lost completeness.
    //
    // The claim is stated *on* the branch (`b ==> …`), which is the shape the
    // `ite_decompose` tier telescopes. Asserting the unbranched `r == 7` instead
    // needs reasoning by cases over the opaque `b` — the case split we deleted;
    // that form lives in `tests/cases/known_limitations/`.
    let input = r#"
function g(a: Int): Int
  ensures result == 7

method client(v: Int, b: Bool)
{
    var r: Int := b ? g(v) : 7
    assert b ==> r == 7
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "the callee's post must still be available on the branch that calls it"
    );
}

#[test]
fn unconditional_call_still_unfolds() {
    // The definitional axiom is no longer an e-class merge when a token is
    // present: it is `ite(tok, f(args) == body, true) == true`, which needs
    // `ite-reduce` then `eq-true-union` to land the merge. This is the
    // dominant shape in the corpus, so it is worth pinning on its own.
    let input = r#"
function inc(x: Int): Int
{ x + 1 }

method client(v: Int)
{
    var a: Int := inc(v)
    assert a == v + 1
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "an unconditional call must still unfold its body"
    );
}

#[test]
fn branch_conditional_call_still_unfolds_under_its_branch() {
    // The guarded definitional equality with the guard collapsing only inside the
    // probe clone — the conditional counterpart of the test above.
    let input = r#"
function inc(x: Int): Int
{ x + 1 }

method client(v: Int, b: Bool)
{
    var a: Int := b ? inc(v) : 0
    assert b ==> a == v + 1
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "a conditional call must unfold under its own branch"
    );
}

#[test]
fn nested_call_definition_reaches_client() {
    // `six`'s body propagates `five%pre` as an ordinary step whose value is never
    // read (`bodyPreconditionPropagation`). Adding that node only makes `five`
    // *materializable*; releasing its truth under `six%pre` is what activates
    // `five`'s own definition, so this is the guard for that plumbing.
    let input = r#"
function five(): Int { 5 }
function six(): Int { five() + 1 }

method client()
{
    assert six() == 6
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "a nested callee's definition must reach a transitive client"
    );
}

#[test]
fn call_with_no_occurrence_never_instantiates() {
    // The other end of the range: with no call at all the token is absent, so the
    // rule never materializes the body and the post never arrives. Correct before
    // this gating too — kept so a future change cannot re-widen the token
    // silently.
    let input = r#"
function g(a: Int): Bool
  ensures false

method client(v: Int)
{
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "an uncalled function's post must not be instantiated"
    );
}

// ---- callee-internal pc on a propagated token ------------------------------
//
// A body that calls `g` under a condition propagates `g%pre(gargs)` so `g` can
// unfold at a client. That token's release carries two guards: the enclosing
// gate (the call's own `f%pre`, or the quantifier's e-class) and the
// body-internal condition guarding the nested call —
// `f_pre(a) ==> (b[x:=a] ==> g_pre(..))`. Without the inner guard, `g`'s own
// axioms fire at args the body never calls it at.

#[test]
fn conditional_nested_call_does_not_release_callee_off_its_branch() {
    // `f(-1)` takes the else arm, so `g` is never called; its `ensures false`
    // must not arrive. The call to `f` is unconditional, so the outer gate is
    // true and the body-internal `x > 0` is the only thing that can confine it.
    let input = r#"
function g(a: Int): Int
  ensures false

function f(x: Int): Int
{ x > 0 ? g(x) : 0 }

method client()
{
    var r: Int := f(-1)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "a nested callee's post must not be released where the body does not call it"
    );
}

#[test]
fn conditional_nested_call_releases_callee_on_its_branch() {
    // Non-vacuity for the test above: at `f(1)` the body really does call `g`,
    // so `ensures false` must still arrive and make the state inconsistent.
    // Guarding must not cost the release where the guard holds.
    let input = r#"
function g(a: Int): Int
  ensures false

function f(x: Int): Int
{ x > 0 ? g(x) : 0 }

method client()
{
    var r: Int := f(1)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "a nested callee's post must still be released where the body calls it"
    );
}

#[test]
fn quantifier_conditional_call_does_not_release_callee_off_its_branch() {
    // The quantifier half: `prepare_body` emits the token for a call inside a
    // forall body, and the release happens at *instantiation*. Instantiating at
    // `i == -1` (via the ground trigger term `h(-1)`) must not fire `g`'s
    // axioms, since the body only calls `g` under `i > 0`. The instantiated body
    // is separately fine — `ite(-1 > 0, .., true)` folds to `true`.
    let input = r#"
function h(a: Int): Int

function g(a: Int): Int
  ensures false

method client()
{
    inhale forall i: Int :: {h(i)} i > 0 ==> g(i) == 5
    var t: Int := h(-1)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "instantiating at a sigma the body excludes must not release the callee"
    );
}

#[test]
fn quantifier_conditional_call_releases_callee_on_its_branch() {
    // Non-vacuity for the test above, at a sigma the body does include.
    let input = r#"
function h(a: Int): Int

function g(a: Int): Int
  ensures false

method client()
{
    inhale forall i: Int :: {h(i)} i > 0 ==> g(i) == 5
    var t: Int := h(3)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "instantiating at a sigma the body includes must still release the callee"
    );
}

#[test]
fn resource_body_conditional_call_does_not_release_callee_off_its_branch() {
    // The third release path: a predicate body's boolean is grafted via
    // `slice_with_tokens`, which releases the propagated token with no *outer*
    // gate (a graft has no enclosing `f%pre`). The body-internal condition is
    // therefore the only guard, and at `r.f == -1` the body does not call `g`.
    let input = r#"
field f: Int

function g(a: Int): Int
  ensures false

predicate p(x: Ref)
{ acc(x.f) && (0 < x.f ==> g(x.f) == 5) }

method client(r: Ref)
{
    inhale acc(r.f) && r.f == -1
    fold p(r)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "a resource body must not release its callee where its condition fails"
    );
}

#[test]
fn resource_body_conditional_call_releases_callee_on_its_branch() {
    // Non-vacuity for the test above. Verified against a temporarily-disabled
    // guard fold: without the guards the `-1` case above wrongly verifies, so
    // this pair genuinely brackets the resource path.
    let input = r#"
field f: Int

function g(a: Int): Int
  ensures false

predicate p(x: Ref)
{ acc(x.f) && (0 < x.f ==> g(x.f) == 5) }

method client(r: Ref)
{
    inhale acc(r.f) && r.f == 3
    fold p(r)
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "a resource body must still release its callee where its condition holds"
    );
}

#[test]
fn dead_arm_verdict_does_not_poison_the_join() {
    // `assume b` makes the `else` arm's cube unsatisfiable, so its `assert false`
    // holds vacuously. That verdict is valid *only* under the arm's cube — and the
    // cube is ambient, not part of the obligation's pc delta, so the memo saw an
    // empty pc and published the bare goal (`false`) into ground, merging `false`
    // with `true` for every later block. Exported under the cube it is the sound
    // `¬b ⇒ false`, which folds away where the cube is unsat.
    let input = r#"
method client(b: Bool)
{
    assume b
    if (b) { assert true } else { assert false }
    assert false
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_err(),
        "a vacuous verdict from a dead arm must not discharge the join's `assert false`"
    );
}

#[test]
fn dead_arm_obligation_is_still_vacuously_discharged() {
    // Non-vacuity for the test above: the dead arm's own `assert false` must
    // still pass — the fix guards the *export*, it does not drop the verdict.
    let input = r#"
method client(b: Bool)
{
    assume b
    if (b) { assert true } else { assert false }
}
"#;
    let program = lower(input);
    assert!(
        verify_named_method(&program, "client").is_ok(),
        "an unreachable arm's obligations hold vacuously"
    );
}
