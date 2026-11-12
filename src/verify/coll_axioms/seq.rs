//! The axioms of `Seq[T]`, adapted from Silicon's
//! `dafny_axioms/sequences.vpr` (see the [module docs](super)).
//!
//! The Dafny axioms' matching-loop cuts (`s != Seq()`, `s != s ++ t`) are left
//! out: the e-graph could almost never prove them (`s != Seq()` rarely has a
//! term), and its generation bound already cuts matching loops.

use super::*;
use crate::vmir::SeqOp;

fn len(s: STerm) -> STerm {
    op(SeqOp::Len, vec![s])
}
fn cat(a: STerm, b: STerm) -> STerm {
    op(SeqOp::Concat, vec![a, b])
}
fn idx(s: STerm, i: STerm) -> STerm {
    op(SeqOp::Index, vec![s, i])
}
fn upd(s: STerm, i: STerm, x: STerm) -> STerm {
    op(SeqOp::Update, vec![s, i, x])
}
fn take(s: STerm, n: STerm) -> STerm {
    op(SeqOp::Take, vec![s, n])
}
fn drop(s: STerm, n: STerm) -> STerm {
    op(SeqOp::Drop, vec![s, n])
}
/// `x in s`.
fn contains(s: STerm, x: STerm) -> STerm {
    op(SeqOp::Contains, vec![s, x])
}
fn range(lo: STerm, hi: STerm) -> STerm {
    op(SeqOp::Range, vec![lo, hi])
}
/// `s == t` of the program, the typed equality (not the axioms' own `==`,
/// which is [`eq`]): only usable as a trigger.
fn seq_eq(s: STerm, t: STerm) -> STerm {
    op(SeqOp::Eq, vec![s, t])
}

/// `undo(shifted, by) == back`: the shifted index of an axiom's conclusion,
/// shifted back, is the index it came from (Dafny's
/// `Seq_add(Seq_sub(n, k), k) == n`). This is not decoration: a solver that
/// matches modulo equalities, not arithmetic, otherwise reads
/// `s[(i - n) + n]` as a new term, and `drop-index1` after `drop-index2` (or
/// `append-index2` after `append-index3`) feeds itself new index terms forever.
fn round_trip(shifted: STerm, by: STerm, back: STerm, undo: fn(STerm, STerm) -> STerm) -> STerm {
    eq(undo(shifted, by), back)
}

/// `0 <= i && i < n`, as two guards.
fn in_range(i: STerm, n: STerm) -> [STerm; 2] {
    [le(int(0), i.clone()), lt(i, n)]
}

/// A sequence axiom over `vars`.
fn seq_ax(name: &'static str, vars: &[VarSort]) -> Ax {
    Ax::new(name, CollKind::Seq, vars)
}

pub(super) fn build() -> Vec<CollAxiom> {
    use VarSort::{Coll as Seq, Elem, Int};
    let (s, t, n) = (v(0), v(1), v(2));
    vec![
        // Start basics
        seq_ax("seq-length-nonneg", &[Seq])
            .trig(vec![len(s.clone())])
            .then([le(int(0), len(s.clone()))])
            .done(),
        seq_ax("seq-empty-length", &[])
            .trig(vec![empty()])
            .then([eq(len(empty()), int(0))])
            .done(),
        seq_ax("seq-length-zero", &[Seq])
            .trig(vec![len(s.clone())])
            .when([eq(len(s.clone()), int(0))])
            .guard_when()
            .then([eq(s.clone(), empty())])
            .done(),
        seq_ax("seq-singleton-length", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(len(unit(v(0))), int(1))])
            .done(),
        seq_ax("seq-append-length", &[Seq, Seq])
            .trig(vec![len(cat(s.clone(), t.clone()))])
            .then([eq(
                len(cat(s.clone(), t.clone())),
                add(len(s.clone()), len(t.clone())),
            )])
            .done(),
        seq_ax("seq-append-empty-left", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(s.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), t.clone())])
            .done(),
        seq_ax("seq-append-empty-right", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(t.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), s.clone())])
            .done(),
        seq_ax("seq-singleton-index", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(idx(unit(v(0)), int(0)), v(0))])
            .done(),
        // End basics
        // Start index-append-update
        seq_ax("seq-append-index1", &[Seq, Seq, Int])
            .trig(vec![idx(cat(s.clone(), t.clone()), n.clone())])
            .trig(vec![idx(s.clone(), n.clone()), cat(s.clone(), t.clone())])
            .when(in_range(n.clone(), len(s.clone())))
            .guard_when()
            .then([eq(
                idx(cat(s.clone(), t.clone()), n.clone()),
                idx(s.clone(), n.clone()),
            )])
            .done(),
        seq_ax("seq-append-index2", &[Seq, Seq, Int])
            .trig(vec![idx(cat(s.clone(), t.clone()), n.clone())])
            .when([
                le(len(s.clone()), n.clone()),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    idx(cat(s.clone(), t.clone()), n.clone()),
                    idx(t.clone(), sub(n.clone(), len(s.clone()))),
                ),
            ])
            .done(),
        seq_ax("seq-append-index3", &[Seq, Seq, Int])
            .trig(vec![idx(t.clone(), n.clone()), cat(s.clone(), t.clone())])
            .when(in_range(n.clone(), len(t.clone())))
            .guard_when()
            .then([
                round_trip(
                    add(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    sub,
                ),
                eq(
                    idx(cat(s.clone(), t.clone()), add(n.clone(), len(s.clone()))),
                    idx(t.clone(), n.clone()),
                ),
            ])
            .done(),
        seq_ax("seq-update-length", &[Seq, Int, Elem])
            .trig(vec![len(upd(s.clone(), v(1), v(2)))])
            .trig(vec![len(s.clone()), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(1), len(s.clone())))
            .guard_when()
            .then([eq(len(upd(s.clone(), v(1), v(2))), len(s.clone()))])
            .done(),
        // `s[i := x][n]`, split on whether `i == n`.
        seq_ax("seq-update-index-same", &[Seq, Int, Elem, Int])
            .trig(vec![idx(upd(s.clone(), v(1), v(2)), v(3))])
            .trig(vec![idx(s.clone(), v(3)), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(3), len(s.clone())))
            .when([eq(v(1), v(3))])
            .guard_when()
            .then([eq(idx(upd(s.clone(), v(1), v(2)), v(3)), v(2))])
            .done(),
        seq_ax("seq-update-index-other", &[Seq, Int, Elem, Int])
            .trig(vec![idx(upd(s.clone(), v(1), v(2)), v(3))])
            .trig(vec![idx(s.clone(), v(3)), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(3), len(s.clone())))
            .when([ne(v(1), v(3))])
            .guard_when()
            .then([eq(
                idx(upd(s.clone(), v(1), v(2)), v(3)),
                idx(s.clone(), v(3)),
            )])
            .done(),
        // End index-append-update
        // Start take/drop. Here `t` is the count `n` (variable 1).
        seq_ax("seq-take-length", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), le(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), t.clone())])
            .done(),
        seq_ax("seq-take-length-over", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        seq_ax("seq-take-length-negative", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), int(0))])
            .done(),
        seq_ax("seq-take-index", &[Seq, Int, Int])
            .trig(vec![idx(take(s.clone(), v(1)), v(2))])
            .trig(vec![idx(s.clone(), v(2)), take(s.clone(), v(1))])
            .when([le(int(0), v(2)), lt(v(2), v(1)), lt(v(2), len(s.clone()))])
            .guard_when()
            .then([eq(idx(take(s.clone(), v(1)), v(2)), idx(s.clone(), v(2)))])
            .done(),
        seq_ax("seq-drop-length", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([le(int(0), t.clone()), le(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                len(drop(s.clone(), t.clone())),
                sub(len(s.clone()), t.clone()),
            )])
            .done(),
        seq_ax("seq-drop-length-over", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), int(0))])
            .done(),
        seq_ax("seq-drop-length-negative", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        seq_ax("seq-drop-index1", &[Seq, Int, Int])
            .trig(vec![idx(drop(s.clone(), v(1)), v(2))])
            .when([
                lt(int(0), v(1)),
                le(int(0), v(2)),
                lt(v(2), sub(len(s.clone()), v(1))),
            ])
            .guard_when()
            .then([
                round_trip(add(v(2), v(1)), v(1), v(2), sub),
                eq(
                    idx(drop(s.clone(), v(1)), v(2)),
                    idx(s.clone(), add(v(2), v(1))),
                ),
            ])
            .done(),
        seq_ax("seq-drop-index2", &[Seq, Int, Int])
            .trig(vec![drop(s.clone(), v(1)), idx(s.clone(), v(2))])
            .when([lt(int(0), v(1)), le(v(1), v(2)), lt(v(2), len(s.clone()))])
            .guard_when()
            .then([
                round_trip(sub(v(2), v(1)), v(1), v(2), add),
                eq(
                    idx(drop(s.clone(), v(1)), sub(v(2), v(1))),
                    idx(s.clone(), v(2)),
                ),
            ])
            .done(),
        seq_ax("seq-append-take1", &[Seq, Seq, Int])
            .trig(vec![take(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                take(cat(s.clone(), t.clone()), n.clone()),
                take(s.clone(), n.clone()),
            )])
            .done(),
        seq_ax("seq-append-take2", &[Seq, Seq, Int])
            .trig(vec![take(cat(s.clone(), t.clone()), n.clone())])
            .when([
                gt(n.clone(), int(0)),
                gt(n.clone(), len(s.clone())),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    take(cat(s.clone(), t.clone()), n.clone()),
                    cat(s.clone(), take(t.clone(), sub(n.clone(), len(s.clone())))),
                ),
            ])
            .done(),
        seq_ax("seq-append-drop1", &[Seq, Seq, Int])
            .trig(vec![drop(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                drop(cat(s.clone(), t.clone()), n.clone()),
                cat(drop(s.clone(), n.clone()), t.clone()),
            )])
            .done(),
        seq_ax("seq-append-drop2", &[Seq, Seq, Int])
            .trig(vec![drop(cat(s.clone(), t.clone()), n.clone())])
            .when([
                gt(n.clone(), int(0)),
                gt(n.clone(), len(s.clone())),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    drop(cat(s.clone(), t.clone()), n.clone()),
                    drop(t.clone(), sub(n.clone(), len(s.clone()))),
                ),
            ])
            .done(),
        // These cut matching loops the axioms above can otherwise introduce, as
        // those create new append terms.
        seq_ax("seq-take-zero", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), empty())])
            .done(),
        seq_ax("seq-drop-zero", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), s.clone())])
            .done(),
        seq_ax("seq-take-all", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), s.clone())])
            .done(),
        seq_ax("seq-drop-all", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), empty())])
            .done(),
        // End take/drop
        // Start contains. Here `t` is the element `x` (variable 1).
        seq_ax("seq-contains-witness", &[Seq, Elem])
            .trig(vec![contains(s.clone(), t.clone())])
            .when([contains(s.clone(), t.clone())])
            .guard_when()
            .then({
                let k = || skolem(Witness::SeqIndexOf, vec![s.clone(), t.clone()]);
                let [lo, hi] = in_range(k(), len(s.clone()));
                [lo, hi, eq(idx(s.clone(), k()), t.clone())]
            })
            .done(),
        seq_ax("seq-contains-index", &[Seq, Elem, Int])
            .trig(vec![
                contains(s.clone(), t.clone()),
                idx(s.clone(), n.clone()),
            ])
            .when(in_range(n.clone(), len(s.clone())))
            .when([eq(idx(s.clone(), n.clone()), t.clone())])
            .guard_when()
            .then([contains(s.clone(), t.clone())])
            .done(),
        // End contains
        // Start equals. `Seq_equal(a, b)` is the program's `SeqOp::Eq`, which
        // shares its e-class with `a == b`, so `Seq_equal_ext` holds by
        // construction and `Seq_equal_def` reduces to its third disjunct:
        // unequal sequences of one length differ at the in-range index
        // `k = Seq_skolem_diff(a, b)`.
        seq_ax("seq-equal-diff", &[Seq, Seq])
            .trig(vec![seq_eq(s.clone(), t.clone())])
            .when([ne(s.clone(), t.clone()), eq(len(s.clone()), len(t.clone()))])
            .guard_when()
            .then({
                let k = || skolem(Witness::SeqDiff, vec![s.clone(), t.clone()]);
                let [lo, hi] = in_range(k(), len(s.clone()));
                [
                    eq(k(), skolem(Witness::SeqDiff, vec![t.clone(), s.clone()])),
                    lo,
                    hi,
                    ne(idx(s.clone(), k()), idx(t.clone(), k())),
                ]
            })
            .done(),
        // The same fact read the other way, which is the one that *proves* an
        // equality: of one length, `a == b` exactly when they agree at `k` (or
        // `k` is out of range, which only happens when they are equal). The
        // implication is spelled as a program's `0 <= i && i < |a| ==> a[i] ==
        // b[i]` lowers, so an instance of such a quantifier at `k` is this term.
        seq_ax("seq-equal-ext", &[Seq, Seq])
            .trig(vec![seq_eq(s.clone(), t.clone())])
            .when([eq(len(s.clone()), len(t.clone()))])
            .guard_when()
            .then({
                let k = || skolem(Witness::SeqDiff, vec![s.clone(), t.clone()]);
                let [lo, hi] = in_range(k(), len(s.clone()));
                [eq(
                    eq(s.clone(), t.clone()),
                    ite(
                        and(lo, hi),
                        eq(idx(s.clone(), k()), idx(t.clone(), k())),
                        STerm::Bool(true),
                    ),
                )]
            })
            .done(),
        // End equals
        // Start extras
        seq_ax("seq-singleton-contains", &[Elem, Elem])
            .trig(vec![contains(unit(v(0)), v(1))])
            .then([eq(contains(unit(v(0)), v(1)), eq(v(0), v(1)))])
            .done(),
        seq_ax("seq-range-length", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([lt(v(0), v(1))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), sub(v(1), v(0)))])
            .int_only(),
        seq_ax("seq-range-length-empty", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([le(v(1), v(0))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), int(0))])
            .int_only(),
        seq_ax("seq-range-index", &[Int, Int, Int])
            .trig(vec![idx(range(v(0), v(1)), v(2))])
            .when(in_range(v(2), sub(v(1), v(0))))
            .guard_when()
            .then([eq(idx(range(v(0), v(1)), v(2)), add(v(0), v(2)))])
            .int_only(),
        seq_ax("seq-range-contains", &[Int, Int, Int])
            .trig(vec![contains(range(v(0), v(1)), v(2))])
            .then([eq(
                contains(range(v(0), v(1)), v(2)),
                and(le(v(0), v(2)), lt(v(2), v(1))),
            )])
            .int_only(),
        // End extras
    ]
}
