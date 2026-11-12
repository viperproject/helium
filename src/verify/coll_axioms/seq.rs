//! The axioms of `Seq[T]`, adapted from Silicon's
//! `dafny_axioms/sequences.vpr` (see the [module docs](super)). The comment
//! above each axiom says what it states, in words and as the Viper formula,
//! with its triggers.
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
        // A sequence never has negative length.
        //   forall s :: {|s|} 0 <= |s|
        seq_ax("seq-length-nonneg", &[Seq])
            .trig(vec![len(s.clone())])
            .then([le(int(0), len(s.clone()))])
            .done(),
        // The empty sequence has length 0.
        //   |Seq()| == 0
        seq_ax("seq-empty-length", &[])
            .trig(vec![empty()])
            .then([eq(len(empty()), int(0))])
            .done(),
        // Only the empty sequence has length 0.
        //   forall s :: {|s|} |s| == 0 ==> s == Seq()
        seq_ax("seq-length-zero", &[Seq])
            .trig(vec![len(s.clone())])
            .when([eq(len(s.clone()), int(0))])
            .guard_when()
            .then([eq(s.clone(), empty())])
            .done(),
        // A singleton has length 1.
        //   forall x :: {Seq(x)} |Seq(x)| == 1
        seq_ax("seq-singleton-length", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(len(unit(v(0))), int(1))])
            .done(),
        // Concatenation adds the lengths.
        //   forall s, t :: {|s ++ t|} |s ++ t| == |s| + |t|
        seq_ax("seq-append-length", &[Seq, Seq])
            .trig(vec![len(cat(s.clone(), t.clone()))])
            .then([eq(
                len(cat(s.clone(), t.clone())),
                add(len(s.clone()), len(t.clone())),
            )])
            .done(),
        // Prepending the empty sequence changes nothing.
        //   forall s, t :: {s ++ t} s == Seq() ==> s ++ t == t
        seq_ax("seq-append-empty-left", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(s.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), t.clone())])
            .done(),
        // Appending the empty sequence changes nothing.
        //   forall s, t :: {s ++ t} t == Seq() ==> s ++ t == s
        seq_ax("seq-append-empty-right", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(t.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), s.clone())])
            .done(),
        // The one element of a singleton is at index 0.
        //   forall x :: {Seq(x)} Seq(x)[0] == x
        seq_ax("seq-singleton-index", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(idx(unit(v(0)), int(0)), v(0))])
            .done(),
        // End basics
        // Start index-append-update
        // An index into the left part of `s ++ t` reads `s`.
        //   forall s, t, n :: {(s ++ t)[n]} {s[n], s ++ t}
        //     0 <= n && n < |s| ==> (s ++ t)[n] == s[n]
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
        // An index into the right part of `s ++ t` reads `t`, shifted back by
        // `|s|`.
        //   forall s, t, n :: {(s ++ t)[n]}
        //     |s| <= n && n < |s ++ t| ==> (s ++ t)[n] == t[n - |s|]
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
        // The same from `t`'s side: `t[n]` is found `|s|` further on in `s ++ t`.
        //   forall s, t, n :: {t[n], s ++ t}
        //     0 <= n && n < |t| ==> (s ++ t)[n + |s|] == t[n]
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
        // An in-range update keeps the length.
        //   forall s, i, x :: {|s[i := x]|} {|s|, s[i := x]}
        //     0 <= i && i < |s| ==> |s[i := x]| == |s|
        seq_ax("seq-update-length", &[Seq, Int, Elem])
            .trig(vec![len(upd(s.clone(), v(1), v(2)))])
            .trig(vec![len(s.clone()), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(1), len(s.clone())))
            .guard_when()
            .then([eq(len(upd(s.clone(), v(1), v(2))), len(s.clone()))])
            .done(),
        // `s[i := x][n]`, split on whether `i == n`.
        // Reading the updated index gives the new element.
        //   forall s, i, x, n :: {s[i := x][n]} {s[n], s[i := x]}
        //     0 <= n && n < |s| && i == n ==> s[i := x][n] == x
        seq_ax("seq-update-index-same", &[Seq, Int, Elem, Int])
            .trig(vec![idx(upd(s.clone(), v(1), v(2)), v(3))])
            .trig(vec![idx(s.clone(), v(3)), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(3), len(s.clone())))
            .when([eq(v(1), v(3))])
            .guard_when()
            .then([eq(idx(upd(s.clone(), v(1), v(2)), v(3)), v(2))])
            .done(),
        // Reading any other index gives the old element.
        //   forall s, i, x, n :: {s[i := x][n]} {s[n], s[i := x]}
        //     0 <= n && n < |s| && i != n ==> s[i := x][n] == s[n]
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
        // Taking `n` elements, with `n` in range, gives length `n`.
        //   forall s, n :: {|s[..n]|} {s[..n], |s|} 0 <= n && n <= |s| ==> |s[..n]| == n
        seq_ax("seq-take-length", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), le(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), t.clone())])
            .done(),
        // Taking more than there are gives all of `s` (Viper clamps `n`).
        //   forall s, n :: {|s[..n]|} {s[..n], |s|} 0 <= n && |s| < n ==> |s[..n]| == |s|
        seq_ax("seq-take-length-over", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        // Taking a negative number of elements gives nothing.
        //   forall s, n :: {|s[..n]|} {s[..n], |s|} n < 0 ==> |s[..n]| == 0
        seq_ax("seq-take-length-negative", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), int(0))])
            .done(),
        // The elements kept by a take are those of `s`.
        //   forall s, n, j :: {s[..n][j]} {s[j], s[..n]}
        //     0 <= j && j < n && j < |s| ==> s[..n][j] == s[j]
        seq_ax("seq-take-index", &[Seq, Int, Int])
            .trig(vec![idx(take(s.clone(), v(1)), v(2))])
            .trig(vec![idx(s.clone(), v(2)), take(s.clone(), v(1))])
            .when([le(int(0), v(2)), lt(v(2), v(1)), lt(v(2), len(s.clone()))])
            .guard_when()
            .then([eq(idx(take(s.clone(), v(1)), v(2)), idx(s.clone(), v(2)))])
            .done(),
        // Dropping `n` elements, with `n` in range, removes `n` from the length.
        //   forall s, n :: {|s[n..]|} {|s|, s[n..]} 0 <= n && n <= |s| ==> |s[n..]| == |s| - n
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
        // Dropping more than there are leaves nothing.
        //   forall s, n :: {|s[n..]|} {|s|, s[n..]} 0 <= n && |s| < n ==> |s[n..]| == 0
        seq_ax("seq-drop-length-over", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), int(0))])
            .done(),
        // Dropping a negative number of elements keeps all of `s`.
        //   forall s, n :: {|s[n..]|} {|s|, s[n..]} n < 0 ==> |s[n..]| == |s|
        seq_ax("seq-drop-length-negative", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        // Index `j` after dropping `n` is index `j + n` of `s`.
        //   forall s, n, j :: {s[n..][j]}
        //     0 < n && 0 <= j && j < |s| - n ==> s[n..][j] == s[j + n]
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
        // The same from `s`'s side: `s[j]` is found `n` earlier after the drop.
        //   forall s, n, j :: {s[n..], s[j]}
        //     0 < n && n <= j && j < |s| ==> s[n..][j - n] == s[j]
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
        // Taking no more than `|s|` from `s ++ t` only takes from `s`.
        //   forall s, t, n :: {(s ++ t)[..n]}
        //     0 < n && n <= |s| ==> (s ++ t)[..n] == s[..n]
        seq_ax("seq-append-take1", &[Seq, Seq, Int])
            .trig(vec![take(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                take(cat(s.clone(), t.clone()), n.clone()),
                take(s.clone(), n.clone()),
            )])
            .done(),
        // Taking past `|s|` from `s ++ t` takes all of `s` and the rest from `t`.
        //   forall s, t, n :: {(s ++ t)[..n]}
        //     0 < n && |s| < n && n < |s ++ t| ==> (s ++ t)[..n] == s ++ t[..n - |s|]
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
        // Dropping no more than `|s|` from `s ++ t` only drops from `s`.
        //   forall s, t, n :: {(s ++ t)[n..]}
        //     0 < n && n <= |s| ==> (s ++ t)[n..] == s[n..] ++ t
        seq_ax("seq-append-drop1", &[Seq, Seq, Int])
            .trig(vec![drop(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                drop(cat(s.clone(), t.clone()), n.clone()),
                cat(drop(s.clone(), n.clone()), t.clone()),
            )])
            .done(),
        // Dropping past `|s|` from `s ++ t` drops all of `s` and the rest from `t`.
        //   forall s, t, n :: {(s ++ t)[n..]}
        //     0 < n && |s| < n && n < |s ++ t| ==> (s ++ t)[n..] == t[n - |s|..]
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
        // Taking zero or fewer elements gives the empty sequence.
        //   forall s, n :: {s[..n]} n <= 0 ==> s[..n] == Seq()
        seq_ax("seq-take-zero", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), empty())])
            .done(),
        // Dropping zero or fewer elements keeps `s`.
        //   forall s, n :: {s[n..]} n <= 0 ==> s[n..] == s
        seq_ax("seq-drop-zero", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), s.clone())])
            .done(),
        // Taking at least `|s|` elements keeps `s`.
        //   forall s, n :: {s[..n]} n >= |s| ==> s[..n] == s
        seq_ax("seq-take-all", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), s.clone())])
            .done(),
        // Dropping at least `|s|` elements gives the empty sequence.
        //   forall s, n :: {s[n..]} n >= |s| ==> s[n..] == Seq()
        seq_ax("seq-drop-all", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), empty())])
            .done(),
        // End take/drop
        // Start contains. Here `t` is the element `x` (variable 1).
        // A member of `s` sits at some in-range index, the witness
        // `k = Seq_skolem(s, x)` (Dafny's `exists i :: s[i] == x`).
        //   forall s, x :: {x in s} x in s ==> 0 <= k && k < |s| && s[k] == x
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
        // An element at an in-range index is a member.
        //   forall s, x, n :: {x in s, s[n]}
        //     0 <= n && n < |s| && s[n] == x ==> x in s
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
        // Two unequal sequences of one length differ at the in-range index
        // `k = Seq_skolem_diff(s, t)`, the same witness either way round.
        //   forall s, t :: {s == t}
        //     s != t && |s| == |t| ==>
        //       k == Seq_skolem_diff(t, s) && 0 <= k && k < |s| && s[k] != t[k]
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
        // equality: of one length, `s == t` exactly when they agree at `k` (or
        // `k` is out of range, which only happens when they are equal). The
        // implication is spelled as a program's `0 <= i && i < |s| ==> s[i] ==
        // t[i]` lowers, so an instance of such a quantifier at `k` is this term.
        //   forall s, t :: {s == t}
        //     |s| == |t| ==> (s == t <==> (0 <= k && k < |s| ==> s[k] == t[k]))
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
        // A singleton contains exactly its element.
        //   forall x, y :: {y in Seq(x)} (y in Seq(x)) <==> x == y
        seq_ax("seq-singleton-contains", &[Elem, Elem])
            .trig(vec![contains(unit(v(0)), v(1))])
            .then([eq(contains(unit(v(0)), v(1)), eq(v(0), v(1)))])
            .done(),
        // A non-empty range `[a..b)` has `b - a` elements.
        //   forall a, b :: {|[a..b)|} a < b ==> |[a..b)| == b - a
        seq_ax("seq-range-length", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([lt(v(0), v(1))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), sub(v(1), v(0)))])
            .int_only(),
        // A range whose upper bound is not above its lower bound is empty.
        //   forall a, b :: {|[a..b)|} b <= a ==> |[a..b)| == 0
        seq_ax("seq-range-length-empty", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([le(v(1), v(0))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), int(0))])
            .int_only(),
        // Index `j` of `[a..b)` is `a + j`.
        //   forall a, b, j :: {[a..b)[j]} 0 <= j && j < b - a ==> [a..b)[j] == a + j
        seq_ax("seq-range-index", &[Int, Int, Int])
            .trig(vec![idx(range(v(0), v(1)), v(2))])
            .when(in_range(v(2), sub(v(1), v(0))))
            .guard_when()
            .then([eq(idx(range(v(0), v(1)), v(2)), add(v(0), v(2)))])
            .int_only(),
        // The members of `[a..b)` are the integers from `a` up to, not including,
        // `b`.
        //   forall a, b, x :: {x in [a..b)} (x in [a..b)) <==> (a <= x && x < b)
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
