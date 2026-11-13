//! The axioms of `Multiset[T]`, adapted from Silicon's
//! `dafny_axioms/multisets.vpr` (see the [module docs](super)). Each keeps its
//! Dafny name, `MS_x_y` as `multiset-x-y`, and its triggers. The comment above
//! each axiom says what it states, in words and as the Viper formula, with
//! `x in m` the count of `x` in `m` (Silicon's `Multiset_count`).
//!
//! Which Dafny axiom became which:
//!
//! | Dafny (`multisets.vpr`)              | here                                                    |
//! |--------------------------------------|---------------------------------------------------------|
//! | `MS_count_nonneg`                    | `multiset-count-nonneg`                                 |
//! | `MS_card_nonneg`                     | `multiset-card-nonneg`                                  |
//! | `MS_empty_count`                     | `multiset-empty-count`                                  |
//! | —                                    | `multiset-empty-card`                                   |
//! | `MS_card_zero`                       | `multiset-card-zero`, `multiset-card-nonzero`           |
//! | `MS_singleton_count`                 | `multiset-singleton-count1`, `multiset-singleton-count2`|
//! | `MS_singleton_card`                  | `multiset-singleton-card`                               |
//! | `MS_unionone_empty`                  | `multiset-unionone-empty`                               |
//! | `MS_unionone_count`                  | `multiset-unionone-count`                               |
//! | `MS_unionone_card`                   | `multiset-unionone-card`                                |
//! | `MS_unionone_positive`               | `multiset-unionone-positive`                            |
//! | `MS_union_count`                     | `multiset-union-count`                                  |
//! | `MS_union_card`                      | `multiset-union-card`                                   |
//! | `MS_intersect_count`                 | `multiset-intersection-count`                           |
//! | `MS_intersect_absorb1`, `_absorb2`   | `multiset-intersection-absorb1`, `-absorb2`             |
//! | `MS_diff_count1`, `MS_diff_count2`   | `multiset-diff-count1`, `multiset-diff-count2`          |
//! | `MS_diff_card`                       | `multiset-diff-card1`, `multiset-diff-card2`            |
//! | `MS_subset_def`                      | `multiset-subset-count`, `multiset-subset-witness`      |
//! | `MS_equal_def`, `MS_equal_ext`       | `multiset-equal-diff`, `multiset-equal-ext`             |
//! | `MS_disjoint_def`                    | left out, see below                                     |
//!
//! Where the domain differs:
//! - `Multiset_count(m, x)` is `MultisetOp::Count` with the multiset first, as
//!   `x in m` is lowered.
//! - `Multiset_unionone` is `MultisetOp::Add`, which only a literal builds:
//!   `Multiset(a, b)` is `Multiset(a)` with `b` added, as Silicon encodes it.
//! - `Math_min` and `Math_clip`, the domain's macros, are written out as the
//!   conditionals they expand to.
//! - The existential of `MS_card_zero` is the witness
//!   [`Witness::MultisetMember`], and the nested `forall` of `MS_subset_def`
//!   is skolemized by [`Witness::MultisetSubset`], a counterexample to the
//!   inclusion, as for sets: `a subset b` is `x in a <= x in b` at that one
//!   element (`multiset-subset-witness`), and an inclusion that holds bounds
//!   every count (`multiset-subset-count`, the nested quantifier with its own
//!   triggers).
//! - `Multiset_equal` is the program's typed equality, as for sets: it shares
//!   its class with `a == b`, so `MS_equal_ext` holds by construction and
//!   `MS_equal_def` becomes `multiset-equal-diff` and `multiset-equal-ext`
//!   around the witness [`Witness::MultisetDiff`] (`Multiset_skolem_diff`).
//! - `Multiset_disjoint` has no axioms here: Viper has no syntax for it, so no
//!   program term can trigger them.
//! - `multiset-empty-card` (`|Multiset()| == 0`) is not in the domain, which
//!   derives it from `MS_card_zero`; the e-graph cannot, and the set table has
//!   the same fact (`set-empty-card`). With it, `MS_card_zero`'s `<==` half
//!   (`m == Multiset() ==> |m| == 0`) follows by congruence, so it has no
//!   axiom of its own; its `==>` half is `multiset-card-zero`.
//! - `MS_diff_card`'s `2 * |a intersection b|` is a sum of the term with
//!   itself, so the axiom stays linear.
//! - Each conjunct of `MS_card_zero`, `MS_singleton_count` and
//!   `MS_diff_card` is an axiom of its own.

use super::*;
use crate::vmir::MultisetOp;

fn card(m: STerm) -> STerm {
    op(MultisetOp::Card, vec![m])
}
/// `x in m`, the count of `x` in `m`.
fn count(m: STerm, x: STerm) -> STerm {
    op(MultisetOp::Count, vec![m, x])
}
fn union(a: STerm, b: STerm) -> STerm {
    op(MultisetOp::Union, vec![a, b])
}
fn inter(a: STerm, b: STerm) -> STerm {
    op(MultisetOp::Intersection, vec![a, b])
}
fn diff(a: STerm, b: STerm) -> STerm {
    op(MultisetOp::Difference, vec![a, b])
}
fn subset(a: STerm, b: STerm) -> STerm {
    op(MultisetOp::Subset, vec![a, b])
}
/// `Multiset_unionone(m, x)`, i.e. `m union Multiset(x)`.
fn add_one(m: STerm, x: STerm) -> STerm {
    op(MultisetOp::Add, vec![m, x])
}
/// `m == n` of the program, the typed equality (not the axioms' own `==`,
/// which is [`eq`]): only usable as a trigger.
fn ms_eq(m: STerm, n: STerm) -> STerm {
    op(MultisetOp::Eq, vec![m, n])
}
/// `Math_min(a, b)`, i.e. `a <= b ? a : b`.
fn min(a: STerm, b: STerm) -> STerm {
    ite(le(a.clone(), b.clone()), a, b)
}
/// `Math_clip(a)`, i.e. `a < 0 ? 0 : a`.
fn clip(a: STerm) -> STerm {
    ite(lt(a.clone(), int(0)), int(0), a)
}

/// A multiset axiom over `vars`.
fn ms_ax(name: &'static str, vars: &[VarSort]) -> Ax {
    Ax::new(name, CollKind::Multiset, vars)
}

pub(super) fn build() -> Vec<CollAxiom> {
    use VarSort::{Coll as Ms, Elem};
    // `a`, `b` are the multisets (variables 0 and 1); the element variables
    // are numbered per axiom.
    let (a, b) = (v(0), v(1));
    vec![
        // Start count, cardinality and the empty multiset
        //
        // No element occurs a negative number of times.
        //   forall ms, x :: {x in ms} (x in ms) >= 0
        ms_ax("multiset-count-nonneg", &[Ms, Elem])
            .trig(vec![count(a.clone(), v(1))])
            .then([ge(count(a.clone(), v(1)), int(0))])
            .done(),
        // A multiset never has negative cardinality.
        //   forall s :: {|s|} 0 <= |s|
        ms_ax("multiset-card-nonneg", &[Ms])
            .trig(vec![card(a.clone())])
            .then([le(int(0), card(a.clone()))])
            .done(),
        // Nothing occurs in the empty multiset.
        //   forall o :: {o in Multiset()} (o in Multiset()) == 0
        ms_ax("multiset-empty-count", &[Elem])
            .trig(vec![count(empty(), v(0))])
            .then([eq(count(empty(), v(0)), int(0))])
            .done(),
        // The empty multiset has cardinality 0. (Not in the Dafny domain; see
        // the module docs.)
        //   |Multiset()| == 0
        ms_ax("multiset-empty-card", &[])
            .trig(vec![empty()])
            .then([eq(card(empty()), int(0))])
            .done(),
        // Only the empty multiset has cardinality 0.
        //   forall s :: {|s|} |s| == 0 ==> s == Multiset()
        ms_ax("multiset-card-zero", &[Ms])
            .trig(vec![card(a.clone())])
            .when([eq(card(a.clone()), int(0))])
            .guard_when()
            .then([eq(a.clone(), empty())])
            .done(),
        // A multiset of nonzero cardinality has an element that occurs in it,
        // the witness `Multiset_skolem_member(s)` (Dafny's
        // `exists x :: 0 < x in s`).
        //   forall s :: {|s|} |s| != 0 ==> 0 < (Multiset_skolem_member(s) in s)
        ms_ax("multiset-card-nonzero", &[Ms])
            .trig(vec![card(a.clone())])
            .when([ne(card(a.clone()), int(0))])
            .guard_when()
            .then([lt(
                int(0),
                count(a.clone(), skolem(Witness::MultisetMember, vec![a.clone()])),
            )])
            .done(),
        // End count, cardinality and the empty multiset
        // Start singleton. `r` is variable 0, `o` variable 1.
        //
        // An element occurs once in a singleton exactly when it is the
        // singleton's element...
        //   forall r, o :: {o in Multiset(r)} ((o in Multiset(r)) == 1) <==> r == o
        ms_ax("multiset-singleton-count1", &[Elem, Elem])
            .trig(vec![count(unit(v(0)), v(1))])
            .then([eq(eq(count(unit(v(0)), v(1)), int(1)), eq(v(0), v(1)))])
            .done(),
        // ...and not at all exactly when it is not.
        //   forall r, o :: {o in Multiset(r)} ((o in Multiset(r)) == 0) <==> r != o
        ms_ax("multiset-singleton-count2", &[Elem, Elem])
            .trig(vec![count(unit(v(0)), v(1))])
            .then([eq(eq(count(unit(v(0)), v(1)), int(0)), ne(v(0), v(1)))])
            .done(),
        // A singleton has cardinality 1, and its element occurs once.
        //   forall r :: {Multiset(r)} |Multiset(r)| == 1 && (r in Multiset(r)) == 1
        ms_ax("multiset-singleton-card", &[Elem])
            .trig(vec![unit(v(0))])
            .then([
                eq(card(unit(v(0))), int(1)),
                eq(count(unit(v(0)), v(0)), int(1)),
            ])
            .done(),
        // A singleton is its element added to the empty multiset.
        //   forall r :: {Multiset(r)} Multiset(r) == add(Multiset(), r)
        ms_ax("multiset-unionone-empty", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(unit(v(0)), add_one(empty(), v(0)))])
            .done(),
        // End singleton
        // Start unionone (`a` with one more `x`, how a literal is built). `x`
        // is variable 1, `o` variable 2.
        //
        // Adding `x` to `a` adds one occurrence of `x` and leaves every other
        // count alone.
        //   forall a, x, o :: {o in add(a, x)} {add(a, x), o in a}
        //     (o in add(a, x)) == (x == o ? (o in a) + 1 : (o in a))
        ms_ax("multiset-unionone-count", &[Ms, Elem, Elem])
            .trig(vec![count(add_one(a.clone(), v(1)), v(2))])
            .trig(vec![add_one(a.clone(), v(1)), count(a.clone(), v(2))])
            .then([eq(
                count(add_one(a.clone(), v(1)), v(2)),
                ite(
                    eq(v(1), v(2)),
                    add(count(a.clone(), v(2)), int(1)),
                    count(a.clone(), v(2)),
                ),
            )])
            .done(),
        // Adding an element raises the cardinality by one, whether or not it
        // already occurs.
        //   forall a, x :: {|add(a, x)|} {add(a, x), |a|} |add(a, x)| == |a| + 1
        ms_ax("multiset-unionone-card", &[Ms, Elem])
            .trig(vec![card(add_one(a.clone(), v(1)))])
            .trig(vec![add_one(a.clone(), v(1)), card(a.clone())])
            .then([eq(
                card(add_one(a.clone(), v(1))),
                add(card(a.clone()), int(1)),
            )])
            .done(),
        // The added element occurs, so the result is not empty.
        //   forall a, x :: {add(a, x)} (x in add(a, x)) > 0 && |add(a, x)| > 0
        ms_ax("multiset-unionone-positive", &[Ms, Elem])
            .trig(vec![add_one(a.clone(), v(1))])
            .then([
                gt(count(add_one(a.clone(), v(1)), v(1)), int(0)),
                gt(card(add_one(a.clone(), v(1))), int(0)),
            ])
            .done(),
        // End unionone
        // Start union and intersection. `o` is variable 2.
        //
        // A union adds the counts of both sides.
        //   forall a, b, o :: {o in a union b} {a union b, o in a, o in b}
        //     (o in a union b) == (o in a) + (o in b)
        ms_ax("multiset-union-count", &[Ms, Ms, Elem])
            .trig(vec![count(union(a.clone(), b.clone()), v(2))])
            .trig(vec![
                union(a.clone(), b.clone()),
                count(a.clone(), v(2)),
                count(b.clone(), v(2)),
            ])
            .then([eq(
                count(union(a.clone(), b.clone()), v(2)),
                add(count(a.clone(), v(2)), count(b.clone(), v(2))),
            )])
            .done(),
        // So it adds their cardinalities too.
        //   forall a, b :: {|a union b|} {|a|, a union b} {|b|, a union b}
        //     |a union b| == |a| + |b|
        ms_ax("multiset-union-card", &[Ms, Ms])
            .trig(vec![card(union(a.clone(), b.clone()))])
            .trig(vec![card(a.clone()), union(a.clone(), b.clone())])
            .trig(vec![card(b.clone()), union(a.clone(), b.clone())])
            .then([eq(
                card(union(a.clone(), b.clone())),
                add(card(a.clone()), card(b.clone())),
            )])
            .done(),
        // An intersection takes the smaller count of the two sides.
        //   forall a, b, o :: {o in a intersection b}
        //     (o in a intersection b) == min(o in a, o in b)
        ms_ax("multiset-intersection-count", &[Ms, Ms, Elem])
            .trig(vec![count(inter(a.clone(), b.clone()), v(2))])
            .then([eq(
                count(inter(a.clone(), b.clone()), v(2)),
                min(count(a.clone(), v(2)), count(b.clone(), v(2))),
            )])
            .done(),
        // Intersecting with `b` again changes nothing.
        //   forall a, b :: {(a intersection b) intersection b}
        //     (a intersection b) intersection b == a intersection b
        ms_ax("multiset-intersection-absorb1", &[Ms, Ms])
            .trig(vec![inter(inter(a.clone(), b.clone()), b.clone())])
            .then([eq(
                inter(inter(a.clone(), b.clone()), b.clone()),
                inter(a.clone(), b.clone()),
            )])
            .done(),
        // Intersecting with `a` again changes nothing.
        //   forall a, b :: {a intersection (a intersection b)}
        //     a intersection (a intersection b) == a intersection b
        ms_ax("multiset-intersection-absorb2", &[Ms, Ms])
            .trig(vec![inter(a.clone(), inter(a.clone(), b.clone()))])
            .then([eq(
                inter(a.clone(), inter(a.clone(), b.clone())),
                inter(a.clone(), b.clone()),
            )])
            .done(),
        // End union and intersection
        // Start difference. `o`/`y` is variable 2.
        //
        // `a setminus b` takes away the count in `b`, down to zero.
        //   forall a, b, o :: {o in a setminus b}
        //     (o in a setminus b) == clip((o in a) - (o in b))
        ms_ax("multiset-diff-count1", &[Ms, Ms, Elem])
            .trig(vec![count(diff(a.clone(), b.clone()), v(2))])
            .then([eq(
                count(diff(a.clone(), b.clone()), v(2)),
                clip(sub(count(a.clone(), v(2)), count(b.clone(), v(2)))),
            )])
            .done(),
        // An element `b` has at least as often as `a` is gone from
        // `a setminus b`. (Triggered on both counts, so it reaches a
        // difference no count of `y` mentions yet.)
        //   forall a, b, y :: {a setminus b, y in b, y in a}
        //     (y in a) <= (y in b) ==> (y in a setminus b) == 0
        ms_ax("multiset-diff-count2", &[Ms, Ms, Elem])
            .trig(vec![
                diff(a.clone(), b.clone()),
                count(b.clone(), v(2)),
                count(a.clone(), v(2)),
            ])
            .when([le(count(a.clone(), v(2)), count(b.clone(), v(2)))])
            .guard_when()
            .then([eq(count(diff(a.clone(), b.clone()), v(2)), int(0))])
            .done(),
        // The union splits into what only `a` has, what only `b` has, and
        // what both have, which the union counts twice.
        //   forall a, b :: {|a setminus b|}
        //     |a setminus b| + |b setminus a| + 2 * |a intersection b| == |a union b|
        ms_ax("multiset-diff-card1", &[Ms, Ms])
            .trig(vec![card(diff(a.clone(), b.clone()))])
            .then([eq(
                add(
                    add(
                        card(diff(a.clone(), b.clone())),
                        card(diff(b.clone(), a.clone())),
                    ),
                    add(
                        card(inter(a.clone(), b.clone())),
                        card(inter(a.clone(), b.clone())),
                    ),
                ),
                card(union(a.clone(), b.clone())),
            )])
            .done(),
        // Taking `b` away from `a` takes away exactly what they share.
        //   forall a, b :: {|a setminus b|} |a setminus b| == |a| - |a intersection b|
        ms_ax("multiset-diff-card2", &[Ms, Ms])
            .trig(vec![card(diff(a.clone(), b.clone()))])
            .then([eq(
                card(diff(a.clone(), b.clone())),
                sub(card(a.clone()), card(inter(a.clone(), b.clone()))),
            )])
            .done(),
        // End difference
        // Start subset. Dafny's `MS_subset_def`,
        //   a subset b <==> forall o :: {o in a} {o in b} (o in a) <= (o in b),
        // skolemized into two axioms (see the module docs). `o` is variable 2.
        //
        // When `a subset b` holds, every count of `a` is at most that of `b`
        // (the nested quantifier, with its triggers).
        //   forall a, b, o :: {a subset b, o in a} {a subset b, o in b}
        //     a subset b ==> (o in a) <= (o in b)
        ms_ax("multiset-subset-count", &[Ms, Ms, Elem])
            .trig(vec![subset(a.clone(), b.clone()), count(a.clone(), v(2))])
            .trig(vec![subset(a.clone(), b.clone()), count(b.clone(), v(2))])
            .when([subset(a.clone(), b.clone())])
            .guard_when()
            .then([le(count(a.clone(), v(2)), count(b.clone(), v(2)))])
            .done(),
        // `a subset b` holds exactly when the one candidate counterexample
        // `w = Multiset_skolem_subset(a, b)` (an element `a` has more often
        // than `b`, if there is one) is not a counterexample. This is the
        // direction that proves an inclusion.
        //   forall a, b :: {a subset b} a subset b <==> (w in a) <= (w in b)
        ms_ax("multiset-subset-witness", &[Ms, Ms])
            .trig(vec![subset(a.clone(), b.clone())])
            .then({
                let w = || skolem(Witness::MultisetSubset, vec![a.clone(), b.clone()]);
                [eq(
                    subset(a.clone(), b.clone()),
                    le(count(a.clone(), w()), count(b.clone(), w())),
                )]
            })
            .done(),
        // End subset
        // Start equals. `Multiset_equal(a, b)` is the program's
        // `MultisetOp::Eq`, which shares its e-class with `a == b`, so
        // `MS_equal_ext` holds by construction and `MS_equal_def` reduces to
        // its second disjunct.
        //
        // Two unequal multisets differ in the count of the element
        // `d = Multiset_skolem_diff(a, b)`, which is the same witness either
        // way round.
        //   forall a, b :: {a == b}
        //     a != b ==> d == Multiset_skolem_diff(b, a) && (d in a) != (d in b)
        ms_ax("multiset-equal-diff", &[Ms, Ms])
            .trig(vec![ms_eq(a.clone(), b.clone())])
            .when([ne(a.clone(), b.clone())])
            .guard_when()
            .then({
                let d = || skolem(Witness::MultisetDiff, vec![a.clone(), b.clone()]);
                [
                    eq(
                        d(),
                        skolem(Witness::MultisetDiff, vec![b.clone(), a.clone()]),
                    ),
                    ne(count(a.clone(), d()), count(b.clone(), d())),
                ]
            })
            .done(),
        // The same fact read the other way, which is the one that *proves* an
        // equality: two multisets are equal exactly when they agree on the
        // count of `d`.
        //   forall a, b :: {a == b} a == b <==> ((d in a) == (d in b))
        ms_ax("multiset-equal-ext", &[Ms, Ms])
            .trig(vec![ms_eq(a.clone(), b.clone())])
            .then({
                let d = || skolem(Witness::MultisetDiff, vec![a.clone(), b.clone()]);
                [eq(
                    eq(a.clone(), b.clone()),
                    eq(count(a.clone(), d()), count(b.clone(), d())),
                )]
            })
            .done(),
        // End equals
    ]
}
