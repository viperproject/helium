//! The axioms of `Set[T]`, adapted from Silicon's `dafny_axioms/sets.vpr`
//! (see the [module docs](super)). Each keeps its Dafny name, `Set_x_y` as
//! `set-x-y`, and its triggers. The comment above each axiom says what it
//! states, in words and as the Viper formula, with `in` the set membership.
//!
//! Where the domain differs:
//! - `Set_in(e, s)` is `SetOp::Contains` with the set first, as `x in s` is
//!   lowered.
//! - `Set_unionone` is `SetOp::Add`, which only a literal builds:
//!   `Set(a, b)` is `Set(a)` with `b` added, as Silicon encodes it.
//! - The existential of `Set_card_zero` is the witness
//!   [`Witness::SetMember`], and the nested `forall` of `Set_subset_def` is
//!   skolemized by [`Witness::SetSubset`], a counterexample to the
//!   inclusion: `a subset b` is `x in a ==> x in b` at that one element
//!   (`set-subset-witness`), and an inclusion that holds carries every member
//!   over (`set-subset-member`, the nested quantifier with its own triggers).
//! - `Set_equal` is the program's typed equality, as for sequences: it shares
//!   its class with `a == b`, so `Set_equal_ext` holds by construction and
//!   `Set_equal_def` becomes `set-equal-diff` and `set-equal-ext`.
//! - `set-empty-card` (`|Set()| == 0`) is not in the domain, which derives it
//!   from `Set_card_zero`'s existential; the e-graph cannot, and the sequence
//!   table has the same fact (`seq-empty-length`).
//! - Each conjunct of `Set_card_zero` and `Set_diff_card` is an axiom of its
//!   own.

use super::*;
use crate::vmir::SetOp;

fn card(s: STerm) -> STerm {
    op(SetOp::Card, vec![s])
}
/// `x in s`.
fn contains(s: STerm, x: STerm) -> STerm {
    op(SetOp::Contains, vec![s, x])
}
fn union(a: STerm, b: STerm) -> STerm {
    op(SetOp::Union, vec![a, b])
}
fn inter(a: STerm, b: STerm) -> STerm {
    op(SetOp::Intersection, vec![a, b])
}
fn diff(a: STerm, b: STerm) -> STerm {
    op(SetOp::Difference, vec![a, b])
}
fn subset(a: STerm, b: STerm) -> STerm {
    op(SetOp::Subset, vec![a, b])
}
/// `Set_unionone(s, x)`, i.e. `s union Set(x)`.
fn add_one(s: STerm, x: STerm) -> STerm {
    op(SetOp::Add, vec![s, x])
}
/// `s == t` of the program, the typed equality (not the axioms' own `==`,
/// which is [`eq`]): only usable as a trigger.
fn set_eq(s: STerm, t: STerm) -> STerm {
    op(SetOp::Eq, vec![s, t])
}

/// A set axiom over `vars`.
fn set_ax(name: &'static str, vars: &[VarSort]) -> Ax {
    Ax::new(name, CollKind::Set, vars)
}

pub(super) fn build() -> Vec<CollAxiom> {
    use VarSort::{Coll as Set, Elem};
    // `a`, `b` are the sets (variables 0 and 1); the element variables are
    // numbered per axiom.
    let (a, b) = (v(0), v(1));
    vec![
        // Start cardinality and the empty set
        //
        // A set never has negative cardinality.
        //   forall a :: {|a|} 0 <= |a|
        set_ax("set-card-nonneg", &[Set])
            .trig(vec![card(a.clone())])
            .then([le(int(0), card(a.clone()))])
            .done(),
        // The empty set has cardinality 0. (Not in the Dafny domain; see the
        // module docs.)
        //   |Set()| == 0
        set_ax("set-empty-card", &[])
            .trig(vec![empty()])
            .then([eq(card(empty()), int(0))])
            .done(),
        // Nothing is a member of the empty set.
        //   forall o :: {o in Set()} !(o in Set())
        set_ax("set-empty-contains-nothing", &[Elem])
            .trig(vec![contains(empty(), v(0))])
            .then([not(contains(empty(), v(0)))])
            .done(),
        // Only the empty set has cardinality 0.
        //   forall a :: {|a|} |a| == 0 ==> a == Set()
        set_ax("set-card-zero", &[Set])
            .trig(vec![card(a.clone())])
            .when([eq(card(a.clone()), int(0))])
            .guard_when()
            .then([eq(a.clone(), empty())])
            .done(),
        // A set of nonzero cardinality has a member, the witness
        // `Set_skolem_member(a)` (Dafny's `exists x :: x in a`).
        //   forall a :: {|a|} |a| != 0 ==> Set_skolem_member(a) in a
        set_ax("set-card-nonzero", &[Set])
            .trig(vec![card(a.clone())])
            .when([ne(card(a.clone()), int(0))])
            .guard_when()
            .then([contains(
                a.clone(),
                skolem(Witness::SetMember, vec![a.clone()]),
            )])
            .done(),
        // End cardinality and the empty set
        // Start singleton. `r` is variable 0, `o` variable 1.
        //
        // A singleton contains its element.
        //   forall r :: {Set(r)} r in Set(r)
        set_ax("set-singleton-contains1", &[Elem])
            .trig(vec![unit(v(0))])
            .then([contains(unit(v(0)), v(0))])
            .done(),
        // A singleton contains exactly its element.
        //   forall r, o :: {o in Set(r)} (o in Set(r)) <==> r == o
        set_ax("set-singleton-contains2", &[Elem, Elem])
            .trig(vec![contains(unit(v(0)), v(1))])
            .then([eq(contains(unit(v(0)), v(1)), eq(v(0), v(1)))])
            .done(),
        // A singleton has cardinality 1.
        //   forall r :: {|Set(r)|} |Set(r)| == 1
        set_ax("set-singleton-card", &[Elem])
            .trig(vec![card(unit(v(0)))])
            .then([eq(card(unit(v(0))), int(1))])
            .done(),
        // End singleton
        // Start unionone (`a` with one element `x` added, how a literal is
        // built). `x` is variable 1, `o`/`y` variable 2.
        //
        // Adding `x` to `a` gives the members of `a` and `x`.
        //   forall a, x, o :: {o in add(a, x)} (o in add(a, x)) <==> (o == x || o in a)
        set_ax("set-unionone-contains1", &[Set, Elem, Elem])
            .trig(vec![contains(add_one(a.clone(), v(1)), v(2))])
            .then([eq(
                contains(add_one(a.clone(), v(1)), v(2)),
                or(eq(v(2), v(1)), contains(a.clone(), v(2))),
            )])
            .done(),
        // The added element is a member.
        //   forall a, x :: {add(a, x)} x in add(a, x)
        set_ax("set-unionone-contains2", &[Set, Elem])
            .trig(vec![add_one(a.clone(), v(1))])
            .then([contains(add_one(a.clone(), v(1)), v(1))])
            .done(),
        // Every member of `a` stays a member after adding `x`.
        //   forall a, x, y :: {add(a, x), y in a} y in a ==> y in add(a, x)
        set_ax("set-unionone-contains3", &[Set, Elem, Elem])
            .trig(vec![add_one(a.clone(), v(1)), contains(a.clone(), v(2))])
            .when([contains(a.clone(), v(2))])
            .guard_when()
            .then([contains(add_one(a.clone(), v(1)), v(2))])
            .done(),
        // Adding an element already in `a` keeps the cardinality.
        //   forall a, x :: {|add(a, x)|} x in a ==> |add(a, x)| == |a|
        set_ax("set-unionone-card1", &[Set, Elem])
            .trig(vec![card(add_one(a.clone(), v(1)))])
            .when([contains(a.clone(), v(1))])
            .guard_when()
            .then([eq(card(add_one(a.clone(), v(1))), card(a.clone()))])
            .done(),
        // Adding an element not in `a` raises the cardinality by one.
        //   forall a, x :: {|add(a, x)|} !(x in a) ==> |add(a, x)| == |a| + 1
        set_ax("set-unionone-card2", &[Set, Elem])
            .trig(vec![card(add_one(a.clone(), v(1)))])
            .when([not(contains(a.clone(), v(1)))])
            .guard_when()
            .then([eq(
                card(add_one(a.clone(), v(1))),
                add(card(a.clone()), int(1)),
            )])
            .done(),
        // End unionone
        // Start union and intersection. `o`/`y` is variable 2.
        //
        // The members of a union are the members of either side.
        //   forall a, b, o :: {o in a union b} (o in a union b) <==> (o in a || o in b)
        set_ax("set-union-contains1", &[Set, Set, Elem])
            .trig(vec![contains(union(a.clone(), b.clone()), v(2))])
            .then([eq(
                contains(union(a.clone(), b.clone()), v(2)),
                or(contains(a.clone(), v(2)), contains(b.clone(), v(2))),
            )])
            .done(),
        // A member of the left side is a member of the union. (Triggered on
        // the member, so a known `y in a` reaches a union that no membership
        // of `y` mentions yet.)
        //   forall a, b, y :: {a union b, y in a} y in a ==> y in a union b
        set_ax("set-union-contains2", &[Set, Set, Elem])
            .trig(vec![union(a.clone(), b.clone()), contains(a.clone(), v(2))])
            .when([contains(a.clone(), v(2))])
            .guard_when()
            .then([contains(union(a.clone(), b.clone()), v(2))])
            .done(),
        // A member of the right side is a member of the union.
        //   forall a, b, y :: {a union b, y in b} y in b ==> y in a union b
        set_ax("set-union-contains3", &[Set, Set, Elem])
            .trig(vec![union(a.clone(), b.clone()), contains(b.clone(), v(2))])
            .when([contains(b.clone(), v(2))])
            .guard_when()
            .then([contains(union(a.clone(), b.clone()), v(2))])
            .done(),
        // The members of an intersection are the members of both sides.
        //   forall a, b, o :: {o in a intersection b} {a intersection b, o in a}
        //                     {a intersection b, o in b}
        //     (o in a intersection b) <==> (o in a && o in b)
        set_ax("set-intersection-contains", &[Set, Set, Elem])
            .trig(vec![contains(inter(a.clone(), b.clone()), v(2))])
            .trig(vec![inter(a.clone(), b.clone()), contains(a.clone(), v(2))])
            .trig(vec![inter(a.clone(), b.clone()), contains(b.clone(), v(2))])
            .then([eq(
                contains(inter(a.clone(), b.clone()), v(2)),
                and(contains(a.clone(), v(2)), contains(b.clone(), v(2))),
            )])
            .done(),
        // Adding `b` to a union that already has it changes nothing.
        //   forall a, b :: {(a union b) union b} (a union b) union b == a union b
        set_ax("set-union-absorb1", &[Set, Set])
            .trig(vec![union(union(a.clone(), b.clone()), b.clone())])
            .then([eq(
                union(union(a.clone(), b.clone()), b.clone()),
                union(a.clone(), b.clone()),
            )])
            .done(),
        // Adding `a` to a union that already has it changes nothing.
        //   forall a, b :: {a union (a union b)} a union (a union b) == a union b
        set_ax("set-union-absorb2", &[Set, Set])
            .trig(vec![union(a.clone(), union(a.clone(), b.clone()))])
            .then([eq(
                union(a.clone(), union(a.clone(), b.clone())),
                union(a.clone(), b.clone()),
            )])
            .done(),
        // Intersecting with `b` again changes nothing.
        //   forall a, b :: {(a intersection b) intersection b}
        //     (a intersection b) intersection b == a intersection b
        set_ax("set-intersection-absorb1", &[Set, Set])
            .trig(vec![inter(inter(a.clone(), b.clone()), b.clone())])
            .then([eq(
                inter(inter(a.clone(), b.clone()), b.clone()),
                inter(a.clone(), b.clone()),
            )])
            .done(),
        // Intersecting with `a` again changes nothing.
        //   forall a, b :: {a intersection (a intersection b)}
        //     a intersection (a intersection b) == a intersection b
        set_ax("set-intersection-absorb2", &[Set, Set])
            .trig(vec![inter(a.clone(), inter(a.clone(), b.clone()))])
            .then([eq(
                inter(a.clone(), inter(a.clone(), b.clone())),
                inter(a.clone(), b.clone()),
            )])
            .done(),
        // Inclusion-exclusion: the union counts the shared members once, the
        // two sides count them twice.
        //   forall a, b :: {|a union b|} {|a intersection b|}
        //     |a union b| + |a intersection b| == |a| + |b|
        set_ax("set-card-union-intersection", &[Set, Set])
            .trig(vec![card(union(a.clone(), b.clone()))])
            .trig(vec![card(inter(a.clone(), b.clone()))])
            .then([eq(
                add(
                    card(union(a.clone(), b.clone())),
                    card(inter(a.clone(), b.clone())),
                ),
                add(card(a.clone()), card(b.clone())),
            )])
            .done(),
        // End union and intersection
        // Start difference. `o`/`y` is variable 2.
        //
        // The members of `a setminus b` are the members of `a` not in `b`.
        //   forall a, b, o :: {o in a setminus b} {a setminus b, o in a}
        //     (o in a setminus b) <==> (o in a && !(o in b))
        set_ax("set-diff-contains1", &[Set, Set, Elem])
            .trig(vec![contains(diff(a.clone(), b.clone()), v(2))])
            .trig(vec![diff(a.clone(), b.clone()), contains(a.clone(), v(2))])
            .then([eq(
                contains(diff(a.clone(), b.clone()), v(2)),
                and(contains(a.clone(), v(2)), not(contains(b.clone(), v(2)))),
            )])
            .done(),
        // A member of `b` is not in `a setminus b`.
        //   forall a, b, y :: {a setminus b, y in b} y in b ==> !(y in a setminus b)
        set_ax("set-diff-contains2", &[Set, Set, Elem])
            .trig(vec![diff(a.clone(), b.clone()), contains(b.clone(), v(2))])
            .when([contains(b.clone(), v(2))])
            .guard_when()
            .then([not(contains(diff(a.clone(), b.clone()), v(2)))])
            .done(),
        // The union splits into the members only in `a`, those only in `b`,
        // and those in both.
        //   forall a, b :: {|a setminus b|}
        //     |a setminus b| + |b setminus a| + |a intersection b| == |a union b|
        set_ax("set-diff-card1", &[Set, Set])
            .trig(vec![card(diff(a.clone(), b.clone()))])
            .then([eq(
                add(
                    add(
                        card(diff(a.clone(), b.clone())),
                        card(diff(b.clone(), a.clone())),
                    ),
                    card(inter(a.clone(), b.clone())),
                ),
                card(union(a.clone(), b.clone())),
            )])
            .done(),
        // Removing `b` from `a` removes exactly the shared members.
        //   forall a, b :: {|a setminus b|} |a setminus b| == |a| - |a intersection b|
        set_ax("set-diff-card2", &[Set, Set])
            .trig(vec![card(diff(a.clone(), b.clone()))])
            .then([eq(
                card(diff(a.clone(), b.clone())),
                sub(card(a.clone()), card(inter(a.clone(), b.clone()))),
            )])
            .done(),
        // End difference
        // Start subset. Dafny's `Set_subset_def`,
        //   a subset b <==> forall o :: {o in a} {o in b} o in a ==> o in b,
        // skolemized into two axioms (see the module docs). `o` is variable 2.
        //
        // When `a subset b` holds, every member of `a` is a member of `b`
        // (the nested quantifier, with its triggers).
        //   forall a, b, o :: {a subset b, o in a} {a subset b, o in b}
        //     a subset b && o in a ==> o in b
        set_ax("set-subset-member", &[Set, Set, Elem])
            .trig(vec![
                subset(a.clone(), b.clone()),
                contains(a.clone(), v(2)),
            ])
            .trig(vec![
                subset(a.clone(), b.clone()),
                contains(b.clone(), v(2)),
            ])
            .when([subset(a.clone(), b.clone()), contains(a.clone(), v(2))])
            .guard_when()
            .then([contains(b.clone(), v(2))])
            .done(),
        // `a subset b` holds exactly when the one candidate counterexample
        // `w = Set_skolem_subset(a, b)` (a member of `a` outside `b`, if
        // there is one) is not a counterexample. This is the direction that
        // proves an inclusion.
        //   forall a, b :: {a subset b}
        //     a subset b <==> (w in a ==> w in b)
        set_ax("set-subset-witness", &[Set, Set])
            .trig(vec![subset(a.clone(), b.clone())])
            .then({
                let w = || skolem(Witness::SetSubset, vec![a.clone(), b.clone()]);
                [eq(
                    subset(a.clone(), b.clone()),
                    implies(contains(a.clone(), w()), contains(b.clone(), w())),
                )]
            })
            .done(),
        // End subset
        // Start equals. `Set_equal(a, b)` is the program's `SetOp::Eq`, which
        // shares its e-class with `a == b`, so `Set_equal_ext` holds by
        // construction and `Set_equal_def` reduces to its second disjunct.
        //
        // Two unequal sets differ at the element `d = Set_skolem_diff(a, b)`,
        // which is the same witness either way round.
        //   forall a, b :: {a == b}
        //     a != b ==> d == Set_skolem_diff(b, a) && (d in a) != (d in b)
        set_ax("set-equal-diff", &[Set, Set])
            .trig(vec![set_eq(a.clone(), b.clone())])
            .when([ne(a.clone(), b.clone())])
            .guard_when()
            .then({
                let d = || skolem(Witness::SetDiff, vec![a.clone(), b.clone()]);
                [
                    eq(d(), skolem(Witness::SetDiff, vec![b.clone(), a.clone()])),
                    ne(contains(a.clone(), d()), contains(b.clone(), d())),
                ]
            })
            .done(),
        // The same fact read the other way, which is the one that *proves* an
        // equality: two sets are equal exactly when they agree on `d`.
        //   forall a, b :: {a == b} a == b <==> ((d in a) == (d in b))
        set_ax("set-equal-ext", &[Set, Set])
            .trig(vec![set_eq(a.clone(), b.clone())])
            .then({
                let d = || skolem(Witness::SetDiff, vec![a.clone(), b.clone()]);
                [eq(
                    eq(a.clone(), b.clone()),
                    eq(contains(a.clone(), d()), contains(b.clone(), d())),
                )]
            })
            .done(),
        // End equals
    ]
}
