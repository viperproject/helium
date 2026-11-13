//! The axioms of `Map[K, V]`, adapted from Silicon's `dafny_axioms/maps.vpr`
//! (see the [module docs](super)). Each keeps its Dafny name, `Map_x_y` as
//! `map-x-y`, and its triggers. The comment above each axiom says what it
//! states, in words and as the Viper formula, with its triggers. In the
//! formulas, `k in domain(m)` is also what the program writes as `k in m`.
//!
//! Which Dafny axiom became which:
//!
//! | Dafny (`maps.vpr`)                | here                                                        |
//! |-----------------------------------|-------------------------------------------------------------|
//! | `Map_card_nonneg`                 | `map-card-nonneg`                                           |
//! | `Map_domain_card`                 | `map-domain-card`                                           |
//! | `Map_empty_domain`                | `map-empty-domain`                                          |
//! | —                                 | `map-empty-card`                                            |
//! | `Map_card_zero`                   | `map-card-zero`, `map-card-nonzero`, `map-card-key`         |
//! | `Map_equal_def`, `Map_equal_ext`  | `map-equal-diff`, `map-equal-ext`                           |
//! | `Map_update_domain`               | `map-update-domain-same`, `map-update-domain-other`, `map-update-value-other` |
//! | `Map_update_card`                 | `map-update-card-existing`, `map-update-card-new`           |
//! | `Map_values_contains` (skolem)    | `map-values-contains`                                       |
//! | `Map_values_nonempty`             | `map-values-nonempty`                                       |
//! | `Map_values_contains` (lookup)    | `map-values-lookup` (the domain names two axioms the same)  |
//! | `Map_disjoint_def1`, `_def2`      | left out, see below                                         |
//!
//! Where the domain differs:
//! - `k in m` is not a function of its own: as in Silicon, it is
//!   `k in domain(m)`, a set membership ([`STerm::OpOn`] on the set of keys),
//!   so the set axioms apply to `domain(m)` and `range(m)` as to any set.
//! - `Map_disjoint` has no axioms here: Viper has no syntax for it, so no
//!   program term can trigger them.
//! - `Map_equal` is the program's typed equality, as for sets: it shares its
//!   class with `m1 == m2`, so `Map_equal_ext` holds by construction, and
//!   `Map_equal_def` becomes `map-equal-diff` and `map-equal-ext` around the
//!   witness [`Witness::MapDiff`], a key where the maps differ.
//! - The existential of `Map_card_zero` is the witness
//!   [`Witness::MapMember`]; `Map_range_domain_skolem` is
//!   [`Witness::MapRangeKey`].
//! - `map-empty-card` (`|Map()| == 0`) is not in the domain, which derives it
//!   from `Map_card_zero`; the e-graph cannot, and the set table has the same
//!   fact (`set-empty-card`).
//! - Each conjunct of `Map_card_zero`, `Map_update_domain` and
//!   `Map_update_card` is an axiom of its own. `Map_card_zero`'s first
//!   conjunct is an equivalence: its `==>` half is `map-card-zero`, and its
//!   `<==` half (`m == Map() ==> |m| == 0`) follows from `map-empty-card` by
//!   congruence, so it has no axiom of its own.
//! - `Map_update_domain`'s `k1 == k2` case states the new value at `k2`;
//!   its `k1 != k2` case is split into the domain and the value.

use super::*;
use crate::vmir::{MapOp, SetOp};

/// The type argument positions of the sets a map axiom tests: `domain(m)` is
/// a set of keys (`Generic(0)`), `range(m)` a set of values (`Generic(1)`).
const KEYS: &[usize] = &[0];
const VALUES: &[usize] = &[1];

fn card(m: STerm) -> STerm {
    op(MapOp::Card, vec![m])
}
/// `m[k]`.
fn lookup(m: STerm, k: STerm) -> STerm {
    op(MapOp::Lookup, vec![m, k])
}
/// `m[k := v]`.
fn update(m: STerm, k: STerm, val: STerm) -> STerm {
    op(MapOp::Update, vec![m, k, val])
}
fn domain(m: STerm) -> STerm {
    op(MapOp::Domain, vec![m])
}
fn range(m: STerm) -> STerm {
    op(MapOp::Range, vec![m])
}
/// `k in m`, i.e. `k in domain(m)`.
fn has_key(m: STerm, k: STerm) -> STerm {
    op_on(SetOp::Contains, KEYS, vec![domain(m), k])
}
/// `v in range(m)`.
fn has_value(m: STerm, val: STerm) -> STerm {
    op_on(SetOp::Contains, VALUES, vec![range(m), val])
}
/// `|domain(m)|`.
fn domain_card(m: STerm) -> STerm {
    op_on(SetOp::Card, KEYS, vec![domain(m)])
}
/// `|range(m)|`.
fn range_card(m: STerm) -> STerm {
    op_on(SetOp::Card, VALUES, vec![range(m)])
}
/// `m1 == m2` of the program, the typed equality (not the axioms' own `==`,
/// which is [`eq`]): only usable as a trigger.
fn map_eq(m1: STerm, m2: STerm) -> STerm {
    op(MapOp::Eq, vec![m1, m2])
}

/// A map axiom over `vars`.
fn map_ax(name: &'static str, vars: &[VarSort]) -> Ax {
    Ax::new(name, CollKind::Map, vars)
}

pub(super) fn build() -> Vec<CollAxiom> {
    use VarSort::{Coll as Map, Key, Value};
    // `m` is the map (variable 0); the other variables are numbered per axiom.
    let m = v(0);
    vec![
        // Start cardinality
        //
        // A map never has negative cardinality.
        //   forall m :: {|m|} 0 <= |m|
        map_ax("map-card-nonneg", &[Map])
            .trig(vec![card(m.clone())])
            .then([le(int(0), card(m.clone()))])
            .done(),
        // A map has as many entries as keys.
        //   forall m :: {|domain(m)|} {|m|} |domain(m)| == |m|
        map_ax("map-domain-card", &[Map])
            .trig(vec![domain_card(m.clone())])
            .trig(vec![card(m.clone())])
            .then([eq(domain_card(m.clone()), card(m.clone()))])
            .done(),
        // End cardinality
        // Start the empty map. `k` is variable 0.
        //
        // The empty map has no keys.
        //   forall k :: {k in domain(Map())} !(k in domain(Map()))
        map_ax("map-empty-domain", &[Key])
            .trig(vec![has_key(empty(), v(0))])
            .then([not(has_key(empty(), v(0)))])
            .done(),
        // The empty map has cardinality 0. (Not in the Dafny domain; see the
        // module docs.)
        //   |Map()| == 0
        map_ax("map-empty-card", &[])
            .trig(vec![empty()])
            .then([eq(card(empty()), int(0))])
            .done(),
        // Only the empty map has cardinality 0.
        //   forall m :: {|m|} |m| == 0 ==> m == Map()
        map_ax("map-card-zero", &[Map])
            .trig(vec![card(m.clone())])
            .when([eq(card(m.clone()), int(0))])
            .guard_when()
            .then([eq(m.clone(), empty())])
            .done(),
        // A map of nonzero cardinality has a key, the witness
        // `Map_skolem_member(m)` (Dafny's `exists u :: u in domain(m)`).
        //   forall m :: {|m|} |m| != 0 ==> Map_skolem_member(m) in domain(m)
        map_ax("map-card-nonzero", &[Map])
            .trig(vec![card(m.clone())])
            .when([ne(card(m.clone()), int(0))])
            .guard_when()
            .then([has_key(
                m.clone(),
                skolem(Witness::MapMember, vec![m.clone()]),
            )])
            .done(),
        // A map with a key has nonzero cardinality.
        //   forall m, u :: {u in domain(m)} u in domain(m) ==> |m| != 0
        map_ax("map-card-key", &[Map, Key])
            .trig(vec![has_key(m.clone(), v(1))])
            .when([has_key(m.clone(), v(1))])
            .guard_when()
            .then([ne(card(m.clone()), int(0))])
            .done(),
        // End the empty map
        // Start equality. `Map_equal(m1, m2)` is the program's `MapOp::Eq`,
        // which shares its e-class with `m1 == m2`, so `Map_equal_ext` holds
        // by construction. `Map_equal_def` (equal domains and equal values
        // make equal maps) is stated through the witness
        // `d = Map_skolem_diff(m1, m2)`, a key where the maps differ if they
        // do: `m1` is variable 0, `m2` variable 1.
        //
        // Different maps differ at `d`: it is a key of one and not the other,
        // or of both with different values. `d` is the same for both orders
        // of the maps, as for sets, so one witness serves `m1 == m2` and
        // `m2 == m1`.
        //   forall m1, m2 :: {m1 == m2} m1 != m2 ==>
        //     d == Map_skolem_diff(m2, m1) &&
        //     ((d in domain(m1)) != (d in domain(m2)) || d in domain(m1) && m1[d] != m2[d])
        map_ax("map-equal-diff", &[Map, Map])
            .trig(vec![map_eq(v(0), v(1))])
            .when([ne(v(0), v(1))])
            .guard_when()
            .then({
                let d = || skolem(Witness::MapDiff, vec![v(0), v(1)]);
                [
                    eq(d(), skolem(Witness::MapDiff, vec![v(1), v(0)])),
                    or(
                        ne(has_key(v(0), d()), has_key(v(1), d())),
                        and(has_key(v(0), d()), ne(lookup(v(0), d()), lookup(v(1), d()))),
                    ),
                ]
            })
            .done(),
        // The same fact read the other way, which is the one that *proves* an
        // equality: maps that agree at `d` are equal. (Were they different,
        // `map-equal-diff` would make them differ at `d`.) Agreeing means
        // the same key membership at `d`, and the same value there if it is a
        // key; equal maps agree everywhere, so the `==` holds both ways.
        //   forall m1, m2 :: {m1 == m2} (m1 == m2) ==
        //     ((d in domain(m1)) == (d in domain(m2)) && (d in domain(m1) ==> m1[d] == m2[d]))
        map_ax("map-equal-ext", &[Map, Map])
            .trig(vec![map_eq(v(0), v(1))])
            .then({
                let d = || skolem(Witness::MapDiff, vec![v(0), v(1)]);
                [eq(
                    eq(v(0), v(1)),
                    and(
                        eq(has_key(v(0), d()), has_key(v(1), d())),
                        implies(has_key(v(0), d()), eq(lookup(v(0), d()), lookup(v(1), d()))),
                    ),
                )]
            })
            .done(),
        // End equality
        // Start update. `k1` is variable 1, `k2` variable 2, `v` variable 3;
        // `u` is `m[k1 := v]`. The three trigger groups are Dafny's.
        //
        // The updated key is a key of `u`, with the new value.
        //   forall m, k1, k2, v :: {k2 in domain(u)} {k2 in domain(m), u} {u[k2]}
        //     k1 == k2 ==> k2 in domain(u) && u[k2] == v
        map_ax("map-update-domain-same", &[Map, Key, Key, Value])
            .trig(update_triggers())
            .trig(vec![has_key(m.clone(), v(2)), upd()])
            .trig(vec![lookup(upd(), v(2))])
            .when([eq(v(1), v(2))])
            .guard_when()
            .then([has_key(upd(), v(2)), eq(lookup(upd(), v(2)), v(3))])
            .done(),
        // Another key is a key of `u` exactly when it is one of `m`.
        //   forall m, k1, k2, v :: (same triggers)
        //     k1 != k2 ==> (k2 in domain(u) <==> k2 in domain(m))
        map_ax("map-update-domain-other", &[Map, Key, Key, Value])
            .trig(update_triggers())
            .trig(vec![has_key(m.clone(), v(2)), upd()])
            .trig(vec![lookup(upd(), v(2))])
            .when([ne(v(1), v(2))])
            .guard_when()
            .then([eq(has_key(upd(), v(2)), has_key(m.clone(), v(2)))])
            .done(),
        // Another key of `m` keeps its value in `u`.
        //   forall m, k1, k2, v :: (same triggers)
        //     k1 != k2 && k2 in domain(m) ==> u[k2] == m[k2]
        map_ax("map-update-value-other", &[Map, Key, Key, Value])
            .trig(update_triggers())
            .trig(vec![has_key(m.clone(), v(2)), upd()])
            .trig(vec![lookup(upd(), v(2))])
            .when([ne(v(1), v(2)), has_key(m.clone(), v(2))])
            .guard_when()
            .then([eq(lookup(upd(), v(2)), lookup(m.clone(), v(2)))])
            .done(),
        // `k` is variable 1, `v` variable 2 from here.
        //
        // Updating a key of `m` keeps its cardinality.
        //   forall m, k, v :: {|m[k := v]|} {|m|, m[k := v]}
        //     k in domain(m) ==> |m[k := v]| == |m|
        map_ax("map-update-card-existing", &[Map, Key, Value])
            .trig(vec![card(update(m.clone(), v(1), v(2)))])
            .trig(vec![card(m.clone()), update(m.clone(), v(1), v(2))])
            .when([has_key(m.clone(), v(1))])
            .guard_when()
            .then([eq(card(update(m.clone(), v(1), v(2))), card(m.clone()))])
            .done(),
        // Updating a new key adds one.
        //   forall m, k, v :: (same triggers)
        //     !(k in domain(m)) ==> |m[k := v]| == |m| + 1
        map_ax("map-update-card-new", &[Map, Key, Value])
            .trig(vec![card(update(m.clone(), v(1), v(2)))])
            .trig(vec![card(m.clone()), update(m.clone(), v(1), v(2))])
            .when([not(has_key(m.clone(), v(1)))])
            .guard_when()
            .then([eq(
                card(update(m.clone(), v(1), v(2))),
                add(card(m.clone()), int(1)),
            )])
            .done(),
        // End update
        // Start values. `v` is variable 1 in `map-values-contains`, `k`
        // variable 1 after it.
        //
        // A value of `m` is the value of some key, the witness
        // `Map_range_domain_skolem(m, v)`.
        //   forall m, v :: {v in range(m)} v in range(m) ==>
        //     Map_range_domain_skolem(m, v) in domain(m) && v == m[Map_range_domain_skolem(m, v)]
        map_ax("map-values-contains", &[Map, Value])
            .trig(vec![has_value(m.clone(), v(1))])
            .when([has_value(m.clone(), v(1))])
            .guard_when()
            .then({
                let k = || skolem(Witness::MapRangeKey, vec![m.clone(), v(1)]);
                [has_key(m.clone(), k()), eq(v(1), lookup(m.clone(), k()))]
            })
            .done(),
        // A map with a key has a value.
        //   forall m, k :: {m[k]} {k in domain(m)} k in domain(m) ==> |range(m)| > 0
        map_ax("map-values-nonempty", &[Map, Key])
            .trig(vec![lookup(m.clone(), v(1))])
            .trig(vec![has_key(m.clone(), v(1))])
            .when([has_key(m.clone(), v(1))])
            .guard_when()
            .then([gt(range_card(m.clone()), int(0))])
            .done(),
        // The value of a key is a value of the map. (Dafny's second
        // `Map_values_contains`. It triggers on the lookup only: the domain
        // dropped the trigger `{k in domain(m)}`, which loops when two maps
        // share a domain, each new value's key giving the other map a new
        // lookup.)
        //   forall m, k :: {m[k]} k in domain(m) ==> m[k] in range(m)
        map_ax("map-values-lookup", &[Map, Key])
            .trig(vec![lookup(m.clone(), v(1))])
            .when([has_key(m.clone(), v(1))])
            .guard_when()
            .then([has_value(m.clone(), lookup(m.clone(), v(1)))])
            .done(),
        // End values
    ]
}

/// `u = m[k1 := v]` of the update axioms: `m`, `k1` and `v` are variables 0,
/// 1 and 3.
fn upd() -> STerm {
    update(v(0), v(1), v(3))
}

/// The first trigger group of the update axioms, `{k2 in domain(u)}`.
fn update_triggers() -> Vec<STerm> {
    vec![has_key(upd(), v(2))]
}
