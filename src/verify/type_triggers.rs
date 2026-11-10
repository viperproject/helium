//! Matching a generic axiom's **type triggers** against the types of a
//! verification unit (see [`crate::verify::unit_types`]).
//!
//! An axiom's `ty_triggers` are *alternative groups* of type patterns, each group a
//! conjunction: a group fires only when **every** pattern in it matches some type of
//! the unit, under one common assignment to the axiom's type parameters. A group in
//! which only some patterns match yields nothing. The result of matching an axiom is
//! the set of distinct complete assignments `Generic(i) ↦ tys[i]` over all groups.

use crate::dhash::HashMap;
use crate::vmir::{MemberId, Type};

use crate::verify::unit_types::UnitTypes;

/// What a type pattern can possibly match, looked up without inspecting any type:
/// the outermost constructor, plus the declaration for a domain / ADT / snapshot.
/// Two types with different heads never unify, so a pattern only has to be tried
/// against the bucket of its own head.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Head {
    Int,
    Bool,
    Real,
    Ref,
    Domain(MemberId),
    Snap(MemberId),
    Option,
    Addr,
}

impl Head {
    /// The head of `ty`. A bare `Generic` has none: it matches every type.
    fn of(ty: &Type) -> Head {
        match ty {
            Type::Int => Head::Int,
            Type::Bool => Head::Bool,
            Type::Real => Head::Real,
            Type::Ref => Head::Ref,
            Type::Domain(id, _) => Head::Domain(*id),
            Type::Snap(id) => Head::Snap(*id),
            Type::Option(_) => Head::Option,
            Type::Addr { .. } => Head::Addr,
            Type::Generic(_) => unreachable!("a bare parameter is matched against every type"),
        }
    }
}

/// The types of one unit, bucketed by [`Head`]. Built once per unit and shared by
/// every axiom, so matching a pattern costs its bucket, not the whole unit.
pub(crate) struct TypeIndex<'t> {
    all: Vec<&'t Type>,
    by_head: HashMap<Head, Vec<&'t Type>>,
}

impl<'t> TypeIndex<'t> {
    pub(crate) fn new(types: &'t UnitTypes) -> Self {
        let mut by_head: HashMap<Head, Vec<&'t Type>> = HashMap::default();
        for ty in types {
            by_head.entry(Head::of(ty)).or_default().push(ty);
        }
        Self {
            all: types.iter().collect(),
            by_head,
        }
    }

    fn candidates(&self, pattern: &Type) -> &[&'t Type] {
        match pattern {
            Type::Generic(_) => &self.all,
            _ => self
                .by_head
                .get(&Head::of(pattern))
                .map_or(&[], Vec::as_slice),
        }
    }
}

/// Every distinct assignment of the `ty_params` type parameters under which some
/// group of `ty_triggers` fully matches `index`, sorted (so instantiation order,
/// and with it the e-graph, does not depend on hash order).
///
/// Panics if a matching group leaves a parameter unbound: `Axiom::infer` only
/// produces triggers that bind every parameter.
pub(crate) fn type_trigger_matches(
    ty_params: usize,
    ty_triggers: &[Vec<Type>],
    index: &TypeIndex<'_>,
) -> Vec<Vec<Type>> {
    let mut out = Vec::new();
    for group in ty_triggers {
        // Rarest bucket first: the join prunes as early as it can, and a pattern
        // with no candidate at all rules the whole group out before any matching.
        let mut order: Vec<(&Type, &[&Type])> = group
            .iter()
            .map(|pattern| (pattern, index.candidates(pattern)))
            .collect();
        if order.iter().any(|(_, bucket)| bucket.is_empty()) {
            continue;
        }
        order.sort_unstable_by_key(|(_, bucket)| bucket.len());
        join(&order, &mut vec![None; ty_params], &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Match `order[0]`, then `order[1..]`, extending `sigma`; push it once every
/// pattern has matched. Matching writes into the assignment as it goes, so each
/// candidate gets its own copy and a failed attempt leaks nothing. Patterns that
/// share a parameter must agree on it (`match_generics` rejects a conflict).
fn join(order: &[(&Type, &[&Type])], sigma: &mut Vec<Option<Type>>, out: &mut Vec<Vec<Type>>) {
    let Some(((pattern, bucket), rest)) = order.split_first() else {
        out.push(
            sigma
                .iter()
                .cloned()
                .collect::<Option<Vec<Type>>>()
                .expect("a type trigger group binds all of the axiom's type parameters"),
        );
        return;
    };
    for candidate in *bucket {
        let mut next = sigma.clone();
        if pattern.match_generics(candidate, &mut next) {
            join(rest, &mut next, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::unit_types::UnitTypes;

    const PAIR: MemberId = MemberId(1);
    const BOX: MemberId = MemberId(2);
    const OTHER: MemberId = MemberId(3);

    fn dom(id: MemberId, args: &[Type]) -> Type {
        Type::Domain(id, args.into())
    }
    fn g(i: usize) -> Type {
        Type::Generic(i)
    }
    fn unit(tys: &[Type]) -> UnitTypes {
        tys.iter().cloned().collect()
    }
    fn matches(n: usize, triggers: &[Vec<Type>], tys: &[Type]) -> Vec<Vec<Type>> {
        let unit = unit(tys);
        type_trigger_matches(n, triggers, &TypeIndex::new(&unit))
    }

    #[test]
    fn single_trigger_yields_each_distinct_instance() {
        let got = matches(
            2,
            &[vec![dom(PAIR, &[g(0), g(1)])]],
            &[
                dom(PAIR, &[Type::Int, Type::Bool]),
                dom(PAIR, &[Type::Bool, Type::Bool]),
                dom(BOX, &[Type::Int]),
            ],
        );
        assert_eq!(
            got,
            vec![vec![Type::Int, Type::Bool], vec![Type::Bool, Type::Bool]]
        );
    }

    #[test]
    fn every_trigger_of_a_group_must_match_under_one_assignment() {
        // `Pair<?0, ?1>` and `Box<?0>`: only `Int` has both.
        let trig = vec![vec![dom(PAIR, &[g(0), g(1)]), dom(BOX, &[g(0)])]];
        let got = matches(
            2,
            &trig,
            &[
                dom(PAIR, &[Type::Int, Type::Bool]),
                dom(PAIR, &[Type::Bool, Type::Bool]),
                dom(BOX, &[Type::Int]),
            ],
        );
        assert_eq!(got, vec![vec![Type::Int, Type::Bool]]);
    }

    #[test]
    fn a_group_with_one_unmatched_trigger_yields_nothing() {
        let trig = vec![vec![dom(PAIR, &[g(0), g(1)]), dom(OTHER, &[g(0)])]];
        assert!(matches(2, &trig, &[dom(PAIR, &[Type::Int, Type::Bool])]).is_empty());
    }

    #[test]
    fn alternative_groups_are_unioned_and_deduplicated() {
        let trig = vec![vec![dom(BOX, &[g(0)])], vec![dom(OTHER, &[g(0)])]];
        let got = matches(
            1,
            &trig,
            &[
                dom(BOX, &[Type::Int]),
                dom(OTHER, &[Type::Int]),
                dom(OTHER, &[Type::Bool]),
            ],
        );
        assert_eq!(got, vec![vec![Type::Int], vec![Type::Bool]]);
    }

    #[test]
    fn a_repeated_parameter_must_agree() {
        let trig = vec![vec![dom(PAIR, &[g(0), g(0)])]];
        let got = matches(
            1,
            &trig,
            &[
                dom(PAIR, &[Type::Int, Type::Bool]),
                dom(PAIR, &[Type::Bool, Type::Bool]),
            ],
        );
        assert_eq!(got, vec![vec![Type::Bool]]);
    }

    #[test]
    fn a_ground_trigger_matches_only_itself() {
        let trig = vec![vec![dom(BOX, &[])]];
        assert_eq!(
            matches(0, &trig, &[dom(BOX, &[])]),
            vec![Vec::<Type>::new()]
        );
        assert!(matches(0, &trig, &[dom(OTHER, &[])]).is_empty());
    }
}
