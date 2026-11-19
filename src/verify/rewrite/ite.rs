//! `ite` reduction: the fused rule that decomposes and collapses conditionals.
//!
//! One rule rather than a dozen, driven by a `classes_by_op` bucket scan instead
//! of a dozen pattern searches. It does two different kinds of inference over the
//! same node loop: **decomposition** (the class's known value constrains its
//! parts) and **reduction** (the parts' known values collapse the class).

use egg::{Applier, EGraph, Id, PatternAst, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::{ConstFold, Fingerprint};
use crate::verify::lang::{Discriminant, Symbolic};
use crate::vmir::Literal;

use super::*;

/// Searcher for the fused ite rule: every e-class holding an `Ite` node, via
/// the `classes_by_op` bucket (no whole-graph scan). One empty subst per class;
/// the applier re-reads the nodes.
#[derive(Clone)]
pub(super) struct IteBucketSearcher;

impl Searcher<Symbolic, ConstFold> for IteBucketSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let Some(classes) = egraph.classes_for_op(&Discriminant::Ite) else {
            return vec![];
        };
        classes
            .take(limit)
            .map(|eclass| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
            .collect()
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        egraph[eclass]
            .nodes
            .iter()
            .any(|n| matches!(n, Symbolic::Ite(..)))
            .then(|| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// The known boolean of a class, per `ConstFold`. Subsumes matching a literal
/// node (the analysis is seeded by literals), so this fires at least wherever
/// the old `true`/`false` patterns did.
pub(super) fn known_bool(egraph: &EGraph<Symbolic, ConstFold>, class: Id) -> Option<bool> {
    match egraph[class].data.known() {
        Some(Literal::Bool(b)) => Some(*b),
        _ => None,
    }
}

pub(super) struct IteReduceApplier;

impl Applier<Symbolic, ConstFold> for IteReduceApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Collect the unions first: node inspection needs `&egraph`.
        enum Target {
            Class(Id),
            True,
            False,
        }
        // An inconsistent class (`true` merged with `false`) has nothing left to
        // derive. Matches go stale within an iteration, so once a graph collapses
        // every remaining match resolves to that one class, and walking its
        // nodes per match would cost the class size times the match count.
        if egraph[eclass].data.is_inconsistent() {
            return Vec::new();
        }
        let mut unions: Vec<(Id, Target)> = Vec::new();
        let self_lit = known_bool(egraph, eclass);
        for node in &egraph[eclass].nodes {
            let Symbolic::Ite([c, t, e]) = node else {
                continue;
            };
            let (c, t, e) = (egraph.find(*c), egraph.find(*t), egraph.find(*e));
            let (t_lit, e_lit) = (known_bool(egraph, t), known_bool(egraph, e));
            // Decompositions: the *class's* proven boolean constrains the parts.
            match self_lit {
                // (a && b) proven true => a, b each true.  a && b is a ? b : false.
                Some(true) if e_lit == Some(false) => {
                    unions.push((c, Target::True));
                    unions.push((t, Target::True));
                }
                // (!x) proven true => x false.  !x is x ? false : true.
                Some(true) if t_lit == Some(false) && e_lit == Some(true) => {
                    unions.push((c, Target::False));
                }
                // (!x) proven false => x true.  Mirror of the arm above; this is
                // how Prusti spells a taken branch (`switchInt` emits
                // `if (value(t) == false) { else } else { then }`, so the arm where
                // the guard holds sits under a negation proven false).
                Some(false) if t_lit == Some(false) && e_lit == Some(true) => {
                    unions.push((c, Target::True));
                }
                // (a || b) proven false => a, b each false.  a || b is a ? true : b.
                Some(false) if t_lit == Some(true) => {
                    unions.push((c, Target::False));
                    unions.push((e, Target::False));
                }
                _ => {}
            }
            // The same decomposition at **any** pinned value, not just booleans
            // and not just literals: if the class's value is pinned to `V` and an
            // arm's is pinned to something [`Fingerprint::differs_from`] separates
            // from `V`, that arm cannot be the one taken, so the condition is
            // pinned to the other side. When *both* arms disagree with `V` the two
            // unions pin `c` to both polarities, which is exactly the
            // contradiction — `ite(c, 1, 2) ≡ 0` is refuted without ever splitting
            // on `c`, and that is how a division guard survives a branch join
            // (`d = c ? 1 : 2` then `d != 0`).
            //
            // The **constructor** half of the fingerprint is what inverts a Prusti
            // enum-snapshot tower: `p_Shape_snap`'s body is a nested ite of
            // `s_Shape_k_cons(..)` arms, and once a snapshot round-trip
            // (`make_generic_Shape` / `make_concrete_Shape`) equates the whole
            // tower with one `s_Shape_1_cons(..)`, each non-matching arm pins its
            // guard false and `ite(false, _, e) ⇒ e` exposes the next level, one
            // per saturation iteration. The bottom arm is a ground constructor, so
            // the last step pins the *matching* guard true — recovering
            // `discr == cons(1)`, which is the permission gate on the variant's
            // footprint. Identical arms (`ite(c, x, x)`) and same-constructor arms
            // pin nothing, which is right: neither says anything about `c`.
            //
            // An arm whose own fingerprint is unknown can still be ruled out: see
            // [`arm_excluded`]. That is what inverts a tower whose bottom arm is
            // not a constructor but Prusti's `s_T_unreachable()`, so that only the
            // tower's guards being exhaustive keeps the bottom arm from being taken.
            if let Some(self_fp) = egraph[eclass].data.fingerprint() {
                let arm_differs = |arm: Id| {
                    egraph[arm]
                        .data
                        .fingerprint()
                        .is_some_and(|fp| fp.differs_from(&self_fp))
                };
                let (t_differs, e_differs) = (arm_differs(t), arm_differs(e));
                if t_differs {
                    unions.push((c, Target::False));
                }
                if e_differs {
                    unions.push((c, Target::True));
                }
                // Not for a boolean class: those are clause towers, which the
                // and/or decompositions above handle, and the `true` class holds
                // every guarded fact, so walking it would cost a walk per fact.
                let value_class = !matches!(self_fp, Fingerprint::Lit(Literal::Bool(_)));
                if value_class && !t_differs && !e_differs && known_bool(egraph, c).is_none() {
                    if arm_excluded(egraph, e, &self_fp, (c, false)) {
                        unions.push((c, Target::True));
                    } else if arm_excluded(egraph, t, &self_fp, (c, true)) {
                        unions.push((c, Target::False));
                    }
                }
            }
            match known_bool(egraph, c) {
                // ite(true, x, y) => x
                Some(true) => unions.push((eclass, Target::Class(t))),
                // ite(false, x, y) => y
                Some(false) => unions.push((eclass, Target::Class(e))),
                None => {}
            }
            // c ? x : x => x
            if t == e {
                unions.push((eclass, Target::Class(t)));
            }
            // c ? true : false => c
            if t_lit == Some(true) && e_lit == Some(false) {
                unions.push((eclass, Target::Class(c)));
            }
            // c ? c : false => c  (b && b)
            if t == c && e_lit == Some(false) {
                unions.push((eclass, Target::Class(c)));
            }
            // c ? true : c => c  (b || b)
            if t_lit == Some(true) && e == c {
                unions.push((eclass, Target::Class(c)));
            }
            // c ? c : true => true
            if t == c && e_lit == Some(true) {
                unions.push((eclass, Target::True));
            }
            // c ? false : c => false
            if t_lit == Some(false) && e == c {
                unions.push((eclass, Target::False));
            }
        }

        let mut changed = Vec::new();
        for (source, target) in unions {
            let target = match target {
                Target::Class(id) => id,
                Target::True => egraph.add(Symbolic::Lit(Literal::Bool(true))),
                Target::False => egraph.add(Symbolic::Lit(Literal::Bool(false))),
            };
            if egraph.union(source, target) {
                changed.push(egraph.find(source));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Whether `arm` — the branch an `ite` takes when `lit = (c, b)` holds, i.e. `c`
/// has value `b` — provably differs from a class pinned to `target`, so that `lit`
/// cannot hold in that class.
///
/// The arm is walked as a case tower: an arm is excluded if its fingerprint
/// differs from `target`; or if it holds an `ite(c', t', e')` with one branch
/// excluded that way (or `c'` decided) and the other branch excluded in turn,
/// under the path extended by the literal that reaches it; or if the path that
/// reaches it is refuted ([`path_refuted`]). Each class is entered once, so a
/// walk costs at most the classes of the tower below `arm`.
///
/// Sound for any `ite`: on a model where the path holds, the class takes one of
/// the walked arms, each of which differs from `target` or is not reachable.
fn arm_excluded(
    egraph: &EGraph<Symbolic, ConstFold>,
    arm: Id,
    target: &Fingerprint<'_>,
    lit: (Id, bool),
) -> bool {
    let differs = |x: Id| {
        egraph[x]
            .data
            .fingerprint()
            .is_some_and(|fp| fp.differs_from(target))
    };
    let mut path: Vec<(Id, bool)> = Vec::new();
    let mut visited = crate::dhash::HashSet::default();
    // Depth-first, with the path shared across the stack: an entry records the
    // path length at its parent and the literal that reaches it from there.
    let mut stack: Vec<(Id, usize, Option<(Id, bool)>)> = vec![(arm, 0, Some(lit))];
    while let Some((x, depth, step)) = stack.pop() {
        path.truncate(depth);
        path.extend(step);
        let x = egraph.find(x);
        if differs(x) {
            continue;
        }
        let mut leaf = true;
        // An arm pinned to `target`'s own value or constructor is no tower to
        // walk: only an unreachable path keeps it from being taken.
        if egraph[x].data.fingerprint().is_none() && visited.insert(x) {
            for node in &egraph[x].nodes {
                let Symbolic::Ite([c, t, e]) = node else {
                    continue;
                };
                let (c, t, e) = (egraph.find(*c), egraph.find(*t), egraph.find(*e));
                let next = match known_bool(egraph, c) {
                    Some(true) => (t, None),
                    Some(false) => (e, None),
                    None if differs(t) => (e, Some((c, false))),
                    None if differs(e) => (t, Some((c, true))),
                    None => continue,
                };
                stack.push((next.0, path.len(), next.1));
                leaf = false;
            }
        }
        if leaf && !path_refuted(egraph, &path) {
            return false;
        }
    }
    true
}

/// Whether the conjunction of `path`'s literals is refuted: its negation, as the
/// nested clause a guard chain folds into ([`fold_guard_ids`]: `ite(g, R, true)`
/// for `g`, `ite(g, true, R)` for `¬g`, innermost `false`), is in the graph and
/// known `true`. A function body's propagated callee tokens are exactly such
/// clauses (`guards ⟹ g%pre`), and the token of a callee whose postcondition is
/// `false` is refuted, so the path to such a call is refuted here. Lookups only:
/// nothing is added.
fn path_refuted(egraph: &EGraph<Symbolic, ConstFold>, path: &[(Id, bool)]) -> bool {
    let (Some(true_), Some(false_)) = (
        egraph.lookup(Symbolic::Lit(Literal::Bool(true))),
        egraph.lookup(Symbolic::Lit(Literal::Bool(false))),
    ) else {
        return false;
    };
    let mut clause = false_;
    for &(g, b) in path.iter().rev() {
        match known_bool(egraph, g) {
            // A literal that holds adds nothing to the conjunction; one that
            // fails refutes it.
            Some(v) if v == b => continue,
            Some(_) => return true,
            None => {}
        }
        let node = match b {
            true => Symbolic::Ite([g, clause, true_]),
            false => Symbolic::Ite([g, true_, clause]),
        };
        match egraph.lookup(node) {
            Some(id) => clause = egraph.find(id),
            None => return false,
        }
    }
    known_bool(egraph, clause) == Some(true)
}

/// Terminating `ite` simplifications, shared by the saturation rule set and the
/// post-`fold`/`unfold` reduction set. Under an assumed branch literal these
/// reduce a gated permission `b ? p : 0` to `p` (resp. `0`).
///
/// Fused into **one** rule that scans the `Ite` op bucket once per iteration and
/// node-checks every shape: as twelve `rw!` patterns they cost ~70% of all search
/// time, since the nested two-level patterns pay a backtracking cross-product
/// over a bucket holding thousands of classes. The fused pass is linear in the
/// bucket (plus the two branch classes' nodes) and union-only.
///
/// Shapes, all decided from a single `Ite` node and its two branch classes:
/// - `ite(true, x, y) ⇒ x`, `ite(false, x, y) ⇒ y` (via `ConstFold` on `c`)
/// - `ite(c, x, x) ⇒ x`
/// - `ite(c, true, false) ⇒ c`
/// - `ite(c, c, false) ⇒ c`, `ite(c, true, c) ⇒ c`
/// - `ite(c, c, true) ⇒ true`, `ite(c, false, c) ⇒ false`
///
/// The two **nested** same-condition shapes (`ite(c, ite(c, x, y), _)` and its
/// mirror) were removed: they needed a scan of a whole branch class, which is why
/// they carried a `NESTED_SCAN_BOUND` cutoff — and that cutoff disabled them on
/// every non-trivial program, since the class they had to search is the `true`
/// class. The permission-side analogue of the same identity is applied
/// structurally at construction by [`ChunkPerm::collapse_same_cond`], with no scan
/// and no budget.
pub(super) fn terminating_ite_rules() -> Vec<Rule> {
    vec![bucket_rule(
        "ite-reduce",
        IteBucketSearcher,
        IteReduceApplier,
    )]
}
