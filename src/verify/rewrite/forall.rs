//! Pure `forall` quantifiers: trigger matching against a prepared term and the
//! rule that instantiates a quantifier body.
//!
//! Builds its instances through `super::recipe`; nothing here flows back the
//! other way.

use std::sync::{Arc, RwLock};

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{Discriminant, FuncId, RecipeId, Symbolic};
use crate::verify::quant::RecipeTable;
use crate::vmir::{Literal, Type};

use super::*;

/// One trigger pattern term, with every head resolved to its verifier `FuncId`
/// (the applier has no registry access). The `vmir::TrigTerm` grammar, flattened
/// to what the e-graph speaks: an `App` matches a `FuncApp` node with the same
/// function and arity, whose type payload **unifies** with `type_args` (a
/// `Generic(i)` there binds the recipe's `i`-th type parameter) and whose
/// argument e-classes match `args` recursively.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum PreparedTerm {
    Bound(usize),
    Capture(usize),
    Lit(Literal),
    App {
        func: FuncId,
        type_args: Box<[Type]>,
        args: Vec<PreparedTerm>,
    },
}

impl PreparedTerm {
    /// Record every `Generic(i)` the term's type arguments mention.
    pub(crate) fn collect_generics(&self, out: &mut crate::dhash::HashSet<usize>) {
        if let PreparedTerm::App {
            type_args, args, ..
        } = self
        {
            type_args.iter().for_each(|t| t.collect_generics(out));
            args.iter().for_each(|a| a.collect_generics(out));
        }
    }

    /// The function a term is rooted at — the searcher's anchor. Only an `App`
    /// can be a top-level trigger term (typecheck-enforced).
    fn root_func(&self) -> Option<FuncId> {
        match self {
            PreparedTerm::App { func, .. } => Some(*func),
            _ => None,
        }
    }
}

/// Match `term` against e-class `class` under the forall node's capture children,
/// extending `sigma` — the value binders *and* the type parameters. Returns every
/// consistent extension (an e-class may hold several nodes matching the pattern's
/// head, each binding σ differently), or an empty vector when the term cannot
/// match.
pub(super) fn match_term(
    egraph: &EGraph<Symbolic, ConstFold>,
    term: &PreparedTerm,
    class: Id,
    caps: &[Id],
    sigma: &Sigma,
) -> Vec<Sigma> {
    let class = egraph.find(class);
    match term {
        PreparedTerm::Bound(i) => match sigma.vals[*i] {
            // A repeated binder must land on the same e-class.
            Some(prev) if egraph.find(prev) != class => vec![],
            Some(_) => vec![sigma.clone()],
            None => {
                let mut next = sigma.clone();
                next.vals[*i] = Some(class);
                vec![next]
            }
        },
        PreparedTerm::Capture(c) => {
            if egraph.find(caps[*c]) == class {
                vec![sigma.clone()]
            } else {
                vec![]
            }
        }
        PreparedTerm::Lit(lit) => {
            if egraph[class]
                .nodes
                .iter()
                .any(|n| matches!(n, Symbolic::Lit(l) if l == lit))
            {
                vec![sigma.clone()]
            } else {
                vec![]
            }
        }
        PreparedTerm::App {
            func,
            type_args,
            args,
        } => {
            let mut out = Vec::new();
            for node in &egraph[class].nodes {
                let Symbolic::FuncApp(f, tys, children) = node else {
                    continue;
                };
                if f != func || tys.len() != type_args.len() || children.len() != args.len() {
                    continue;
                }
                // Unify the pattern's types with the node's ground ones. Matching
                // writes into σ as it goes, so each candidate node gets its own
                // copy: a failed attempt must not leak its bindings. A ground
                // pattern degenerates to plain equality.
                let mut typed = sigma.clone();
                if !type_args
                    .iter()
                    .zip(tys.iter())
                    .all(|(pat, ground)| pat.match_generics(ground, &mut typed.tys))
                {
                    continue;
                }
                // Thread σ left to right across the arguments, branching on every
                // consistent way each of them matches.
                let mut partials = vec![typed];
                for (arg, &child) in args.iter().zip(children.iter()) {
                    partials = partials
                        .iter()
                        .flat_map(|s| match_term(egraph, arg, child, caps, s))
                        .collect();
                    if partials.is_empty() {
                        break;
                    }
                }
                out.extend(partials);
            }
            out
        }
    }
}

/// Every σ that matches `term` somewhere in the e-graph, extending `sigma`. Scans
/// the classes holding an application of the term's root function.
pub(super) fn match_term_anywhere(
    egraph: &EGraph<Symbolic, ConstFold>,
    term: &PreparedTerm,
    caps: &[Id],
    sigma: &Sigma,
) -> Vec<Sigma> {
    let Some(root) = term.root_func() else {
        return vec![];
    };
    let Some(classes) = egraph.classes_for_op(&Discriminant::FuncApp(root)) else {
        return vec![];
    };
    classes
        .flat_map(|class| match_term(egraph, term, class, caps, sigma))
        .collect()
}

/// Searcher for the **single** quantifier-instantiation rule: every e-class
/// holding a `Forall` node, found through `classes_for_op` (one bucket per
/// recipe — indexed, no whole-graph scan). Quantifiers are *data*, not rules, so
/// a `forall` materialized mid-run is picked up on the next iteration; egg
/// forbids injecting rules mid-`Runner`.
pub(super) struct ForallSearcher;

impl Searcher<Symbolic, ConstFold> for ForallSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        // Every quantifier shares one `classes_by_op` bucket, so the scan is
        // proportional to the `forall`s *present in this graph* — not to the
        // recipes the program happens to contain.
        let Some(classes) = egraph.classes_for_op(&Discriminant::Forall) else {
            return Vec::new();
        };
        classes
            .take(limit)
            .map(|eclass| SearchMatches {
                eclass,
                // One empty subst: the applier reads the node itself (recipe
                // + captures), which an egg `Subst` cannot carry.
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
            .any(|n| matches!(n, Symbolic::Forall(..)))
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

/// Applier: for the matched forall e-class, take each `Forall(recipe, tys, caps)`
/// node in it and pair it with every σ under which one of its recipe's trigger
/// groups matches the graph. σ is the value binders **and** the type parameters;
/// the latter start from the node's own `tys` (a nested quantifier arrives with
/// them fixed) and are otherwise bound by the triggers' type arguments. Instantiate
/// the body at `caps ++ σ` and add the **guarded** clause `Ite(forall, res[caps, σ], true) == true`: the instance is released only
/// once that forall e-class merges `true` (existing `ite-true` rule), so
/// instantiating is sound regardless of the quantifier's truth.
///
/// Memoized per `(recipe, types, caps ++ σ)` (canonicalized at insert) — a
/// saturation-cost guard only, since instances are idempotent. The types are part
/// of the key: `emp<Int>()` and `emp<Bool>()` have the same (empty) value σ.
///
/// The table is interned into as bodies are walked, so this holds it behind the
/// shared lock rather than a snapshot. Only reads happen here: a nested recipe is
/// interned innermost-first with its encloser, on the eval walk, never from inside
/// a rule.
pub(super) struct ForallApplier {
    pub(super) table: Arc<RwLock<RecipeTable>>,
    pub(super) memo: Arc<Memo<(RecipeId, Vec<Type>, Vec<Id>)>>,
}

impl Applier<Symbolic, ConstFold> for ForallApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Distinct capture tuples in one e-class are distinct quantifiers (each
        // yields its own instances); the guard is this e-class either way.
        let quants: Vec<(RecipeId, Box<[Type]>, Box<[Id]>)> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::Forall(rid, tys, caps) => Some((*rid, tys.clone(), caps.clone())),
                _ => None,
            })
            .collect();

        let table = self.table.read().expect("recipe table lock");
        // Collect the instances first: `build_instance` needs `&mut egraph`.
        let mut instances: Vec<(RecipeId, Vec<Type>, Vec<Id>)> = Vec::new();
        for (rid, node_tys, caps) in &quants {
            let recipe = table.get(*rid);
            let caps: Vec<Id> = caps.iter().map(|&c| egraph.find(c)).collect();
            // A node carrying types is already instantiated at them; an open one
            // (empty payload, `n_ty` parameters) leaves every type to the triggers.
            let seed_tys: Vec<Option<Type>> = if node_tys.is_empty() {
                vec![None; recipe.n_ty]
            } else {
                node_tys.iter().cloned().map(Some).collect()
            };
            for group in &recipe.groups {
                let Some((anchor, rest)) = group.split_first() else {
                    continue;
                };
                let empty = Sigma {
                    vals: vec![None; recipe.n_bound],
                    tys: seed_tys.clone(),
                };
                let mut sigmas = match_term_anywhere(egraph, anchor, &caps, &empty);
                for term in rest {
                    sigmas = sigmas
                        .iter()
                        .flat_map(|s| match_term_anywhere(egraph, term, &caps, s))
                        .collect();
                    if sigmas.is_empty() {
                        break;
                    }
                }
                for sigma in sigmas {
                    // A group covers every binder (typecheck-enforced), so a
                    // complete match leaves no value slot open. A type parameter
                    // no trigger mentions can stay open: such a σ has no
                    // instance (skipping it is sound, merely incomplete).
                    let Some(sigma_vals): Option<Vec<Id>> = sigma.vals.into_iter().collect()
                    else {
                        continue;
                    };
                    let Some(sigma_tys): Option<Vec<Type>> = sigma.tys.into_iter().collect()
                    else {
                        continue;
                    };
                    // The seed is the body's leading temps: captures then binders.
                    let mut vals = caps.clone();
                    vals.extend(sigma_vals);
                    if self.memo.insert((*rid, sigma_tys.clone(), vals.clone())) {
                        instances.push((*rid, sigma_tys, vals));
                    }
                }
            }
        }

        let mut changed = Vec::new();
        for (rid, tys, vals) in instances {
            let recipe = table.get(rid);
            debug_assert_eq!(
                vals.len(),
                recipe.n_caps + recipe.n_bound,
                "instance seed is captures ++ sigma"
            );
            // `eclass` as the token guard: a `g%pre` token this body propagates is
            // released only under the quantifier's own truth, the same gate the
            // instance itself sits behind below. A quantifier that is merely
            // *present* materializes its body but activates no callee's axioms.
            let instance_vals = build_instance_vals_typed(
                egraph,
                &recipe.insts,
                &vals,
                &tys,
                &mut changed,
                Some(eclass),
            );
            let res = resolve_val(egraph, &instance_vals, &recipe.res);
            let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
            let guard = egraph.add(Symbolic::Ite([eclass, res, true_]));
            if egraph.union(guard, true_) {
                changed.push(egraph.find(guard));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// The one rule that instantiates **every** `forall` in the program. Quantifiers
/// are e-nodes, so this replaces the old per-quantifier, per-trigger-group rule
/// minting entirely.
pub(crate) fn forall_rule(table: Arc<RwLock<RecipeTable>>) -> Rule {
    let searcher = ForallSearcher;
    let applier = ForallApplier {
        table,
        memo: Arc::new(Memo::new()),
    };
    timed(Rewrite::new("forall-instantiate", searcher, applier).expect("forall rule"))
}

/// A partial substitution: one slot per value binder, and one per type parameter
/// of the recipe (`Generic(i)` maps to `tys[i]`).
#[derive(Clone)]
pub(super) struct Sigma {
    pub(super) vals: Vec<Option<Id>>,
    pub(super) tys: Vec<Option<Type>>,
}
