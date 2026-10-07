//! Pure `forall` quantifiers: trigger matching against a prepared term and the
//! rule that instantiates a quantifier body.
//!
//! Builds its instances through `super::recipe`; nothing here flows back the
//! other way.

use std::sync::{Arc, RwLock};

use egg::{
    Applier, Changes, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{Discriminant, FuncId, RecipeId, Symbolic};
use crate::verify::quant::RecipeTable;
use crate::vmir::{Literal, Type};

use super::*;

/// One trigger pattern term, with every head resolved to its verifier `FuncId`
/// (the applier has no registry access). The `vmir::TrigTerm` grammar, flattened
/// to what the e-graph speaks: an `App` matches a `FuncApp` node with the same
/// function, type payload and arity, whose argument e-classes match `args`
/// recursively.
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
/// extending `sigma`. Returns every consistent extension (an e-class may hold
/// several nodes matching the pattern's head, each binding σ differently), or an
/// empty vector when the term cannot match.
pub(super) fn match_term(
    egraph: &EGraph<Symbolic, ConstFold>,
    term: &PreparedTerm,
    class: Id,
    caps: &[Id],
    sigma: &Sigma,
) -> Vec<Sigma> {
    let class = egraph.find(class);
    match term {
        PreparedTerm::Bound(i) => match sigma[*i] {
            // A repeated binder must land on the same e-class.
            Some(prev) if egraph.find(prev) != class => vec![],
            Some(_) => vec![sigma.clone()],
            None => {
                let mut next = sigma.clone();
                next[*i] = Some(class);
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
                if f != func || tys != type_args || children.len() != args.len() {
                    continue;
                }
                // Thread σ left to right across the arguments, branching on every
                // consistent way each of them matches.
                let mut partials = vec![sigma.clone()];
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
pub(super) struct ForallSearcher {
    pub(super) table: Arc<RwLock<RecipeTable>>,
}

/// The variable a match of [`ForallSearcher::search_changes`] binds to the class
/// where a trigger term may match anew.
fn anchor() -> Var {
    var("?anchor")
}

impl PreparedTerm {
    /// Adds the functions and literals of this term to `funcs` and `lits`, and
    /// returns its height (an `App` over leaves is 1).
    fn shape(&self, funcs: &mut Vec<FuncId>, lits: &mut Vec<Literal>) -> usize {
        match self {
            PreparedTerm::Bound(_) | PreparedTerm::Capture(_) => 0,
            PreparedTerm::Lit(lit) => {
                lits.push(lit.clone());
                1
            }
            PreparedTerm::App { func, args, .. } => {
                funcs.push(*func);
                1 + args.iter().map(|a| a.shape(funcs, lits)).max().unwrap_or(0)
            }
        }
    }
}

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

    /// A quantifier new to its class (or whose captures changed) is instantiated in
    /// full. Every other quantifier only at the classes where one of its trigger
    /// terms may match anew: those holding a changed application or literal of a
    /// trigger, and the classes above them through trigger applications, up to the
    /// height of the tallest trigger term. A match binds such a class to
    /// [`anchor`]; the applier joins the term rooted there with the group's other
    /// terms matched anywhere.
    fn search_changes(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        changes: &Changes<Symbolic>,
        _limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let mut found: Vec<(Id, Subst)> = changes
            .nodes(&Discriminant::Forall)
            .map(|(class, _)| (*class, Subst::default()))
            .collect();
        let Some(quants) = egraph.classes_for_op(&Discriminant::Forall) else {
            return group_matches(found);
        };
        // Each quantifier class with the root functions of its trigger terms, and
        // the shape of all of them.
        let table = self.table.read().expect("recipe table lock");
        let (mut funcs, mut lits, mut height) = (Vec::new(), Vec::new(), 0);
        let mut roots_of: Vec<(Id, Vec<FuncId>)> = Vec::new();
        for q in quants {
            let mut roots = Vec::new();
            for node in &egraph[q].nodes {
                let Symbolic::Forall(rid, _) = node else {
                    continue;
                };
                for term in table.get(*rid).groups.iter().flatten() {
                    height = height.max(term.shape(&mut funcs, &mut lits));
                    roots.extend(term.root_func());
                }
            }
            roots.sort_unstable();
            roots.dedup();
            roots_of.push((q, roots));
        }
        drop(table);
        funcs.sort_unstable();
        funcs.dedup();
        let in_trigger = |n: &Symbolic| match n {
            Symbolic::FuncApp(f, ..) => funcs.binary_search(f).is_ok(),
            Symbolic::Lit(lit) => lits.contains(lit),
            _ => false,
        };
        let mut level: Vec<Id> = changes
            .iter()
            .filter(|(_, n)| in_trigger(n))
            .map(|(class, _)| *class)
            .collect();
        let mut anchors: Vec<Id> = Vec::new();
        let is_trigger_app = |n: &Symbolic| matches!(n, Symbolic::FuncApp(..)) && in_trigger(n);
        for _ in 0..height {
            level.sort_unstable();
            level.dedup();
            anchors.extend(&level);
            level = level
                .iter()
                .flat_map(|&class| parents_where(egraph, class, is_trigger_app))
                .map(|(up, _)| up)
                .collect();
        }
        anchors.sort_unstable();
        anchors.dedup();
        // The root functions each anchor holds an application of.
        let anchor_funcs: Vec<(Id, Vec<FuncId>)> = anchors
            .into_iter()
            .map(|a| {
                let mut fs: Vec<FuncId> = egraph[a]
                    .nodes
                    .iter()
                    .filter_map(|n| match n {
                        Symbolic::FuncApp(f, ..) => Some(*f),
                        _ => None,
                    })
                    .collect();
                fs.dedup();
                (a, fs)
            })
            .filter(|(_, fs)| !fs.is_empty())
            .collect();
        for (q, roots) in &roots_of {
            for (a, fs) in &anchor_funcs {
                if fs.iter().any(|f| roots.binary_search(f).is_ok()) {
                    let mut subst = Subst::default();
                    subst.insert(anchor(), *a);
                    found.push((*q, subst));
                }
            }
        }
        group_matches(found)
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

/// Applier: for the matched forall e-class, take each `Forall(recipe, caps)` node
/// in it and pair it with every σ under which one of its recipe's trigger groups
/// matches the graph. Instantiate the body at `caps ++ σ` and add the **guarded**
/// clause `Ite(forall, res[caps, σ], true) == true`: the instance is released only
/// once that forall e-class merges `true` (existing `ite-true` rule), so
/// instantiating is sound regardless of the quantifier's truth.
///
/// Memoized per `(recipe, caps ++ σ)` (canonicalized at insert) — a saturation-cost
/// guard only, since instances are idempotent.
///
/// The table is interned into as bodies are walked, so this holds it behind the
/// shared lock rather than a snapshot. Only reads happen here: a nested recipe is
/// interned innermost-first with its encloser, on the eval walk, never from inside
/// a rule.
pub(super) struct ForallApplier {
    pub(super) table: Arc<RwLock<RecipeTable>>,
    pub(super) memo: Arc<Memo<(RecipeId, Vec<Id>)>>,
}

impl Applier<Symbolic, ConstFold> for ForallApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // With an anchor, only the σ under which some trigger term matches at the
        // anchor class are new (see `ForallSearcher::search_changes`).
        let anchor = subst.get(anchor()).map(|&a| egraph.find(a));
        // Distinct capture tuples in one e-class are distinct quantifiers (each
        // yields its own instances); the guard is this e-class either way.
        let quants: Vec<(RecipeId, Box<[Id]>)> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::Forall(rid, caps) => Some((*rid, caps.clone())),
                _ => None,
            })
            .collect();

        let table = self.table.read().expect("recipe table lock");
        // Collect the instances first: `build_instance` needs `&mut egraph`.
        let mut instances: Vec<(RecipeId, Vec<Id>)> = Vec::new();
        for (rid, caps) in &quants {
            let recipe = table.get(*rid);
            let caps: Vec<Id> = caps.iter().map(|&c| egraph.find(c)).collect();
            for group in &recipe.groups {
                let empty: Sigma = vec![None; recipe.n_bound];
                // Which term is matched first, and where: the whole group's first
                // term anywhere, or (anchored) each term at the anchor in turn.
                let starts: Vec<(usize, Vec<Sigma>)> = match anchor {
                    None => match group.first() {
                        Some(first) => vec![(0, match_term_anywhere(egraph, first, &caps, &empty))],
                        None => vec![],
                    },
                    Some(at) => (0..group.len())
                        .map(|i| (i, match_term(egraph, &group[i], at, &caps, &empty)))
                        .collect(),
                };
                let mut sigmas = Vec::new();
                for (first, mut partial) in starts {
                    for (i, term) in group.iter().enumerate() {
                        if i == first || partial.is_empty() {
                            continue;
                        }
                        partial = partial
                            .iter()
                            .flat_map(|s| match_term_anywhere(egraph, term, &caps, s))
                            .collect();
                    }
                    sigmas.extend(partial);
                }
                for sigma in sigmas {
                    // A group covers every binder (typecheck-enforced), so a
                    // complete match leaves no slot open.
                    let Some(sigma): Option<Vec<Id>> = sigma.into_iter().collect() else {
                        continue;
                    };
                    // The seed is the body's leading temps: captures then binders.
                    let mut vals = caps.clone();
                    vals.extend(sigma);
                    if self.memo.insert((*rid, vals.clone())) {
                        instances.push((*rid, vals));
                    }
                }
            }
        }

        let mut changed = Vec::new();
        for (rid, vals) in instances {
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
            let instance_vals = build_instance_vals_guarded(
                egraph,
                &recipe.insts,
                &vals,
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
    let searcher = ForallSearcher {
        table: Arc::clone(&table),
    };
    let applier = ForallApplier {
        table,
        memo: Arc::new(Memo::new()),
    };
    timed(Rewrite::new("forall-instantiate", searcher, applier).expect("forall rule"))
}

/// A partial bound-variable substitution, one slot per binder.
pub(super) type Sigma = Vec<Option<Id>>;
