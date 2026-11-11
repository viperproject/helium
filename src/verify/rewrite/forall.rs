//! Pure `forall` quantifiers: trigger matching against a prepared term and the
//! rule that instantiates a quantifier body.
//!
//! Builds its instances through `super::recipe`; nothing here flows back the
//! other way.

use std::sync::{Arc, RwLock};

use egg::{
    Applier, EGraph, Id, Language, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use crate::verify::analysis::{ConstFold, Data};
use crate::verify::lang::{Discriminant, FuncId, RecipeId, SeqAux, Symbolic};
use crate::verify::quant::{Guard, RecipeTable};
use crate::verify::stats;
use crate::vmir::{BinOp, Literal, SeqOp, Type, Val};

use super::*;

/// The head of a trigger application, resolved to what the e-graph speaks.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum TermHead {
    /// A `FuncApp` with this function and type payload.
    Func {
        func: FuncId,
        type_args: Box<[Type]>,
    },
    /// A `Symbolic::Seq` of this element type and operation.
    Seq(Type, SeqOp),
    /// A `Symbolic::SeqLit` of this element type.
    SeqLit(Type),
    /// A `Symbolic::SeqAux` of this element type and kind.
    SeqAux(Type, SeqAux),
}

impl TermHead {
    /// The `classes_by_op` bucket holding this head's applications.
    fn discriminant(&self) -> Discriminant {
        match self {
            TermHead::Func { func, .. } => Discriminant::FuncApp(*func),
            TermHead::Seq(_, op) => Discriminant::Seq(*op),
            TermHead::SeqLit(_) => Discriminant::SeqLit,
            TermHead::SeqAux(_, aux) => Discriminant::SeqAux(*aux),
        }
    }

    /// When `node` is an application of this head: the head's pattern types, the
    /// node's types (to be matched by [`match_types`], which binds a generic
    /// quantifier's open type parameters), and the node's children.
    fn children_of<'s, 'a>(
        &'s self,
        node: &'a Symbolic,
    ) -> Option<(&'s [Type], &'a [Type], &'a [Id])> {
        match (self, node) {
            (TermHead::Func { func, type_args }, Symbolic::FuncApp(f, tys, children))
                if f == func =>
            {
                Some((type_args, tys, children))
            }
            (TermHead::Seq(elem_ty, op), Symbolic::Seq(e, o, children)) if o == op => Some((
                std::slice::from_ref(elem_ty),
                std::slice::from_ref(&**e),
                children,
            )),
            (TermHead::SeqLit(elem_ty), Symbolic::SeqLit(e, elem)) => Some((
                std::slice::from_ref(elem_ty),
                std::slice::from_ref(&**e),
                std::slice::from_ref(elem),
            )),
            (TermHead::SeqAux(elem_ty, aux), Symbolic::SeqAux(e, a, children)) if a == aux => {
                Some((
                    std::slice::from_ref(elem_ty),
                    std::slice::from_ref(&**e),
                    children,
                ))
            }
            _ => None,
        }
    }
}

/// One trigger pattern term, with every head resolved (the applier has no
/// registry access). The `vmir::TrigTerm` grammar, flattened to what the e-graph
/// speaks: an `App` matches a node of its head with the same arity, whose
/// argument e-classes match `args` recursively.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum PreparedTerm {
    Bound(usize),
    Capture(usize),
    Lit(Literal),
    App {
        head: TermHead,
        args: Vec<PreparedTerm>,
    },
}

impl PreparedTerm {
    /// The head a term is rooted at, which is the searcher's anchor. Only an
    /// `App` can be a top-level trigger term (typecheck-enforced).
    fn root_head(&self) -> Option<&TermHead> {
        match self {
            PreparedTerm::App { head, .. } => Some(head),
            _ => None,
        }
    }
}

/// Match `term` against e-class `class` under the forall node's capture children,
/// extending `sigma`. Returns every consistent extension (an e-class may hold
/// several nodes matching the pattern's head, each binding σ differently), or an
/// empty vector when the term cannot match. An application also fixes the
/// generic quantifier's open type parameters its type arguments mention. Each
/// extension is at least as deep as every class it matched (see
/// [`Sigma::generation`]).
pub(super) fn match_term(
    egraph: &EGraph<Symbolic, ConstFold>,
    term: &PreparedTerm,
    class: Id,
    caps: &[Id],
    sigma: &Sigma,
) -> Vec<Sigma> {
    let class = egraph.find(class);
    let matched = || sigma.through(egraph, class);
    match term {
        PreparedTerm::Bound(i) => match sigma.vals[*i] {
            // A repeated binder must land on the same e-class.
            Some(prev) if egraph.find(prev) != class => vec![],
            Some(_) => vec![matched()],
            None => {
                let mut next = matched();
                next.vals[*i] = Some(class);
                vec![next]
            }
        },
        PreparedTerm::Capture(c) => {
            if egraph.find(caps[*c]) == class {
                vec![matched()]
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
                vec![matched()]
            } else {
                vec![]
            }
        }
        PreparedTerm::App { head, args } => {
            let mut out = Vec::new();
            for node in &egraph[class].nodes {
                let Some((pats, tys, children)) = head.children_of(node) else {
                    continue;
                };
                if children.len() != args.len() {
                    continue;
                }
                let Some(sigma) = match_types(pats, tys, &matched()) else {
                    continue;
                };
                // Thread σ left to right across the arguments, branching on every
                // consistent way each of them matches.
                let mut partials = vec![sigma];
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

/// `sigma` extended so that the pattern type arguments `pats` equal the ground
/// `ground`, fixing open type parameters as needed; `None` on a mismatch. For a
/// ground quantifier (no type payload) this is plain equality.
fn match_types(pats: &[Type], ground: &[Type], sigma: &Sigma) -> Option<Sigma> {
    if pats.len() != ground.len() {
        return None;
    }
    if sigma.tys.is_empty() {
        return (pats == ground).then(|| sigma.clone());
    }
    // Fixed parameters are substituted in; an open one stays `Generic(i)` and is
    // bound by the match.
    let mut bound: Vec<Option<Type>> = vec![None; sigma.tys.len()];
    for (p, g) in pats.iter().zip(ground) {
        if !p.subst_generics(&sigma.tys).match_generics(g, &mut bound) {
            return None;
        }
    }
    let mut next = sigma.clone();
    for (slot, t) in next.tys.iter_mut().zip(bound) {
        if let Some(t) = t {
            *slot = t;
        }
    }
    Some(next)
}

/// Every σ that matches `term` somewhere in the e-graph, extending `sigma`. Scans
/// the classes holding an application of the term's root head.
pub(super) fn match_term_anywhere(
    egraph: &EGraph<Symbolic, ConstFold>,
    term: &PreparedTerm,
    caps: &[Id],
    sigma: &Sigma,
) -> Vec<Sigma> {
    let Some(root) = term.root_head() else {
        return vec![];
    };
    let Some(classes) = egraph.classes_for_op(&root.discriminant()) else {
        return vec![];
    };
    // Between a union and the next rebuild the index still lists the class that
    // was merged away, which resolves to the survivor: once the classes of many
    // applications merge (each proven `true`), the survivor would be matched once
    // per class it absorbed, every match walking all of their nodes.
    let mut seen = HashSet::default();
    classes
        .map(|class| egraph.find(class))
        .filter(|&class| seen.insert(class))
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

/// Applier: for the matched forall e-class, take each `Forall(recipe, caps)` node
/// in it and pair it with every σ under which one of its recipe's trigger groups
/// matches the graph. Instantiate the body at `caps ++ σ` and add the **guarded**
/// clause `Ite(forall, res[caps, σ], true) == true`: the instance is released only
/// once that forall e-class merges `true` (existing `ite-true` rule), so
/// instantiating is sound regardless of the quantifier's truth.
///
/// A generic quantifier (non-empty type payload) is instantiated at the types
/// its payload and the matched trigger terms fix: its body is built with them
/// substituted in, so every node it adds is ground (or, for a nested quantifier,
/// carries the fixed types in its own payload).
///
/// Memoized per `(recipe, types, caps ++ σ)` (canonicalized at insert) — a
/// saturation-cost guard only, since instances are idempotent.
///
/// An instance's generation is one more than the deepest class its trigger
/// matched (captures included); one deeper than [`MAX_GENERATION`] is deferred:
/// not built and not memoized, so it is built once a union makes a matched class
/// shallower.
///
/// The table is interned into as bodies are walked, so this holds it behind the
/// shared lock rather than a snapshot. Only reads happen here: a nested recipe is
/// interned innermost-first with its encloser, on the eval walk, never from inside
/// a rule.
pub(super) struct ForallApplier {
    pub(super) table: Arc<RwLock<RecipeTable>>,
    pub(super) memo: Arc<Memo<(RecipeId, Vec<Type>, Vec<Id>)>>,
}

/// One body step evaluated against the e-graph **without adding to it**: the
/// e-class that already holds it, if any, and the value it folds to.
struct Probe {
    class: Option<Id>,
    value: Data,
}

impl Probe {
    fn of_class(egraph: &EGraph<Symbolic, ConstFold>, class: Id) -> Self {
        Probe {
            class: Some(egraph.find(class)),
            value: egraph[class].data.value.clone(),
        }
    }

    fn of_lit(egraph: &EGraph<Symbolic, ConstFold>, lit: &Literal) -> Self {
        Probe {
            class: egraph.lookup(Symbolic::Lit(lit.clone())),
            value: Data::Known(lit.clone()),
        }
    }

    /// The probe of `node`, whose child `i` is `ops[i]`: the existing class when
    /// every child has one and the graph already holds the node, else just the
    /// folded value (so `5 > 0` is true at σ = 5 even though the comparison
    /// has never been built).
    fn of_node(egraph: &EGraph<Symbolic, ConstFold>, node: Symbolic, ops: &[&Probe]) -> Self {
        // `x == x` holds and `x < x` fails whether or not the comparison was
        // ever built.
        if let Symbolic::Binary(op @ (BinOp::Eq | BinOp::LtI | BinOp::LtR), _) = node
            && let [Probe { class: Some(l), .. }, Probe { class: Some(r), .. }] = ops
            && l == r
        {
            return Probe::of_lit(egraph, &Literal::Bool(op == BinOp::Eq));
        }
        let real: Option<Vec<Id>> = ops.iter().map(|p| p.class).collect();
        let class = real
            .and_then(|real| egraph.lookup(node.clone().map_children(|i| real[usize::from(i)])));
        match class {
            // The class's own value also reflects the unions it took part in
            // (a guard assumed elsewhere is merged with `true`).
            Some(class) if !matches!(egraph[class].data.value, Data::Unknown) => {
                Probe::of_class(egraph, class)
            }
            class => Probe {
                class: class.map(|c| egraph.find(c)),
                value: egraph
                    .analysis
                    .fold_value(&node, |i: &Id| &ops[usize::from(*i)].value),
            },
        }
    }
}

/// Whether every conjunct of `guard` is already `true` at the seed `vals`
/// (captures ++ σ) and the types `tys` the match fixed. Evaluates the guard's
/// steps with lookups only: a step is known when the graph already holds it or
/// when it constant-folds from known operands, and a conjunct holds when it is
/// known to be `true`.
fn guard_holds(
    egraph: &EGraph<Symbolic, ConstFold>,
    guard: &Guard,
    vals: &[Id],
    tys: &[Type],
) -> bool {
    let mut probes: Vec<Probe> = vals.iter().map(|&c| Probe::of_class(egraph, c)).collect();
    for inst in &guard.insts {
        let Some((node, args)) = probe_shape(inst, tys) else {
            continue;
        };
        // A literal operand gets a probe of its own; a temp reads its slot.
        let lits: Vec<Probe> = args
            .iter()
            .filter_map(|v| match v {
                Val::Literal(l) => Some(Probe::of_lit(egraph, l)),
                Val::Temp(_) => None,
            })
            .collect();
        let mut lits = lits.iter();
        let ops: Vec<&Probe> = args
            .iter()
            .map(|v| match v {
                Val::Temp(n) => &probes[*n],
                Val::Literal(_) => lits.next().expect("one probe per literal operand"),
            })
            .collect();
        let probe = Probe::of_node(egraph, node, &ops);
        probes.push(probe);
    }
    guard.conds.iter().all(|c| {
        let value = match c {
            Val::Temp(n) => probes[*n].value.clone(),
            Val::Literal(l) => Data::Known(l.clone()),
        };
        matches!(value, Data::Known(Literal::Bool(true)))
    })
}

/// The e-node a value-producing step adds, at the types `tys`, with its
/// children numbered `0..n` as indices into the returned operands; `None` for a
/// step that occupies no temp. Mirrors `build_instance_vals_impl`.
fn probe_shape<'a>(inst: &'a AxiomInst, tys: &[Type]) -> Option<(Symbolic, Vec<&'a Val>)> {
    let ix = |n: usize| -> Box<[Id]> { (0..n).map(Id::from).collect() };
    let ty = |t: &Type| Box::new(t.subst_generics(tys));
    let [i0, i1, i2] = [0, 1, 2].map(Id::from);
    Some(match inst {
        AxiomInst::Val(p) => match p {
            AxiomPure::Binary(op, l, r) => (Symbolic::Binary(*op, [i0, i1]), vec![l, r]),
            AxiomPure::Ternary(c, t, e) => (Symbolic::Ite([i0, i1, i2]), vec![c, t, e]),
            AxiomPure::RealCast(v) => (Symbolic::RealCast(i0), vec![v]),
            AxiomPure::App {
                func,
                type_args,
                args,
            } => (
                Symbolic::FuncApp(
                    *func,
                    type_args.iter().map(|t| t.subst_generics(tys)).collect(),
                    ix(args.len()),
                ),
                args.iter().collect(),
            ),
            AxiomPure::Seq { elem_ty, op, args } => (
                Symbolic::Seq(ty(elem_ty), *op, ix(args.len())),
                args.iter().collect(),
            ),
            AxiomPure::SeqLit { elem_ty, elem } => (Symbolic::SeqLit(ty(elem_ty), i0), vec![elem]),
            // Its value is the equality; the tag built beside it has no bearing
            // on that value.
            AxiomPure::SeqEq { l, r, .. } => (Symbolic::Binary(BinOp::Eq, [i0, i1]), vec![l, r]),
        },
        AxiomInst::Forall {
            recipe,
            type_args,
            caps,
        } => (
            Symbolic::Forall(
                *recipe,
                type_args.iter().map(|t| t.subst_generics(tys)).collect(),
                ix(caps.len()),
            ),
            caps.iter().collect(),
        ),
        AxiomInst::Assume(_) | AxiomInst::Token { .. } => return None,
    })
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
        let mut instances: Vec<(RecipeId, Vec<Type>, Vec<Id>, u32)> = Vec::new();
        let mut deferred = 0;
        let mut skipped = 0;
        for (rid, tys, caps) in &quants {
            let recipe = table.get(*rid);
            let caps: Vec<Id> = caps.iter().map(|&c| egraph.find(c)).collect();
            let empty = caps.iter().fold(
                Sigma {
                    vals: vec![None; recipe.n_bound],
                    tys: tys.to_vec(),
                    generation: 0,
                },
                |s, &c| s.through(egraph, c),
            );
            for group in &recipe.groups {
                let Some((anchor, rest)) = group.split_first() else {
                    continue;
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
                    let generation = sigma.generation + 1;
                    // A group covers every binder (typecheck-enforced), so a
                    // complete match leaves no slot open.
                    let Some(bound): Option<Vec<Id>> = sigma.vals.into_iter().collect() else {
                        continue;
                    };
                    // Each group, with what the payload already fixes, binds every
                    // type parameter this level uses (translation-enforced,
                    // `check_typed`); one only a nested quantifier uses may stay open.
                    assert!(
                        recipe.own_tys.iter().all(|&i| is_ground(&sigma.tys[i])),
                        "a trigger match of a generic quantifier must fix the types it uses"
                    );
                    if generation > MAX_GENERATION {
                        deferred += 1;
                        continue;
                    }
                    // The seed is the body's leading temps: captures then binders.
                    let mut vals = caps.clone();
                    vals.extend(bound);
                    // Like a deferred instance, a skipped one is not memoized:
                    // it is built once its guard becomes true.
                    if let Some(guard) = &recipe.guard
                        && !guard_holds(egraph, guard, &vals, &sigma.tys)
                    {
                        skipped += 1;
                        continue;
                    }
                    if self.memo.insert((*rid, sigma.tys.clone(), vals.clone())) {
                        instances.push((*rid, sigma.tys, vals, generation));
                    }
                }
            }
        }
        if deferred > 0 {
            stats::bump(|s| s.inst_deferred += deferred);
        }
        if skipped > 0 {
            stats::bump(|s| s.inst_guard_skips += skipped);
        }
        if let Some(peak) = instances.iter().map(|i| i.3).max() {
            stats::bump(|s| s.inst_generation_peak = s.inst_generation_peak.max(peak.into()));
        }

        let mut changed = Vec::new();
        for (rid, tys, vals, generation) in instances {
            let recipe = table.get(rid);
            let insts = if tys.is_empty() {
                std::borrow::Cow::Borrowed(&recipe.insts)
            } else {
                std::borrow::Cow::Owned(subst_types(&recipe.insts, &tys))
            };
            debug_assert_eq!(
                vals.len(),
                recipe.n_caps + recipe.n_bound,
                "instance seed is captures ++ sigma"
            );
            // Every node the instance adds is created at its generation.
            let outer = egraph.analysis.building.replace(generation);
            // `eclass` as the token guard: a `g%pre` token this body propagates is
            // released only under the quantifier's own truth, the same gate the
            // instance itself sits behind below. A quantifier that is merely
            // *present* materializes its body but activates no callee's axioms.
            let instance_vals =
                build_instance_vals_guarded(egraph, &insts, &vals, &mut changed, Some(eclass));
            let res = resolve_val(egraph, &instance_vals, &recipe.res);
            let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
            let guard = egraph.add(Symbolic::Ite([eclass, res, true_]));
            egraph.analysis.building.set(outer);
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

/// A partial substitution: one slot per binder, and the quantifier's type
/// payload being fixed (see [`Symbolic::Forall`]: entry `i` is open while it is
/// `Generic(i)`; empty for a ground quantifier).
#[derive(Clone)]
pub(super) struct Sigma {
    pub(super) vals: Vec<Option<Id>>,
    pub(super) tys: Vec<Type>,
    /// The deepest [generation](crate::verify::analysis::ClassData::generation) of
    /// the classes matched so far.
    pub(super) generation: u32,
}

impl Sigma {
    /// This substitution, having also matched `class`.
    fn through(&self, egraph: &EGraph<Symbolic, ConstFold>, class: Id) -> Sigma {
        Sigma {
            vals: self.vals.clone(),
            tys: self.tys.clone(),
            generation: self.generation.max(egraph[class].data.generation),
        }
    }
}

/// The deepest instance generation built. A termination bound: a matching loop
/// (an instance that builds a new term its own trigger matches) has no fixpoint,
/// and there are finitely many instances of each generation, since every class
/// of generation `g` holds a node built by an instance of generation at most `g`
/// or by the other rules from such classes (which terminate on their own).
///
/// Sized above the depth proofs need: the deepest measured is 11 (a crate's
/// type-level arithmetic; a bound of 8 loses proofs there). Z3's defaults
/// instantiate eagerly up to cost 10 and at a final check up to 20. Below the
/// iteration limit, so a matching loop ends its run at a fixpoint.
pub(crate) const MAX_GENERATION: u32 = 16;

/// Whether `t` mentions no type parameter.
fn is_ground(t: &Type) -> bool {
    let mut g = crate::dhash::HashSet::default();
    t.collect_generics(&mut g);
    g.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After many applications' classes merge, and before the rebuild that
    /// refreshes the op index, a trigger still matches each application once:
    /// the index's merged-away entries resolve to the class already matched.
    #[test]
    fn a_class_merged_mid_iteration_is_matched_once() {
        let mut g = EGraph::<Symbolic, ConstFold>::default();
        let f = FuncId(0);
        let n = 50;
        let apps: Vec<Id> = (0..n)
            .map(|i| {
                let x = g.add(Symbolic::Fresh(i));
                g.add(Symbolic::FuncApp(f, Box::new([]), Box::new([x])))
            })
            .collect();
        g.rebuild();
        for &app in &apps[1..] {
            g.union(apps[0], app);
        }
        let trigger = PreparedTerm::App {
            head: TermHead::Func {
                func: f,
                type_args: Box::new([]),
            },
            args: vec![PreparedTerm::Bound(0)],
        };
        let empty = Sigma {
            vals: vec![None],
            tys: vec![],
            generation: 0,
        };
        let sigmas = match_term_anywhere(&g, &trigger, &[], &empty);
        assert_eq!(sigmas.len(), n as usize);
    }

    /// `forall x :: {f(x)} f(g(x))` from `f(a)`: the instance at `x = g^k(a)` has
    /// generation `k + 1`, so the chain stops with `g^B(a)` built
    /// (`B = MAX_GENERATION`) and its instance deferred, at a fixpoint. The
    /// deferred instance is not remembered as done: once unions with
    /// generation-0 terms make the match shallow, it is built.
    #[test]
    fn deferred_instance_is_built_once_its_binding_becomes_shallow() {
        use crate::verify::quant::{QuantRecipe, RecipeTable};
        use crate::vmir::Val;
        let (f, g_) = (FuncId(0), FuncId(1));
        let app = |func, args| {
            AxiomInst::Val(AxiomPure::App {
                func,
                type_args: vec![],
                args,
            })
        };
        let (table, rid) = RecipeTable::of(QuantRecipe {
            n_caps: 0,
            n_bound: 1,
            n_tys: 0,
            own_tys: vec![],
            groups: vec![vec![PreparedTerm::App {
                head: TermHead::Func {
                    func: f,
                    type_args: Box::new([]),
                },
                args: vec![PreparedTerm::Bound(0)],
            }]],
            insts: vec![app(g_, vec![Val::Temp(0)]), app(f, vec![Val::Temp(1)])],
            res: Val::Literal(Literal::Bool(true)),
            guard: None,
        });
        let rules = [forall_rule(Arc::new(RwLock::new(table)))];
        let mut g = EGraph::<Symbolic, ConstFold>::default();
        let a = g.add(Symbolic::Fresh(0));
        g.add(Symbolic::FuncApp(f, Box::new([]), Box::new([a])));
        let q = g.add(Symbolic::Forall(rid, Box::new([]), Box::new([])));
        let t = g.add(Symbolic::Lit(Literal::Bool(true)));
        g.union(q, t);
        g.rebuild();
        let run = |g| {
            let r = egg::Runner::default()
                .with_egraph(g)
                .with_iter_limit(10 * MAX_GENERATION as usize)
                .run(&rules);
            assert!(matches!(r.stop_reason, Some(egg::StopReason::Saturated)));
            r.egraph
        };
        let g_of = |g: &EGraph<Symbolic, ConstFold>, x| {
            g.lookup(Symbolic::FuncApp(g_, Box::new([]), Box::new([x])))
        };
        let mut g = run(g);
        let mut deep = a;
        for _ in 0..MAX_GENERATION {
            deep = g_of(&g, deep).expect("within the bound");
        }
        assert_eq!(g[deep].data.generation, MAX_GENERATION);
        assert!(g_of(&g, deep).is_none(), "one past the bound is deferred");
        // The program's own `f(b)`, with `b == g^B(a)`: congruence makes the
        // whole trigger match generation 0.
        let named = g.add(Symbolic::Fresh(1));
        g.add(Symbolic::FuncApp(f, Box::new([]), Box::new([named])));
        g.union(deep, named);
        g.rebuild();
        let g = run(g);
        assert!(g_of(&g, deep).is_some());
    }
}
