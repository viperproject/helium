//! ADT rules: constructor/destructor/tag reductions, minted per concept as the
//! `FuncRegistry` grows, and the shared op-indexed searcher they use.

use crate::dhash::{HashMap, HashSet};
use std::rc::Rc;

use egg::{
    Applier, Changes, EGraph, Id, Language, PatternAst, Rewrite, SearchMatches, Searcher, Subst,
    Symbol, Var,
};

use crate::verify::analysis::{ConstFold, Data};
use crate::verify::lang::{Discriminant, FuncId, Symbolic};
use crate::vmir::{Literal, Type};

use super::*;

/// Build the projection reduction `accessor(ctor(a0..an)) ⇒ a_index` for a
/// single member id, including ones minted after `VerifyContext` construction
/// (monomorphic Option instances).
pub fn proj_rule(accessor: FuncId, ctor: FuncId, index: usize) -> Rule {
    timed(
        Rewrite::new(
            format!("proj-{}", accessor.0),
            UnaryAppSearcher {
                func: accessor,
                reads: ArgReads::Projection,
            },
            ProjApplier { ctor, index },
        )
        .expect("valid proj rewrite"),
    )
}

/// Build the injectivity rule for one constructor: an e-class holding two
/// applications of the same constructor unions their arguments pairwise
/// (`C(a..) ≡ C(b..) ⟹ aᵢ ≡ bᵢ`). Sound because ADT constructors are free.
///
/// Congruence runs only *forward*, and [`proj_rule`] recovers the backward
/// direction only where a `projᵢ` application exists. Two constructor terms can
/// land in one class with no projection over them, and then the component
/// equalities are reachable only through this rule.
pub fn inj_rule(ctor: FuncId) -> Rule {
    timed(
        Rewrite::new(
            format!("inj-{}", ctor.0),
            AxiomTriggerSearcher {
                func: ctor,
                token: None,
            },
            InjApplier { ctor },
        )
        .expect("valid injectivity rewrite"),
    )
}

/// Build the discriminator reduction `tag_fn(ctor_C(..)) ⇒ index_C` for a single
/// (possibly synthesised) tag function. Companion to [`proj_rule`].
pub fn tag_rule(tag_fn: FuncId, ctor_tags: HashMap<FuncId, usize>) -> Rule {
    let mut ctors: Vec<FuncId> = ctor_tags.keys().copied().collect();
    ctors.sort_unstable();
    timed(
        Rewrite::new(
            format!("tag-{}", tag_fn.0),
            UnaryAppSearcher {
                func: tag_fn,
                reads: ArgReads::Constructors(ctors),
            },
            TagApplier { ctor_tags },
        )
        .expect("valid tag rewrite"),
    )
}

/// The pattern variable bound to a tag call's argument.
pub(super) fn tag_x() -> Var {
    var("?x")
}

/// Searcher for a unary application `func(base)`: matches any `FuncApp(func, _,
/// [base])` node (its ground type args live in the discriminant and are ignored
/// here) and binds `?x` to the single value arg. FuncApp isn't string-matchable,
/// so this is hand-written. Shared by the tag and projection reductions.
pub(super) struct UnaryAppSearcher {
    pub(super) func: FuncId,
    pub(super) reads: ArgReads,
}

/// What a unary-application rule reads in its argument's class, so which changes
/// there make it search again.
pub(super) enum ArgReads {
    /// Whether the class holds an application of one of these constructors
    /// ([`TagApplier`]).
    Constructors(Vec<FuncId>),
    /// The constructor applications and `ite`s the projection extracts through,
    /// down `ite` arms ([`ProjApplier::project`]).
    Projection,
}

/// The unary applications over each class whose projection may have changed,
/// by function: see [`projected`].
type Projected = HashMap<FuncId, Vec<(Id, Subst)>>;

thread_local! {
    /// [`projected`] for the change set with this serial. One projection rule is
    /// minted per accessor, and each would otherwise walk the same changes.
    static PROJECTED: std::cell::RefCell<Option<(u64, Rc<Projected>)>> =
        const { std::cell::RefCell::new(None) };
}

/// The unary applications, by function, over each class whose projection (see
/// [`ProjApplier::project`]) may have changed: the classes that gained a
/// constructor application or an `ite`, and every class above them through `ite`
/// arms. A class whose data is a literal holds no constructor value and is no arm
/// of one, so the walk does not enter it (the `true` class, which holds every
/// proven fact, among them).
fn projected(egraph: &EGraph<Symbolic, ConstFold>, changes: &Changes<Symbolic>) -> Rc<Projected> {
    if let Some(hit) = PROJECTED.with(|p| {
        let p = p.borrow();
        p.as_ref()
            .filter(|(serial, _)| *serial == changes.serial())
            .map(|(_, out)| Rc::clone(out))
    }) {
        return hit;
    }
    let opaque = |class: Id| egraph[class].data.known().is_some();
    let mut stack: Vec<Id> = changes
        .iter()
        .filter(|(class, node)| match node {
            Symbolic::Ite(..) => true,
            Symbolic::FuncApp(..) => matches!(egraph[*class].data, Data::Ctor(..)),
            _ => false,
        })
        .map(|(class, _)| *class)
        .filter(|&class| !opaque(class))
        .collect();
    let mut seen = HashSet::default();
    let mut out = Projected::default();
    let x = [tag_x()];
    while let Some(class) = stack.pop() {
        if !seen.insert(class) {
            continue;
        }
        let ups = parents_where(egraph, class, |n| {
            matches!(n, Symbolic::Ite(..))
                || matches!(n, Symbolic::FuncApp(_, _, a) if a.len() == 1)
        });
        for (up, node) in ups {
            match node {
                Symbolic::Ite([_, t, e]) if (t == class || e == class) && !opaque(up) => {
                    stack.push(up)
                }
                Symbolic::FuncApp(f, _, ref a) if a[0] == class => {
                    out.entry(f).or_default().push((up, flat_subst(&x, &node)))
                }
                _ => {}
            }
        }
    }
    let out = Rc::new(out);
    PROJECTED.with(|p| *p.borrow_mut() = Some((changes.serial(), Rc::clone(&out))));
    out
}

impl Searcher<Symbolic, ConstFold> for UnaryAppSearcher {
    /// Seed only from the e-classes that contain a node with this concept's
    /// operator (egg's `classes_by_op` op-index), instead of egg's default
    /// whole-e-graph scan. Sound because the ground type instantiation is **not**
    /// in the discriminant (it lives in the enode payload), so a single
    /// `Discriminant::FuncApp(func)` bucket holds every instantiation of `func`.
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let Some(ids) = egraph.classes_for_op(&Discriminant::FuncApp(self.func)) else {
            return vec![];
        };
        let mut ms = Vec::new();
        let mut limit = limit;
        for eclass in ids {
            if limit == 0 {
                break;
            }
            if let Some(m) = self.search_eclass_with_limit(egraph, eclass, limit) {
                limit -= m.substs.len();
                ms.push(m);
            }
        }
        ms
    }

    fn search_changes(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        changes: &Changes<Symbolic>,
        _limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let x = [tag_x()];
        let mut found: Vec<(Id, Subst)> = changes
            .nodes(&Discriminant::FuncApp(self.func))
            .filter(|(_, n)| n.children().len() == 1)
            .map(|(class, node)| (*class, flat_subst(&x, node)))
            .collect();
        match &self.reads {
            ArgReads::Constructors(ctors) => {
                let mut args: Vec<Id> = ctors
                    .iter()
                    .flat_map(|c| changes.nodes(&Discriminant::FuncApp(*c)))
                    .map(|(class, _)| *class)
                    .collect();
                args.sort_unstable();
                args.dedup();
                let is_app = |n: &Symbolic| matches!(n, Symbolic::FuncApp(f, _, a) if *f == self.func && a.len() == 1);
                for arg in args {
                    for (app, node) in parents_where(egraph, arg, is_app) {
                        found.push((app, flat_subst(&x, &node)));
                    }
                }
            }
            ArgReads::Projection => {
                let projected = projected(egraph, changes);
                found.extend(projected.get(&self.func).into_iter().flatten().cloned());
            }
        }
        group_matches(found)
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        let mut substs = Vec::new();
        for node in &egraph[eclass].nodes {
            if let Symbolic::FuncApp(f, _, args) = node
                && *f == self.func
                && args.len() == 1
            {
                let mut subst = Subst::default();
                subst.insert(tag_x(), args[0]);
                substs.push(subst);
                if substs.len() >= limit {
                    break;
                }
            }
        }
        if substs.is_empty() {
            None
        } else {
            Some(SearchMatches {
                eclass,
                substs,
                ast: None,
            })
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![tag_x()]
    }
}

/// Applier for the tag reduction: if the argument's e-class holds a constructor
/// of this ADT, union the `tag(..)` e-class with the constructor's tag literal.
pub(super) struct TagApplier {
    pub(super) ctor_tags: HashMap<FuncId, usize>,
}

impl Applier<Symbolic, ConstFold> for TagApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let xc = egraph.find(subst[tag_x()]);
        let mut tag = None;
        for node in &egraph[xc].nodes {
            if let Symbolic::FuncApp(c, _, _) = node
                && let Some(&t) = self.ctor_tags.get(c)
            {
                tag = Some(t);
                break;
            }
        }
        let Some(t) = tag else { return vec![] };
        let lit = egraph.add(Symbolic::Lit(Literal::Int(num::BigInt::from(t))));
        if egraph.union(eclass, lit) {
            vec![egraph.find(eclass)]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![tag_x()]
    }
}

/// Applier for [`inj_rule`]: unions the arguments of every pair of same-ctor
/// applications sharing the matched e-class. Grouped by type args — two
/// instantiations of a generic constructor are different operators, and only
/// same-operator applications are congruent.
pub(super) struct InjApplier {
    pub(super) ctor: FuncId,
}

impl Applier<Symbolic, ConstFold> for InjApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // One representative application per type instantiation; every later one
        // is unioned against it argument-wise (transitivity covers the rest).
        let mut reps: Vec<(Box<[Type]>, Box<[Id]>)> = Vec::new();
        let mut pairs: Vec<(Id, Id)> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::FuncApp(f, tys, args) = node else {
                continue;
            };
            if *f != self.ctor {
                continue;
            }
            match reps.iter().find(|(t, _)| t == tys) {
                Some((_, rep)) => pairs.extend(rep.iter().copied().zip(args.iter().copied())),
                None => reps.push((tys.clone(), args.clone())),
            }
        }
        let mut changed = Vec::new();
        for (a, b) in pairs {
            if egraph.union(a, b) {
                changed.push(egraph.find(a));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for the projection reduction: if the argument's e-class holds the
/// matching constructor `ctor`, union the `accessor(..)` e-class with that
/// constructor's `index`-th value argument. (Type args are not children, so the
/// value args start at 0.)
pub(super) struct ProjApplier {
    pub(super) ctor: FuncId,
    pub(super) index: usize,
}

impl ProjApplier {
    /// Extract the projected field out of `class`, returning the e-class of the
    /// result (adding `ite` nodes as needed). Either the ctor is directly present
    /// (return the field arg), or the class holds an `ite` whose both arms extract
    /// recursively — the projection commutes into the `ite`
    /// (`projᵢ(ite(c, cons(a..), cons(b..))) ⇒ ite(c, aᵢ, bᵢ)`). The commuting arm
    /// connects an enum discriminator's boxed `ite` body to a switch comparing the
    /// *unboxed* value; without it `proj∘cons` never fires, since the arg class
    /// holds an `Ite` rather than the ctor.
    ///
    /// The argument class is frequently a **shared** `ite`-DAG, so `memo` caches
    /// the per-class result — otherwise a DAG with `d` shared classes is walked
    /// over up to `2^d` distinct paths. `seen` is the in-progress cycle guard: a
    /// class whose extraction depends on an active cycle is *not* memoized (its
    /// `acyclic` return is `false`).
    ///
    /// Returns `(result, acyclic)`: `result` is the extracted class (or `None` if
    /// not projectable); `acyclic` is `false` iff the computation short-circuited
    /// on an in-progress class, marking the result as not cacheable by callers.
    fn project(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        class: Id,
        memo: &mut HashMap<Id, Option<Id>>,
        seen: &mut Vec<Id>,
    ) -> (Option<Id>, bool) {
        let class = egraph.find(class);
        if let Some(&r) = memo.get(&class) {
            return (r, true);
        }
        if seen.contains(&class) {
            return (None, false);
        }
        seen.push(class);
        // Read-only scan: a direct ctor hit wins; otherwise collect the `ite`
        // triples in node order. Copy the ids out so the `&egraph` borrow ends
        // before the recursive `&mut egraph` calls below.
        let mut field = None;
        let mut ites: Vec<[Id; 3]> = Vec::new();
        for node in &egraph[class].nodes {
            match node {
                Symbolic::FuncApp(c, _, args) if *c == self.ctor && self.index < args.len() => {
                    field = Some(args[self.index]);
                    break;
                }
                Symbolic::Ite([c, t, e]) => ites.push([*c, *t, *e]),
                _ => {}
            }
        }
        let mut result = None;
        let mut acyclic = true;
        if let Some(f) = field {
            result = Some(f);
        } else {
            for [c, t, e] in ites {
                let (tr, ta) = self.project(egraph, t, memo, seen);
                acyclic &= ta;
                let Some(tid) = tr else { continue };
                let (er, ea) = self.project(egraph, e, memo, seen);
                acyclic &= ea;
                let Some(eid) = er else { continue };
                result = Some(egraph.add(Symbolic::Ite([c, tid, eid])));
                break;
            }
        }
        seen.pop();
        if acyclic {
            memo.insert(class, result);
        }
        (result, acyclic)
    }
}

impl Applier<Symbolic, ConstFold> for ProjApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let xc = egraph.find(subst[tag_x()]);
        let (Some(field), _) = self.project(egraph, xc, &mut HashMap::default(), &mut Vec::new())
        else {
            return vec![];
        };
        if egraph.union(eclass, field) {
            vec![egraph.find(eclass)]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![tag_x()]
    }
}

#[cfg(test)]
mod bench {
    //! Micro-benchmark: the production [`UnaryAppSearcher`] (seeds from egg's
    //! `classes_by_op` op-index via its `search_with_limit` override) vs an
    //! otherwise-identical searcher that uses egg's default whole-e-graph scan.
    //!
    //! Run with: `cargo test --lib --release rewrite::bench -- --ignored --nocapture`.
    use super::*;
    use crate::verify::analysis::ConstFold;
    use egg::Runner;
    use std::time::{Duration, Instant};

    /// Same per-e-class match logic as [`UnaryAppSearcher`] but **without** the
    /// `search_with_limit` override — so it inherits egg's default, which scans
    /// every e-class in the graph. This is the "before" baseline.
    struct ScanSearcher {
        func: FuncId,
    }

    impl Searcher<Symbolic, ConstFold> for ScanSearcher {
        fn search_eclass_with_limit(
            &self,
            egraph: &EGraph<Symbolic, ConstFold>,
            eclass: Id,
            limit: usize,
        ) -> Option<SearchMatches<'_, Symbolic>> {
            let mut substs = Vec::new();
            for node in &egraph[eclass].nodes {
                if let Symbolic::FuncApp(f, _, args) = node
                    && *f == self.func
                    && args.len() == 1
                {
                    let mut subst = Subst::default();
                    subst.insert(tag_x(), args[0]);
                    substs.push(subst);
                    if substs.len() >= limit {
                        break;
                    }
                }
            }
            if substs.is_empty() {
                None
            } else {
                Some(SearchMatches {
                    eclass,
                    substs,
                    ast: None,
                })
            }
        }

        fn vars(&self) -> Vec<Var> {
            vec![tag_x()]
        }
    }

    /// Build an e-graph with `n_concepts` distinct ADT-like concepts — each a
    /// `proj(cons(x))` tower with its own `(cons, proj)` `FuncId`s — plus
    /// `n_filler` unrelated singleton e-classes that match no proj rule (the work
    /// the whole-graph scan wastes time on). Returns the graph and the concepts.
    fn build(
        n_concepts: usize,
        n_filler: usize,
    ) -> (EGraph<Symbolic, ConstFold>, Vec<(FuncId, FuncId)>) {
        let mut g = EGraph::<Symbolic, ConstFold>::default();
        let mut concepts = Vec::with_capacity(n_concepts);
        for i in 0..n_concepts {
            let cons_id = FuncId(1000 + 2 * i);
            let proj_id = FuncId(1000 + 2 * i + 1);
            let x = g.add(Symbolic::Fresh(i as u32));
            let cons = g.add(Symbolic::FuncApp(cons_id, Box::new([]), Box::new([x])));
            g.add(Symbolic::FuncApp(proj_id, Box::new([]), Box::new([cons])));
            concepts.push((cons_id, proj_id));
        }
        for j in 0..n_filler {
            g.add(Symbolic::Fresh((n_concepts + j) as u32));
        }
        g.rebuild();
        (g, concepts)
    }

    /// Best wall-clock of 5 saturation runs over a fresh clone of `g`.
    fn best_of_5(g: &EGraph<Symbolic, ConstFold>, rules: &[Rule]) -> Duration {
        let mut best = Duration::MAX;
        for _ in 0..5 {
            let eg = g.clone();
            let start = Instant::now();
            let runner = Runner::default()
                .with_node_limit(10_000_000)
                .with_iter_limit(100)
                .with_time_limit(Duration::from_secs(120))
                .with_egraph(eg)
                .run(rules);
            best = best.min(start.elapsed());
            std::hint::black_box(runner.egraph.total_size());
        }
        best
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly with --ignored --nocapture"]
    fn indexed_searcher_beats_whole_graph_scan() {
        let n_concepts = 500;
        let n_filler = 20_000;
        let (g, concepts) = build(n_concepts, n_filler);

        let indexed: Vec<Rule> = concepts.iter().map(|(c, p)| proj_rule(*p, *c, 0)).collect();
        let naive: Vec<Rule> = concepts
            .iter()
            .map(|(c, p)| {
                Rewrite::new(
                    format!("scan-{}", p.0),
                    ScanSearcher { func: *p },
                    ProjApplier { ctor: *c, index: 0 },
                )
                .expect("valid rule")
            })
            .collect();

        let t_naive = best_of_5(&g, &naive);
        let t_indexed = best_of_5(&g, &indexed);

        eprintln!(
            "concepts={n_concepts} filler_eclasses={n_filler} rules={}",
            concepts.len()
        );
        eprintln!("whole-graph scan : {t_naive:?}");
        eprintln!("classes_by_op    : {t_indexed:?}");
        eprintln!(
            "speedup          : {:.1}x",
            t_naive.as_secs_f64() / t_indexed.as_secs_f64().max(f64::MIN_POSITIVE)
        );
        assert!(
            t_indexed < t_naive,
            "indexed seeding should beat the whole-graph scan"
        );
    }
}
