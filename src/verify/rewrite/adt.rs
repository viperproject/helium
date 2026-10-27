//! ADT rules: constructor/destructor/tag reductions, minted per concept as the
//! `FuncRegistry` grows, and the shared op-indexed searcher they use.

use std::sync::Arc;
use crate::dhash::{HashMap, HashSet};

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

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
            UnaryAppSearcher { func: accessor },
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
            AxiomTriggerSearcher { func: ctor },
            InjApplier { ctor },
        )
        .expect("valid injectivity rewrite"),
    )
}

/// Build the discriminator reduction `tag_fn(ctor_C(..)) ⇒ index_C` for a single
/// (possibly synthesised) tag function. Companion to [`proj_rule`].
pub fn tag_rule(tag_fn: FuncId, ctor_tags: HashMap<FuncId, usize>) -> Rule {
    timed(
        Rewrite::new(
            format!("tag-{}", tag_fn.0),
            UnaryAppSearcher { func: tag_fn },
            TagApplier { ctor_tags },
        )
        .expect("valid tag rewrite"),
    )
}

/// Build the tester reduction `is_C(D(..)) ⇒ (C == D)` for the tester of variant
/// `variant`. `ctor_tags` maps each constructor of the head to its variant index,
/// as for [`tag_rule`].
pub fn is_rule(is_fn: FuncId, ctor : FuncId) -> Rule {
    timed(
        Rewrite::new(
            format!("is-{}", is_fn.0),
            UnaryAppSearcher { func: is_fn },
            IsApplier { ctor },
        )
            .expect("valid is rewrite"),
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


/// Applier for [`is_rule`]: if the argument class holds a constructor, unions
/// the tester with `true` when it is variant `variant` and `false` otherwise.
pub(super) struct IsApplier {
    /// The constructor this tester tests for (`cons(adt, variant)`).
    pub(super) ctor: FuncId,
}

impl Applier<Symbolic, ConstFold> for IsApplier {
    fn apply_one(&self, egraph: &mut EGraph<Symbolic, ConstFold>, eclass: Id,
                 subst: &Subst, _: Option<&PatternAst<Symbolic>>, _: Symbol) -> Vec<Id> {
        let xc = egraph.find(subst[tag_x()]);
        let Data::Ctor(c, _) = &egraph[xc].data else { return vec![] };
        let lit = egraph.add(Symbolic::Lit(Literal::Bool(*c == self.ctor)));
        if egraph.union(eclass, lit) { vec![egraph.find(eclass)] } else { vec![] }
    }
    fn vars(&self) -> Vec<Var> { vec![tag_x()] }
}


/// Exclusivity / last-variant for the testers of one ADT head: `is_C(x) = true`
/// makes every other existing `is_D(x)` false; all-but-one false makes the last
/// one true. Lookup only, never adds a tester node, so it cannot grow the graph.
///
/// **One rule per head**, not one per tester: the decided testers of every
/// variant share the `true` and `false` classes, so per-tester rules would each
/// walk those classes again. `testers` is indexed by variant; the rule is named
/// after its first tester.
pub fn is_excl_rule(testers: Arc<[FuncId]>) -> Rule {
    let variant_of = testers.iter().enumerate().map(|(v, &t)| (t, v)).collect();
    timed(
        Rewrite::new(
            format!("is-excl-{}", testers[0].0),
            DecidedTesterSearcher { testers: testers.clone() },
            IsExclApplier { testers, variant_of },
        )
            .expect("valid is-excl rewrite"),
    )
}

/// Searcher for [`is_excl_rule`]: every class that holds a tester of the head
/// and whose value is known. Reads the testers' `classes_by_op` buckets and one
/// class datum per class; inspects no node. One empty subst per class; the
/// applier re-reads the nodes.
#[derive(Clone)]
pub(super) struct DecidedTesterSearcher {
    pub(super) testers: Arc<[FuncId]>,
}

impl Searcher<Symbolic, ConstFold> for DecidedTesterSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        (known_bool(egraph, eclass).is_some()
            && egraph[eclass]
                .nodes
                .iter()
                .any(|n| matches!(n, Symbolic::FuncApp(f, ..) if self.testers.contains(f))))
        .then(|| SearchMatches {
            eclass,
            substs: vec![Subst::default()],
            ast: None,
        })
    }

    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        // `true` and `false` hold testers of several variants, so they sit in
        // several buckets: report each class once.
        let mut seen = HashSet::default();
        self.testers
            .iter()
            .filter_map(|&t| egraph.classes_for_op(&Discriminant::FuncApp(t)))
            .flatten()
            .filter(|&c| known_bool(egraph, c).is_some() && seen.insert(c))
            .take(limit)
            .map(|eclass| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
            .collect()
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

pub(super) struct IsExclApplier {
    /// All testers of the head, indexed by variant.
    pub(super) testers: Arc<[FuncId]>,
    /// Tester → its variant, the inverse of `testers`.
    pub(super) variant_of: HashMap<FuncId, usize>,
}

impl IsExclApplier {
    /// Applies the rule to every tester node of `eclass` in **one** walk.
    ///
    /// A decided tester sits in the `true` or `false` class, which holds one
    /// such node per decided value and every assumed fact besides. Applying per
    /// match and re-scanning the class to recover the matched node's type args
    /// cost `|class|` per match, so `|class|²` per iteration. Here each node is
    /// read once and carries its own variant, type args and argument.
    fn apply_class(&self, egraph: &mut EGraph<Symbolic, ConstFold>, eclass: Id) -> Vec<Id> {
        let Some(&Literal::Bool(holds)) = egraph[eclass].data.known() else { return vec![] };
        // Collect first: the lookups and unions below need `&mut egraph`.
        let apps: Vec<(usize, Box<[Type]>, Id)> = egraph[eclass].nodes.iter().filter_map(|n| match n {
            Symbolic::FuncApp(f, tys, args) => {
                self.variant_of.get(f).map(|&v| (v, tys.clone(), args[0]))
            }
            _ => None,
        }).collect();
        let mut changed = vec![];
        for (variant, tys, x) in apps {
            self.apply_app(egraph, holds, variant, tys, x, &mut changed);
        }
        changed
    }

    /// The rule for one tester application `is_C[tys](x)`, `C` the `variant`-th,
    /// known to be `holds`.
    fn apply_app(&self, egraph: &mut EGraph<Symbolic, ConstFold>, holds: bool,
                 variant: usize, tys: Box<[Type]>, x: Id, changed: &mut Vec<Id>) {
        // Existing sibling tester classes, by variant (None = not in the graph).
        // One probe node re-pointed at each tester, so a lookup allocates nothing.
        let mut probe = Symbolic::FuncApp(self.testers[variant], tys, Box::new([x]));
        let sib: Vec<Option<Id>> = self.testers.iter().map(|&t| {
            if let Symbolic::FuncApp(f, ..) = &mut probe { *f = t; }
            egraph.lookup(&mut probe)
        }).collect();

        let mut union_lit = |eg: &mut EGraph<_, _>, id: Id, b: bool| {
            let lit = eg.add(Symbolic::Lit(Literal::Bool(b)));
            if eg.union(id, lit) { changed.push(eg.find(id)); }
        };

        if holds {
            // Exclusivity.
            for (d, id) in sib.iter().enumerate() {
                if d != variant && let Some(id) = id { union_lit(egraph, *id, false); }
            }
        } else {
            // Last variant: every other variant has a tester known false.
            let is_false = |eg: &EGraph<_, ConstFold>, id: Id| eg[id].data.known() == Some(&Literal::Bool(false));
            let open: Vec<usize> = (0..sib.len())
                .filter(|&d| !sib[d].is_some_and(|id| is_false(egraph, id)))
                .collect();
            if let [e] = open[..] && let Some(id) = sib[e] {
                union_lit(egraph, id, true);
            }
        }
    }
}

impl Applier<Symbolic, ConstFold> for IsExclApplier {
    /// One [`IsExclApplier::apply_class`] per matched class. Matches that an
    /// earlier rule of this iteration merged into one class are walked once.
    fn apply_matches(&self, egraph: &mut EGraph<Symbolic, ConstFold>,
                     matches: &[SearchMatches<Symbolic>], _: Symbol) -> Vec<Id> {
        let mut walked = HashSet::default();
        let mut changed = vec![];
        for m in matches {
            if walked.insert(egraph.find(m.eclass)) {
                changed.extend(self.apply_class(egraph, m.eclass));
            }
        }
        changed
    }

    fn apply_one(&self, egraph: &mut EGraph<Symbolic, ConstFold>, eclass: Id,
                 _: &Subst, _: Option<&PatternAst<Symbolic>>, _: Symbol) -> Vec<Id> {
        self.apply_class(egraph, eclass)
    }
    fn vars(&self) -> Vec<Var> { vec![] }
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
mod is_excl_tests {
    //! [`is_excl_rule`] on a graph where every decided tester sits in the
    //! `true` / `false` class — the shape that made a per-match class scan
    //! quadratic.
    //!
    //! Benchmark: `cargo test --lib --release is_excl_tests -- --ignored --nocapture`.
    use super::*;
    use egg::Runner;
    use std::time::Instant;

    const TESTERS: [FuncId; 3] = [FuncId(9000), FuncId(9001), FuncId(9002)];

    fn rules() -> Vec<Rule> {
        vec![is_excl_rule(Arc::new(TESTERS))]
    }

    /// `n` values with all three testers applied. `pin(i)` gives, per variant,
    /// the literal value `i`'s tester is assumed to be (`None` = undecided).
    fn graph(
        n: usize,
        pin: impl Fn(usize) -> [Option<bool>; 3],
    ) -> (EGraph<Symbolic, ConstFold>, Vec<[Id; 3]>) {
        let mut g = EGraph::<Symbolic, ConstFold>::default();
        // Give both literals a parent, as every real graph does (`ite(pc, fact,
        // true)`). egg keeps the class with more parents as the root; with none,
        // each merge into `false` would make the tester the root and grow a
        // union-find chain that the non-compressing `find` then walks per lookup.
        let t = g.add(Symbolic::Lit(Literal::Bool(true)));
        let f = g.add(Symbolic::Lit(Literal::Bool(false)));
        g.add(Symbolic::Ite([t, f, t]));
        let mut apps = Vec::with_capacity(n);
        for i in 0..n {
            let x = g.add(Symbolic::Fresh(i as u32));
            let ids = TESTERS.map(|f| g.add(Symbolic::FuncApp(f, Box::new([]), Box::new([x]))));
            for (id, pin) in ids.iter().zip(pin(i)) {
                if let Some(b) = pin {
                    let lit = g.add(Symbolic::Lit(Literal::Bool(b)));
                    g.union(*id, lit);
                }
            }
            apps.push(ids);
        }
        g.rebuild();
        (g, apps)
    }

    fn known(g: &EGraph<Symbolic, ConstFold>, id: Id) -> Option<bool> {
        match g[id].data.known() {
            Some(Literal::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    fn saturate(g: EGraph<Symbolic, ConstFold>) -> EGraph<Symbolic, ConstFold> {
        Runner::default()
            .with_node_limit(usize::MAX)
            .with_time_limit(std::time::Duration::MAX)
            .with_egraph(g)
            .with_iter_limit(10)
            .run(&rules())
            .egraph
    }

    #[test]
    fn exclusivity_refutes_every_sibling() {
        let (g, apps) = graph(50, |_| [Some(true), None, None]);
        let g = saturate(g);
        for [a, b, c] in apps {
            assert_eq!(known(&g, a), Some(true));
            assert_eq!(known(&g, b), Some(false));
            assert_eq!(known(&g, c), Some(false));
        }
    }

    #[test]
    fn last_variant_is_forced() {
        let (g, apps) = graph(50, |_| [Some(false), Some(false), None]);
        let g = saturate(g);
        for [_, _, c] in apps {
            assert_eq!(known(&g, c), Some(true));
        }
    }

    #[test]
    fn two_open_variants_decide_nothing() {
        let (g, apps) = graph(50, |_| [Some(false), None, None]);
        let g = saturate(g);
        for [_, b, c] in apps {
            assert_eq!(known(&g, b), None);
            assert_eq!(known(&g, c), None);
        }
    }

    /// Value `i` is variant `i % 3`: `true` and `false` each hold testers of
    /// every variant.
    fn mixed(i: usize) -> [Option<bool>; 3] {
        let mut pin = [None; 3];
        pin[i % 3] = Some(true);
        pin
    }

    #[test]
    fn mixed_variants_refute_their_siblings() {
        let (g, apps) = graph(60, mixed);
        let g = saturate(g);
        for (i, ids) in apps.iter().enumerate() {
            for (v, &id) in ids.iter().enumerate() {
                assert_eq!(known(&g, id), Some(v == i % 3));
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly with --ignored --nocapture"]
    fn decided_testers_scale_linearly() {
        for n in [2_000, 4_000, 8_000] {
            let (g, _) = graph(n, mixed);
            take_rule_timing();
            let start = Instant::now();
            let g = saturate(g);
            let total = start.elapsed();
            let (search, apply) = take_rule_timing()
                .values()
                .fold((0.0, 0.0), |(s, a), t| (s + t.search, a + t.apply));
            eprintln!(
                "n={n:>5}  total {total:>10.1?}  search {:>8.1}ms  apply {:>8.1}ms",
                search * 1e3,
                apply * 1e3
            );
            std::hint::black_box(g.total_size());
        }
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
