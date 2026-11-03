//! ADT rules: projection, accessor and tester reductions plus tester
//! exclusivity, minted per concept by the `FuncRegistry`, and the shared
//! op-indexed searcher they use.

use std::collections::BTreeSet;
use crate::dhash::{HashMap, HashSet};
use std::sync::Arc;

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::{AdtData, ConstFold, Data};
use crate::verify::lang::{Discriminant, FuncId, Symbolic};
use crate::vmir::{Literal, MemberId, Type};

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

/// Build the accessor rule for one constructor: every application
/// `C(a₀, …, aₙ)` gets its accessors, `C.i(C(a₀, …, aₙ)) ≡ aᵢ`. `accessors` is
/// indexed by field.
///
/// This subsumes injectivity: two applications `C(a..) ≡ C(b..)` share one class
/// `x`, so they share the one node `C.i(x)`, which is then equal to both `aᵢ`
/// and `bᵢ`. Congruence alone runs only *forward*; the accessor terms give the
/// backward direction even where the program never projects.
pub fn accessor_rule(ctor: FuncId, accessors: Box<[FuncId]>) -> Rule {
    timed(
        Rewrite::new(
            format!("accessor-{}", ctor.0),
            AxiomTriggerSearcher { func: ctor },
            AccessorApplier { ctor, accessors },
        )
        .expect("valid accessor rewrite"),
    )
}

/// Build the tester reduction `is_C(D(..)) ⇒ (C == D)` for the tester `is_fn`
/// of constructor `ctor`: a tester applied to a value whose class holds a
/// constructor of the head folds to `true` or `false`. Sound because ADT
/// constructors are free.
pub fn is_rule(is_fn: FuncId, ctor: FuncId) -> Rule {
    timed(
        Rewrite::new(
            format!("is-{}", is_fn.0),
            UnaryAppSearcher { func: is_fn },
            IsApplier { ctor },
        )
        .expect("valid is rewrite"),
    )
}

/// The pattern variable bound to the argument of a unary application.
pub(super) fn arg_x() -> Var {
    var("?x")
}

/// Searcher for a unary application `func(base)`: matches any `FuncApp(func, _,
/// [base])` node (its ground type args live in the discriminant and are ignored
/// here) and binds `?x` to the single value arg. FuncApp isn't string-matchable,
/// so this is hand-written. Shared by the tester and projection reductions.
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
                subst.insert(arg_x(), args[0]);
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
        vec![arg_x()]
    }
}

/// Applier for [`is_rule`]: if the argument class holds a constructor (read
/// from its `ConstFold` data, no node scan), unions the tester with `true` when
/// that constructor is `ctor` and with `false` otherwise.
pub(super) struct IsApplier {
    /// The constructor this tester tests for (`cons(adt, variant)`).
    pub(super) ctor: FuncId,
}

impl Applier<Symbolic, ConstFold> for IsApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _: Option<&PatternAst<Symbolic>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xc = egraph.find(subst[arg_x()]);
        let Data::Adt(adt) = &egraph[xc].data else {
            return vec![];
        };
        let holds = if !adt.possible.contains(&self.ctor) {
            false
        } else if adt.possible.len() == 1 {
            true
        } else {
            return vec![]
        };
        let lit = egraph.add(Symbolic::Lit(Literal::Bool(holds)));
        if egraph.union(eclass, lit) {
            vec![egraph.find(eclass)]
        } else {
            vec![]
        }
    }
    fn vars(&self) -> Vec<Var> {
        vec![arg_x()]
    }
}


pub fn variant_narrow_rule(head: MemberId, ctors: Arc<[FuncId]>, testers: Arc<[FuncId]>) -> Rule {
    let variant_of = testers.iter().enumerate().map(|(v, &t)| (t, v)).collect();
    timed(
        Rewrite::new(
            format!("variant-narrow-{}", head.0),
            DecidedTesterSearcher {
                testers: testers.clone(),
            },
            VariantNarrowApplier {
                head,
                ctors,
                variant_of,
            },
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

pub(super) struct VariantNarrowApplier {
    pub(super) head: MemberId,
    /// All constructors of the head, indexed by variant.
    pub(super) ctors: Arc<[FuncId]>,
    pub(super) variant_of: HashMap<FuncId, usize>,
}

impl VariantNarrowApplier {

    fn apply_class(&self, egraph: &mut EGraph<Symbolic, ConstFold>, eclass: Id) -> Vec<Id> {
        let Some(&Literal::Bool(holds)) = egraph[eclass].data.known() else {
            return vec![];
        };
        let apps: Vec<(usize, Box<[Type]>, Id)> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::FuncApp(f, tys, args) => {
                    self.variant_of.get(f).map(|&v| (v, tys.clone(), args[0]))
                }
                _ => None,
            })
            .collect();
        let mut changed = vec![];
        for (variant, tys, x) in apps {
            self.apply_app(egraph, holds, variant, tys, x, &mut changed);
        }
        changed
    }
    fn apply_app(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        holds: bool,
        variant: usize,
        tys: Box<[Type]>,
        x: Id,
        changed: &mut Vec<Id>,
    ) {
        let x = egraph.find(x);
        // What is known so far; `Unknown` means every variant is still possible.
        let current: BTreeSet<FuncId> = match &egraph[x].data {
            Data::Adt(adt) => adt.possible.clone(),
            Data::Unknown => self.ctors.iter().copied().collect(),
            // Already contradictory: nothing to narrow.
            Data::Inconsistent => return,
            Data::Known(lit) => panic!("type error: tester applied to literal {lit:?}"),
        };

        let ctor = self.ctors[variant];
        let narrowed : BTreeSet<FuncId> = if holds {
            current.iter().filter(|&c| *c == ctor).copied().collect()
        } else {
            current.iter().filter(|&c| *c != ctor).copied().collect()
        };

        // only write when something changed
        let was_unknown = matches!(egraph[x].data, Data::Unknown);
        if !was_unknown && narrowed.len() == current.len() {
            return;
        }

        let data = if narrowed.is_empty() {
            Data::Inconsistent
        } else {
            Data::Adt(AdtData { head: self.head, tys, possible: narrowed })
        };
        egraph.set_analysis_data(x, data);
        changed.push(x);
    }
}

impl Applier<Symbolic, ConstFold> for VariantNarrowApplier {
    /// One [`VariantNarrowApplier::apply_class`] per matched class. Matches that an
    /// earlier rule of this iteration merged into one class are walked once.
    fn apply_matches(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        matches: &[SearchMatches<Symbolic>],
        _: Symbol,
    ) -> Vec<Id> {
        let mut walked = HashSet::default();
        let mut changed = vec![];
        for m in matches {
            if walked.insert(egraph.find(m.eclass)) {
                changed.extend(self.apply_class(egraph, m.eclass));
            }
        }
        changed
    }

    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _: &Subst,
        _: Option<&PatternAst<Symbolic>>,
        _: Symbol,
    ) -> Vec<Id> {
        self.apply_class(egraph, eclass)
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
        let Some(&Literal::Bool(holds)) = egraph[eclass].data.known() else {
            return vec![];
        };
        // Collect first: the lookups and unions below need `&mut egraph`.
        let apps: Vec<(usize, Box<[Type]>, Id)> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::FuncApp(f, tys, args) => {
                    self.variant_of.get(f).map(|&v| (v, tys.clone(), args[0]))
                }
                _ => None,
            })
            .collect();
        let mut changed = vec![];
        for (variant, tys, x) in apps {
            self.apply_app(egraph, holds, variant, tys, x, &mut changed);
        }
        changed
    }

    /// The rule for one tester application `is_C[tys](x)`, `C` the `variant`-th,
    /// known to be `holds`.
    fn apply_app(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        holds: bool,
        variant: usize,
        tys: Box<[Type]>,
        x: Id,
        changed: &mut Vec<Id>,
    ) {
        // Existing sibling tester classes, by variant (None = not in the graph).
        // One probe node re-pointed at each tester, so a lookup allocates nothing.
        let mut probe = Symbolic::FuncApp(self.testers[variant], tys, Box::new([x]));
        let sib: Vec<Option<Id>> = self
            .testers
            .iter()
            .map(|&t| {
                if let Symbolic::FuncApp(f, ..) = &mut probe {
                    *f = t;
                }
                egraph.lookup(&mut probe)
            })
            .collect();

        let mut union_lit = |eg: &mut EGraph<_, _>, id: Id, b: bool| {
            let lit = eg.add(Symbolic::Lit(Literal::Bool(b)));
            if eg.union(id, lit) {
                changed.push(eg.find(id));
            }
        };

        if holds {
            // Exclusivity.
            for (d, id) in sib.iter().enumerate() {
                if d != variant
                    && let Some(id) = id
                {
                    union_lit(egraph, *id, false);
                }
            }
        } else {
            // Last variant: every other variant has a tester known false.
            let is_false = |eg: &EGraph<_, ConstFold>, id: Id| {
                eg[id].data.known() == Some(&Literal::Bool(false))
            };
            let open: Vec<usize> = (0..sib.len())
                .filter(|&d| !sib[d].is_some_and(|id| is_false(egraph, id)))
                .collect();
            if let [e] = open[..]
                && let Some(id) = sib[e]
            {
                union_lit(egraph, id, true);
            }
        }
    }
}

impl Applier<Symbolic, ConstFold> for IsExclApplier {
    /// One [`IsExclApplier::apply_class`] per matched class. Matches that an
    /// earlier rule of this iteration merged into one class are walked once.
    fn apply_matches(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        matches: &[SearchMatches<Symbolic>],
        _: Symbol,
    ) -> Vec<Id> {
        let mut walked = HashSet::default();
        let mut changed = vec![];
        for m in matches {
            if walked.insert(egraph.find(m.eclass)) {
                changed.extend(self.apply_class(egraph, m.eclass));
            }
        }
        changed
    }

    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _: &Subst,
        _: Option<&PatternAst<Symbolic>>,
        _: Symbol,
    ) -> Vec<Id> {
        self.apply_class(egraph, eclass)
    }
    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for [`accessor_rule`]: for every application of `ctor` in the matched
/// e-class, adds `accessorᵢ(class)` at the application's own type args and
/// unions it with the `i`-th argument. The accessor node is hash-consed, so after
/// the first application it is a lookup and a no-op union.
pub(super) struct AccessorApplier {
    pub(super) ctor: FuncId,
    /// The constructor's field accessors, indexed by field.
    pub(super) accessors: Box<[FuncId]>,
}

impl Applier<Symbolic, ConstFold> for AccessorApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let x = egraph.find(eclass);
        // Collect first: the adds and unions below need `&mut egraph`. An
        // accessor is applied at its constructor's type args, so each
        // application carries the ones its accessor terms need.
        let apps: Vec<(Box<[Type]>, Box<[Id]>)> = egraph[x]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::FuncApp(f, tys, args) if *f == self.ctor => {
                    Some((tys.clone(), args.clone()))
                }
                _ => None,
            })
            .collect();
        let mut changed = Vec::new();
        for (tys, args) in apps {
            assert_eq!(
                args.len(),
                self.accessors.len(),
                "constructor {:?} applied to {} args but has {} accessors",
                self.ctor,
                args.len(),
                self.accessors.len()
            );
            for (&accessor, &arg) in self.accessors.iter().zip(&args) {
                let app = egraph.add(Symbolic::FuncApp(accessor, tys.clone(), Box::new([x])));
                if egraph.union(app, arg) {
                    changed.push(egraph.find(arg));
                }
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
        let xc = egraph.find(subst[arg_x()]);
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
        vec![arg_x()]
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
                    subst.insert(arg_x(), args[0]);
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
            vec![arg_x()]
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
