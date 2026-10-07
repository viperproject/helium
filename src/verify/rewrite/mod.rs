//! Structural egg rewrite rules for the verifier.

use crate::dhash::HashSet;

use egg::{Rewrite, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::Symbolic;

pub(crate) mod adt;
pub(crate) mod arith;
pub(crate) mod diseq;
pub(crate) mod forall;
pub(crate) mod function;
pub(crate) mod ite;
pub(crate) mod memo;
pub(crate) mod recipe;
pub(crate) mod timing;

pub use adt::{inj_rule, proj_rule, tag_rule};
pub(crate) use forall::{PreparedTerm, forall_rule};
pub(crate) use function::{function_post_rule, function_rule, post_rule};
pub(crate) use memo::{Memo, ScratchScope, new_memo_unit, new_scope_id};
pub(crate) use recipe::{
    AxiomInst, AxiomPure, build_instance_releasing_tokens, build_instance_vals_guarded,
};
pub(crate) use timing::take_rule_timing;

pub(in crate::verify::rewrite) use arith::*;
pub(in crate::verify::rewrite) use diseq::*;
pub(in crate::verify::rewrite) use function::*;
pub(in crate::verify::rewrite) use ite::*;
pub(in crate::verify::rewrite) use recipe::*;
pub(in crate::verify::rewrite) use timing::*;

type Rule = Rewrite<Symbolic, ConstFold>;

/// Runs a **bucket** rule — an op-bucket searcher yielding one empty subst per
/// class, whose applier re-reads the class's nodes — over one iteration's
/// matches without walking a class once per match.
///
/// Matches go stale within an iteration: once many matched classes merge into
/// one (thousands of clauses all proven `true`), every later match resolves to
/// that class, and walking the whole class per match is quadratic. Instead each
/// matched root is walked once, and then exactly the classes whose nodes read
/// differently since — as the unions of the walks report them through
/// `ConstFold::union_log` — are walked again, pass by pass, until a pass makes
/// no union. So derivations still chain within the iteration (an `ite(c, x, y)`
/// reduces once a walk pins `c`), at a cost proportional to what changed.
///
/// **Invariant: the applier only ever sees a class its searcher matches.** The
/// logged classes were never matched — they include the parents of an absorbed
/// class, whatever their op, and classes whose known value just changed — so
/// every walk first re-asks the searcher (`search_eclass`) about the class, and
/// an applier may rely on the searcher's filter exactly as under a plain egg
/// rule. The searcher is the single source of truth for that filter: build bucket
/// rules with [`bucket_rule`], which hands the same searcher to the rule and to
/// the wrapper.
///
/// Only for rules whose derivations terminate on their own (unions of existing
/// classes, a bounded set of new terms): the passes run to this rule's local
/// fixpoint inside one iteration, beyond the runner's per-iteration limits.
pub(super) struct PerClass<S, A> {
    searcher: S,
    applier: A,
}

/// A bucket rule: `searcher` picks the classes (one empty subst each) and
/// `applier` re-reads a picked class's nodes, run under [`PerClass`].
pub(super) fn bucket_rule<S, A>(name: &str, searcher: S, applier: A) -> Rule
where
    S: egg::Searcher<Symbolic, ConstFold> + Clone + Send + Sync + 'static,
    A: egg::Applier<Symbolic, ConstFold> + Send + Sync + 'static,
{
    let applier = PerClass {
        searcher: searcher.clone(),
        applier,
    };
    Rewrite::new(name, searcher, applier).expect("bucket rule")
}

impl<S: egg::Searcher<Symbolic, ConstFold>, A: egg::Applier<Symbolic, ConstFold>> PerClass<S, A> {
    /// Applies the rule to `class` if the searcher matches it now.
    fn walk(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        class: egg::Id,
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        if self.searcher.search_eclass(egraph, class).is_none() {
            return Vec::new();
        }
        self.applier
            .apply_one(egraph, class, &egg::Subst::default(), None, rule_name)
    }
}

impl<S: egg::Searcher<Symbolic, ConstFold>, A: egg::Applier<Symbolic, ConstFold>>
    egg::Applier<Symbolic, ConstFold> for PerClass<S, A>
{
    fn apply_matches(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        matches: &[egg::SearchMatches<Symbolic>],
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        let mut changed = Vec::new();
        let mut walk = |egraph: &mut egg::EGraph<Symbolic, ConstFold>, root| {
            changed.extend(self.walk(egraph, root, rule_name));
        };
        // Matches an earlier rule of this iteration merged share a root: take
        // each such root once. The rest are distinct roots already.
        let mut merged_before = HashSet::default();
        let first_pass: Vec<egg::Id> = matches
            .iter()
            .filter_map(|m| {
                let root = egraph.find(m.eclass);
                (root == m.eclass || merged_before.insert(root)).then_some(root)
            })
            .collect();
        *egraph.analysis.union_log.borrow_mut() = Some(Vec::new());
        for root in first_pass {
            // One merged away during this pass is covered by the log, which
            // recorded the root it merged into.
            if egraph.find(root) == root {
                walk(egraph, root);
            }
        }
        loop {
            let logged = std::mem::take(egraph.analysis.union_log.borrow_mut().as_mut().unwrap());
            if logged.is_empty() {
                break;
            }
            let mut this_pass = HashSet::default();
            for id in logged {
                let root = egraph.find(id);
                if this_pass.insert(root) {
                    walk(egraph, root);
                }
            }
        }
        *egraph.analysis.union_log.borrow_mut() = None;
        changed
    }

    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        eclass: egg::Id,
        _subst: &egg::Subst,
        _searcher_ast: Option<&egg::PatternAst<Symbolic>>,
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        self.walk(egraph, eclass, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        self.applier.vars()
    }
}

fn var(name: &str) -> Var {
    name.parse().expect("valid pattern var")
}

/// The static structural rule set. Per-ADT cons/proj/tag reductions are minted
/// by the registry (`verify::mono`) and appended by `VerifyContext::new`.
pub fn rules() -> Vec<Rule> {
    static_rules().into_iter().filter(kept).map(timed).collect()
}

/// Ablation gate: `SILVER_OXIDE_DROP_RULES=name1,name2` removes those rules from
/// the saturation and reduction sets. Dropping a rewrite is incomplete, never
/// unsound.
fn kept(rule: &Rule) -> bool {
    use std::sync::OnceLock;
    static DROP: OnceLock<HashSet<String>> = OnceLock::new();
    let drop = DROP.get_or_init(|| {
        std::env::var("SILVER_OXIDE_DROP_RULES")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });
    drop.is_empty() || !drop.contains(rule.name.as_str())
}

/// The terminating structural reductions used to **normalize** the e-graph after
/// heap-producing ops (`fold`/`unfold`). The registry's ADT reductions are
/// appended by `VerifyContext::new`. Kept separate from [`rules`] so that
/// *non-terminating* rules run only during full saturation.
pub fn reduce_rules() -> Vec<Rule> {
    terminating_ite_rules()
        .into_iter()
        .filter(kept)
        .map(timed)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmir::{BinOp, Literal};
    use egg::{EGraph, Id, Runner, SearchMatches, Subst};
    use num::{BigInt, BigRational};

    /// Every rule built by [`bucket_rule`].
    const BUCKET_RULES: [&str; 8] = [
        "ite-reduce",
        "eq-false-mirror",
        "eq-false-then",
        "eq-false-else",
        "contra-congruence",
        "distinguishing-observation",
        "lt-asymmetry-int",
        "lt-asymmetry-real",
    ];

    struct Order {
        /// `0 < a`, assumed true.
        assumed: Id,
        /// `a < 0`, its mirror.
        mirror: Id,
        /// `b < a`, undecided, with its mirror `a < b` in the graph.
        open: Id,
        open_mirror: Id,
    }

    /// The order facts of one sort: `0 < a` proven, its mirror present (and
    /// refuted when `refute`), and an undecided pair of mirrors.
    fn order(g: &mut EGraph<Symbolic, ConstFold>, op: BinOp, zero: Literal, refute: bool) -> Order {
        let base = 2 * g.number_of_classes() as u32;
        let (a, b) = (
            g.add(Symbolic::Fresh(base)),
            g.add(Symbolic::Fresh(base + 1)),
        );
        let zero = g.add(Symbolic::Lit(zero));
        let assumed = g.add(Symbolic::Binary(op, [zero, a]));
        let mirror = g.add(Symbolic::Binary(op, [a, zero]));
        let open = g.add(Symbolic::Binary(op, [b, a]));
        let open_mirror = g.add(Symbolic::Binary(op, [a, b]));
        let t = g.add(Symbolic::Lit(Literal::Bool(true)));
        g.union(assumed, t);
        if refute {
            let f = g.add(Symbolic::Lit(Literal::Bool(false)));
            g.union(mirror, f);
        }
        Order {
            assumed,
            mirror,
            open,
            open_mirror,
        }
    }

    fn graph(refute: bool) -> (EGraph<Symbolic, ConstFold>, Vec<Order>) {
        let mut g = EGraph::<Symbolic, ConstFold>::default();
        let int = order(&mut g, BinOp::LtI, Literal::Int(BigInt::from(0)), refute);
        let real = order(
            &mut g,
            BinOp::LtR,
            Literal::Real(BigRational::from(BigInt::from(0))),
            refute,
        );
        g.rebuild();
        (g, vec![int, real])
    }

    fn known(g: &EGraph<Symbolic, ConstFold>, id: Id) -> Option<bool> {
        match g[id].data.known() {
            Some(Literal::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    fn assert_sound(g: &EGraph<Symbolic, ConstFold>, orders: &[Order], rule: &str) {
        let t = g.lookup(Symbolic::Lit(Literal::Bool(true))).unwrap();
        let f = g.lookup(Symbolic::Lit(Literal::Bool(false))).unwrap();
        assert_ne!(g.find(t), g.find(f), "{rule}: true merged with false");
        for o in orders {
            assert_eq!(known(g, o.assumed), Some(true), "{rule}: assumption lost");
            assert_eq!(known(g, o.open), None, "{rule}: open `b < a` decided");
            assert_eq!(
                known(g, o.open_mirror),
                None,
                "{rule}: open `a < b` decided"
            );
        }
    }

    /// The [`PerClass`] invariant: a bucket rule applied to a class its searcher
    /// does not match derives nothing. The union log hands the walk such classes
    /// (parents of an absorbed class, a class just pinned `false`); lt-asymmetry
    /// applied to the `false` class holding the refuted mirror `a < 0` used to
    /// refute the assumption `0 < a` itself.
    #[test]
    fn bucket_rules_skip_classes_their_searcher_rejects() {
        let rules = static_rules();
        for name in BUCKET_RULES {
            let rule = rules
                .iter()
                .find(|r| r.name.as_str() == name)
                .unwrap_or_else(|| panic!("no bucket rule {name}"));
            let (mut g, orders) = graph(true);
            let rejected: Vec<Id> = g
                .classes()
                .map(|c| c.id)
                .filter(|&c| rule.searcher.search_eclass(&g, c).is_none())
                .collect();
            assert!(!rejected.is_empty());
            for c in rejected {
                let m = SearchMatches {
                    eclass: c,
                    substs: vec![Subst::default()],
                    ast: None,
                };
                rule.applier.apply_matches(&mut g, &[m], rule.name);
                g.rebuild();
                assert_sound(&g, &orders, name);
            }
        }
    }

    /// End to end: saturating `0 < a` with its mirror present refutes the mirror
    /// and nothing else (`requires none < a; assert !(a < none); assert false`).
    #[test]
    fn lt_asymmetry_refutes_only_the_mirror() {
        let (g, orders) = graph(false);
        let runner = Runner::default()
            .with_egraph(g)
            .with_iter_limit(10)
            .run(&rules());
        let g = &runner.egraph;
        for o in &orders {
            assert_eq!(known(g, o.mirror), Some(false));
        }
        assert_sound(g, &orders, "saturation");
    }
}
