//! Structural egg rewrite rules for the verifier.

use crate::dhash::HashSet;

use egg::{Language, Rewrite, Var};

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

/// Applies a rule's matches, then — within the same iteration — the matches its
/// own changes create: rebuild, search only what this rule's last pass changed
/// (semi-naive), apply, until a pass changes nothing. So derivations chain within
/// the iteration (an `ite(c, x, y)` reduces once a pass pins `c`, a disequality
/// unwinds a whole `ite` tower) at a cost proportional to what changed.
///
/// Only for rules whose derivations terminate on their own (unions of existing
/// classes, a bounded set of new terms): the passes run to this rule's local
/// fixpoint inside one iteration, beyond the runner's per-iteration limits.
pub(super) struct LocalFixpoint<S, A> {
    pub(super) searcher: S,
    pub(super) applier: A,
}

/// A rule whose application runs to its own fixpoint (see [`LocalFixpoint`]).
pub(super) fn local_fixpoint_rule<S, A>(name: &str, searcher: S, applier: A) -> Rule
where
    S: egg::Searcher<Symbolic, ConstFold> + Clone + Send + Sync + 'static,
    A: egg::Applier<Symbolic, ConstFold> + Send + Sync + 'static,
{
    let applier = LocalFixpoint {
        searcher: searcher.clone(),
        applier,
    };
    Rewrite::new(name, searcher, applier).expect("local fixpoint rule")
}

impl<S, A> egg::Applier<Symbolic, ConstFold> for LocalFixpoint<S, A>
where
    S: egg::Searcher<Symbolic, ConstFold>,
    A: egg::Applier<Symbolic, ConstFold>,
{
    fn apply_matches(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        matches: &[egg::SearchMatches<Symbolic>],
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        let mut changed = Vec::new();
        let mut apply = |egraph: &mut egg::EGraph<Symbolic, ConstFold>,
                         matches: &[egg::SearchMatches<Symbolic>]| {
            for m in matches {
                for subst in &m.substs {
                    let out = self
                        .applier
                        .apply_one(egraph, m.eclass, subst, None, rule_name);
                    changed.extend(out);
                }
            }
        };
        // A subscriber of its own keeps the log of each pass, whoever runs the rule.
        let passes = egg::Symbol::from(format!("{rule_name} passes"));
        egraph.track_changes();
        let mut pass_start = egraph.change_pos().expect("changes are tracked");
        egraph.mark_seen(passes, pass_start);
        apply(egraph, matches);
        loop {
            egraph.rebuild();
            let changes = egraph
                .changes_since(pass_start)
                .expect("a pass's changes are kept");
            if changes.is_empty() {
                break;
            }
            pass_start = egraph.change_pos().expect("changes are tracked");
            egraph.mark_seen(passes, pass_start);
            let found = self.searcher.search_changes(egraph, &changes, usize::MAX);
            if found.is_empty() {
                break;
            }
            apply(egraph, &found);
        }
        egraph.unsubscribe(passes);
        changed
    }

    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        eclass: egg::Id,
        subst: &egg::Subst,
        searcher_ast: Option<&egg::PatternAst<Symbolic>>,
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        self.applier
            .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        self.applier.vars()
    }
}

/// Groups `(class, subst)` matches by class, in id order and without duplicates.
pub(super) fn group_matches(
    mut found: Vec<(egg::Id, egg::Subst)>,
) -> Vec<egg::SearchMatches<'static, Symbolic>> {
    found.sort_unstable();
    found.dedup();
    let mut out: Vec<egg::SearchMatches<'static, Symbolic>> = Vec::new();
    for (eclass, subst) in found {
        match out.last_mut() {
            Some(m) if m.eclass == eclass => m.substs.push(subst),
            _ => out.push(egg::SearchMatches {
                eclass,
                substs: vec![subst],
                ast: None,
            }),
        }
    }
    out
}

/// The matches in `found`, one `(class, subst)` per substitution.
pub(super) fn ungroup(
    found: Vec<egg::SearchMatches<'_, Symbolic>>,
) -> impl Iterator<Item = (egg::Id, egg::Subst)> + '_ {
    found
        .into_iter()
        .flat_map(|m| m.substs.into_iter().map(move |s| (m.eclass, s)))
}

/// The substitution binding `vars` to the children of `node`, in order: the match
/// of a flat pattern `(op ?v0 ?v1 ..)` at `node`.
pub(super) fn flat_subst(vars: &[Var], node: &Symbolic) -> egg::Subst {
    let mut subst = egg::Subst::default();
    for (v, &c) in vars.iter().zip(node.children()) {
        subst.insert(*v, c);
    }
    subst
}

/// The parents of `class` whose node `keep` accepts, as `(class, node)` with the
/// node canonical. `keep` sees the node as added (only its operator and payload
/// are meaningful there; its children may be stale).
pub(super) fn parents_where(
    egraph: &egg::EGraph<Symbolic, ConstFold>,
    class: egg::Id,
    keep: impl Fn(&Symbolic) -> bool,
) -> Vec<(egg::Id, Symbolic)> {
    let class = egraph.find(class);
    egraph[class]
        .parents()
        .filter(|&p| keep(egraph.id_to_node(p)))
        .map(|p| {
            let node = egraph.id_to_node(p).clone();
            (egraph.find(p), node.map_children(|c| egraph.find(c)))
        })
        .collect()
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
    use egg::{EGraph, Id, Runner};
    use num::{BigInt, BigRational};

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

    /// The invariant `PerClass` used to break: an applier runs only on what its
    /// searcher matched. Every rule's matches on a graph holding a refuted mirror
    /// `a < 0` (and a `false` class holding it) derive nothing unsound; the
    /// asymmetry applied to that `false` class used to refute the assumption
    /// `0 < a` itself.
    #[test]
    fn rules_apply_only_to_their_searchers_matches() {
        for rule in static_rules() {
            let (mut g, orders) = graph(true);
            let found = rule.search(&g);
            rule.apply(&mut g, &found);
            g.rebuild();
            assert_sound(&g, &orders, rule.name.as_str());
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
