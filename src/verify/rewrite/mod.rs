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
/// Only for rules whose derivations terminate on their own (unions of existing
/// classes, a bounded set of new terms): the passes run to this rule's local
/// fixpoint inside one iteration, beyond the runner's per-iteration limits.
pub(super) struct PerClass<A>(pub(super) A);

impl<A: egg::Applier<Symbolic, ConstFold>> egg::Applier<Symbolic, ConstFold> for PerClass<A> {
    fn apply_matches(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        matches: &[egg::SearchMatches<Symbolic>],
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        let mut changed = Vec::new();
        let mut walk = |egraph: &mut egg::EGraph<Symbolic, ConstFold>, root| {
            let out = self
                .0
                .apply_one(egraph, root, &egg::Subst::default(), None, rule_name);
            changed.extend(out);
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
        subst: &egg::Subst,
        searcher_ast: Option<&egg::PatternAst<Symbolic>>,
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        self.0
            .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        self.0.vars()
    }
}

fn var(name: &str) -> Var {
    name.parse().expect("valid pattern var")
}

/// The static structural rule set. Per-ADT cons/proj/tag reductions are minted
/// by the registry (`verify::mono`) and appended by `VerifyContext::new`.
pub fn rules() -> Vec<Rule> {
    static_rules().into_iter().map(timed).collect()
}

/// The terminating structural reductions used to **normalize** the e-graph after
/// heap-producing ops (`fold`/`unfold`). The registry's ADT reductions are
/// appended by `VerifyContext::new`. Kept separate from [`rules`] so that
/// *non-terminating* rules run only during full saturation.
pub fn reduce_rules() -> Vec<Rule> {
    terminating_ite_rules().into_iter().map(timed).collect()
}
