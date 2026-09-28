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
