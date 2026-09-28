//! Per-rule wall-clock instrumentation.
//!
//! [`timed`] wraps a rule so its searcher and applier report into a thread-local
//! sink, drained into `VerifyStats` at the end of a run. Pure observation: search
//! results and applications are delegated unchanged.

use crate::dhash::HashMap;
use std::sync::Arc;

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::Symbolic;

use super::*;

/// Drain the accumulated per-rule timing (resets the sink).
pub(crate) fn take_rule_timing() -> std::collections::BTreeMap<String, RuleTime> {
    RULE_TIMING.with(|t| {
        std::mem::take(&mut *t.borrow_mut())
            .into_iter()
            .map(|(name, time)| (name.as_str().to_string(), time))
            .collect()
    })
}

pub(super) fn note_time(name: Symbol, search: f64, apply: f64) {
    RULE_TIMING.with(|t| {
        let mut map = t.borrow_mut();
        let entry = map.entry(name).or_default();
        entry.search += search;
        entry.apply += apply;
    });
}

/// Wrap a rule so its searcher/applier report wall-clock time into the
/// thread-local sink. Pure observation — search results and applications are
/// delegated unchanged.
pub(super) fn timed(rw: Rule) -> Rule {
    let name = rw.name;
    Rewrite::new(
        name,
        TimedSearcher {
            name,
            inner: rw.searcher,
        },
        TimedApplier {
            name,
            inner: rw.applier,
        },
    )
    .expect("wrapping preserves var bindings")
}

pub(super) struct TimedSearcher {
    pub(super) name: Symbol,
    pub(super) inner: Arc<dyn Searcher<Symbolic, ConstFold> + Send + Sync>,
}

impl Searcher<Symbolic, ConstFold> for TimedSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let start = std::time::Instant::now();
        let out = self.inner.search_with_limit(egraph, limit);
        note_time(self.name, start.elapsed().as_secs_f64(), 0.0);
        out
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        let start = std::time::Instant::now();
        let out = self.inner.search_eclass_with_limit(egraph, eclass, limit);
        note_time(self.name, start.elapsed().as_secs_f64(), 0.0);
        out
    }

    fn vars(&self) -> Vec<Var> {
        self.inner.vars()
    }
}

pub(super) struct TimedApplier {
    pub(super) name: Symbol,
    pub(super) inner: Arc<dyn Applier<Symbolic, ConstFold> + Send + Sync>,
}

impl Applier<Symbolic, ConstFold> for TimedApplier {
    fn apply_matches(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        matches: &[SearchMatches<Symbolic>],
        rule_name: Symbol,
    ) -> Vec<Id> {
        let start = std::time::Instant::now();
        let out = self.inner.apply_matches(egraph, matches, rule_name);
        note_time(self.name, 0.0, start.elapsed().as_secs_f64());
        out
    }

    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Symbolic>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        self.inner
            .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        self.inner.vars()
    }
}

// ---- Per-rule timing --------------------------------------------------------

thread_local! {
    /// Per-rule search/apply wall clock, accumulated by [`timed`] wrappers and
    /// drained into `VerifyStats` at the end of a run. Thread-local so parallel
    /// tests don't bleed into each other; one verification runs on one thread.
    /// Keyed by the interned rule `Symbol` (Copy) — stringified only at drain.
    static RULE_TIMING: std::cell::RefCell<HashMap<Symbol, RuleTime>> =
        std::cell::RefCell::new(HashMap::default());
}

use crate::verify::stats::RuleTime;
