//! Cost metrics for one `verify::verify` run, used by the performance
//! regression tests. The *deterministic* counters (egg is deterministic for a
//! fixed rule set + input) are gated as exact-match snapshots; the timing
//! fields are a non-gating trend (see `plans/verification-perf-regression.md`).

use std::cell::RefCell;
use std::collections::BTreeMap;

thread_local! {
    /// The current run's cost metrics. Thread-local for the same reason the
    /// per-rule timing sink is (`rewrite::timing`): one verification runs on one
    /// thread, and parallel tests must not bleed into each other.
    ///
    /// A sink rather than a field, so counting needs no `&mut` threaded through
    /// the call graph. It previously lived on `FuncRegistry` purely because the
    /// allocator happened to be the per-run shared state already being passed
    /// around — which made `&mut FuncRegistry` look load-bearing at 22 call sites
    /// when only the ADT id minting actually needs it.
    static STATS: RefCell<VerifyStats> = RefCell::new(VerifyStats::default());
}

/// Read or update the run's stats. Do not call recursively — the closure runs
/// while the sink is mutably borrowed.
pub(crate) fn with_stats<T>(f: impl FnOnce(&mut VerifyStats) -> T) -> T {
    STATS.with(|s| f(&mut s.borrow_mut()))
}

/// Bump a counter. Sugar for the overwhelmingly common [`with_stats`] use.
pub(crate) fn bump(f: impl FnOnce(&mut VerifyStats)) {
    with_stats(f);
}

/// Clear the sink. Called at the start of a run so residue from a previous run
/// on this thread (or one that panicked mid-way) cannot leak in.
pub(crate) fn reset_stats() {
    STATS.with(|s| *s.borrow_mut() = VerifyStats::default());
}

/// Drain the run's stats, folding in the per-rule timing. Resets the sink.
pub(crate) fn take_stats() -> VerifyStats {
    let mut stats = STATS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    stats.rule_timing.0 = crate::verify::rewrite::take_rule_timing();
    stats
}

/// The work counters now, to report a trace scope's share of them with
/// [`work_since`].
pub(crate) fn work_now() -> [(&'static str, u64); 8] {
    with_stats(|s| s.work())
}

/// Each work counter's growth since `start`, as trace event fields.
pub(crate) fn work_since(
    start: &[(&'static str, u64); 8],
) -> Vec<(&'static str, crate::json::Json)> {
    work_now()
        .iter()
        .zip(start)
        .map(|((name, now), (_, then))| (*name, (now - then).into()))
        .collect()
}

/// Wall-clock saturation time, broken into egg's phases. Non-deterministic —
/// reported for trends, never compared in the gating snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timing {
    /// e-matching: finding rule left-hand-side matches.
    pub search: f64,
    /// applying the matched rewrites.
    pub apply: f64,
    /// congruence-closure rebuild after unions.
    pub rebuild: f64,
}

impl Timing {
    pub fn total(&self) -> f64 {
        self.search + self.apply + self.rebuild
    }
}

/// Verifier work performed over a run. The non-`timing` fields are deterministic
/// and form the gated cost snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyStats {
    /// `saturate()` calls (full rule set).
    pub saturations: u64,
    /// `reduce()` calls (terminating reductions only).
    pub reduces: u64,
    /// Scratch full-rule-set (non-ground) saturations on a throwaway clone —
    /// `probe`-tier goal probes and forall-WD checks (`run_probe`). Disjoint from
    /// `saturations` (which counts only live persistent-graph runs).
    pub probe_saturations: u64,
    /// egg `Runner` iterations spent inside `probe_saturations` (subset of
    /// `sat_iterations`).
    pub probe_iterations: u64,
    /// Terminating-reduction-only runs on a throwaway clone (`run_reduce`) — the
    /// middle rung of the `ite_decompose` ladder. Paired with `probe_saturations`:
    /// a chain that telescopes cleanly spends neither.
    pub probe_reduces: u64,
    /// Rule runs abandoned by a `Runner` hook — the goal was settled, or the graph
    /// went contradictory, before the fixpoint. Counts iterations *not* spent.
    pub probe_early_stops: u64,
    /// Per-block scratch e-graph: ground clones taken to build a block scratch (one per block that reaches
    /// the `probe` tier), full-rule-set saturations of that shared scratch, and the egg
    /// iterations they cost. `block_scratch_freehits` counts obligations
    /// discharged straight off the saturated scratch with no per-obligation
    /// clone (their pc was already implied by the block cube). All non-gated.
    pub block_scratch_clones: u64,
    pub block_scratch_saturations: u64,
    pub block_scratch_iterations: u64,
    pub block_scratch_freehits: u64,
    /// total egg `Runner` iterations across all saturations/reductions/probes.
    pub sat_iterations: u64,
    /// Rule runs stopped by the iteration or node limit rather than by reaching
    /// a fixpoint (or by a hook). Each is a silent truncation: whatever the
    /// missing iterations would have derived is absent from the verdict.
    pub sat_bound_stops: u64,
    /// peak e-graph size observed in any iteration.
    pub egraph_nodes_peak: usize,
    pub egraph_classes_peak: usize,
    /// Body instructions fully processed (obligations discharged + evaluated)
    /// across all verification units. A progress marker: a failing member stops
    /// counting at its failing instruction, so a higher count on the same input
    /// means the run got further.
    pub insts_processed: u64,
    /// total rule applications, and a per-rule breakdown.
    pub rule_applications: u64,
    pub per_rule: BTreeMap<String, u64>,
    /// `prove_under_pc` calls, and how many reached the expensive `probe`
    /// clone+saturate path (the clearest deterioration signal).
    pub prove_calls: u64,
    /// Obligations closed by each tier of `prove_under_pc`, named as that method
    /// documents them and listed here in execution order. `prove_inconsistent` and
    /// `prove_dead_block` are the vacuous verdicts (contradictory graph /
    /// unreachable block); `prove_goal_true` and `prove_memo` are the two O(1)
    /// e-graph hits; `prove_saturate` is the post-`saturate()` re-check. Together
    /// they say how thin the `prove_probe` population really is — the premise of
    /// the lazy-scratch design (see `design/block-vmir/82-*.md`). Non-gated.
    pub prove_inconsistent: u64,
    pub prove_dead_block: u64,
    pub prove_goal_true: u64,
    pub prove_memo: u64,
    pub prove_saturate: u64,
    pub prove_probe: u64,
    /// Goals discharged by the last tier — non-forking `ite`-goal decomposition
    /// (a constant branch reduces the goal to its other branch, no case split).
    pub prove_ite_decompose: u64,
    /// Non-deterministic timing (excluded from `Eq` / the gated snapshot).
    pub timing: TimingTrend,
    /// Per-e-graph wall clock (ground vs block scratch vs probes vs clones).
    pub graph_timing: GraphTimingTrend,
    /// Per-rule search/apply wall clock (excluded from `Eq` / the snapshot).
    pub rule_timing: RuleTimingTrend,
}

/// `Timing` wrapper whose `PartialEq`/`Eq` ignore the floats, so `VerifyStats`
/// can derive `Eq` and be compared by its deterministic fields alone.
#[derive(Debug, Clone, Copy, Default)]
pub struct TimingTrend(pub Timing);

impl PartialEq for TimingTrend {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for TimingTrend {}

/// Wall-clock seconds attributed to each e-graph a run maintains, so a
/// two-egraph run can be read as "how much went to ground vs the block scratch".
/// Non-deterministic — a trend, never gated.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphTiming {
    /// `saturate()` + `reduce()` on the persistent ground graph.
    pub ground: f64,
    /// `saturate_scratch()` + `reduce_scratch()` on the per-block scratch.
    pub scratch: f64,
    /// `run_probe()` — throwaway clones (`probe`-tier goal probes, WD checks).
    pub probe: f64,
    /// Building a block scratch: the ground clone plus the cube unions/rebuild.
    pub scratch_clone: f64,
}

/// [`GraphTiming`] wrapper, `Eq`-transparent like [`TimingTrend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphTimingTrend(pub GraphTiming);

impl PartialEq for GraphTimingTrend {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for GraphTimingTrend {}

/// Wall-clock seconds one rule spent in its searcher/applier over the whole
/// run. Non-deterministic — a trend, never gated.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleTime {
    pub search: f64,
    pub apply: f64,
}

/// Per-rule timing map wrapper, `Eq`-transparent like [`TimingTrend`] so the
/// gated snapshot stays purely deterministic. `Debug` is a count summary — the
/// full map is rendered by `verify --breakdown`.
#[derive(Clone, Default)]
pub struct RuleTimingTrend(pub BTreeMap<String, RuleTime>);

impl std::fmt::Debug for RuleTimingTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuleTimingTrend({} rules)", self.0.len())
    }
}

impl PartialEq for RuleTimingTrend {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for RuleTimingTrend {}

impl VerifyStats {
    /// A stable, human-readable rendering of the **deterministic** fields only
    /// (timing excluded), one `key=value` per line, sorted. This is the gated
    /// cost snapshot compared against a committed baseline.
    pub fn snapshot_string(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("saturations={}\n", self.saturations));
        s.push_str(&format!("reduces={}\n", self.reduces));
        s.push_str(&format!("sat_iterations={}\n", self.sat_iterations));
        s.push_str(&format!("egraph_nodes_peak={}\n", self.egraph_nodes_peak));
        s.push_str(&format!(
            "egraph_classes_peak={}\n",
            self.egraph_classes_peak
        ));
        s.push_str(&format!("rule_applications={}\n", self.rule_applications));
        s.push_str(&format!("prove_calls={}\n", self.prove_calls));
        s.push_str(&format!("prove_probe={}\n", self.prove_probe));
        // `per_rule` is a BTreeMap → already sorted, hence deterministic.
        for (rule, n) in &self.per_rule {
            s.push_str(&format!("rule.{rule}={n}\n"));
        }
        s
    }

    /// The deterministic work counters a trace scope reports the growth of
    /// (`member`/`block` end events).
    pub(crate) fn work(&self) -> [(&'static str, u64); 8] {
        [
            ("insts", self.insts_processed),
            ("prove_calls", self.prove_calls),
            ("saturations", self.saturations),
            ("reduces", self.reduces),
            ("probe_runs", self.probe_saturations + self.probe_reduces),
            ("scratch_saturations", self.block_scratch_saturations),
            ("sat_iterations", self.sat_iterations),
            ("rule_applications", self.rule_applications),
        ]
    }

    /// Fold one finished `Runner`'s iterations into the stats.
    pub(crate) fn record_run(&mut self, iterations: &[egg::Iteration<()>]) {
        for it in iterations {
            self.sat_iterations += 1;
            self.egraph_nodes_peak = self.egraph_nodes_peak.max(it.egraph_nodes);
            self.egraph_classes_peak = self.egraph_classes_peak.max(it.egraph_classes);
            for (rule, n) in &it.applied {
                self.rule_applications += *n as u64;
                *self.per_rule.entry(rule.to_string()).or_default() += *n as u64;
            }
            self.timing.0.search += it.search_time;
            self.timing.0.apply += it.apply_time;
            self.timing.0.rebuild += it.rebuild_time;
        }
    }
}

impl VerifyStats {
    /// Every field, deterministic counters and timing alike, as JSON for
    /// `verify --json`. Destructured exhaustively so a new counter cannot be
    /// added without deciding how it is reported.
    pub fn to_json(&self) -> crate::json::Json {
        use crate::json::Json;
        let VerifyStats {
            saturations,
            reduces,
            probe_saturations,
            probe_iterations,
            probe_reduces,
            probe_early_stops,
            block_scratch_clones,
            block_scratch_saturations,
            block_scratch_iterations,
            block_scratch_freehits,
            sat_iterations,
            sat_bound_stops,
            egraph_nodes_peak,
            egraph_classes_peak,
            insts_processed,
            rule_applications,
            per_rule,
            prove_calls,
            prove_inconsistent,
            prove_dead_block,
            prove_goal_true,
            prove_memo,
            prove_saturate,
            prove_probe,
            prove_ite_decompose,
            timing,
            graph_timing,
            rule_timing,
        } = self;
        let n = |v: &u64| Json::from(*v);
        Json::obj([
            ("saturations", n(saturations)),
            ("reduces", n(reduces)),
            ("probe_saturations", n(probe_saturations)),
            ("probe_iterations", n(probe_iterations)),
            ("probe_reduces", n(probe_reduces)),
            ("probe_early_stops", n(probe_early_stops)),
            ("block_scratch_clones", n(block_scratch_clones)),
            ("block_scratch_saturations", n(block_scratch_saturations)),
            ("block_scratch_iterations", n(block_scratch_iterations)),
            ("block_scratch_freehits", n(block_scratch_freehits)),
            ("sat_iterations", n(sat_iterations)),
            ("sat_bound_stops", n(sat_bound_stops)),
            ("egraph_nodes_peak", Json::from(*egraph_nodes_peak)),
            ("egraph_classes_peak", Json::from(*egraph_classes_peak)),
            ("insts_processed", n(insts_processed)),
            ("rule_applications", n(rule_applications)),
            ("prove_calls", n(prove_calls)),
            ("prove_inconsistent", n(prove_inconsistent)),
            ("prove_dead_block", n(prove_dead_block)),
            ("prove_goal_true", n(prove_goal_true)),
            ("prove_memo", n(prove_memo)),
            ("prove_saturate", n(prove_saturate)),
            ("prove_probe", n(prove_probe)),
            ("prove_ite_decompose", n(prove_ite_decompose)),
            (
                "per_rule",
                Json::obj(per_rule.iter().map(|(k, v)| (k.clone(), n(v)))),
            ),
            (
                "timing",
                Json::obj([
                    ("search", Json::from(timing.0.search)),
                    ("apply", Json::from(timing.0.apply)),
                    ("rebuild", Json::from(timing.0.rebuild)),
                ]),
            ),
            (
                "graph_timing",
                Json::obj([
                    ("ground", Json::from(graph_timing.0.ground)),
                    ("scratch", Json::from(graph_timing.0.scratch)),
                    ("probe", Json::from(graph_timing.0.probe)),
                    ("scratch_clone", Json::from(graph_timing.0.scratch_clone)),
                ]),
            ),
            (
                "rule_timing",
                Json::obj(rule_timing.0.iter().map(|(k, t)| {
                    (
                        k.clone(),
                        Json::obj([
                            ("search", Json::from(t.search)),
                            ("apply", Json::from(t.apply)),
                        ]),
                    )
                })),
            ),
        ])
    }
}
