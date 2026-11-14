use crate::verify::analysis::ConstFold;
use crate::verify::lang::Symbolic;
use crate::verify::stats;
use crate::vmir::Literal;

/// [`VerifyContext::is_inconsistent`](super::VerifyContext::is_inconsistent) for a detached graph (a probe, or the block
/// scratch). `false` when either boolean literal is absent: a graph that never
/// mentioned one cannot have merged them.
pub(super) fn graph_inconsistent(egraph: &egg::EGraph<Symbolic, ConstFold>) -> bool {
    let (Some(t), Some(f)) = (
        egraph.lookup(Symbolic::Lit(Literal::Bool(true))),
        egraph.lookup(Symbolic::Lit(Literal::Bool(false))),
    ) else {
        return false;
    };
    egraph.find(t) == egraph.find(f)
}

/// One egg run over `rules`. Returns the graph and the iteration log — the
/// caller records the log into the stats once the rule borrow is released.
pub(super) fn run_rules<'r>(
    egraph: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
    iter_limit: Option<usize>,
) -> (egg::EGraph<Symbolic, ConstFold>, Vec<egg::Iteration<()>>) {
    run_rules_until(egraph, rules, iter_limit, None)
}

/// As [`run_rules`], but stopping as soon as `goal` is settled.
///
/// egg checks a `Runner`'s hooks at the start of every iteration and treats an
/// `Err` as a stop, so this costs one class lookup per iteration and saves every
/// iteration after the answer exists.
///
/// **`goal` is only sound on a throwaway graph.** The returned e-graph is *not*
/// saturated, so a caller that records it as clean — `VerifyContext::saturate`
/// (which sets `self.clean`) or `saturate_scratch` (which clears `dirty` on a graph
/// shared by every obligation in the block) — would make later obligations read a
/// fixpoint that was never reached. Pass `None` there and `Some` only for a probe
/// that dies with the call.
///
/// The *inconsistency* stop below carries no such condition and so is installed for
/// every caller: a contradictory graph proves everything, which is exactly why
/// `run_rules` refuses to start on one, so abandoning its saturation cannot change
/// a verdict either.
pub(super) fn run_rules_until<'r>(
    egraph: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
    iter_limit: Option<usize>,
    goal: Option<egg::Id>,
) -> (egg::EGraph<Symbolic, ConstFold>, Vec<egg::Iteration<()>>) {
    // A contradictory graph proves everything, so no rule can change any verdict
    // it yields — stop running them. This is the single choke point for every
    // graph (ground saturate/reduce, scratch saturate/reduce, every probe), and
    // it is per-graph: a deliberately contradictory probe short-circuits without
    // touching a consistent ground graph.
    if graph_inconsistent(&egraph) {
        return (egraph, Vec::new());
    }
    // The observation cache describes the graph this run walks; a run on a
    // different graph must not read it (ground and its clones share ids).
    crate::verify::rewrite::diseq::new_scan_generation();
    // Explicit limits: egg's defaults (30 iterations, 10k nodes, **5 seconds**)
    // are SILENT truncation points — a run that hits one simply stops
    // mid-saturation and the caller sees an ordinary "not proven", which surfaced
    // as a false insufficient-permission at ~20 match arms (one tower level
    // collapses per iteration, so deep-but-terminating collapses need iterations
    // ∝ depth).
    //
    // The time limit is the worst of the three, because it makes a *verdict*
    // depend on wall clock: `enum_v8_p2::m_e_guarded` was measured failing on a
    // 119s run and verifying on a 149s one, same binary and input. That also
    // silently falsifies the premise `perf_regression` rests on ("egg is
    // deterministic for a fixed rule set + input") for any program whose
    // saturation approaches it. Disabled outright — the node and iteration limits
    // are the real backstops, and unlike a clock they are reproducible.
    let mut runner = egg::Runner::default()
        .with_scheduler(egg::SimpleScheduler)
        .with_node_limit(100_000)
        .with_iter_limit(iter_limit.unwrap_or(100))
        .with_time_limit(std::time::Duration::MAX)
        .with_egraph(egraph)
        // Contradiction reached mid-run: every remaining iteration is spent
        // elaborating a graph that already proves everything.
        .with_hook(|r| {
            if graph_inconsistent(&r.egraph) {
                stats::bump(|s| s.probe_early_stops += 1);
                return Err("inconsistent".to_owned());
            }
            Ok(())
        });
    if let Some(goal) = goal {
        runner = runner.with_hook(move |r| {
            // `ConstFold` settles a class without merging it into `true`, so ask
            // the analysis rather than comparing canonical ids.
            if matches!(
                r.egraph[r.egraph.find(goal)].data.known(),
                Some(Literal::Bool(true))
            ) {
                stats::bump(|s| s.probe_early_stops += 1);
                return Err("goal proven".to_owned());
            }
            Ok(())
        });
    }
    let runner = runner.run(rules);
    // A bound stop is a truncation the verdict cannot see. Counted here rather
    // than in `record_run`: egg reports the iteration limit only on the runner,
    // never on its last `Iteration`. Hook stops (`Other`) and fixpoints
    // (`Saturated`) are not truncations; `TimeLimit` is disabled above.
    if matches!(
        runner.stop_reason,
        Some(egg::StopReason::IterationLimit(_) | egg::StopReason::NodeLimit(_))
    ) {
        stats::bump(|s| s.sat_bound_stops += 1);
    }
    (runner.egraph, runner.iterations)
}

/// One full run of `rules`, recorded in the stats. Memo scoping is ambient (see
/// `rewrite::Memo`): the caller's scope decides where instances are memoized.
pub(super) fn saturate_flat<'r>(
    egraph: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
) -> egg::EGraph<Symbolic, ConstFold> {
    let (egraph, iterations) = run_rules(egraph, rules, None);
    stats::bump(|s| s.record_run(&iterations));
    egraph
}

/// Saturate a detached probe e-graph with `rules`, inside a scratch memo scope
/// (its instantiations die with the probe; the live base memo lets it skip
/// rebuilding every already-live instance).
pub(super) fn run_probe<'r>(
    probe: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
) -> egg::EGraph<Symbolic, ConstFold> {
    let _scope = crate::verify::rewrite::ScratchScope::enter();
    let t = std::time::Instant::now();
    let iters_before = stats::with_stats(|s| s.sat_iterations);
    let out = saturate_flat(probe, rules);
    stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
    stats::bump(|s| s.probe_saturations += 1);
    stats::bump(|s| s.probe_iterations += s.sat_iterations - iters_before);
    out
}

/// As [`run_probe`], but stopping as soon as `goal` is settled — for a probe
/// whose whole purpose is that one question. Sound only because the graph is a
/// throwaway; see [`run_rules_until`].
pub(super) fn run_probe_until<'r>(
    probe: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
    goal: egg::Id,
) -> egg::EGraph<Symbolic, ConstFold> {
    let _scope = crate::verify::rewrite::ScratchScope::enter();
    let t = std::time::Instant::now();
    let iters_before = stats::with_stats(|s| s.sat_iterations);
    let (out, iterations) = run_rules_until(probe, rules, None, Some(goal));
    stats::bump(|s| s.record_run(&iterations));
    stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
    stats::bump(|s| s.probe_saturations += 1);
    stats::bump(|s| s.probe_iterations += s.sat_iterations - iters_before);
    out
}

/// Run `rules` (the terminating reductions) over a detached probe — the `reduce`
/// counterpart of [`run_probe`], in a scratch memo scope of its own.
pub(super) fn run_reduce<'r>(
    probe: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
) -> egg::EGraph<Symbolic, ConstFold> {
    let _scope = crate::verify::rewrite::ScratchScope::enter();
    let t = std::time::Instant::now();
    let (out, iterations) = run_rules(probe, rules, None);
    stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
    stats::bump(|s| s.probe_reduces += 1);
    stats::bump(|s| s.record_run(&iterations));
    out
}
