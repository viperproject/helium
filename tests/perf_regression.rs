//! Verification-cost regression gate. For each `benchmarks/*.vpr` and each
//! point in [`SCALING_POINTS`] (all of which must verify clean), capture the
//! verifier's *deterministic* cost metrics and compare them against a committed
//! baseline under `benchmarks/baseline/<name>.txt` (scaling points:
//! `benchmarks/baseline/scaling/<stem>.txt`).
//!
//! egg is deterministic for a fixed rule set + input, so the metrics are the same
//! on every machine. The aggregate counters ([`GATED`]) must stay within a band:
//! growing past [`GROWTH`]× fails as a regression, shrinking past [`SHRINK`]×
//! fails so the baseline gets refreshed and keeps guarding the new level. Moves
//! of at most [`SLACK`] never fail. Refresh baselines deliberately after a
//! justified change:
//!
//! ```text
//! UPDATE_PERF_BASELINE=1 cargo test --test perf_regression
//! ```
//!
//! and review the resulting diff. See `plans/verification-perf-regression.md`.

use std::path::{Path, PathBuf};

use silver_oxide::pipeline;

const GATED: &[&str] = &[
    "saturations",
    "reduces",
    "sat_iterations",
    "egraph_nodes_peak",
    "egraph_classes_peak",
    "rule_applications",
    "prove_calls",
    "prove_probe",
];
const GROWTH: f64 = 1.5;
const SHRINK: f64 = 2.0;
const SLACK: u64 = 10;

fn benchmarks_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks")
}

/// Small points of the generated families in `benchmarks/scaling/` whose
/// counters grow faster than linearly in their knob. Each family contributes two
/// points, the second one knob step up, so a steeper growth rate pushes the
/// larger point's counters out of band even where the smaller one stays put.
/// Families whose counters are linear but whose time is not (per-operation cost
/// growing with graph size) are invisible to this gate and are left out.
const SCALING_POINTS: &[&str] = &[
    "pcalias_k4",
    "pcalias_k5",
    "enum_tag_v8",
    "enum_tag_v16",
    "option_d2",
    "option_d4",
    "struct_w4",
    "struct_w8",
    "tuple_w2",
    "tuple_w4",
];

/// `(name, program, baseline)` for every gated benchmark.
fn collect_benchmarks() -> Vec<(String, PathBuf, PathBuf)> {
    let dir = benchmarks_dir();
    let baseline = dir.join("baseline");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("benchmarks/ dir exists")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "vpr"))
        .collect();
    v.sort();
    let flat = v.into_iter().map(|p| {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        let base = baseline.join(format!("{name}.txt"));
        (name, p, base)
    });
    let scaling = SCALING_POINTS.iter().map(|stem| {
        (
            format!("scaling/{stem}"),
            dir.join("scaling/vpr").join(format!("{stem}.vpr")),
            baseline.join("scaling").join(format!("{stem}.txt")),
        )
    });
    flat.chain(scaling).collect()
}

/// Render a `before → after` line diff so a regression is readable.
fn diff(baseline: &str, current: &str) -> String {
    let base: std::collections::BTreeMap<&str, &str> =
        baseline.lines().filter_map(|l| l.split_once('=')).collect();
    let cur: std::collections::BTreeMap<&str, &str> =
        current.lines().filter_map(|l| l.split_once('=')).collect();
    let mut keys: Vec<&str> = base.keys().chain(cur.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    let mut out = String::new();
    for k in keys {
        let (b, c) = (base.get(k), cur.get(k));
        if b != c {
            out.push_str(&format!(
                "  {k}: {} → {}\n",
                b.copied().unwrap_or("(absent)"),
                c.copied().unwrap_or("(absent)")
            ));
        }
    }
    out
}

/// The gated counters outside the band, one line each.
fn out_of_band(baseline: &str, current: &str) -> Vec<String> {
    let get = |s: &str, k: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(k)?.strip_prefix('=')?.parse::<u64>().ok())
            .unwrap_or(0)
    };
    let mut out = Vec::new();
    for k in GATED {
        let (b, c) = (get(baseline, k), get(current, k));
        if c > b + SLACK && c as f64 > b as f64 * GROWTH {
            out.push(format!("  {k}: {b} → {c} regressed past {GROWTH}×\n"));
        } else if b > c + SLACK && b as f64 > c as f64 * SHRINK {
            out.push(format!(
                "  {k}: {b} → {c} improved past {SHRINK}×; refresh the baseline\n"
            ));
        }
    }
    out
}

#[test]
fn verification_cost_matches_baseline() {
    let update = std::env::var_os("UPDATE_PERF_BASELINE").is_some();
    let benches = collect_benchmarks();
    assert!(!benches.is_empty(), "no benchmarks found");

    let mut failures = Vec::new();
    for (name, bench, baseline_path) in benches {
        let (results, _timings, _member_times, stats) =
            pipeline::run_file_timed(&bench).unwrap_or_else(|e| panic!("{name}: pipeline {e}"));
        // A benchmark must verify clean — a failing program has no stable cost.
        for (unit, outcome) in &results {
            assert!(
                outcome.is_ok(),
                "{name}: unit `{unit}` failed to verify: {outcome:?} \
                 (benchmarks must verify OK)"
            );
        }

        let current = stats.snapshot_string();

        if update {
            std::fs::create_dir_all(baseline_path.parent().unwrap()).expect("baseline dir");
            std::fs::write(&baseline_path, &current).expect("write baseline");
            continue;
        }

        let baseline = std::fs::read_to_string(&baseline_path).unwrap_or_else(|_| {
            panic!(
                "{name}: missing baseline {}; run `UPDATE_PERF_BASELINE=1 cargo test \
                 --test perf_regression` to create it",
                baseline_path.display()
            )
        });
        let violations = out_of_band(&baseline, &current);
        if !violations.is_empty() {
            failures.push(format!(
                "{name}: verification cost out of band\n{}full diff:\n{}",
                violations.concat(),
                diff(&baseline, &current)
            ));
        }
    }

    if update {
        eprintln!("[perf] baselines updated; review the diff before committing");
        return;
    }
    assert!(
        failures.is_empty(),
        "verification cost regressed (rerun with UPDATE_PERF_BASELINE=1 if intended):\n\n{}",
        failures.join("\n")
    );
}
