//! Parse + typecheck + translate + verify.
//!
//! Usage: `cargo run --bin verify -- [--breakdown] [--json] [--viper-metrics]
//! [--trace=CATEGORIES] [--trace-file=PATH] [--viz=DIR] cases/foo.vpr`
//!
//! `--breakdown` (`-b`) prints per-member verify times, slowest first.
//!
//! `--json` prints one JSON object on stdout instead of the text rows: every
//! member's status, the phase timings, per-member verify times, the full
//! [`VerifyStats`](silver_oxide::verify::VerifyStats), peak memory and the git
//! commit the binary was built from. Nothing is printed on stderr unless
//! `--breakdown` asks for it. This is the interface the `bench` runner reads.
//!
//! `--viper-metrics` prints shape metrics of the parsed program (sizes, member
//! counts, fold/unfold, inhale/exhale, `acc`, quantifiers, labels and gotos,
//! calls; totals and per member) as JSON, without verifying. Combined with
//! `--json` the metrics are included in the verification report instead.
//!
//! `--trace=CATEGORIES` writes a structured trace of the verifier's work, one JSON
//! object per line, for the comma-separated categories (`--trace=help` lists them;
//! see `silver_oxide::trace`). It goes to stderr, or to `--trace-file=PATH`.
//!
//! `--viz=DIR` writes Graphviz snapshots of the e-graph and heap after every
//! instruction, one PDF per member, plus the dependency graph, into `DIR`.
//!
//! Exit status is 1 if any row is not `[OK]` — an unproved obligation, an
//! unsupported construct, a declaration skipped because one it depends on was
//! rejected, or a file that could not be parsed at all. 0 only when every unit
//! of the file verified. `--viper-metrics` alone exits 1 only on a parse error.

use silver_oxide::json::Json;
use silver_oxide::pipeline::{MemberResult, PhaseTimings, PipelineError};
use silver_oxide::trace::{Categories, Category, TraceConfig};
use silver_oxide::verify::VerifyStats;
use silver_oxide::viper::metrics::ViperMetrics;
use silver_oxide::{peak_memory, pipeline, viper_parser};
use std::{path::Path, process::ExitCode, time::Duration};

/// Version of the `--json` output layout. Bump on an incompatible change.
const JSON_SCHEMA: u64 = 1;

/// What [`pipeline::run_file_timed`] returns.
type Outcome = Result<
    (
        Vec<MemberResult>,
        PhaseTimings,
        Vec<(String, Duration)>,
        VerifyStats,
    ),
    PipelineError,
>;

struct Args {
    file: String,
    breakdown: bool,
    json: bool,
    metrics: bool,
    trace: Categories,
    trace_file: String,
    viz: Option<String>,
}

const USAGE: &str = "usage: verify [--breakdown] [--json] [--viper-metrics] \
     [--trace=CATEGORIES] [--trace-file=PATH] [--viz=DIR] <file.vpr>";

fn parse_args() -> Result<Args, String> {
    let (mut file, mut breakdown, mut json, mut metrics) = (None, false, false, false);
    let (mut trace, mut trace_file, mut viz) = (Categories::NONE, "-".to_string(), None);
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--breakdown" | "-b" => breakdown = true,
            "--json" => json = true,
            "--viper-metrics" => metrics = true,
            _ => {
                if let Some(spec) = arg.strip_prefix("--trace=") {
                    if spec == "help" {
                        return Err(trace_help());
                    }
                    trace =
                        Categories::parse(spec).map_err(|e| format!("{e}\n{}", trace_help()))?;
                } else if let Some(path) = arg.strip_prefix("--trace-file=") {
                    trace_file = path.to_string();
                } else if let Some(dir) = arg.strip_prefix("--viz=") {
                    viz = Some(dir.to_string());
                } else if arg.starts_with("--") {
                    return Err(format!("unknown option `{arg}`\n{USAGE}"));
                } else {
                    file = Some(arg);
                }
            }
        }
    }
    Ok(Args {
        file: file.ok_or(USAGE)?,
        breakdown,
        json,
        metrics,
        trace,
        trace_file,
        viz,
    })
}

/// The `--trace` categories, one per line.
fn trace_help() -> String {
    let mut s = String::from("trace categories (comma-separated; `all` = all but `time`):");
    for (_, name, about) in Category::ALL {
        s.push_str(&format!("\n  {name:<8} {about}"));
    }
    s
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    if args.metrics && !args.json {
        return report_metrics(&args.file);
    }
    let out = match silver_oxide::trace::open_output(&args.trace_file) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("cannot write trace to {}: {e}", args.trace_file);
            return ExitCode::FAILURE;
        }
    };
    let config = TraceConfig {
        categories: args.trace,
        out,
        viz_dir: args.viz.as_ref().map(Into::into),
    };
    let outcome =
        silver_oxide::trace::with_trace(config, || pipeline::run_file_timed(Path::new(&args.file)));
    if args.json {
        report_json(&args, &outcome)
    } else {
        report_text(&outcome, args.breakdown)
    }
}

fn exit_code(clean: bool) -> ExitCode {
    if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn build_info() -> Vec<(&'static str, Json)> {
    vec![
        ("schema", Json::from(JSON_SCHEMA)),
        ("commit", Json::from(env!("HELIUM_GIT_COMMIT"))),
        ("dirty", Json::from(env!("HELIUM_GIT_DIRTY") == "true")),
    ]
}

fn viper_metrics(file: &str) -> Result<ViperMetrics, String> {
    let source = std::fs::read_to_string(file).map_err(|e| format!("IO: {e}"))?;
    let program = viper_parser::vpr_program(&source).map_err(|e| format!("parse: {e}"))?;
    Ok(ViperMetrics::of(&source, &program))
}

/// `--viper-metrics` alone: the program's shape, without verifying it.
fn report_metrics(file: &str) -> ExitCode {
    let mut out = build_info();
    out.push(("file", Json::from(file)));
    let ok = match viper_metrics(file) {
        Ok(m) => {
            out.push(("viper_metrics", m.to_json()));
            true
        }
        Err(e) => {
            out.push(("error", Json::from(e)));
            false
        }
    };
    println!("{}", Json::obj(out));
    exit_code(ok)
}

/// `--json`: one JSON object on stdout (see the module docs).
fn report_json(args: &Args, outcome: &Outcome) -> ExitCode {
    let mut out = build_info();
    out.push(("file", Json::from(args.file.as_str())));
    let clean = match outcome {
        Err(e) => {
            out.push(("ok", Json::Bool(false)));
            out.push(("error", Json::from(e.to_string())));
            false
        }
        Ok((results, timings, member_times, stats)) => {
            let clean = results.iter().all(|(_, s)| s.is_ok());
            out.push(("ok", Json::Bool(clean)));
            out.push(("error", Json::Null));
            out.push(("results", results_json(results)));
            out.push((
                "phases",
                Json::obj(timings.phases.iter().map(|(name, d)| (*name, secs(*d)))),
            ));
            out.push(("total", secs(timings.total)));
            out.push((
                "member_times",
                Json::obj(
                    member_times
                        .iter()
                        .map(|(name, d)| (name.clone(), secs(*d))),
                ),
            ));
            out.push(("stats", stats.to_json()));
            clean
        }
    };
    out.push((
        "peak_rss_mb",
        Json::opt(peak_memory::peak_rss_bytes(), |b| {
            Json::from(b as f64 / (1024.0 * 1024.0))
        }),
    ));
    if args.metrics {
        let m = viper_metrics(&args.file);
        out.push(("viper_metrics", m.map_or(Json::Null, |m| m.to_json())));
    }
    println!("{}", Json::obj(out));
    if let (true, Ok((_, _, member_times, stats))) = (args.breakdown, outcome) {
        print_breakdown(member_times, stats);
    }
    exit_code(clean)
}

fn secs(d: Duration) -> Json {
    Json::from(d.as_secs_f64())
}

/// Each member as `{name, status, detail}`; `detail` is null when empty.
fn results_json(results: &[MemberResult]) -> Json {
    Json::Arr(
        results
            .iter()
            .map(|(name, status)| {
                let detail = status.to_string();
                let detail = if detail.is_empty() {
                    Json::Null
                } else {
                    Json::from(detail)
                };
                Json::obj([
                    ("name", Json::from(name.as_str())),
                    ("status", Json::from(status.tag())),
                    ("detail", detail),
                ])
            })
            .collect(),
    )
}

/// The default text rows on stdout, timings and counters on stderr.
fn report_text(outcome: &Outcome, breakdown: bool) -> ExitCode {
    let (results, timings, member_times, stats) = match outcome {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[PIPELINE-ERROR] {e}");
            return ExitCode::FAILURE;
        }
    };
    if results.is_empty() {
        println!("[INFO] no method bodies to verify");
    }
    for (name, status) in results {
        let tag = status.tag();
        let detail = status.to_string();
        if detail.is_empty() {
            println!("  [{tag}] {name}");
        } else {
            println!("  [{tag}] {name}: {detail}");
        }
    }
    eprintln!(
        "[TIMING]
{timings}"
    );
    if breakdown {
        print_breakdown(member_times, stats);
    }
    eprintln!("[STATS] {stats:?}");
    exit_code(results.iter().all(|(_, s)| s.is_ok()))
}

fn print_breakdown(member_times: &[(String, Duration)], stats: &VerifyStats) {
    let mut rows = member_times.to_vec();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    eprintln!("[VERIFY-BREAKDOWN] (slowest first)");
    for (name, dur) in &rows {
        eprintln!("  {name:<24} {dur:>10.3?}");
    }
    let mut rules: Vec<_> = stats.rule_timing.0.iter().collect();
    rules.sort_by(|a, b| (b.1.search + b.1.apply).total_cmp(&(a.1.search + a.1.apply)));
    eprintln!("[RULE-TIMING] (search+apply, slowest first)");
    for (name, t) in rules.iter().take(20) {
        eprintln!(
            "  {name:<40} search {:>8.1}ms  apply {:>8.1}ms",
            t.search * 1e3,
            t.apply * 1e3
        );
    }
}
