//! Parse + typecheck + translate + verify.
//!
//! Usage: `cargo run --bin verify -- [--breakdown] [--json] [--viper-metrics] cases/foo.vpr`
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
//! Exit status is 1 if any row is not `[OK]` — an unproved obligation, an
//! unsupported construct, a declaration skipped because one it depends on was
//! rejected, or a file that could not be parsed at all. 0 only when every unit
//! of the file verified. `--viper-metrics` alone exits 1 only on a parse error.

use silver_oxide::json::Json;
use silver_oxide::viper::metrics::ViperMetrics;
use silver_oxide::{peak_memory, pipeline, viper_parser};
use std::{path::Path, process::ExitCode};

/// Version of the `--json` output layout. Bump on an incompatible change.
const JSON_SCHEMA: u64 = 1;

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

fn main() -> ExitCode {
    let mut file = None;
    let mut breakdown = false;
    let mut json = false;
    let mut metrics = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--breakdown" | "-b" => breakdown = true,
            "--json" => json = true,
            "--viper-metrics" => metrics = true,
            _ => file = Some(arg),
        }
    }
    let Some(file) = file else {
        eprintln!("usage: verify [--breakdown] [--json] [--viper-metrics] <file.vpr>");
        return ExitCode::FAILURE;
    };

    if metrics && !json {
        let mut out = build_info();
        out.push(("file", Json::from(file.as_str())));
        let code = match viper_metrics(&file) {
            Ok(m) => {
                out.push(("viper_metrics", m.to_json()));
                ExitCode::SUCCESS
            }
            Err(e) => {
                out.push(("error", Json::from(e)));
                ExitCode::FAILURE
            }
        };
        println!("{}", Json::obj(out));
        return code;
    }

    let outcome = pipeline::run_file_timed(Path::new(&file));

    if json {
        let mut out = build_info();
        out.push(("file", Json::from(file.as_str())));
        let clean = match &outcome {
            Err(e) => {
                out.push(("ok", Json::Bool(false)));
                out.push(("error", Json::from(e.to_string())));
                false
            }
            Ok((results, timings, member_times, stats)) => {
                let clean = results.iter().all(|(_, s)| s.is_ok());
                out.push(("ok", Json::Bool(clean)));
                out.push(("error", Json::Null));
                out.push((
                    "results",
                    Json::Arr(
                        results
                            .iter()
                            .map(|(name, status)| {
                                let detail = status.to_string();
                                Json::obj([
                                    ("name", Json::from(name.as_str())),
                                    ("status", Json::from(status.tag())),
                                    (
                                        "detail",
                                        if detail.is_empty() {
                                            Json::Null
                                        } else {
                                            Json::from(detail)
                                        },
                                    ),
                                ])
                            })
                            .collect(),
                    ),
                ));
                out.push((
                    "phases",
                    Json::obj(
                        timings
                            .phases
                            .iter()
                            .map(|(name, d)| (*name, Json::from(d.as_secs_f64()))),
                    ),
                ));
                out.push(("total", Json::from(timings.total.as_secs_f64())));
                out.push((
                    "member_times",
                    Json::obj(
                        member_times
                            .iter()
                            .map(|(name, d)| (name.clone(), Json::from(d.as_secs_f64()))),
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
        if metrics {
            out.push((
                "viper_metrics",
                match viper_metrics(&file) {
                    Ok(m) => m.to_json(),
                    Err(_) => Json::Null,
                },
            ));
        }
        println!("{}", Json::obj(out));
        if breakdown {
            if let Ok((_, _, member_times, stats)) = &outcome {
                print_breakdown(member_times, stats);
            }
        }
        return if clean {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    match outcome {
        Err(e) => {
            eprintln!("[PIPELINE-ERROR] {e}");
            ExitCode::FAILURE
        }
        Ok((results, timings, member_times, stats)) => {
            if results.is_empty() {
                println!("[INFO] no method bodies to verify");
            }
            let mut clean = true;
            for (name, status) in &results {
                clean &= status.is_ok();
                let tag = status.tag();
                let detail = status.to_string();
                if detail.is_empty() {
                    println!("  [{tag}] {name}");
                } else {
                    println!("  [{tag}] {name}: {detail}");
                }
            }
            eprintln!("[TIMING]\n{timings}");
            if breakdown {
                print_breakdown(&member_times, &stats);
            }
            eprintln!("[STATS] {stats:?}");
            if clean {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

fn print_breakdown(
    member_times: &[(String, std::time::Duration)],
    stats: &silver_oxide::verify::VerifyStats,
) {
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
