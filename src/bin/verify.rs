//! Parse + typecheck + translate + verify.
//!
//! Usage: `cargo run --bin verify -- [--breakdown] cases/foo.vpr`
//!
//! `--breakdown` (`-b`) prints per-member verify times, slowest first.
//!
//! Exit status is 1 if any row is not `[OK]` — an unproved obligation, an
//! unsupported construct, a declaration skipped because one it depends on was
//! rejected, or a file that could not be parsed at all. 0 only when every unit
//! of the file verified.

use silver_oxide::pipeline;
use std::{path::Path, process::ExitCode};

fn main() -> ExitCode {
    let mut file = None;
    let mut breakdown = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--breakdown" | "-b" => breakdown = true,
            _ => file = Some(arg),
        }
    }
    let Some(file) = file else {
        eprintln!("usage: verify [--breakdown] <file.vpr>");
        return ExitCode::FAILURE;
    };

    match pipeline::run_file_timed(Path::new(&file)) {
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
                let mut rows = member_times.clone();
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
            eprintln!("[STATS] {stats:?}");
            if clean {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
