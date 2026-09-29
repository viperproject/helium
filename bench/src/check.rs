//! `bench check-suites`: validate the layout of every suite before anything is
//! measured.
//!
//! Errors (a run refuses to start): a `suite.json` that does not parse, a
//! family pattern that does not match its stems, a malformed
//! `expected_failures.txt`, a `.rs` rustc rejects. Warnings (reported, never
//! fatal): a `.rs` with no `.vpr` yet, a `.vpr` older than its `.rs`.

use std::path::Path;
use std::time::Duration;

use crate::measure;
use crate::run::RustcOptions;
use crate::suites::{self, Suite};

#[derive(Debug, Default)]
pub struct Report {
    pub suites: Vec<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn check(benchmarks: &Path, rustc: Option<&RustcOptions>, scratch: &Path) -> Report {
    let mut report = Report::default();
    let rustc = rustc.map(RustcOptions::resolve);
    let rustc = rustc.as_ref();
    for suite in suites::discover(benchmarks) {
        report.suites.push(format!(
            "{} ({} files, {} measurable{})",
            suite.name,
            suite.files.len(),
            suite.measurable().count(),
            if suite.families.is_empty() {
                String::new()
            } else {
                format!(
                    ", families: {}",
                    suite
                        .families
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ));
        report.errors.extend(suite.errors.iter().cloned());
        report.warnings.extend(suite.warnings.iter().cloned());
        check_staleness(&suite, &mut report);
        if let Some(rustc) = rustc {
            check_rustc(&suite, rustc, scratch, &mut report);
        }
    }
    report
}

/// A `.vpr` older than its `.rs` is probably a stale encoding. Checkout
/// order can make mtimes lie, so this is a warning.
fn check_staleness(suite: &Suite, report: &mut Report) {
    for f in &suite.files {
        let (Some(rs), Some(vpr)) = (&f.rs, &f.vpr) else {
            continue;
        };
        let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        if let (Some(r), Some(v)) = (mtime(rs), mtime(vpr)) {
            if v < r {
                report.warnings.push(format!(
                    "{}: vpr/{}.vpr is older than src/{}.rs (stale encoding?)",
                    suite.name, f.stem, f.stem
                ));
            }
        }
    }
}

fn check_rustc(suite: &Suite, rustc: &RustcOptions, scratch: &Path, report: &mut Report) {
    let out_dir = scratch.join("check-rustc");
    let _ = std::fs::create_dir_all(&out_dir);
    for f in &suite.files {
        let Some(rs) = &f.rs else { continue };
        let mut c = rustc.command();
        c.args(suite.rustc_args())
            .args(["--emit=metadata", "--cap-lints", "allow", "--out-dir"])
            .arg(&out_dir)
            .arg(rs);
        match measure::run_once(&mut c, Duration::from_secs_f64(suite.timeout_s()), scratch) {
            Ok(s) if s.code == Some(0) => {}
            Ok(s) => report.errors.push(format!(
                "{}: rustc rejects src/{}.rs: {}",
                suite.name,
                f.stem,
                measure::tail(&s.stderr, 300)
            )),
            Err(e) => report
                .errors
                .push(format!("{}: cannot run rustc: {e}", suite.name)),
        }
    }
}
