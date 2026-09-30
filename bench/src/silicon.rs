//! Viper's Silicon on the same `.vpr` Helium verifies: the reference verifier
//! for the same input.
//!
//! One pinned fat jar is used, identified by its SHA-256 (and the version line
//! it prints) in every result. Two times per file: `silicon_wall`, the whole
//! process including JVM startup (what a user waits for), and
//! `silicon_verify`, the time Silicon itself reports (the fair comparison
//! with `helium_verify`). Silicon's verdict is recorded per member, by mapping
//! each error's source line to the declaration containing it.
//!
//! Silicon's result depends only on the `.vpr`, the jar and its arguments,
//! never on our commit, so results are cached by `(vpr sha256, jar sha256,
//! arguments)` and Silicon is rerun only when one of those changes (see
//! [`Cache`] for what is kept).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::measure::{self, Sample, Status, Timing};

/// How to run Silicon.
#[derive(Debug, Clone)]
pub struct Silicon {
    pub java: PathBuf,
    pub jar: PathBuf,
    pub jar_sha256: String,
    pub jvm_args: Vec<String>,
    pub args: Vec<String>,
    /// The version line Silicon printed on a probe run (`Silicon 1.1 (abc@x)`).
    pub version: Option<String>,
}

impl Silicon {
    pub fn new(
        java: PathBuf,
        jar: PathBuf,
        jvm_args: Vec<String>,
        args: Vec<String>,
    ) -> std::io::Result<Self> {
        let jar_sha256 = crate::sha256_file(&jar)?;
        Ok(Silicon {
            java,
            jar,
            jar_sha256,
            jvm_args,
            args,
            version: None,
        })
    }

    /// `"<version line>@sha256:<jar hash>"`, recorded in every result.
    pub fn id(&self) -> String {
        format!(
            "{}@sha256:{}",
            self.version.as_deref().unwrap_or("silicon"),
            self.jar_sha256
        )
    }

    /// Everything besides the `.vpr` and the jar that shapes a cached result:
    /// the JVM and Silicon arguments (`-Xss` decides whether Silicon crashes,
    /// Silicon's own flags what it reports). A short hash, part of the cache
    /// key.
    pub fn config_id(&self) -> String {
        let config = format!("jvm {:?}\0silicon {:?}", self.jvm_args, self.args);
        crate::sha256_bytes(config.as_bytes())[..16].to_string()
    }

    pub fn command(&self, vpr: &Path) -> Command {
        let mut c = Command::new(&self.java);
        c.args(&self.jvm_args)
            .arg("-jar")
            .arg(&self.jar)
            .args(&self.args)
            .arg(vpr);
        c
    }
}

/// One Silicon error, attributed to the member it is in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SiliconError {
    pub member: Option<String>,
    pub line: Option<u64>,
    pub message: String,
}

/// What Silicon made of one `.vpr`: the cached unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiliconResult {
    pub silicon: String,
    pub vpr_sha256: String,
    pub status: Status,
    /// Whether the whole file verified; `None` when Silicon did not finish.
    pub verified: Option<bool>,
    pub wall: TimingData,
    pub verify: TimingData,
    pub peak_rss_mb: Option<f64>,
    pub errors: Vec<SiliconError>,
    pub failed_members: BTreeSet<String>,
    pub message: Option<String>,
    /// The per-run timeout, seconds (what a cached timeout is valid for).
    #[serde(default)]
    pub timeout_s: Option<f64>,
}

/// [`Timing`] in a form that round-trips through the cache file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingData {
    pub median: Option<f64>,
    pub mad: Option<f64>,
    pub runs: Vec<f64>,
}

impl From<&Timing> for TimingData {
    fn from(t: &Timing) -> Self {
        TimingData {
            median: t.median,
            mad: t.mad,
            runs: t.runs.clone(),
        }
    }
}

static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^Silicon \S+ \(.*\)").unwrap());
/// Silicon's summary line. Its time is `12.34s` under a minute, `01m:05s`
/// under an hour and `1h:02m:03s` above (silver's `formatMillisReadably`);
/// `ms` is accepted too.
static FINISHED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^Silicon (finished verification successfully|found (\d+) errors?) in (?:(\d+)h:)?(?:(\d+)m:)?([0-9.]+)\s*(m?s)",
    )
    .unwrap()
});
/// One error line: `  [0] <message> (<file>@<line>.<col>)`. The position is
/// often a range, `@320.11--321.30`; the error belongs to its start line.
/// From ten errors on, Silicon pads the index: `[ 0]` ... `[10]`.
static ERROR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*\[\s*\d+\]\s+(.*?)\s*\(([^()@]*)@(\d+)\.(\d+)(?:--\d+\.\d+)?\)\s*$")
        .unwrap()
});
static DECL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(method|function|predicate|domain|field|adt|define|import)\s+([A-Za-z_$][\w$']*)")
        .unwrap()
});

/// Top-level declarations of a Viper source with their first line (1-based),
/// in order. A line belongs to the last declaration that starts at or before
/// it.
pub fn declaration_lines(source: &str) -> Vec<(u64, String)> {
    source
        .lines()
        .enumerate()
        .filter_map(|(i, l)| DECL.captures(l).map(|c| (i as u64 + 1, c[2].to_string())))
        .collect()
}

fn member_at(decls: &[(u64, String)], line: u64) -> Option<String> {
    decls
        .iter()
        .rev()
        .find(|(l, _)| *l <= line)
        .map(|(_, n)| n.clone())
}

/// Parsed Silicon output.
#[derive(Debug, Clone, PartialEq)]
pub struct SiliconOutput {
    pub version: Option<String>,
    pub verified: Option<bool>,
    /// Seconds Silicon reports for verification.
    pub verify_time: Option<f64>,
    pub errors: Vec<SiliconError>,
}

pub fn parse_output(out: &str, decls: &[(u64, String)]) -> SiliconOutput {
    let finished = FINISHED.captures(out);
    let verify_time = finished.as_ref().and_then(|c| {
        let part = |i: usize| {
            c.get(i)
                .map_or(Some(0.0), |m| m.as_str().parse::<f64>().ok())
        };
        let x: f64 = c[5].parse().ok()?;
        let secs = if &c[6] == "ms" { x / 1000.0 } else { x };
        Some(part(3)? * 3600.0 + part(4)? * 60.0 + secs)
    });
    let verified = finished.as_ref().map(|c| c[1].starts_with("finished"));
    let mut errors: Vec<SiliconError> = ERROR
        .captures_iter(out)
        .map(|c| {
            let line = c[3].parse().ok();
            SiliconError {
                member: line.and_then(|l| member_at(decls, l)),
                line,
                message: c[1].to_string(),
            }
        })
        .collect();
    // Errors Silicon counted but we could not read stand in as one error
    // outside every member, so no member is taken to verify: a dropped error
    // would otherwise hide a soundness disagreement.
    let reported = finished
        .as_ref()
        .and_then(|c| c.get(2))
        .and_then(|n| n.as_str().parse::<usize>().ok());
    if let Some(n) = reported.filter(|&n| n > errors.len()) {
        errors.push(SiliconError {
            member: None,
            line: None,
            message: format!(
                "{} of Silicon's {n} errors could not be read",
                n - errors.len()
            ),
        });
    }
    SiliconOutput {
        version: VERSION.find(out).map(|m| m.as_str().to_string()),
        verified,
        verify_time,
        errors,
    }
}

/// Run Silicon on `vpr`: `warmup` untimed runs, then `runs` timed ones.
pub fn measure(
    silicon: &mut Silicon,
    vpr: &Path,
    vpr_sha256: &str,
    warmup: usize,
    runs: usize,
    timeout: Duration,
    scratch: &Path,
) -> std::io::Result<SiliconResult> {
    let source = std::fs::read_to_string(vpr)?;
    let decls = declaration_lines(&source);
    let finished = |s: &Sample| FINISHED.is_match(&s.stdout) || FINISHED.is_match(&s.stderr);
    let samples = measure::repeat(
        || silicon.command(vpr),
        warmup,
        runs,
        timeout,
        scratch,
        finished,
    )?;
    let wall = measure::wall_timing(&samples, finished);

    // A timed-out sample's output is not its own verdict, even when it holds
    // a summary line: only runs that finished inside the timeout count.
    let parsed: Vec<SiliconOutput> = samples
        .iter()
        .filter(|s| !s.timed_out && finished(s))
        .map(|s| parse_output(&format!("{}\n{}", s.stdout, s.stderr), &decls))
        .collect();
    if silicon.version.is_none() {
        silicon.version = parsed.iter().find_map(|p| p.version.clone());
    }
    let verify_runs: Vec<f64> = parsed.iter().filter_map(|p| p.verify_time).collect();
    let verify = Timing::from_runs(wall.status, verify_runs, &[], None);
    let last = parsed.last();
    let errors = last.map(|p| p.errors.clone()).unwrap_or_default();
    Ok(SiliconResult {
        silicon: silicon.id(),
        vpr_sha256: vpr_sha256.to_string(),
        status: wall.status,
        verified: last.and_then(|p| p.verified),
        wall: (&wall).into(),
        verify: (&verify).into(),
        peak_rss_mb: wall.peak_rss_mb,
        failed_members: errors.iter().filter_map(|e| e.member.clone()).collect(),
        errors,
        message: wall.message.clone(),
        timeout_s: Some(timeout.as_secs_f64()),
    })
}

/// Silicon results keyed by `"<vpr sha256>|<jar sha256>|<config>"` (see
/// [`Silicon::config_id`]).
///
/// What is kept: finished results, and timeouts with the timeout they hit.
/// Errors (a crash, an out-of-memory JVM) are not kept, since they may be
/// the machine's fault rather than the file's, and are measured again.
pub type Cache = crate::cache::Cache<SiliconResult>;

impl Cache {
    pub fn key(vpr_sha256: &str, jar_sha256: &str, config: &str) -> String {
        format!("{vpr_sha256}|{jar_sha256}|{config}")
    }

    /// Whether to keep `r` in the cache.
    pub fn keeps(r: &SiliconResult) -> bool {
        r.status != Status::Error
    }

    /// Whether `hit` answers a run of `runs` timed runs with `timeout`: a
    /// finished result needs at least as many runs, a timeout must have hit a
    /// limit no shorter than this one (it would time out again).
    pub fn reusable(hit: &SiliconResult, runs: usize, timeout: Duration) -> bool {
        match hit.status {
            Status::Ok => hit.wall.runs.len() >= runs,
            Status::Timeout => hit.timeout_s.is_some_and(|t| t >= timeout.as_secs_f64()),
            Status::Error => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "field f: Int\n\nmethod m_a(x: Ref)\n{\n  assert false\n}\n\nmethod m_b()\n{\n}\n\nfunction g(): Int\n{ 1 }\n";

    #[test]
    fn finds_declarations() {
        let d = declaration_lines(SRC);
        assert_eq!(
            d,
            [
                (1, "f".into()),
                (3, "m_a".into()),
                (8, "m_b".into()),
                (12, "g".into())
            ]
        );
        assert_eq!(member_at(&d, 5).as_deref(), Some("m_a"));
        assert_eq!(member_at(&d, 9).as_deref(), Some("m_b"));
    }

    #[test]
    fn parses_success_and_failure() {
        let d = declaration_lines(SRC);
        let ok = "Silicon 1.1-SNAPSHOT (4dfc6b5b@(detached))\nSilicon finished verification successfully in 2.87s.\n";
        let p = parse_output(ok, &d);
        assert_eq!(
            p.version.as_deref(),
            Some("Silicon 1.1-SNAPSHOT (4dfc6b5b@(detached))")
        );
        assert_eq!((p.verified, p.verify_time), (Some(true), Some(2.87)));
        assert!(p.errors.is_empty());

        for (line, secs) in [
            ("in 870ms.", 0.87),
            ("in 01m:05s.", 65.0),
            ("in 1h:02m:03s.", 3723.0),
        ] {
            let p = parse_output(
                &format!("Silicon finished verification successfully {line}\n"),
                &d,
            );
            assert_eq!(
                (p.verified, p.verify_time),
                (Some(true), Some(secs)),
                "{line}"
            );
        }

        let bad = "Silicon 1.1-SNAPSHOT (4dfc6b5b@(detached))\nSilicon found 1 error in 3.10s:\n  [0] Assert might fail. Assertion false might not hold. (x.vpr@5.3)\n";
        let p = parse_output(bad, &d);
        assert_eq!((p.verified, p.verify_time), (Some(false), Some(3.1)));
        assert_eq!(
            p.errors,
            [SiliconError {
                member: Some("m_a".into()),
                line: Some(5),
                message: "Assert might fail. Assertion false might not hold.".into()
            }]
        );
    }

    /// Silicon usually reports a range (`@9.5--10.12`), with parentheses in
    /// the message and Windows line endings; the error goes to the member
    /// holding the range's start line.
    #[test]
    fn parses_error_ranges() {
        let d = declaration_lines(SRC);
        let out = "Silicon found 1 error in 18.56s:\r\n  [0] Postcondition of m_b might not hold. There might be insufficient permission to access p(get(old(x))) (x.vpr@9.5--10.12)\r\n";
        let p = parse_output(out, &d);
        assert_eq!(p.verified, Some(false));
        assert_eq!(
            p.errors,
            [SiliconError {
                member: Some("m_b".into()),
                line: Some(9),
                message: "Postcondition of m_b might not hold. There might be insufficient permission to access p(get(old(x)))".into()
            }]
        );
    }

    /// From ten errors on, Silicon pads the index (`[ 0]`); every error must
    /// still be read, or the members it names pass as verified.
    #[test]
    fn parses_padded_error_indices() {
        let d = declaration_lines(SRC);
        let mut out = "Silicon found 11 errors in 31.37s:\n".to_string();
        for i in 0..11 {
            let line = if i == 0 { 5 } else { 9 };
            out += &format!("  [{i:>2}] Assert might fail. (x.vpr@{line}.10--{line}.38)\n");
        }
        let p = parse_output(&out, &d);
        assert_eq!(p.errors.len(), 11);
        assert_eq!(p.errors[0].member.as_deref(), Some("m_a"));
        assert!(p.errors.iter().all(|e| e.member.is_some()));
    }

    /// Errors Silicon counts but we cannot read become one unplaced error,
    /// so no member is taken to verify.
    #[test]
    fn unread_errors_are_unplaced() {
        let d = declaration_lines(SRC);
        let out = "Silicon found 3 errors in 1.00s:\n  [0] Assert might fail. (x.vpr@5.3)\n  [1] Something new {x.vpr, line 9}\n";
        let p = parse_output(out, &d);
        assert_eq!(p.errors.len(), 2);
        assert_eq!(p.errors[0].member.as_deref(), Some("m_a"));
        assert_eq!(p.errors[1].member, None);
        assert_eq!(
            p.errors[1].message,
            "2 of Silicon's 3 errors could not be read"
        );
    }

    fn result(status: Status, runs: usize, timeout_s: Option<f64>) -> SiliconResult {
        SiliconResult {
            silicon: "s".into(),
            vpr_sha256: "v".into(),
            status,
            verified: (status == Status::Ok).then_some(true),
            wall: TimingData {
                median: None,
                mad: None,
                runs: vec![1.0; runs],
            },
            verify: TimingData {
                median: None,
                mad: None,
                runs: vec![],
            },
            peak_rss_mb: None,
            errors: vec![],
            failed_members: BTreeSet::new(),
            message: None,
            timeout_s,
        }
    }

    #[test]
    fn cache_keeps_and_reuses() {
        let t = |s| Duration::from_secs(s);
        // Finished: reused for as many runs as it has, not more.
        let ok = result(Status::Ok, 5, Some(300.0));
        assert!(Cache::keeps(&ok));
        assert!(Cache::reusable(&ok, 5, t(300)));
        assert!(!Cache::reusable(&ok, 6, t(300)));
        // Timed out: reused only while the timeout is no longer.
        let timeout = result(Status::Timeout, 1, Some(300.0));
        assert!(Cache::keeps(&timeout));
        assert!(Cache::reusable(&timeout, 5, t(300)));
        assert!(Cache::reusable(&timeout, 5, t(60)));
        assert!(!Cache::reusable(&timeout, 5, t(600)));
        // A timeout cached before the limit was recorded is measured again.
        assert!(!Cache::reusable(
            &result(Status::Timeout, 1, None),
            5,
            t(60)
        ));
        // Errors are never kept.
        let error = result(Status::Error, 1, Some(300.0));
        assert!(!Cache::keeps(&error));
        assert!(!Cache::reusable(&error, 1, t(300)));
    }

    #[test]
    fn config_is_part_of_the_key() {
        let s = |jvm: &str| Silicon {
            java: "java".into(),
            jar: "s.jar".into(),
            jar_sha256: "j".into(),
            jvm_args: vec![jvm.into()],
            args: vec![],
            version: None,
        };
        assert_eq!(s("-Xss128m").config_id(), s("-Xss128m").config_id());
        assert_ne!(s("-Xss128m").config_id(), s("-Xss512m").config_id());
    }
}
