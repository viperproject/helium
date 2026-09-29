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
//! Silicon's result depends only on the `.vpr` and the jar, never on our
//! commit, so results are cached by `(vpr sha256, jar sha256)` and Silicon is
//! rerun only when an encoding or the jar changes.

use std::collections::{BTreeMap, BTreeSet};
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
static ERROR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*\[\d+\]\s+(.*?)\s*\(([^()@]*)@(\d+)\.(\d+)\)\s*$").unwrap()
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
    let errors = ERROR
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

    let parsed: Vec<SiliconOutput> = samples
        .iter()
        .filter(|s| finished(s))
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
    })
}

/// Silicon results keyed by `"<vpr sha256>|<jar sha256>"`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Cache {
    pub schema: u32,
    pub entries: BTreeMap<String, SiliconResult>,
}

impl Cache {
    pub fn key(vpr_sha256: &str, jar_sha256: &str) -> String {
        format!("{vpr_sha256}|{jar_sha256}")
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(std::io::Error::other),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Cache {
                schema: 1,
                ..Default::default()
            }),
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(tmp, path)
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
}
