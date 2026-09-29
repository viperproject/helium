//! The benchmark runner behind `bench` (see `plans/regression-pipeline.md`).
//!
//! - [`suites`]: discovering suites under `benchmarks/` by directory layout;
//! - [`measure`]: timing one process (wall time, peak memory, timeout);
//! - [`helium`], [`silicon`]: running and reading the two verifiers;
//! - [`rust_metrics`]: per-function shape metrics from a `syn` parse;
//! - [`run`]: measuring every file and joining it all into one run JSON;
//! - [`check`]: `bench check-suites`.

pub mod check;
pub mod helium;
pub mod measure;
pub mod run;
pub mod rust_metrics;
pub mod silicon;
pub mod suites;

use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

/// Run `git` in `repo`, returning trimmed stdout on success.
pub fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// This machine's name.
pub fn hostname() -> String {
    for var in ["BENCH_HOST", "HOSTNAME", "COMPUTERNAME"] {
        if let Ok(h) = std::env::var(var) {
            if !h.trim().is_empty() {
                return h.trim().to_string();
            }
        }
    }
    Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    utc_from_unix(secs)
}

fn utc_from_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn utc_formatting() {
        assert_eq!(super::utc_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::utc_from_unix(1_790_000_000), "2026-09-21T14:13:20Z");
        assert_eq!(super::utc_from_unix(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn sha256() {
        assert_eq!(
            super::sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
