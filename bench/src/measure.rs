//! Timing one command: run it as its own process, with a timeout, and record
//! its wall time, peak memory and output.
//!
//! Every measurement is a separate process run one after another, never in
//! parallel. Output goes to files rather than pipes, so a chatty process
//! cannot block on a full pipe while it is being timed.

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;

/// One run of a command.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Seconds from spawn to exit (or to the kill, on timeout).
    pub wall: f64,
    pub peak_rss_mb: Option<f64>,
    /// Exit code; `None` when killed (timeout or signal).
    pub code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Run `cmd` once. `scratch` is a directory for the output files.
pub fn run_once(cmd: &mut Command, timeout: Duration, scratch: &Path) -> std::io::Result<Sample> {
    std::fs::create_dir_all(scratch)?;
    let out_path = scratch.join("stdout.txt");
    let err_path = scratch.join("stderr.txt");
    cmd.stdin(Stdio::null())
        .stdout(File::create(&out_path)?)
        .stderr(File::create(&err_path)?);

    let start = Instant::now();
    let child = cmd.spawn()?;
    let (wall, peak, code, timed_out) = imp::wait(child, start, timeout)?;
    Ok(Sample {
        wall,
        peak_rss_mb: peak.map(|b| b as f64 / (1024.0 * 1024.0)),
        code,
        timed_out,
        stdout: String::from_utf8_lossy(&std::fs::read(&out_path)?).into_owned(),
        stderr: String::from_utf8_lossy(&std::fs::read(&err_path)?).into_owned(),
    })
}

#[cfg(unix)]
mod imp {
    use std::process::Child;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// `wait4` gives the exit status and the child's own peak RSS in one call;
    /// it runs on a helper thread so the main thread can time out and kill.
    pub fn wait(
        child: Child,
        start: Instant,
        timeout: Duration,
    ) -> std::io::Result<(f64, Option<u64>, Option<i32>, bool)> {
        let pid = child.id() as libc::pid_t;
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let mut status = 0;
            // SAFETY: `rusage` is plain old data, zeroed is a valid value.
            let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
            // SAFETY: `pid` is our un-reaped child; both out-pointers are valid.
            let r = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
            let end = Instant::now();
            let _ = tx.send((r, status, usage.ru_maxrss, end));
        });
        let (r, status, maxrss, end, timed_out) = match rx.recv_timeout(timeout) {
            Ok((r, s, m, e)) => (r, s, m, e, false),
            Err(_) => {
                // SAFETY: the child has not been reaped (the waiter has not
                // returned), so `pid` still names it.
                unsafe { libc::kill(pid, libc::SIGKILL) };
                let (r, s, m, _) = rx.recv().expect("waiter thread");
                (r, s, m, start + timeout, true)
            }
        };
        let _ = waiter.join();
        // The child is reaped; dropping the handle does not wait on it again.
        drop(child);
        if r < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // `ru_maxrss` is KiB on Linux, bytes on macOS.
        let peak = if cfg!(target_os = "macos") {
            maxrss as u64
        } else {
            maxrss as u64 * 1024
        };
        let code = libc::WIFEXITED(status).then(|| libc::WEXITSTATUS(status));
        Ok(((end - start).as_secs_f64(), Some(peak), code, timed_out))
    }
}

#[cfg(windows)]
mod imp {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{HANDLE, WAIT_TIMEOUT};
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    pub fn wait(
        mut child: Child,
        start: Instant,
        timeout: Duration,
    ) -> std::io::Result<(f64, Option<u64>, Option<i32>, bool)> {
        let handle = child.as_raw_handle() as HANDLE;
        let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
        // SAFETY: `handle` is the live process handle owned by `child`.
        let waited = unsafe { WaitForSingleObject(handle, ms) };
        let end = Instant::now();
        if waited == WAIT_TIMEOUT {
            child.kill()?;
            child.wait()?;
            return Ok((timeout.as_secs_f64(), None, None, true));
        }
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: the process has exited but its handle is still open, which
        // keeps its accounting readable; `counters` is sized as declared.
        let ok = unsafe { GetProcessMemoryInfo(handle, &mut counters, counters.cb) };
        let peak = (ok != 0).then_some(counters.PeakWorkingSetSize as u64);
        let status = child.wait()?;
        Ok(((end - start).as_secs_f64(), peak, status.code(), false))
    }
}

/// Median and median absolute deviation of `xs` (not empty).
pub fn median_mad(xs: &[f64]) -> (f64, f64) {
    let med = median(xs);
    let dev: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    (med, median(&dev))
}

pub fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// What a timed column records: `ok`, `timeout` (recorded, not dropped),
/// `error` (the command failed; its times are still kept) or `skipped`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Timeout,
    Error,
}

/// Summary of the timed runs of one command.
#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    pub status: Status,
    pub median: Option<f64>,
    pub mad: Option<f64>,
    pub runs: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl Timing {
    pub fn from_runs(
        status: Status,
        runs: Vec<f64>,
        peaks: &[f64],
        message: Option<String>,
    ) -> Self {
        let (median, mad) = if runs.is_empty() {
            (None, None)
        } else {
            let (m, d) = median_mad(&runs);
            (Some(m), Some(d))
        };
        Timing {
            status,
            median,
            mad,
            runs,
            peak_rss_mb: (!peaks.is_empty()).then(|| self::median(peaks)),
            message,
        }
    }
}

/// Run `make()` `warmup` times untimed, then `runs` times timed, stopping at
/// the first timeout (a timed-out command is not retried five times over).
/// `ok` decides whether a finished sample succeeded.
pub fn repeat(
    mut make: impl FnMut() -> Command,
    warmup: usize,
    runs: usize,
    timeout: Duration,
    scratch: &Path,
    ok: impl Fn(&Sample) -> bool,
) -> std::io::Result<Vec<Sample>> {
    let mut samples = Vec::new();
    for i in 0..warmup + runs {
        let s = run_once(&mut make(), timeout, scratch)?;
        let stop = s.timed_out || !ok(&s);
        if i >= warmup || stop {
            samples.push(s);
        }
        if stop {
            break;
        }
    }
    Ok(samples)
}

/// Summarise samples from [`repeat`] into a [`Timing`] of their wall times.
pub fn wall_timing(samples: &[Sample], ok: impl Fn(&Sample) -> bool) -> Timing {
    let last = samples.last();
    let status = match last {
        Some(s) if s.timed_out => Status::Timeout,
        Some(s) if !ok(s) => Status::Error,
        _ => Status::Ok,
    };
    let message = match status {
        Status::Error => last.map(|s| tail(&format!("{}{}", s.stderr, s.stdout), 400)),
        _ => None,
    };
    let peaks: Vec<f64> = samples.iter().filter_map(|s| s.peak_rss_mb).collect();
    Timing::from_runs(
        status,
        samples.iter().map(|s| s.wall).collect(),
        &peaks,
        message,
    )
}

/// The last `n` bytes of `s`, on a char boundary.
pub fn tail(s: &str, n: usize) -> String {
    let s = s.trim();
    let mut start = s.len().saturating_sub(n);
    while !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_mad() {
        assert_eq!(median_mad(&[3.0, 1.0, 2.0]), (2.0, 1.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 10.0]), 2.5);
    }

    fn sleeper(secs: &str) -> Command {
        if cfg!(windows) {
            let mut c = Command::new("powershell");
            c.args([
                "-NoProfile",
                "-Command",
                &format!("Start-Sleep -Seconds {secs}"),
            ]);
            c
        } else {
            let mut c = Command::new("sleep");
            c.arg(secs);
            c
        }
    }

    #[test]
    fn times_out_and_kills() {
        let dir = std::env::temp_dir().join(format!("bench-measure-{}", std::process::id()));
        let s = run_once(&mut sleeper("20"), Duration::from_millis(1500), &dir).unwrap();
        assert!(s.timed_out);
        assert!(s.wall < 10.0);
        assert_eq!(s.code, None);
    }

    #[test]
    fn records_exit_code_output_and_memory() {
        let dir = std::env::temp_dir().join(format!("bench-measure2-{}", std::process::id()));
        let mut c = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "echo hi && exit 3"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "echo hi; exit 3"]);
            c
        };
        let s = run_once(&mut c, Duration::from_secs(30), &dir).unwrap();
        assert!(!s.timed_out);
        assert_eq!(s.code, Some(3));
        assert_eq!(s.stdout.trim(), "hi");
        assert!(s.peak_rss_mb.unwrap() > 0.1);
    }
}
