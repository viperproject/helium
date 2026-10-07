//! Timing one command: run it as its own process, with a timeout, and record
//! its wall time, peak memory and output.
//!
//! Every measurement is a separate process run one after another, never in
//! parallel. Output goes to files rather than pipes, so a chatty process
//! cannot block on a full pipe while it is being timed.
//!
//! The command runs in its own process tree (a process group on Unix, a job
//! object on Windows), and the whole tree is killed on a timeout and cleaned
//! up after exit. Killing only the direct child is not enough: `java` on
//! Windows is often a launcher that starts the real JVM, and Silicon starts
//! z3s; left running, they steal CPU from the next measurement and keep
//! writing into the output files the next run reads.

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Serialize;

/// One run of a command.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Seconds from start to exit (or to the kill, on timeout).
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

    let (wall, peak, code, timed_out) = imp::run(cmd, timeout)?;
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
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// The process group a [`spawn_tree`] child leads.
    pub struct Guard(libc::pid_t);

    impl Guard {
        pub fn kill(&self) {
            // SAFETY: a plain syscall on a process group id; with the group
            // already empty it fails harmlessly.
            unsafe { libc::kill(-self.0, libc::SIGKILL) };
        }
    }

    pub fn spawn_tree(cmd: &mut Command) -> std::io::Result<(Child, Option<Guard>)> {
        let child = cmd.process_group(0).spawn()?;
        let pid = child.id() as libc::pid_t;
        Ok((child, Some(Guard(pid))))
    }

    /// The child leads a new process group, so one `kill(-pgid)` reaches
    /// everything it started. (A Ctrl-C at the terminal then no longer
    /// reaches the tree, but the timeout does.)
    ///
    /// `wait4` gives the exit status and the child's own peak RSS in one call;
    /// it runs on a helper thread so the main thread can time out and kill.
    pub fn run(
        cmd: &mut Command,
        timeout: Duration,
    ) -> std::io::Result<(f64, Option<u64>, Option<i32>, bool)> {
        let start = Instant::now();
        let child = cmd.process_group(0).spawn()?;
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
                // returned), so its pid still names its process group.
                unsafe { libc::kill(-pid, libc::SIGKILL) };
                let (r, s, m, _) = rx.recv().expect("waiter thread");
                (r, s, m, start + timeout, true)
            }
        };
        let _ = waiter.join();
        // Whatever the child left running. A pid is not reused while it still
        // names a process group, so this cannot hit a stranger; with the
        // group already empty it fails harmlessly.
        // SAFETY: a plain syscall on a process group id.
        unsafe { libc::kill(-pid, libc::SIGKILL) };
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
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{
        CREATE_SUSPENDED, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME, WaitForSingleObject,
    };

    /// A job object holding the child and everything it starts. Dropping it
    /// kills whatever is still running in it.
    pub struct Job(HANDLE);

    pub type Guard = Job;

    pub fn spawn_tree(cmd: &mut Command) -> std::io::Result<(Child, Option<Guard>)> {
        spawn(cmd).map(|(child, job, _)| (child, job))
    }

    impl Job {
        fn new() -> Option<Job> {
            // SAFETY: no security attributes, no name.
            let h = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if h.is_null() {
                return None;
            }
            let job = Job(h);
            // SAFETY: plain old data, zeroed is a valid value.
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            // Also covers `bench` itself dying: closing the last handle kills
            // the tree.
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `info` is the structure the information class names,
            // passed with its size.
            let ok = unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&raw const info).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            (ok != 0).then_some(job)
        }

        pub fn kill(&self) {
            // SAFETY: `self.0` is a live job handle.
            unsafe { TerminateJobObject(self.0, 1) };
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            self.kill();
            // SAFETY: we own the handle and close it once.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Resume every thread of the (suspended, freshly created) process `pid`;
    /// false when none was found.
    fn resume(pid: u32) -> bool {
        // SAFETY: a system-wide thread snapshot; the handle is closed below.
        let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snap == INVALID_HANDLE_VALUE {
            return false;
        }
        // SAFETY: plain old data; `dwSize` is set as the API requires.
        let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut resumed = false;
        // SAFETY: `snap` is a valid snapshot and `entry` is sized.
        let mut more = unsafe { Thread32First(snap, &mut entry) } != 0;
        while more {
            if entry.th32OwnerProcessID == pid {
                // SAFETY: opening a thread by id; the handle is closed below.
                let t = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
                if !t.is_null() {
                    // SAFETY: `t` is a live thread handle with resume access.
                    resumed |= unsafe { ResumeThread(t) } != u32::MAX;
                    // SAFETY: closing the handle we opened.
                    unsafe { CloseHandle(t) };
                }
            }
            // SAFETY: as for `Thread32First`.
            more = unsafe { Thread32Next(snap, &mut entry) } != 0;
        }
        // SAFETY: closing the snapshot we took.
        unsafe { CloseHandle(snap) };
        resumed
    }

    /// Start `cmd` inside a fresh job. The process is created suspended and
    /// resumed only once it is in the job, so nothing it starts can escape.
    /// Without a job (creation failed, or assignment was refused) it runs as
    /// a plain child and only it is killed on a timeout.
    fn spawn(cmd: &mut Command) -> std::io::Result<(Child, Option<Job>, Instant)> {
        let Some(job) = Job::new() else {
            let start = Instant::now();
            return Ok((cmd.spawn()?, None, start));
        };
        let mut child = cmd.creation_flags(CREATE_SUSPENDED).spawn()?;
        // SAFETY: both handles are live.
        let assigned =
            unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle() as HANDLE) } != 0;
        // The clock starts at the resume: creating the suspended process is
        // not the command's own time.
        let start = Instant::now();
        if !resume(child.id()) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::other("could not resume the new process"));
        }
        Ok((child, assigned.then_some(job), start))
    }

    pub fn run(
        cmd: &mut Command,
        timeout: Duration,
    ) -> std::io::Result<(f64, Option<u64>, Option<i32>, bool)> {
        let (mut child, job, start) = spawn(cmd)?;
        let handle = child.as_raw_handle() as HANDLE;
        let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
        // SAFETY: `handle` is the live process handle owned by `child`.
        let waited = unsafe { WaitForSingleObject(handle, ms) };
        let end = Instant::now();
        if waited == WAIT_TIMEOUT {
            match &job {
                Some(job) => job.kill(),
                None => child.kill()?,
            }
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
        // Dropping the job kills anything the child left running.
        drop(job);
        Ok(((end - start).as_secs_f64(), peak, status.code(), false))
    }
}

/// A long-lived child in its own process tree, as [`run_once`] runs its
/// commands: [`Tree::kill`] reaches everything it started (a JVM's z3s too),
/// and dropping the tree kills it.
pub struct Tree {
    child: std::process::Child,
    guard: Option<imp::Guard>,
}

impl Tree {
    pub fn spawn(cmd: &mut Command) -> std::io::Result<Tree> {
        let (child, guard) = imp::spawn_tree(cmd)?;
        Ok(Tree { child, guard })
    }

    pub fn child(&mut self) -> &mut std::process::Child {
        &mut self.child
    }

    pub fn kill(&mut self) {
        match &self.guard {
            Some(g) => g.kill(),
            None => {
                let _ = self.child.kill();
            }
        }
        let _ = self.child.wait();
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        self.kill();
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
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Timing {
    pub status: Status,
    pub median: Option<f64>,
    pub mad: Option<f64>,
    pub runs: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Taken from a cache rather than measured in this run.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cached: bool,
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
            cached: false,
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

    /// A timeout kills the whole process tree, not just the direct child:
    /// `java` on Windows is often a launcher whose JVM (and its z3s) would
    /// otherwise keep running, stealing CPU from later measurements and
    /// writing into their output files.
    #[test]
    fn timeout_kills_grandchildren() {
        let dir = std::env::temp_dir().join(format!("bench-measure3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The grandchild marks that it started, then writes `late` after 4 s,
        // past the parent's 3 s timeout.
        let mut c = if cfg!(windows) {
            let mut c = Command::new("powershell");
            c.args([
                "-NoProfile",
                "-Command",
                "Start-Process -WindowStyle Hidden powershell -ArgumentList \
                 '-NoProfile','-Command','Set-Content started x; Start-Sleep 4; Set-Content late x'; \
                 Start-Sleep 30",
            ]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "(touch started; sleep 4; touch late) & sleep 30"]);
            c
        };
        c.current_dir(&dir);
        let s = run_once(&mut c, Duration::from_secs(3), &dir.join("out")).unwrap();
        assert!(s.timed_out);
        assert!(dir.join("started").exists(), "the grandchild never started");
        std::thread::sleep(Duration::from_secs(5));
        assert!(
            !dir.join("late").exists(),
            "the grandchild outlived the timeout"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
