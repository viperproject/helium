//! Silicon in a warm JVM: how every Silicon column is measured.
//!
//! In a cold `java -jar` run, Silicon's own clock starts before it creates
//! its verifier, so on small files its time is mostly class loading, the
//! JIT's first passes and Z3's start, about two seconds that do not depend on
//! the file. Cold runs are therefore not measured at all. One JVM runs every file
//! through Silicon's own command-line path (`SilFrontend.execute`, a fresh
//! verifier and Z3 per file, as a cold run), after it has been warmed up on
//! files of a separate corpus: Silicon's own test files, minus any file whose
//! content is also a benchmark (by SHA-256), so no benchmark file has been
//! seen by the JIT before it is timed. The time is the one Silicon reports,
//! read from the same summary line as a cold run.
//!
//! The driver is `bench/silicon/SiliconWarm.java`, run from source
//! (`java -cp silicon.jar SiliconWarm.java`), so it needs no build step and
//! always matches the jar. A timeout or a crash ends the JVM; the next file
//! starts a new one and warms it up again.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::measure::{self, Status, Timing, Tree};
use crate::silicon::{self, Silicon, SiliconResult};

/// The driver's source, written next to the scratch files when it starts.
pub const DRIVER: &str = include_str!("../silicon/SiliconWarm.java");

/// How to warm Silicon up.
#[derive(Debug, Clone)]
pub struct WarmOptions {
    /// Directories searched (recursively) for warm-up `.vpr` files.
    pub corpus: Vec<PathBuf>,
    /// Seconds spent warming up each new JVM.
    pub warmup_s: f64,
    /// Silicon's own `--timeout` for each warm-up file, so one slow file
    /// cannot take the whole budget.
    pub file_timeout_s: u64,
}

/// What the run file records about the warm-up.
#[derive(Debug, Clone, Serialize)]
pub struct WarmInfo {
    /// Part of the Silicon cache key: the driver, the warm-up files and the
    /// budget.
    pub id: String,
    pub warmup_files: usize,
    /// Warm-up files left out because a benchmark has the same content.
    pub excluded: usize,
    pub warmup_s: f64,
    pub file_timeout_s: u64,
    /// JVMs started in this run, and the warm-up files each one verified.
    pub jvms: usize,
    pub warmup_runs: Vec<usize>,
}

struct Driver {
    tree: Tree,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

/// One answer from the driver.
enum Reply {
    /// The output before the `@@END` line, and that line's fields.
    Done(String, Vec<String>),
    TimedOut,
    /// The JVM exited (a crash, an out-of-memory error), with what it printed.
    Died(String),
}

impl Driver {
    fn send(&mut self, line: &str) -> std::io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn reply(&mut self, timeout: Duration) -> Reply {
        let deadline = Instant::now() + timeout;
        let mut out = String::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => match line.strip_prefix("@@END ") {
                    Some(end) => {
                        return Reply::Done(out, end.split(' ').map(String::from).collect());
                    }
                    None => {
                        out.push_str(&line);
                        out.push('\n');
                    }
                },
                Err(RecvTimeoutError::Timeout) => return Reply::TimedOut,
                Err(RecvTimeoutError::Disconnected) => return Reply::Died(out),
            }
        }
    }
}

pub struct WarmSilicon {
    silicon: Silicon,
    opts: WarmOptions,
    files: Vec<PathBuf>,
    pub info: WarmInfo,
    scratch: PathBuf,
    driver: Option<Driver>,
}

/// Every `.vpr` under `dirs`, in path order.
fn vpr_files(dirs: &[PathBuf]) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "vpr") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    for d in dirs {
        walk(d, &mut out);
    }
    out.sort();
    out
}

impl WarmSilicon {
    /// Collect the warm-up files, leaving out any with the same content as
    /// one of `benchmarks` (the `.vpr` files of every suite, whether measured
    /// in this run or not, so the warm-up set does not change with
    /// `--suite`/`--only`).
    pub fn new(
        silicon: Silicon,
        opts: WarmOptions,
        benchmarks: &[PathBuf],
        scratch: &Path,
    ) -> Result<WarmSilicon, String> {
        let candidates = vpr_files(&opts.corpus);
        if candidates.is_empty() {
            return Err(format!(
                "no .vpr files under the Silicon warm-up corpus {:?}",
                opts.corpus
            ));
        }
        // Only files of equal size can have equal content: hash just those.
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).ok();
        let sizes: BTreeSet<u64> = candidates.iter().filter_map(|p| size(p)).collect();
        let bench_shas: BTreeSet<String> = benchmarks
            .iter()
            .filter(|p| size(p).is_some_and(|s| sizes.contains(&s)))
            .filter_map(|p| crate::sha256_file(p).ok())
            .collect();
        let mut files = Vec::new();
        let mut shas = Vec::new();
        let mut excluded = 0;
        for p in candidates {
            let sha = crate::sha256_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if bench_shas.contains(&sha) {
                excluded += 1;
            } else {
                files.push(p);
                shas.push(sha);
            }
        }
        if files.is_empty() {
            return Err("every Silicon warm-up file is also a benchmark".into());
        }
        let id_input = format!(
            "{}\0{}\0{}\0{}",
            crate::sha256_bytes(DRIVER.as_bytes()),
            shas.join(","),
            opts.warmup_s,
            opts.file_timeout_s
        );
        let info = WarmInfo {
            id: crate::sha256_bytes(id_input.as_bytes())[..16].to_string(),
            warmup_files: files.len(),
            excluded,
            warmup_s: opts.warmup_s,
            file_timeout_s: opts.file_timeout_s,
            jvms: 0,
            warmup_runs: Vec::new(),
        };
        Ok(WarmSilicon {
            silicon,
            opts,
            files,
            info,
            scratch: scratch.join("silicon-warm"),
            driver: None,
        })
    }

    /// The Silicon cache key of a warm result for `vpr_sha256`.
    pub fn cache_key(&self, vpr_sha256: &str) -> String {
        format!(
            "{}|warm:{}",
            silicon::Cache::key(
                vpr_sha256,
                &self.silicon.jar_sha256,
                &self.silicon.config_id()
            ),
            self.info.id
        )
    }

    /// A running, warmed-up driver.
    fn driver(&mut self) -> Result<&mut Driver, String> {
        if self.driver.is_none() {
            self.driver = Some(self.start()?);
        }
        Ok(self.driver.as_mut().unwrap())
    }

    fn start(&mut self) -> Result<Driver, String> {
        let io = |e: std::io::Error| format!("silicon warm driver: {e}");
        std::fs::create_dir_all(&self.scratch).map_err(io)?;
        let source = self.scratch.join("SiliconWarm.java");
        std::fs::write(&source, DRIVER).map_err(io)?;
        let list = self.scratch.join("warmup.txt");
        let listed: String = self
            .files
            .iter()
            .map(|p| format!("{}\n", p.display()))
            .collect();
        std::fs::write(&list, listed).map_err(io)?;
        let stderr = std::fs::File::create(self.scratch.join("stderr.txt")).map_err(io)?;

        let mut cmd = Command::new(&self.silicon.java);
        cmd.args(&self.silicon.jvm_args)
            .arg("-cp")
            .arg(&self.silicon.jar)
            .arg(&source)
            .args(&self.silicon.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr);
        let mut tree = Tree::spawn(&mut cmd).map_err(io)?;
        let stdin = tree.child().stdin.take().expect("piped");
        let stdout = tree.child().stdout.take().expect("piped");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut d = Driver { tree, stdin, lines };
        let fail = |what: &str, r: Reply| match r {
            Reply::TimedOut => format!("silicon warm driver: no answer to {what}"),
            Reply::Died(out) | Reply::Done(out, _) => format!(
                "silicon warm driver: {what} failed: {}",
                measure::tail(&out, 400)
            ),
        };
        // Compiling the driver and loading Silicon take a few seconds.
        match d.reply(Duration::from_secs(180)) {
            Reply::Done(_, end) if end.first().is_some_and(|e| e == "READY") => {}
            r => return Err(fail("start", r)),
        }
        crate::run::progress(format!(
            "  silicon warm: warming up a new JVM for {}s on {} files",
            self.opts.warmup_s,
            self.files.len()
        ));
        d.send(&format!(
            "WARM\t{}\t{}\t{}",
            self.opts.warmup_s,
            self.opts.file_timeout_s,
            list.display()
        ))
        .map_err(io)?;
        // The budget is checked between files: allow one file past it, and
        // Silicon's own timeout is approximate.
        let limit = self.opts.warmup_s + 4.0 * self.opts.file_timeout_s as f64 + 120.0;
        match d.reply(Duration::from_secs_f64(limit)) {
            Reply::Done(_, end) if end.first().is_some_and(|e| e == "WARM") => {
                let runs = end.get(1).and_then(|n| n.parse().ok()).unwrap_or(0);
                self.info.jvms += 1;
                self.info.warmup_runs.push(runs);
            }
            r => return Err(fail("warm-up", r)),
        }
        Ok(d)
    }

    /// Verify `vpr` `runs` times in the warm JVM, stopping at the first run
    /// that does not finish. No per-file warm-up runs: the JVM is warm from
    /// the corpus, and repeating the file first would let the JIT specialise
    /// on it, which a cold run cannot.
    pub fn measure(
        &mut self,
        vpr: &Path,
        vpr_sha256: &str,
        runs: usize,
        timeout: Duration,
    ) -> Result<SiliconResult, String> {
        let source = std::fs::read_to_string(vpr).map_err(|e| e.to_string())?;
        let decls = silicon::declaration_lines(&source);
        let mut walls = Vec::new();
        let mut parsed = Vec::new();
        let mut status = Status::Ok;
        let mut message = None;
        for _ in 0..runs {
            let d = self.driver()?;
            d.send(&format!("RUN\t{}", vpr.display()))
                .map_err(|e| format!("silicon warm driver: {e}"))?;
            match d.reply(timeout) {
                Reply::Done(out, end) => {
                    let p = silicon::parse_output(&out, &decls);
                    if self.silicon.version.is_none() {
                        self.silicon.version = p.version.clone();
                    }
                    let driver_ok = end.get(1).is_some_and(|s| s == "ok");
                    if !driver_ok || p.verify_time.is_none() {
                        status = Status::Error;
                        message = Some(measure::tail(&out, 400));
                        break;
                    }
                    walls.push(end.get(2).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN));
                    parsed.push(p);
                }
                Reply::TimedOut => {
                    status = Status::Timeout;
                    walls.push(timeout.as_secs_f64());
                    self.stop();
                    break;
                }
                Reply::Died(out) => {
                    status = Status::Error;
                    message = Some(format!("the JVM exited: {}", measure::tail(&out, 400)));
                    self.stop();
                    break;
                }
            }
        }
        let verify_runs: Vec<f64> = parsed.iter().filter_map(|p| p.verify_time).collect();
        let wall = Timing::from_runs(status, walls, &[], message.clone());
        let verify = Timing::from_runs(status, verify_runs, &[], None);
        let last = parsed.last();
        let errors = last.map(|p| p.errors.clone()).unwrap_or_default();
        Ok(SiliconResult {
            silicon: self.silicon.id(),
            vpr_sha256: vpr_sha256.to_string(),
            status,
            verified: if status == Status::Ok {
                last.and_then(|p| p.verified)
            } else {
                None
            },
            wall: (&wall).into(),
            verify: (&verify).into(),
            peak_rss_mb: None,
            failed_members: errors.iter().filter_map(|e| e.member.clone()).collect(),
            errors,
            message,
            timeout_s: Some(timeout.as_secs_f64()),
        })
    }

    /// `"<version line>@sha256:<jar hash>"`, once a run has printed the
    /// version (or a cached result has carried it).
    pub fn silicon_id(&self) -> String {
        self.silicon.id()
    }

    /// Take the version line from a cached result's `silicon` id, when no run
    /// of this JVM has printed it yet.
    pub fn adopt_version(&mut self, id: &str) {
        if self.silicon.version.is_none() {
            self.silicon.version = id
                .rsplit_once("@sha256:")
                .map(|(v, _)| v.to_string())
                .filter(|v| v != "silicon");
        }
    }

    /// End the JVM (and its z3s).
    pub fn stop(&mut self) {
        if let Some(mut d) = self.driver.take() {
            let _ = d.send("QUIT");
            d.tree.kill();
        }
    }
}

impl Drop for WarmSilicon {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silicon() -> Silicon {
        Silicon {
            java: "java".into(),
            jar: "s.jar".into(),
            jar_sha256: "j".into(),
            jvm_args: vec![],
            args: vec![],
            version: None,
        }
    }

    /// A warm-up file with a benchmark's content is left out, wherever it
    /// lives; the cache key follows the warm-up set and the budget.
    #[test]
    fn warmup_corpus_leaves_out_benchmarks() {
        let dir = std::env::temp_dir().join(format!("bench-warm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = dir.join("corpus");
        std::fs::create_dir_all(corpus.join("sub")).unwrap();
        std::fs::write(corpus.join("a.vpr"), "method a() {}").unwrap();
        std::fs::write(corpus.join("sub/b.vpr"), "method b() {}").unwrap();
        std::fs::write(corpus.join("notes.txt"), "method b() {}").unwrap();
        let bench = dir.join("bench.vpr");
        std::fs::write(&bench, "method b() {}").unwrap();
        let opts = |warmup_s| WarmOptions {
            corpus: vec![corpus.clone()],
            warmup_s,
            file_timeout_s: 10,
        };

        let w = WarmSilicon::new(silicon(), opts(60.0), &[bench.clone()], &dir).unwrap();
        assert_eq!(w.files, [corpus.join("a.vpr")]);
        assert_eq!((w.info.warmup_files, w.info.excluded), (1, 1));
        assert!(w.cache_key("v").ends_with(&format!("|warm:{}", w.info.id)));

        let all = WarmSilicon::new(silicon(), opts(60.0), &[], &dir).unwrap();
        assert_eq!(all.info.warmup_files, 2);
        assert_ne!(all.info.id, w.info.id);
        let longer = WarmSilicon::new(silicon(), opts(120.0), &[bench.clone()], &dir).unwrap();
        assert_ne!(longer.info.id, w.info.id);

        std::fs::write(corpus.join("a.vpr"), "method b() {}").unwrap();
        assert!(WarmSilicon::new(silicon(), opts(60.0), &[bench], &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
