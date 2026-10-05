//! `bench run`: measure every file of every selected suite and write one run
//! JSON (schema 1, see `plans/regression-pipeline.md`).
//!
//! Per file, in order: fingerprints, Rust metrics (`syn`), Viper metrics
//! (`verify --viper-metrics`), rustc (`--emit=metadata`),
//! Helium (`verify --json`), Silicon (or its cache), then the per-member join:
//! Viper method `m_f` ↔ Rust function `f`, with Helium's and Silicon's verdicts
//! side by side.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::helium::{self, HeliumRun};
use crate::measure::{self, Sample, Status, Timing};
use crate::rust_metrics::{self, FileMetrics, FnMetrics};
use crate::silicon::{self, Silicon, SiliconResult};
use crate::suites::{self, Suite, SuiteFile};

pub const SCHEMA: u32 = 1;

/// Everything `bench run` needs to know.
#[derive(Debug, Clone)]
pub struct Options {
    pub benchmarks: PathBuf,
    /// The `verify` binary being measured.
    pub verify: PathBuf,
    /// The `verify` used for `--viper-metrics` (defaults to `verify`; differs
    /// when backfilling with a build older than the flag).
    pub metrics_verify: PathBuf,
    /// The repository whose commit is being measured.
    pub repo: PathBuf,
    pub commit: Option<String>,
    pub host: String,
    pub suites: Vec<String>,
    pub only: Vec<String>,
    pub warmup: usize,
    pub runs: usize,
    pub timeout: Option<f64>,
    pub rustc: Option<RustcOptions>,
    /// rustc timings reused across runs (read and updated).
    pub rustc_cache: Option<PathBuf>,
    pub silicon: Option<Silicon>,
    pub silicon_cache: Option<PathBuf>,
    pub scratch: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RustcOptions {
    pub rustc: PathBuf,
    /// `+toolchain` for a rustup proxy, if pinned.
    pub toolchain: Option<String>,
}

impl RustcOptions {
    pub fn command(&self) -> Command {
        let mut c = Command::new(&self.rustc);
        if let Some(tc) = &self.toolchain {
            c.arg(format!("+{tc}"));
        }
        c
    }

    /// Bypass the rustup proxy: time the toolchain's own `rustc`, found from
    /// its sysroot. The proxy adds tens of milliseconds and its own process to
    /// every run, which is noise next to a 100 ms `rustc_check`. The toolchain
    /// is resolved once, from the current directory (so `rust-toolchain.toml`
    /// applies) unless `+toolchain` pins it.
    pub fn resolve(&self) -> RustcOptions {
        let direct = self
            .command()
            .args(["--print", "sysroot"])
            .output()
            .ok()
            .and_then(|o| {
                let sysroot = PathBuf::from(String::from_utf8_lossy(&o.stdout).trim());
                let exe =
                    sysroot
                        .join("bin")
                        .join(if cfg!(windows) { "rustc.exe" } else { "rustc" });
                (o.status.success() && exe.is_file()).then_some(exe)
            });
        match direct {
            Some(rustc) => RustcOptions {
                rustc,
                toolchain: None,
            },
            None => self.clone(),
        }
    }

    pub fn version(&self) -> Option<String> {
        let out = self.command().arg("-vV").output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines().next().map(|l| l.trim().to_string())
    }
}

/// Whether this compiler takes `-Z` flags: nightly and locally built ones do.
pub fn rustc_is_nightly(version: &str) -> bool {
    version.contains("-nightly") || version.contains("-dev")
}

/// One file's rustc measurement: the cached unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustcTiming {
    /// The `rustc` process, end to end (`rustc_check`).
    #[serde(flatten)]
    pub check: Timing,
    /// rustc's own `-Z time-passes` total and passes, from the same runs
    /// (`rustc_self`); `None` for a compiler that is not nightly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub own: Option<PhasedTiming>,
}

/// rustc timings. Like Silicon's, they depend only on the `.rs`, the
/// compiler and its arguments, never on the Helium commit, so each is measured
/// once and reused. Keyed by `"<rs sha256>|<rustc -vV>|<args>"`; the version
/// line carries the compiler's commit hash, and the arguments include
/// `-Z time-passes` when it is passed.
pub type RustcCache = crate::cache::Cache<RustcTiming>;

impl RustcCache {
    pub fn key(rs_sha256: &str, rustc_version: &str, args: &[String]) -> String {
        format!("{rs_sha256}|{rustc_version}|{}", args.join(" "))
    }
}

// ── Output ──

#[derive(Debug, Serialize)]
pub struct RunFile {
    pub schema: u32,
    pub commit: String,
    pub dirty: bool,
    pub date: String,
    pub commit_date: Option<String>,
    pub subject: Option<String>,
    pub host: String,
    pub tools: Tools,
    pub settings: Settings,
    pub suites: BTreeMap<String, SuiteInfo>,
    pub files: Vec<FileResult>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Tools {
    pub rustc: Option<String>,
    pub rustc_toolchain: Option<String>,
    pub silicon: Option<String>,
    /// The commit `verify` reports it was built from (JSON builds only).
    pub verify_commit: Option<String>,
    pub verify_json: bool,
}

#[derive(Debug, Serialize)]
pub struct Settings {
    pub warmup: usize,
    pub runs: usize,
    pub timeout_s: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SuiteInfo {
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub families: Vec<FamilyInfo>,
    pub rustc_args: Vec<String>,
    pub silicon: bool,
    pub files: usize,
}

#[derive(Debug, Serialize)]
pub struct FamilyInfo {
    pub name: String,
    /// The stem regex, so the site can place files of older runs (measured
    /// before the family was declared) in the family too.
    pub pattern: String,
    pub knobs: Vec<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct Fingerprints {
    pub rs: Option<String>,
    pub vpr: String,
}

#[derive(Debug, Serialize)]
pub struct FileResult {
    pub suite: String,
    pub stem: String,
    pub family: Option<String>,
    pub knobs: Option<BTreeMap<String, f64>>,
    pub sha256: Fingerprints,
    pub rust_metrics: Option<FileRustMetrics>,
    pub viper_metrics: Option<Value>,
    /// Viper lines per Rust line: how much Prusti's encoding blows up.
    pub blowup: Option<f64>,
    pub times: Times,
    pub peak_rss_mb: BTreeMap<String, f64>,
    /// A file-level Helium error (parse failure): no members.
    pub helium_error: Option<String>,
    pub coverage: BTreeMap<String, u64>,
    pub stats: Option<Value>,
    pub silicon: Option<SiliconSummary>,
    pub members: Vec<Member>,
}

/// File-level Rust metrics; per-function metrics live on the members.
#[derive(Debug, Serialize)]
pub struct FileRustMetrics {
    pub loc: u64,
    pub fns: u64,
    pub structs: u64,
    pub enums: u64,
    pub totals: serde_json::Map<String, Value>,
}

#[derive(Debug, Serialize, Default)]
pub struct Times {
    /// The `rustc` process, end to end.
    pub rustc_check: Option<Timing>,
    /// rustc's own `-Z time-passes` total (no process startup), with the
    /// median of each pass. Passes nest, so they do not add up to the total.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc_self: Option<PhasedTiming>,
    /// `verify`'s own pipeline total (no process startup), with phase medians.
    pub helium_verify: Option<PhasedTiming>,
    /// The `verify` process, end to end.
    pub helium_wall: Option<Timing>,
    pub silicon_wall: Option<SiliconTiming>,
    pub silicon_verify: Option<SiliconTiming>,
}

/// A tool's self-reported total, with the median of each phase it reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasedTiming {
    #[serde(flatten)]
    pub timing: Timing,
    pub phases: BTreeMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct SiliconTiming {
    pub status: Status,
    pub median: Option<f64>,
    pub mad: Option<f64>,
    pub runs: Vec<f64>,
    pub cached: bool,
}

#[derive(Debug, Serialize)]
pub struct SiliconSummary {
    pub verified: Option<bool>,
    pub status: Status,
    pub errors: Vec<silicon::SiliconError>,
    pub cached: bool,
    pub message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Member {
    pub name: String,
    pub rust_fn: Option<String>,
    /// How `rust_fn` was found: `exact` (`m_f` ↔ `f`) or `fuzzy`.
    #[serde(rename = "match")]
    pub match_kind: Option<&'static str>,
    pub helium: String,
    pub silicon: Option<&'static str>,
    pub expected_failure: bool,
    /// `incompleteness` (Helium FAIL, Silicon OK) or `soundness` (Helium OK,
    /// Silicon FAIL: needs a look).
    pub disagreement: Option<&'static str>,
    pub time: Option<f64>,
    pub rust_metrics: Option<FnMetrics>,
    pub viper_metrics: Option<Value>,
}

// ── Joining Viper members to Rust functions ──

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The Rust function a Viper member encodes. Prusti names the encoding of `f`
/// `m_f`; that exact rule is tried first (on the plain name, unique among the
/// file's functions). Failing that — `impl` methods and module paths — the
/// qualified name is compared with separators removed, then as a unique
/// suffix. Those are reported as `fuzzy` so the caller can log them.
pub fn rust_fn_for<'a>(
    member: &str,
    fns: &'a BTreeMap<String, FnMetrics>,
) -> Option<(&'a String, &'static str)> {
    let base = member.strip_prefix("m_")?;
    let base = base.split('#').next().unwrap_or(base);
    let exact: Vec<&String> = fns
        .keys()
        .filter(|q| q.rsplit("::").next() == Some(base))
        .collect();
    if exact.len() == 1 {
        return Some((exact[0], "exact"));
    }
    let nb = normalize(base);
    if let Some(q) = fns.keys().find(|q| normalize(q) == nb) {
        return Some((q, "fuzzy"));
    }
    let mut suffix: Vec<&String> = fns
        .keys()
        .filter(|q| {
            let plain = normalize(q.rsplit("::").next().unwrap_or(q));
            !plain.is_empty() && nb.ends_with(&plain)
        })
        .collect();
    suffix.sort_by_key(|q| std::cmp::Reverse(q.len()));
    match suffix.as_slice() {
        [one] => Some((one, "fuzzy")),
        [a, b, ..] if normalize(a).len() > normalize(b).len() => Some((a, "fuzzy")),
        _ => None,
    }
}

// ── Measuring ──

fn progress(msg: impl AsRef<str>) {
    eprintln!("[bench] {}", msg.as_ref());
}

fn fmt_time(t: Option<f64>) -> String {
    t.map_or("-".into(), |x| format!("{x:.3}s"))
}

struct Ctx<'a> {
    opts: &'a Options,
    /// `opts.rustc`, resolved past the rustup proxy.
    rustc: Option<RustcOptions>,
    /// Its `rustc -vV` line, part of the rustc cache key.
    rustc_version: Option<String>,
    rustc_cache: Option<RustcCache>,
    silicon_cache: Option<silicon::Cache>,
    silicon: Option<Silicon>,
    verify_commit: Option<String>,
    verify_json: bool,
    warnings: Vec<String>,
}

pub fn run(opts: &Options) -> Result<RunFile, String> {
    let mut suites_found = suites::discover(&opts.benchmarks);
    if suites_found.is_empty() {
        return Err(format!("no suites under {}", opts.benchmarks.display()));
    }
    if !opts.suites.is_empty() {
        let known: BTreeSet<&str> = suites_found.iter().map(|s| s.name.as_str()).collect();
        for s in &opts.suites {
            if !known.contains(s.as_str()) {
                return Err(format!(
                    "unknown suite `{s}` (have: {})",
                    known.into_iter().collect::<Vec<_>>().join(", ")
                ));
            }
        }
        suites_found.retain(|s| opts.suites.contains(&s.name));
    }
    let errors: Vec<&String> = suites_found.iter().flat_map(|s| &s.errors).collect();
    if !errors.is_empty() {
        return Err(format!(
            "suite errors (run `bench check-suites`):\n  {}",
            errors
                .iter()
                .map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        ));
    }

    let load_err = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    let silicon_cache = match &opts.silicon_cache {
        Some(p) if opts.silicon.is_some() => {
            Some(silicon::Cache::load(p).map_err(|e| load_err(p, e))?)
        }
        _ => None,
    };
    let rustc_cache = match &opts.rustc_cache {
        Some(p) if opts.rustc.is_some() => Some(RustcCache::load(p).map_err(|e| load_err(p, e))?),
        _ => None,
    };
    let rustc = opts.rustc.as_ref().map(RustcOptions::resolve);
    let mut ctx = Ctx {
        opts,
        rustc_version: rustc.as_ref().and_then(RustcOptions::version),
        rustc,
        rustc_cache,
        silicon_cache,
        silicon: opts.silicon.clone(),
        verify_commit: None,
        verify_json: true,
        warnings: suites_found
            .iter()
            .flat_map(|s| s.warnings.clone())
            .collect(),
    };
    for w in &ctx.warnings {
        progress(format!("warning: {w}"));
    }

    // The commit is fixed before measuring: a commit made while a long run is
    // in progress must not be credited with it.
    let repo = &opts.repo;
    let commit = opts
        .commit
        .clone()
        .or_else(|| crate::git(repo, &["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    let dirty = opts.commit.is_none()
        && crate::git(repo, &["status", "--porcelain", "--untracked-files=no"])
            .is_some_and(|s| !s.is_empty());
    let commit_date = crate::git(repo, &["show", "-s", "--format=%cI", &commit]);
    let subject = crate::git(repo, &["show", "-s", "--format=%s", &commit]);

    let mut files = Vec::new();
    for suite in &suites_found {
        for file in suite.measurable() {
            let key = format!("{}/{}", suite.name, file.stem);
            if !opts.only.is_empty() && !opts.only.iter().any(|o| o == &key || o == &suite.name) {
                continue;
            }
            let result = measure_file(&mut ctx, suite, file).map_err(|e| format!("{key}: {e}"))?;
            files.push(result);
            // Save the caches as we go: a crash an hour in keeps what was
            // measured.
            if let (Some(cache), Some(path)) = (&ctx.silicon_cache, &opts.silicon_cache) {
                cache.save(path).map_err(|e| load_err(path, e))?;
            }
            if let (Some(cache), Some(path)) = (&ctx.rustc_cache, &opts.rustc_cache) {
                cache.save(path).map_err(|e| load_err(path, e))?;
            }
        }
    }

    Ok(RunFile {
        schema: SCHEMA,
        commit,
        dirty,
        date: crate::utc_now(),
        commit_date,
        subject,
        host: opts.host.clone(),
        tools: Tools {
            rustc: ctx.rustc_version.clone(),
            rustc_toolchain: opts.rustc.as_ref().and_then(|r| r.toolchain.clone()),
            silicon: ctx.silicon.as_ref().map(Silicon::id),
            verify_commit: ctx.verify_commit,
            verify_json: ctx.verify_json,
        },
        settings: Settings {
            warmup: opts.warmup,
            runs: opts.runs,
            timeout_s: opts.timeout,
        },
        suites: suites_found
            .iter()
            .map(|s| {
                (
                    s.name.clone(),
                    SuiteInfo {
                        description: s.config.description.clone(),
                        tags: s.config.tags.clone(),
                        families: s
                            .families
                            .iter()
                            .map(|f| FamilyInfo {
                                name: f.name.clone(),
                                pattern: f.regex.as_str().to_string(),
                                knobs: f.knobs.clone(),
                            })
                            .collect(),
                        rustc_args: s.rustc_args(),
                        silicon: s.silicon(),
                        files: s.measurable().count(),
                    },
                )
            })
            .collect(),
        files,
        warnings: ctx.warnings,
    })
}

fn measure_file(ctx: &mut Ctx, suite: &Suite, file: &SuiteFile) -> Result<FileResult, String> {
    let opts = ctx.opts;
    let vpr = file.vpr.as_ref().expect("measurable");
    let timeout = Duration::from_secs_f64(opts.timeout.unwrap_or_else(|| suite.timeout_s()));
    let scratch = opts.scratch.join("work");
    let io = |e: std::io::Error| e.to_string();
    progress(format!("{}/{}", suite.name, file.stem));

    let vpr_sha = crate::sha256_file(vpr).map_err(io)?;
    let rs_source = file
        .rs
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()
        .map_err(io)?;
    let rs_sha = rs_source
        .as_ref()
        .map(|s| crate::sha256_bytes(s.as_bytes()));

    // ── Rust metrics ──
    let (rust, rust_error) = match &rs_source {
        Some(src) => match rust_metrics::file_metrics(src) {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e)),
        },
        None => (None, None),
    };
    if let Some(e) = rust_error {
        ctx.warnings
            .push(format!("{}/{}: Rust metrics: {e}", suite.name, file.stem));
    }

    // ── Viper metrics ──
    let viper_metrics = viper_metrics(&opts.metrics_verify, vpr, &scratch);
    let mut viper_members: BTreeMap<String, Value> = BTreeMap::new();
    let viper_file = viper_metrics.map(|mut v| {
        if let Some(Value::Array(members)) = v.as_object_mut().and_then(|o| o.remove("members")) {
            for m in members {
                if let (Some(name), Some(counts)) = (m["name"].as_str(), m.get("counts")) {
                    viper_members.insert(name.to_string(), counts.clone());
                }
            }
        }
        v
    });
    let blowup = match (&viper_file, &rust) {
        (Some(v), Some(r)) if r.loc > 0 => v["loc"].as_f64().map(|l| l / r.loc as f64),
        _ => None,
    };

    let mut times = Times::default();
    let mut peaks = BTreeMap::new();

    // ── rustc ──
    if let (Some(rustc), Some(rs)) = (ctx.rustc.clone(), &file.rs) {
        let mut args = suite.rustc_args();
        // A nightly rustc also reports its own time, from the same runs.
        let time_passes = ctx.rustc_version.as_deref().is_some_and(rustc_is_nightly);
        if time_passes {
            args.extend(["-Z".to_string(), "time-passes".to_string()]);
        }
        // Only successful timings are cached; a failure is measured again, and
        // so is a timing of fewer runs than this run asks for.
        let key = rs_sha
            .as_deref()
            .zip(ctx.rustc_version.as_deref())
            .map(|(sha, version)| RustcCache::key(sha, version, &args));
        let hit = key
            .as_ref()
            .and_then(|k| ctx.rustc_cache.as_ref()?.entries.get(k).cloned())
            .filter(|t| t.check.status == Status::Ok && t.check.runs.len() >= opts.runs);
        let RustcTiming { check: t, own } = match hit {
            Some(t) => RustcTiming {
                check: Timing {
                    cached: true,
                    ..t.check
                },
                own: t.own.map(|o| PhasedTiming {
                    timing: Timing {
                        cached: true,
                        ..o.timing
                    },
                    ..o
                }),
            },
            None => {
                let t = time_rustc(&rustc, &args, rs, opts, timeout, &scratch).map_err(io)?;
                if let (Some(k), Some(cache)) = (key, ctx.rustc_cache.as_mut()) {
                    if t.check.status == Status::Ok {
                        cache.entries.insert(k, t.clone());
                    }
                }
                t
            }
        };
        if let Some(p) = t.peak_rss_mb {
            peaks.insert("rustc_check".to_string(), p);
        }
        if t.status != Status::Ok {
            ctx.warnings.push(format!(
                "{}/{}: rustc_check: {:?}{}",
                suite.name,
                file.stem,
                t.status,
                t.message
                    .as_deref()
                    .map(|m| format!(": {m}"))
                    .unwrap_or_default()
            ));
        }
        times.rustc_check = Some(t);
        times.rustc_self = own;
    }

    // ── Helium ──
    let mut text_mode = false;
    let parsed_ok = |s: &Sample| helium::parse(&s.stdout, &s.stderr).is_some();
    let mut samples = Vec::new();
    for i in 0..opts.warmup + opts.runs {
        let mut c = Command::new(&opts.verify);
        c.arg("--json");
        if text_mode {
            c.arg("--breakdown");
        }
        c.arg(vpr);
        let s = measure::run_once(&mut c, timeout, &scratch).map_err(io)?;
        let run = helium::parse(&s.stdout, &s.stderr);
        if i == 0 && run.as_ref().is_some_and(|r| !r.json) && !text_mode {
            // An old build: rerun the warm-up with --breakdown for member times.
            text_mode = true;
            ctx.verify_json = false;
            let mut c = Command::new(&opts.verify);
            c.args(["--json", "--breakdown"]).arg(vpr);
            let s = measure::run_once(&mut c, timeout, &scratch).map_err(io)?;
            if opts.warmup == 0 {
                samples.push(s);
            }
            continue;
        }
        let stop = s.timed_out || run.is_none();
        if i >= opts.warmup || stop {
            samples.push(s);
        }
        if stop {
            break;
        }
    }
    let wall = measure::wall_timing(&samples, parsed_ok);
    let runs: Vec<HeliumRun> = samples
        .iter()
        .filter_map(|s| helium::parse(&s.stdout, &s.stderr))
        .collect();
    if let Some(p) = wall.peak_rss_mb {
        peaks.insert("helium".into(), p);
    }
    if ctx.verify_commit.is_none() && ctx.verify_json {
        ctx.verify_commit = samples.iter().find_map(|s| {
            serde_json::from_str::<Value>(s.stdout.lines().next()?).ok()?["commit"]
                .as_str()
                .map(String::from)
        });
    }
    let totals: Vec<f64> = runs.iter().filter_map(|r| r.total).collect();
    let mut phases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut member_times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &runs {
        for (k, v) in &r.phases {
            phases.entry(k.clone()).or_default().push(*v);
        }
        for (k, v) in &r.member_times {
            member_times.entry(k.clone()).or_default().push(*v);
        }
    }
    times.helium_verify = Some(PhasedTiming {
        timing: Timing::from_runs(wall.status, totals, &[], None),
        phases: phases
            .iter()
            .map(|(k, v)| (k.clone(), measure::median(v)))
            .collect(),
    });
    let last = runs.last().cloned().unwrap_or_default();
    if wall.status != Status::Ok {
        ctx.warnings.push(format!(
            "{}/{}: helium: {:?}{}",
            suite.name,
            file.stem,
            wall.status,
            wall.message
                .as_deref()
                .map(|m| format!(": {m}"))
                .unwrap_or_default()
        ));
    }
    times.helium_wall = Some(wall);

    // ── Silicon ──
    let mut silicon_result: Option<(SiliconResult, bool)> = None;
    if suite.silicon() {
        if let Some(sil) = ctx.silicon.as_mut() {
            let key = silicon::Cache::key(&vpr_sha, &sil.jar_sha256, &sil.config_id());
            let hit = ctx
                .silicon_cache
                .as_ref()
                .and_then(|c| c.entries.get(&key))
                .filter(|hit| silicon::Cache::reusable(hit, opts.runs, timeout));
            match hit {
                Some(hit) => {
                    if sil.version.is_none() {
                        sil.version = hit
                            .silicon
                            .rsplit_once("@sha256:")
                            .map(|(v, _)| v.to_string());
                    }
                    silicon_result = Some((hit.clone(), true));
                }
                None => {
                    let r = silicon::measure(
                        sil,
                        vpr,
                        &vpr_sha,
                        opts.warmup,
                        opts.runs,
                        timeout,
                        &scratch,
                    )
                    .map_err(|e| format!("silicon: {e}"))?;
                    if let Some(cache) = ctx.silicon_cache.as_mut() {
                        if silicon::Cache::keeps(&r) {
                            cache.entries.insert(key, r.clone());
                        } else {
                            // Not a stale answer for the next run either.
                            cache.entries.remove(&key);
                        }
                    }
                    silicon_result = Some((r, false));
                }
            }
        }
    }
    if let Some((r, cached)) = &silicon_result {
        let t = |d: &silicon::TimingData| SiliconTiming {
            status: r.status,
            median: d.median,
            mad: d.mad,
            runs: d.runs.clone(),
            cached: *cached,
        };
        times.silicon_wall = Some(t(&r.wall));
        times.silicon_verify = Some(t(&r.verify));
        if let Some(p) = r.peak_rss_mb {
            peaks.insert("silicon".into(), p);
        }
    }

    // ── Members ──
    let fns = rust.as_ref().map(|r| &r.functions);
    let mut coverage: BTreeMap<String, u64> = BTreeMap::new();
    let row_names: BTreeSet<&str> = last.results.iter().map(|(n, _)| n.as_str()).collect();
    let members = last
        .results
        .iter()
        .map(|(name, status)| {
            *coverage.entry(status.clone()).or_default() += 1;
            let base = name.split('#').next().unwrap_or(name);
            // `m_f#requires` / `m_f#ensures` check `m_f`'s contract. The shape
            // metrics go on the member that stands for the declaration itself
            // (`m_f`, or the `#` row when there is no plain one), so each Rust
            // function is one data point, not three.
            let primary = base == name || !row_names.contains(base);
            let joined = if primary {
                fns.and_then(|f| rust_fn_for(name, f))
            } else {
                None
            };
            if let Some((q, "fuzzy")) = joined {
                progress(format!(
                    "  join: fuzzy match {}/{}: {name} -> {q}",
                    suite.name, file.stem
                ));
            }
            let sil = silicon_result.as_ref().and_then(|(r, _)| {
                r.verified?;
                if r.failed_members.contains(base) {
                    // Silicon reports per declaration: a failure in `m_f` does
                    // not say whether `m_f#requires` / `#ensures` (its contract
                    // checks) fail, so those rows stay unknown.
                    (base == name).then_some("FAIL")
                } else if silicon_unattributed(r) {
                    // A rejection we could not place: no member is known OK.
                    None
                } else {
                    Some("OK")
                }
            });
            let disagreement = match (status.as_str(), sil) {
                ("FAIL", Some("OK")) => Some("incompleteness"),
                ("OK", Some("FAIL")) => Some("soundness"),
                _ => None,
            };
            Member {
                name: name.clone(),
                rust_fn: joined.map(|(q, _)| q.clone()),
                match_kind: joined.map(|(_, k)| k),
                helium: status.clone(),
                silicon: sil,
                expected_failure: suite
                    .expected_failures
                    .contains(&(file.stem.clone(), name.clone())),
                disagreement,
                time: member_times.get(name).map(|v| measure::median(v)),
                rust_metrics: joined.map(|(q, _)| fns.unwrap()[q].clone()),
                viper_metrics: if primary {
                    viper_members.get(base).cloned()
                } else {
                    None
                },
            }
        })
        .collect::<Vec<_>>();
    if let Some((r, _)) = &silicon_result
        && silicon_unattributed(r)
    {
        ctx.warnings.push(format!(
            "{}/{}: Silicon rejects the file but not every error could be placed in a member; \
             per-member Silicon verdicts left unknown",
            suite.name, file.stem
        ));
    }
    for m in &members {
        if m.disagreement == Some("soundness") {
            ctx.warnings.push(format!(
                "{}/{}: {} verifies with Helium but Silicon rejects it (soundness check needed)",
                suite.name, file.stem, m.name
            ));
        }
    }

    let t = &times;
    let cached = |c: bool| if c { " (cached)" } else { "" };
    progress(format!(
        "  helium {} | rustc_check {}{} | silicon_verify {}{}",
        fmt_time(t.helium_verify.as_ref().and_then(|h| h.timing.median)),
        fmt_time(t.rustc_check.as_ref().and_then(|x| x.median)),
        cached(t.rustc_check.as_ref().is_some_and(|x| x.cached)),
        fmt_time(t.silicon_verify.as_ref().and_then(|x| x.median)),
        cached(t.silicon_verify.as_ref().is_some_and(|s| s.cached)),
    ));

    let family = suite.family_of(&file.stem);
    Ok(FileResult {
        suite: suite.name.clone(),
        stem: file.stem.clone(),
        family: family.as_ref().map(|(f, _)| f.name.clone()),
        knobs: family.map(|(_, k)| k),
        sha256: Fingerprints {
            rs: rs_sha,
            vpr: vpr_sha,
        },
        rust_metrics: rust.map(|r: FileMetrics| FileRustMetrics {
            loc: r.loc,
            fns: r.fns,
            structs: r.structs,
            enums: r.enums,
            totals: r.totals,
        }),
        viper_metrics: viper_file,
        blowup,
        times,
        peak_rss_mb: peaks,
        helium_error: last.error.clone(),
        coverage,
        stats: last.stats.clone().map(trim_rule_timing),
        silicon: silicon_result.map(|(r, cached)| SiliconSummary {
            verified: r.verified,
            status: r.status,
            errors: r.errors,
            cached,
            message: r.message,
        }),
        members,
    })
}

/// Silicon rejected the file, but some of its errors are not placed in a
/// member (none parsed, or a line outside every declaration). Then no member
/// can be called OK: that would report Helium incomplete where Silicon failed.
fn silicon_unattributed(r: &SiliconResult) -> bool {
    r.verified == Some(false)
        && (r.errors.is_empty() || r.errors.iter().any(|e| e.member.is_none()))
}

/// Time `rustc --emit=metadata` (type and borrow checking only, like
/// `cargo check`) on `rs`: warm-up runs, then the timed ones. When `args`
/// carry `-Z time-passes`, rustc's own report of each run is kept too.
fn time_rustc(
    rustc: &RustcOptions,
    args: &[String],
    rs: &Path,
    opts: &Options,
    timeout: Duration,
    scratch: &Path,
) -> std::io::Result<RustcTiming> {
    let out_dir = scratch.join("rustc-out");
    std::fs::create_dir_all(&out_dir)?;
    let make = || {
        let mut c = rustc.command();
        c.args(args)
            .args(["--emit=metadata", "--cap-lints", "allow", "--out-dir"])
            .arg(&out_dir)
            .arg(rs);
        c
    };
    let succeeded = |s: &Sample| s.code == Some(0);
    let samples = measure::repeat(make, opts.warmup, opts.runs, timeout, scratch, succeeded)?;
    let check = measure::wall_timing(&samples, succeeded);
    let own = args.iter().any(|a| a == "time-passes").then(|| {
        let reports: Vec<TimePasses> = samples
            .iter()
            .filter(|s| !s.timed_out && succeeded(s))
            .filter_map(|s| parse_time_passes(&s.stderr))
            .collect();
        let mut passes: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for r in &reports {
            for (k, v) in &r.passes {
                passes.entry(k.clone()).or_default().push(*v);
            }
        }
        PhasedTiming {
            timing: Timing::from_runs(
                check.status,
                reports.iter().map(|r| r.total).collect(),
                &[],
                None,
            ),
            phases: passes
                .iter()
                .map(|(k, v)| (k.clone(), measure::median(v)))
                .collect(),
        }
    });
    Ok(RustcTiming { check, own })
}

/// What one `rustc -Z time-passes` run reported about itself.
#[derive(Debug, Clone, PartialEq)]
struct TimePasses {
    /// The `total` line: the compiler session, without process startup.
    total: f64,
    /// Seconds per pass; a pass reported more than once is summed.
    passes: BTreeMap<String, f64>,
}

/// Read the `-Z time-passes` lines on stderr (`time:   0.130; rss:   29MB ->
/// 36MB (  +7MB)`, a tab, the pass name); `None` without a `total` line.
fn parse_time_passes(stderr: &str) -> Option<TimePasses> {
    let mut total = None;
    let mut passes = BTreeMap::new();
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("time:") else {
            continue;
        };
        let Some((secs, rest)) = rest.split_once(';') else {
            continue;
        };
        let (Ok(secs), Some(name)) = (secs.trim().parse::<f64>(), rest.split_whitespace().last())
        else {
            continue;
        };
        if name == "total" {
            total = Some(secs);
        } else {
            *passes.entry(name.to_string()).or_insert(0.0) += secs;
        }
    }
    Some(TimePasses {
        total: total?,
        passes,
    })
}

/// Keep the 20 slowest rules of `stats.rule_timing`: the full map is a few
/// hundred entries of wall-clock noise per file, and the slow tail is what a
/// reader looks for. The deterministic `per_rule` counts are kept whole.
fn trim_rule_timing(mut stats: Value) -> Value {
    if let Some(Value::Object(rules)) = stats.get_mut("rule_timing") {
        let cost =
            |v: &Value| v["search"].as_f64().unwrap_or(0.0) + v["apply"].as_f64().unwrap_or(0.0);
        let mut all: Vec<(String, Value)> = std::mem::take(rules).into_iter().collect();
        all.sort_by(|a, b| cost(&b.1).total_cmp(&cost(&a.1)));
        rules.extend(all.into_iter().take(20));
    }
    stats
}

/// `verify --viper-metrics`, or `None` for a build without the flag.
fn viper_metrics(verify: &Path, vpr: &Path, scratch: &Path) -> Option<Value> {
    let mut c = Command::new(verify);
    c.arg("--viper-metrics").arg(vpr);
    let s = measure::run_once(&mut c, Duration::from_secs(600), scratch).ok()?;
    let v: Value = serde_json::from_str(s.stdout.lines().next()?).ok()?;
    v.get("viper_metrics").cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_time_passes() {
        let err = "time:   0.033; rss:   18MB ->   19MB (   +1MB)\tparse_crate\n\
                   time:   0.000; rss:   27MB ->   27MB (   +0MB)\tdrop_ast\n\
                   time:   0.002; rss:   28MB ->   28MB (   +0MB)\tdrop_ast\n\
                   warning: something else\n\
                   time:   0.720; rss:   14MB ->   30MB (  +16MB)\ttotal\n";
        let r = parse_time_passes(err).unwrap();
        assert_eq!(r.total, 0.72);
        assert_eq!(r.passes["parse_crate"], 0.033);
        assert_eq!(r.passes["drop_ast"], 0.002);
        assert!(!r.passes.contains_key("total"));
        let no_total = "time:   0.1; rss: 1MB -> 1MB (+0MB)\tparse_crate\n";
        assert_eq!(parse_time_passes(no_total), None);
    }

    #[test]
    fn only_nightly_rustc_takes_z_flags() {
        assert!(rustc_is_nightly(
            "rustc 1.100.0-nightly (e71c0f1e3 2026-08-18)"
        ));
        assert!(rustc_is_nightly("rustc 1.100.0-dev"));
        assert!(!rustc_is_nightly("rustc 1.92.0 (ded5c06cf 2025-12-08)"));
    }

    fn fns(names: &[&str]) -> BTreeMap<String, FnMetrics> {
        names
            .iter()
            .map(|n| (n.to_string(), FnMetrics::default()))
            .collect()
    }

    #[test]
    fn joins_members_to_functions() {
        let f = fns(&[
            "body_step",
            "Grid::total",
            "geo::area",
            "step",
            "Vec2::step",
        ]);
        assert_eq!(
            rust_fn_for("m_body_step", &f),
            Some((&"body_step".to_string(), "exact"))
        );
        assert_eq!(
            rust_fn_for("m_Grid$total", &f),
            Some((&"Grid::total".to_string(), "fuzzy"))
        );
        assert_eq!(
            rust_fn_for("m_geo$$area", &f),
            Some((&"geo::area".to_string(), "fuzzy"))
        );
        // `step` is ambiguous by plain name, but `Vec2$step` is not.
        assert_eq!(
            rust_fn_for("m_Vec2$step", &f),
            Some((&"Vec2::step".to_string(), "fuzzy"))
        );
        assert_eq!(rust_fn_for("p_Bool_assign", &f), None);
        assert_eq!(rust_fn_for("m_unknown", &f), None);
    }
}
