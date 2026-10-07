//! The benchmark runner. See `benchmarks/README.md` and
//! `plans/regression-pipeline.md`.
//!
//! ```text
//! bench check-suites [--benchmarks DIR] [suite options] [--no-rustc] [--rustc PATH] [--rustc-toolchain TC]
//! bench run [--out FILE] [suite options] [options]
//! bench rust-metrics FILE.rs
//! ```
//!
//! Suite options, for both commands. The external suites and the run-wide
//! `.vpr` size limit come from `tools/bench/config.json`, overlaid with its
//! `config.local.json` (as `tools/bench/run.py` reads them); an external suite
//! whose directory is missing on this machine is left out with a note.
//!
//! ```text
//! --config FILE           the config to read (default: tools/bench/config.json, if present)
//! --no-config             read no config: only the suites under --benchmarks
//! --external-suites JSON  more suites: {"name": {"path": DIR, <suite.json fields>}}; a name
//!                         also in the config replaces it there
//! --max-vpr-mb MB         skip .vpr files larger than this (a suite's own max_vpr_mb wins)
//! ```
//!
//! `bench run` options:
//!
//! ```text
//! --benchmarks DIR        suites root (default: benchmarks)
//! --verify PATH           the verify binary to measure (default: target/release/verify)
//! --metrics-verify PATH   verify used for --viper-metrics (default: --verify)
//! --repo DIR              repository whose commit is recorded (default: .)
//! --commit SHA            record this commit instead of asking git
//! --host NAME             host name to record (default: this machine's)
//! --suite NAME            only this suite (repeatable)
//! --only ENTRY            only these files (repeatable): a suite, SUITE/STEM, a stem (the path
//!                         inside the suite directory), any path ending in one, or a .vpr's path
//! --only-file FILE        --only for every line of FILE (blank lines and `#` comments skipped)
//! --warmup N              untimed runs first (default 1)
//! --runs N                timed runs (default 5)
//! --timeout SECS          per-run timeout (default: suite.json, else 300)
//! --rustc PATH            rustc to time (default: rustc)
//! --rustc-toolchain TC    pinned toolchain, passed as +TC
//! --no-rustc              skip the rustc columns
//! --rustc-cache FILE      rustc timings cache (read and updated)
//! --silicon-jar PATH      Silicon fat jar (no jar: no Silicon columns)
//! --java PATH             java binary (default: java)
//! --jvm-arg ARG           extra JVM argument (repeatable; default -Xss128m)
//! --silicon-arg ARG       extra Silicon argument (repeatable)
//! --silicon-cache FILE    Silicon results cache (read and updated)
//! --silicon-warm DIR      also time Silicon in one JVM warmed up on the .vpr files under
//!                         DIR (repeatable; files equal to a benchmark are left out)
//! --silicon-warmup-s SECS warm-up budget of each new warm JVM (default 60)
//! --silicon-warmup-file-timeout SECS
//!                         Silicon's --timeout for each warm-up file (default 10)
//! --scratch DIR           scratch directory (default: system temp)
//! --out FILE              write the run JSON here (default: stdout)
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bench::run::{Options, RustcOptions};
use bench::silicon::Silicon;
use bench::silicon_warm::WarmOptions;

fn usage() -> ExitCode {
    eprintln!(
        "usage: bench check-suites [--benchmarks DIR] [--config FILE | --no-config] [--external-suites JSON]\n                          \
         [--max-vpr-mb MB] [--no-rustc] [--rustc PATH] [--rustc-toolchain TC]\n       \
         bench run [--out FILE] [options]   (see the source header or benchmarks/README.md)\n       \
         bench rust-metrics FILE.rs"
    );
    ExitCode::from(2)
}

struct Args {
    rest: std::vec::IntoIter<String>,
}

impl Args {
    fn value(&mut self, flag: &str) -> Result<String, String> {
        self.rest
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))
    }

    fn number(&mut self, flag: &str) -> Result<f64, String> {
        let v = self.value(flag)?;
        v.parse().map_err(|_| format!("{flag}: not a number: {v}"))
    }
}

fn default_verify() -> PathBuf {
    let exe = if cfg!(windows) {
        "verify.exe"
    } else {
        "verify"
    };
    PathBuf::from("target").join("release").join(exe)
}

fn default_scratch() -> PathBuf {
    std::env::temp_dir().join(format!("helium-bench-{}", std::process::id()))
}

fn main() -> ExitCode {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        return usage();
    }
    let cmd = argv.remove(0);
    let mut args = Args {
        rest: argv.into_iter(),
    };
    let result = match cmd.as_str() {
        "check-suites" => check_suites(&mut args),
        "run" => run(&mut args),
        "rust-metrics" => rust_metrics(&mut args),
        "-h" | "--help" | "help" => return usage(),
        other => Err(format!("unknown command `{other}`")),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("bench: {e}");
            ExitCode::from(2)
        }
    }
}

/// The suite options, shared by both commands; [`SuiteArgs::discovery`]
/// combines them with the config.
#[derive(Default)]
struct SuiteArgs {
    explicit: bench::suites::Discovery,
    config: Option<PathBuf>,
    no_config: bool,
}

impl SuiteArgs {
    /// Take `flag` if it is a suite option.
    fn take(&mut self, flag: &str, args: &mut Args) -> Result<bool, String> {
        match flag {
            "--external-suites" => {
                let json = args.value(flag)?;
                self.explicit
                    .external
                    .extend(bench::suites::External::parse_all(&json, Path::new("."))?);
            }
            "--max-vpr-mb" => self.explicit.max_vpr_mb = Some(args.number(flag)?),
            "--config" => self.config = Some(args.value(flag)?.into()),
            "--no-config" => self.no_config = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The config's suites and limit, with the explicit options on top.
    fn discovery(self) -> Result<bench::suites::Discovery, String> {
        let mut d = match (&self.config, self.no_config) {
            (_, true) => bench::suites::Discovery::default(),
            (Some(path), false) => config_discovery(path)?,
            (None, false) => {
                let default = Path::new("tools/bench/config.json");
                if default.is_file() {
                    config_discovery(default)?
                } else {
                    bench::suites::Discovery::default()
                }
            }
        };
        let names: Vec<String> = self
            .explicit
            .external
            .iter()
            .map(|e| e.name.clone())
            .collect();
        d.external.retain(|e| !names.contains(&e.name));
        d.external.extend(self.explicit.external);
        if self.explicit.max_vpr_mb.is_some() {
            d.max_vpr_mb = self.explicit.max_vpr_mb;
        }
        Ok(d)
    }
}

/// `external_suites` and `max_vpr_mb` of a `tools/bench/config.json`, with
/// the top-level keys of a `config.local.json` next to it replacing its own.
/// Relative paths are taken from the repository root (two levels up).
fn config_discovery(path: &Path) -> Result<bench::suites::Discovery, String> {
    let read = |p: &Path| -> Result<serde_json::Map<String, serde_json::Value>, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
    };
    let mut cfg = read(path)?;
    let local = path.with_file_name("config.local.json");
    if local.is_file() {
        cfg.extend(read(&local)?);
    }
    let root = path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut external = match cfg.get("external_suites") {
        Some(v) if !v.is_null() => bench::suites::External::parse_all(&v.to_string(), root)
            .map_err(|e| format!("{}: {e}", path.display()))?,
        _ => Vec::new(),
    };
    external.retain(|e| {
        let present = e.path.is_dir();
        if !present {
            eprintln!(
                "[bench] note: external suite `{}` skipped: no directory {}",
                e.name,
                e.path.display()
            );
        }
        present
    });
    Ok(bench::suites::Discovery {
        external,
        max_vpr_mb: cfg.get("max_vpr_mb").and_then(serde_json::Value::as_f64),
    })
}

fn check_suites(args: &mut Args) -> Result<ExitCode, String> {
    let mut benchmarks = PathBuf::from("benchmarks");
    let mut suite_args = SuiteArgs::default();
    let mut rustc = Some(RustcOptions {
        rustc: "rustc".into(),
        toolchain: None,
    });
    while let Some(a) = args.rest.next() {
        match a.as_str() {
            "--benchmarks" => benchmarks = args.value(&a)?.into(),
            "--no-rustc" => rustc = None,
            "--rustc" => {
                if let Some(r) = rustc.as_mut() {
                    r.rustc = args.value(&a)?.into();
                }
            }
            "--rustc-toolchain" => {
                if let Some(r) = rustc.as_mut() {
                    r.toolchain = Some(args.value(&a)?);
                }
            }
            _ if suite_args.take(&a, args)? => {}
            _ => return Err(format!("check-suites: unknown argument `{a}`")),
        }
    }
    let discovery = suite_args.discovery()?;
    let scratch = default_scratch();
    let report = bench::check::check(&benchmarks, &discovery, rustc.as_ref(), &scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    for s in &report.suites {
        println!("suite    {s}");
    }
    for w in &report.warnings {
        println!("warning  {w}");
    }
    for e in &report.errors {
        println!("ERROR    {e}");
    }
    println!(
        "\n{} suites, {} errors, {} warnings",
        report.suites.len(),
        report.errors.len(),
        report.warnings.len()
    );
    Ok(if report.errors.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run(args: &mut Args) -> Result<ExitCode, String> {
    let mut opts = Options {
        benchmarks: "benchmarks".into(),
        discovery: Default::default(),
        verify: default_verify(),
        metrics_verify: PathBuf::new(),
        repo: ".".into(),
        commit: None,
        host: bench::hostname(),
        suites: Vec::new(),
        only: Vec::new(),
        warmup: 1,
        runs: 5,
        timeout: None,
        rustc: Some(RustcOptions {
            rustc: "rustc".into(),
            toolchain: None,
        }),
        rustc_cache: None,
        silicon: None,
        silicon_cache: None,
        silicon_warm: None,
        scratch: default_scratch(),
    };
    let mut out: Option<PathBuf> = None;
    let mut jar: Option<PathBuf> = None;
    let mut java = PathBuf::from("java");
    let mut jvm_args: Vec<String> = Vec::new();
    let mut silicon_args: Vec<String> = Vec::new();
    let mut warm_corpus: Vec<PathBuf> = Vec::new();
    let mut warmup_s = 60.0;
    let mut warmup_file_timeout = 10.0;
    let mut suite_args = SuiteArgs::default();
    while let Some(a) = args.rest.next() {
        match a.as_str() {
            "--benchmarks" => opts.benchmarks = args.value(&a)?.into(),
            "--verify" => opts.verify = args.value(&a)?.into(),
            "--metrics-verify" => opts.metrics_verify = args.value(&a)?.into(),
            "--repo" => opts.repo = args.value(&a)?.into(),
            "--commit" => opts.commit = Some(args.value(&a)?),
            "--host" => opts.host = args.value(&a)?,
            "--suite" => opts.suites.push(args.value(&a)?),
            "--only" => opts.only.push(args.value(&a)?),
            "--only-file" => {
                let path = args.value(&a)?;
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                opts.only.extend(
                    text.lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .map(String::from),
                );
            }
            "--warmup" => opts.warmup = args.number(&a)? as usize,
            "--runs" => opts.runs = (args.number(&a)? as usize).max(1),
            "--timeout" => opts.timeout = Some(args.number(&a)?),
            "--rustc" => {
                let p = args.value(&a)?;
                opts.rustc
                    .get_or_insert(RustcOptions {
                        rustc: "rustc".into(),
                        toolchain: None,
                    })
                    .rustc = p.into();
            }
            "--rustc-toolchain" => {
                let tc = args.value(&a)?;
                opts.rustc
                    .get_or_insert(RustcOptions {
                        rustc: "rustc".into(),
                        toolchain: None,
                    })
                    .toolchain = Some(tc);
            }
            "--no-rustc" => opts.rustc = None,
            "--silicon-jar" => jar = Some(args.value(&a)?.into()),
            "--java" => java = args.value(&a)?.into(),
            "--jvm-arg" => jvm_args.push(args.value(&a)?),
            "--silicon-arg" => silicon_args.push(args.value(&a)?),
            "--rustc-cache" => opts.rustc_cache = Some(args.value(&a)?.into()),
            "--silicon-cache" => opts.silicon_cache = Some(args.value(&a)?.into()),
            "--silicon-warm" => warm_corpus.push(args.value(&a)?.into()),
            "--silicon-warmup-s" => warmup_s = args.number(&a)?,
            "--silicon-warmup-file-timeout" => warmup_file_timeout = args.number(&a)?,
            "--scratch" => opts.scratch = args.value(&a)?.into(),
            "--out" => out = Some(args.value(&a)?.into()),
            _ if suite_args.take(&a, args)? => {}
            _ => return Err(format!("run: unknown argument `{a}`")),
        }
    }
    opts.discovery = suite_args.discovery()?;
    if opts.metrics_verify.as_os_str().is_empty() {
        opts.metrics_verify = opts.verify.clone();
    }
    if !opts.verify.is_file() {
        return Err(format!(
            "no verify binary at {} (cargo build --release --bin verify)",
            opts.verify.display()
        ));
    }
    if let Some(jar) = jar {
        if jvm_args.is_empty() {
            jvm_args.push("-Xss128m".into());
        }
        opts.silicon = Some(
            Silicon::new(java, jar.clone(), jvm_args, silicon_args)
                .map_err(|e| format!("{}: {e}", jar.display()))?,
        );
        if !warm_corpus.is_empty() {
            opts.silicon_warm = Some(WarmOptions {
                corpus: warm_corpus,
                warmup_s,
                file_timeout_s: (warmup_file_timeout as u64).max(1),
            });
        }
    } else {
        eprintln!("[bench] no --silicon-jar: Silicon columns are skipped");
    }

    let result = bench::run::run(&opts);
    let _ = std::fs::remove_dir_all(&opts.scratch);
    let run = result?;
    // Compact: run files are committed once per commit, and indentation is
    // most of their size.
    let json = serde_json::to_string(&run).map_err(|e| e.to_string())?;
    match out {
        Some(path) => {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("[bench] wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(ExitCode::SUCCESS)
}

fn rust_metrics(args: &mut Args) -> Result<ExitCode, String> {
    let path = args.value("rust-metrics")?;
    let src = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let m = bench::rust_metrics::file_metrics(&src).map_err(|e| format!("{path}: {e}"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&m).map_err(|e| e.to_string())?
    );
    Ok(ExitCode::SUCCESS)
}
