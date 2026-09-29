//! The benchmark runner. See `benchmarks/README.md` and
//! `plans/regression-pipeline.md`.
//!
//! ```text
//! bench check-suites [--benchmarks DIR] [--no-rustc] [--rustc PATH] [--rustc-toolchain TC]
//! bench run [--out FILE] [options]
//! bench rust-metrics FILE.rs
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
//! --only SUITE/STEM       only this file (repeatable)
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
//! --scratch DIR           scratch directory (default: system temp)
//! --out FILE              write the run JSON here (default: stdout)
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use bench::run::{Options, RustcOptions};
use bench::silicon::Silicon;

fn usage() -> ExitCode {
    eprintln!(
        "usage: bench check-suites [--benchmarks DIR] [--no-rustc] [--rustc PATH] [--rustc-toolchain TC]\n       \
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

fn check_suites(args: &mut Args) -> Result<ExitCode, String> {
    let mut benchmarks = PathBuf::from("benchmarks");
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
            _ => return Err(format!("check-suites: unknown argument `{a}`")),
        }
    }
    let scratch = default_scratch();
    let report = bench::check::check(&benchmarks, rustc.as_ref(), &scratch);
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
        scratch: default_scratch(),
    };
    let mut out: Option<PathBuf> = None;
    let mut jar: Option<PathBuf> = None;
    let mut java = PathBuf::from("java");
    let mut jvm_args: Vec<String> = Vec::new();
    let mut silicon_args: Vec<String> = Vec::new();
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
            "--scratch" => opts.scratch = args.value(&a)?.into(),
            "--out" => out = Some(args.value(&a)?.into()),
            _ => return Err(format!("run: unknown argument `{a}`")),
        }
    }
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
