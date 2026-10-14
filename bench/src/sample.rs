//! `bench sample`: measure the selected files in a random order, for as long
//! as it is left running.
//!
//! For an evaluation with no fixed end: the files of the selected suites are
//! shuffled once (with a recorded seed) and measured one after another in that
//! order, Helium and Silicon on each, exactly as `bench run` measures a file.
//! Whenever the run stops, the files measured so far are a uniform random
//! sample of the selection, so their shares (verified, failed, ...) and
//! timings estimate those of the whole selection; the summary gives each share
//! with a 95% confidence interval.
//!
//! The output directory (`--out`) holds
//!
//! - `sample.json`: the seed and what is measured (written once);
//! - `order.txt`: the random order, one `suite/stem` per line;
//! - `files.jsonl`: one `bench run` file result per line, appended as soon as
//!   the file is measured;
//! - `summary.json`: the estimates so far, replaced after every file.
//!
//! So the run can end at any moment (Ctrl-C, a closed terminal, a reboot)
//! and loses at most the file in progress. Running the same command again
//! continues where it stopped, in the same order; that needs the same `verify`
//! binary and settings, since results of two builds must not be mixed (a new
//! build gets a new `--out`, or `--fresh` to start over).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::run::{Options, Selection, Session, progress};
use crate::suites::SuiteFile;

pub struct SampleOptions {
    /// Where the sample lives.
    pub out: PathBuf,
    /// Seed of a new sample's order (default: from the clock).
    pub seed: Option<u64>,
    /// Discard an existing sample in `out` and start a new one.
    pub fresh: bool,
    /// Measure at most this many more files in this invocation.
    pub limit: Option<usize>,
}

/// How a sample ended.
pub enum Outcome {
    /// Every file of the selection is measured.
    Complete,
    /// `--limit` files were measured.
    Limited,
    /// Stopped by Ctrl-C; the file in progress was dropped.
    Interrupted,
}

/// What a sample is a sample of, fixed when it is created: continuing it is
/// only allowed while all of this stays the same.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Meta {
    seed: u64,
    created: String,
    /// The `verify` binary measured, by content.
    verify_sha256: String,
    commit: String,
    dirty: bool,
    warmup: usize,
    runs: usize,
    timeout_s: Option<f64>,
    /// Silicon's configuration (`None`: Helium only).
    silicon: Option<String>,
    suites: Vec<String>,
    only: Vec<String>,
    population: usize,
}

impl Meta {
    /// The fields that must match for a sample to be continued.
    fn settings(&self) -> Value {
        json!([
            self.verify_sha256,
            self.warmup,
            self.runs,
            self.timeout_s,
            self.silicon,
            self.suites,
            self.only
        ])
    }
}

const META: &str = "sample.json";
const ORDER: &str = "order.txt";
const FILES: &str = "files.jsonl";
const SUMMARY: &str = "summary.json";

pub fn sample(opts: &Options, sopts: &SampleOptions) -> Result<Outcome, String> {
    crate::catch_interrupt();
    let mut session = Session::new(opts)?;

    // ── The selection: every measurable file of the selected suites, as `bench run` picks them ──
    let suite_names: Vec<String> = session.suites.iter().map(|s| s.name.clone()).collect();
    let names: Vec<&str> = suite_names.iter().map(String::as_str).collect();
    let selection = Selection::new(&opts.only, &names);
    let mut population: BTreeMap<String, (usize, SuiteFile)> = BTreeMap::new();
    for (i, suite) in session.suites.iter().enumerate() {
        for file in suite.measurable() {
            if opts.only.is_empty() || !selection.matches(&suite.name, file).is_empty() {
                population.insert(format!("{}/{}", suite.name, file.stem), (i, file.clone()));
            }
        }
    }
    if population.is_empty() {
        return Err("no files selected".into());
    }

    let out = &sopts.out;
    let io = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    std::fs::create_dir_all(out).map_err(|e| io(out, e))?;
    let current = Meta {
        seed: sopts.seed.unwrap_or_else(clock_seed),
        created: crate::utc_now(),
        verify_sha256: crate::sha256_file(&opts.verify).map_err(|e| io(&opts.verify, e))?,
        commit: session.commit.clone(),
        dirty: session.dirty,
        warmup: opts.warmup,
        runs: opts.runs,
        timeout_s: opts.timeout,
        silicon: opts.silicon.as_ref().map(|s| s.config_id()),
        suites: opts.suites.clone(),
        only: opts.only.clone(),
        population: population.len(),
    };

    // ── A new sample, or the one in `out` ──
    let meta_path = out.join(META);
    if sopts.fresh {
        for name in [META, ORDER, FILES, SUMMARY] {
            let p = out.join(name);
            if p.exists() {
                std::fs::remove_file(&p).map_err(|e| io(&p, e))?;
            }
        }
    }
    let (meta, order) = if meta_path.is_file() {
        let text = std::fs::read_to_string(&meta_path).map_err(|e| io(&meta_path, e))?;
        let meta: Meta =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", meta_path.display()))?;
        if meta.settings() != current.settings() {
            return Err(format!(
                "{} holds a sample of another `verify` build or other settings \
                 (it measured commit {}): use another --out, or --fresh to discard it",
                out.display(),
                &meta.commit[..meta.commit.len().min(10)]
            ));
        }
        if sopts.seed.is_some_and(|s| s != meta.seed) {
            return Err(format!(
                "the sample in {} has seed {}: use another --out, or --fresh to discard it",
                out.display(),
                meta.seed
            ));
        }
        let order_path = out.join(ORDER);
        let order: Vec<String> = std::fs::read_to_string(&order_path)
            .map_err(|e| io(&order_path, e))?
            .lines()
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        (meta, order)
    } else {
        let mut order: Vec<String> = population.keys().cloned().collect();
        shuffle(&mut order, current.seed);
        write_atomic(
            &out.join(ORDER),
            &order.iter().map(|k| format!("{k}\n")).collect::<String>(),
        )?;
        let text = serde_json::to_string_pretty(&current).map_err(|e| e.to_string())?;
        write_atomic(&meta_path, &(text + "\n"))?;
        progress(format!(
            "sample: new, {} files in a random order (seed {}) -> {}",
            order.len(),
            current.seed,
            out.display()
        ));
        (current, order)
    };

    let gone: Vec<&String> = order
        .iter()
        .filter(|k| !population.contains_key(*k))
        .collect();
    if !gone.is_empty() {
        progress(format!(
            "warning: {} files of the sample no longer exist and are skipped (first: {})",
            gone.len(),
            gone[0]
        ));
    }
    let in_order: BTreeSet<&String> = order.iter().collect();
    let new = population.keys().filter(|k| !in_order.contains(k)).count();
    if new > 0 {
        progress(format!(
            "warning: {new} files are not in the sample's order (added since it was created); \
             --fresh includes them"
        ));
    }

    // ── What is measured already ──
    let files_path = out.join(FILES);
    let mut records = read_records(&files_path)?;
    let done: BTreeSet<String> = records.iter().map(|r| r.key.clone()).collect();
    let order: Vec<String> = order
        .into_iter()
        .filter(|k| population.contains_key(k))
        .collect();
    let mut todo: Vec<String> = order
        .iter()
        .filter(|k| !done.contains(*k))
        .cloned()
        .collect();
    if !done.is_empty() {
        progress(format!(
            "sample: continuing, {} of {} files measured (seed {})",
            records.len(),
            order.len(),
            meta.seed
        ));
    }
    if let Some(limit) = sopts.limit {
        todo.truncate(limit);
    }
    let summary_path = out.join(SUMMARY);
    let write_summary = |records: &[Record], session: &Session| -> Result<Value, String> {
        let s = summarize(records, order.len(), &meta, session);
        let text = serde_json::to_string_pretty(&s).map_err(|e| e.to_string())?;
        write_atomic(&summary_path, &(text + "\n"))?;
        Ok(s)
    };
    write_summary(&records, &session)?;

    // ── Measure, one file after another ──
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&files_path)
        .map_err(|e| io(&files_path, e))?;
    for key in &todo {
        if crate::interrupted() {
            break;
        }
        let (suite, file) = &population[key];
        let result = session.measure(*suite, file);
        if crate::interrupted() {
            // Its processes got the Ctrl-C too: whatever it measured is not a result.
            progress(format!(
                "interrupted: {key} dropped, it is measured first next time"
            ));
            break;
        }
        let value = serde_json::to_value(result?).map_err(|e| e.to_string())?;
        let line = serde_json::to_string(&value).map_err(|e| e.to_string())?;
        log.write_all(format!("{line}\n").as_bytes())
            .and_then(|()| log.sync_data())
            .map_err(|e| io(&files_path, e))?;
        session.save_caches()?;
        records.push(Record::from_result(&value));
        let s = write_summary(&records, &session)?;
        progress(format!("sample: {}", headline(&s)));
    }

    if crate::interrupted() {
        progress(format!(
            "sample: stopped with {} of {} files measured -> {}",
            records.len(),
            order.len(),
            summary_path.display()
        ));
        Ok(Outcome::Interrupted)
    } else if records.len() >= order.len() {
        progress(format!("sample: complete -> {}", summary_path.display()));
        Ok(Outcome::Complete)
    } else {
        progress(format!(
            "sample: {} of {} files measured -> {}",
            records.len(),
            order.len(),
            summary_path.display()
        ));
        Ok(Outcome::Limited)
    }
}

// ── The random order ──

fn clock_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    // Keep it short enough to type back in.
    splitmix(&mut (nanos ^ u64::from(std::process::id()))) % 1_000_000_000
}

/// SplitMix64: a small, well-mixed generator, so the order is reproducible
/// from the seed without a dependency.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fisher–Yates with `seed`.
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    for i in (1..items.len()).rev() {
        let j = ((u128::from(splitmix(&mut state)) * (i as u128 + 1)) >> 64) as usize;
        items.swap(i, j);
    }
}

// ── Reading the results back ──

/// What the summary needs of one file result.
#[derive(Debug, Clone)]
struct Record {
    key: String,
    /// The suite, plus the stem's first directory if it has one (a crate of
    /// a split crate corpus).
    group: String,
    helium: Verdict,
    silicon: Option<Verdict>,
    helium_s: Option<f64>,
    silicon_s: Option<f64>,
    members: BTreeMap<String, u64>,
    incompleteness: u64,
    soundness: u64,
}

/// A file's verdict from one verifier.
#[derive(Debug, Clone, PartialEq)]
enum Verdict {
    Verified,
    Failed,
    /// Neither, and why: `timeout`, `error`, or the member statuses that
    /// stopped it (`UNSUPPORTED`, `SKIP`, ...).
    Inconclusive(String),
}

impl Verdict {
    fn name(&self) -> &str {
        match self {
            Verdict::Verified => "verified",
            Verdict::Failed => "failed",
            Verdict::Inconclusive(_) => "inconclusive",
        }
    }
}

const VERDICTS: [&str; 3] = ["verified", "failed", "inconclusive"];

impl Record {
    fn from_result(v: &Value) -> Record {
        let suite = v["suite"].as_str().unwrap_or_default();
        let stem = v["stem"].as_str().unwrap_or_default();
        let group = match stem.split_once('/') {
            Some((dir, _)) => format!("{suite}/{dir}"),
            None => suite.to_string(),
        };
        let members: BTreeMap<String, u64> = v["coverage"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, n)| (k.clone(), n.as_u64().unwrap_or(0)))
                    .collect()
            })
            .unwrap_or_default();
        let status = |t: &Value| t["status"].as_str().unwrap_or("ok").to_string();
        let helium = {
            let wall = &v["times"]["helium_wall"];
            if !wall.is_null() && status(wall) != "ok" {
                Verdict::Inconclusive(status(wall))
            } else if v["helium_error"].is_string() {
                Verdict::Inconclusive("error".into())
            } else if members.get("FAIL").is_some_and(|n| *n > 0) {
                Verdict::Failed
            } else if members.is_empty() {
                Verdict::Inconclusive("no members".into())
            } else {
                let other: Vec<&str> = members
                    .iter()
                    .filter(|(k, n)| k.as_str() != "OK" && **n > 0)
                    .map(|(k, _)| k.as_str())
                    .collect();
                if other.is_empty() {
                    Verdict::Verified
                } else {
                    Verdict::Inconclusive(other.join("+"))
                }
            }
        };
        let silicon = v["silicon"].as_object().map(|s| {
            let st = s.get("status").and_then(Value::as_str).unwrap_or("ok");
            match (st, s.get("verified").and_then(Value::as_bool)) {
                ("ok", Some(true)) => Verdict::Verified,
                ("ok", Some(false)) => Verdict::Failed,
                ("ok", None) => Verdict::Inconclusive("unknown".into()),
                (other, _) => Verdict::Inconclusive(other.to_string()),
            }
        });
        let time = |col: &str| {
            let t = &v["times"][col];
            (status(t) == "ok").then(|| t["median"].as_f64()).flatten()
        };
        let count = |kind: &str| {
            v["members"].as_array().map_or(0, |ms| {
                ms.iter().filter(|m| m["disagreement"] == kind).count() as u64
            })
        };
        Record {
            key: format!("{suite}/{stem}"),
            group,
            helium,
            silicon,
            helium_s: time("helium_verify"),
            silicon_s: time("silicon_verify"),
            members,
            incompleteness: count("incompleteness"),
            soundness: count("soundness"),
        }
    }
}

/// The results in `files.jsonl`. A last line cut short (the run was killed
/// while writing it) is removed, so the file can be appended to again.
fn read_records(path: &Path) -> Result<Vec<Record>, String> {
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    let mut records = Vec::new();
    let mut seen = BTreeSet::new();
    let mut valid = 0;
    for line in text.split_inclusive('\n') {
        let Ok(v) = serde_json::from_str::<Value>(line.trim_end()) else {
            break;
        };
        if !line.ends_with('\n') {
            break;
        }
        let r = Record::from_result(&v);
        if seen.insert(r.key.clone()) {
            records.push(r);
        }
        valid += line.len();
    }
    if valid < text.len() {
        progress(format!(
            "warning: {}: dropping {} bytes of an incomplete last result",
            path.display(),
            text.len() - valid
        ));
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(io)?;
        f.set_len(valid as u64).map_err(io)?;
    }
    Ok(records)
}

// ── The summary ──

const Z95: f64 = 1.959_963_984_540_054;

/// The 95% Wilson score interval of the proportion `k / n`.
fn wilson(k: usize, n: usize) -> Option<[f64; 2]> {
    if n == 0 {
        return None;
    }
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    Some([
        round4((center - half).max(0.0)),
        round4((center + half).min(1.0)),
    ])
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// A share of the sample, and what it projects to for the whole selection.
fn share(k: usize, n: usize, population: usize) -> Value {
    json!({
        "count": k,
        "share": (n > 0).then(|| round4(k as f64 / n as f64)),
        "ci95": wilson(k, n),
        "projected": (n > 0).then(|| (k as f64 / n as f64 * population as f64).round() as u64),
    })
}

fn verdicts<'r>(verdicts: impl Iterator<Item = &'r Verdict>, population: usize) -> Value {
    let all: Vec<&Verdict> = verdicts.collect();
    let n = all.len();
    let mut out = serde_json::Map::new();
    out.insert("files".into(), json!(n));
    for name in VERDICTS {
        let k = all.iter().filter(|v| v.name() == name).count();
        out.insert(name.into(), share(k, n, population));
    }
    let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
    for v in &all {
        if let Verdict::Inconclusive(why) = v {
            *reasons.entry(why).or_default() += 1;
        }
    }
    out.insert("inconclusive_reasons".into(), json!(reasons));
    Value::Object(out)
}

fn time_stats(mut xs: Vec<f64>, population: usize) -> Value {
    if xs.is_empty() {
        return Value::Null;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    let total: f64 = xs.iter().sum();
    json!({
        "files": n,
        "total": total,
        "mean": total / n as f64,
        "median": crate::measure::median(&xs),
        "p90": xs[(0.9 * (n - 1) as f64) as usize],
        "max": xs[n - 1],
        // The mean times every file of the selection: the time a full run
        // would spend on the files that finish.
        "projected_total": total / n as f64 * population as f64,
    })
}

/// The geometric mean of `ratios` with a 95% interval (normal approximation
/// on the logarithms).
fn geomean_ci(ratios: &[f64]) -> Value {
    let logs: Vec<f64> = ratios
        .iter()
        .filter(|r| **r > 0.0)
        .map(|r| r.ln())
        .collect();
    let n = logs.len();
    if n == 0 {
        return Value::Null;
    }
    let mean = logs.iter().sum::<f64>() / n as f64;
    let ci = (n > 1).then(|| {
        let var = logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let half = Z95 * (var / n as f64).sqrt();
        [(mean - half).exp(), (mean + half).exp()]
    });
    json!({"files": n, "geomean": mean.exp(), "ci95": ci})
}

fn summarize(records: &[Record], population: usize, meta: &Meta, session: &Session) -> Value {
    let n = records.len();
    let silicon = meta.silicon.is_some();
    let mut agreement: BTreeMap<String, usize> = BTreeMap::new();
    let mut members: BTreeMap<String, u64> = BTreeMap::new();
    let mut groups: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in records {
        if let Some(s) = &r.silicon {
            *agreement
                .entry(format!("helium {} / silicon {}", r.helium.name(), s.name()))
                .or_default() += 1;
        }
        for (k, c) in &r.members {
            *members.entry(k.clone()).or_default() += c;
        }
        groups.entry(&r.group).or_default().push(r);
    }
    let speedups: Vec<f64> = records
        .iter()
        .filter_map(|r| Some(r.silicon_s? / r.helium_s?))
        .collect();
    let group_counts = |rs: &[&Record], pick: &dyn Fn(&Record) -> Option<Verdict>| -> Value {
        let mut c: BTreeMap<&str, usize> = VERDICTS.iter().map(|v| (*v, 0)).collect();
        for r in rs {
            if let Some(v) = pick(r) {
                *c.get_mut(v.name()).unwrap() += 1;
            }
        }
        json!(c)
    };
    json!({
        "seed": meta.seed,
        "created": meta.created,
        "updated": crate::utc_now(),
        "commit": meta.commit,
        "dirty": meta.dirty,
        "host": session.host(),
        "tools": session.tools(),
        "settings": {"warmup": meta.warmup, "runs": meta.runs, "timeout_s": meta.timeout_s,
                     "suites": meta.suites, "only": meta.only},
        "population": population,
        "measured": n,
        "remaining": population.saturating_sub(n),
        "share_measured": (population > 0).then(|| round4(n as f64 / population as f64)),
        "helium": verdicts(records.iter().map(|r| &r.helium), population),
        "silicon": if silicon {
            verdicts(records.iter().filter_map(|r| r.silicon.as_ref()), population)
        } else {
            Value::Null
        },
        "agreement": if silicon { json!(agreement) } else { Value::Null },
        "members": {
            "helium": members,
            "incompleteness": records.iter().map(|r| r.incompleteness).sum::<u64>(),
            "soundness": records.iter().map(|r| r.soundness).sum::<u64>(),
            // Helium verifies what Silicon rejects: each needs a look.
            "soundness_files": records.iter().filter(|r| r.soundness > 0)
                .map(|r| r.key.as_str()).collect::<Vec<_>>(),
        },
        "time_s": {
            "helium_verify": time_stats(records.iter().filter_map(|r| r.helium_s).collect(), population),
            "silicon_verify": time_stats(records.iter().filter_map(|r| r.silicon_s).collect(), population),
            // Silicon's time over Helium's, per file both finished.
            "speedup": geomean_ci(&speedups),
        },
        "groups": groups.iter().map(|(g, rs)| (g.to_string(), json!({
            "measured": rs.len(),
            "helium": group_counts(rs, &|r| Some(r.helium.clone())),
            "silicon": if silicon { group_counts(rs, &|r| r.silicon.clone()) } else { Value::Null },
        }))).collect::<serde_json::Map<_, _>>(),
    })
}

/// One line on the state of the sample.
fn headline(s: &Value) -> String {
    let pct = |v: &Value| -> String {
        match (v["share"].as_f64(), v["ci95"].as_array()) {
            (Some(p), Some(ci)) => format!(
                "{:.1}% [{:.1}-{:.1}]",
                100.0 * p,
                100.0 * ci[0].as_f64().unwrap_or(0.0),
                100.0 * ci[1].as_f64().unwrap_or(0.0)
            ),
            _ => "-".into(),
        }
    };
    let mut line = format!(
        "{}/{} measured | helium verified {}, failed {}",
        s["measured"],
        s["population"],
        pct(&s["helium"]["verified"]),
        pct(&s["helium"]["failed"])
    );
    if !s["silicon"].is_null() {
        line += &format!(
            " | silicon verified {}, failed {}",
            pct(&s["silicon"]["verified"]),
            pct(&s["silicon"]["failed"])
        );
    }
    if let Some(g) = s["time_s"]["speedup"]["geomean"].as_f64() {
        line += &format!(" | speedup x{g:.2}");
    }
    line
}

/// Write `text` to `path` through a temporary file and a rename, so a reader
/// (or a run killed halfway) never sees half a file.
fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut last = None;
    // On Windows a reader holding the file open makes the rename fail for a moment.
    for attempt in 0..10 {
        match std::fs::rename(&tmp, path) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(50 * (attempt + 1)));
            }
        }
    }
    Err(format!("{}: {}", path.display(), last.unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_a_seeded_permutation() {
        let mut a: Vec<u32> = (0..100).collect();
        let mut b = a.clone();
        shuffle(&mut a, 42);
        shuffle(&mut b, 42);
        assert_eq!(a, b);
        assert_ne!(a, (0..100).collect::<Vec<_>>());
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        let mut c: Vec<u32> = (0..100).collect();
        shuffle(&mut c, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn wilson_interval() {
        assert_eq!(wilson(0, 0), None);
        let [lo, hi] = wilson(50, 100).unwrap();
        assert!((lo - 0.4038).abs() < 1e-3 && (hi - 0.5962).abs() < 1e-3);
        assert_eq!(wilson(0, 10).unwrap()[0], 0.0);
        assert_eq!(wilson(10, 10).unwrap()[1], 1.0);
    }

    #[test]
    fn reads_verdicts_from_file_results() {
        let r = Record::from_result(&json!({
            "suite": "split", "stem": "anyhow/0001_f",
            "times": {"helium_wall": {"status": "ok"}, "helium_verify": {"status": "ok", "median": 0.5},
                      "silicon_verify": {"status": "ok", "median": 2.0}},
            "helium_error": null, "coverage": {"OK": 2, "FAIL": 1},
            "silicon": {"status": "ok", "verified": true},
            "members": [{"disagreement": "incompleteness"}, {"disagreement": null}],
        }));
        assert_eq!(r.group, "split/anyhow");
        assert_eq!(r.helium, Verdict::Failed);
        assert_eq!(r.silicon, Some(Verdict::Verified));
        assert_eq!((r.helium_s, r.silicon_s), (Some(0.5), Some(2.0)));
        assert_eq!((r.incompleteness, r.soundness), (1, 0));

        let r = Record::from_result(&json!({
            "suite": "viper", "stem": "x",
            "times": {"helium_wall": {"status": "timeout"}},
            "coverage": {}, "silicon": {"status": "timeout", "verified": null}, "members": [],
        }));
        assert_eq!(r.group, "viper");
        assert_eq!(r.helium, Verdict::Inconclusive("timeout".into()));
        assert_eq!(r.silicon, Some(Verdict::Inconclusive("timeout".into())));

        let r = Record::from_result(&json!({
            "suite": "viper", "stem": "y", "times": {"helium_wall": {"status": "ok"}},
            "coverage": {"OK": 3, "UNSUPPORTED": 1}, "silicon": null, "members": [],
        }));
        assert_eq!(r.helium, Verdict::Inconclusive("UNSUPPORTED".into()));
        assert_eq!(r.silicon, None);
    }

    #[test]
    fn drops_a_torn_last_line() {
        let dir = std::env::temp_dir().join(format!("bench-sample-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILES);
        let line = r#"{"suite":"s","stem":"a","times":{},"coverage":{"OK":1},"members":[]}"#;
        std::fs::write(&path, format!("{line}\n{{\"suite\":\"s\",\"st")).unwrap();
        let records = read_records(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, "s/a");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{line}\n"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
