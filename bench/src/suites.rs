//! Benchmark suites, discovered by directory layout.
//!
//! A **suite** is a directory under `benchmarks/` laid out as
//!
//! ```text
//! benchmarks/<suite>/
//!   src/<stem>.rs             Rust source (optional: a suite may be Viper-only)
//!   vpr/<stem>.vpr            its Viper encoding
//!   suite.json                optional settings (see [`SuiteConfig`])
//!   expected_failures.txt     optional, `<stem> <member>` per line
//! ```
//!
//! Two flat, Viper-only forms are recognised as well: `.vpr` files directly in
//! `benchmarks/` form the suite `viper`, and a subdirectory of a suite that
//! holds `.vpr` files directly (`panic_free/isolate/`) forms the suite
//! `<suite>/<sub>`. There is no central list: adding a directory adds a suite.
//!
//! Suites can also live outside `benchmarks/` ([`External`], configured in
//! `tools/bench/config.json`). Besides the layouts above, an external suite
//! may be a **crate corpus**: one benchmark per whole crate,
//!
//! ```text
//! <dir>/
//!   crates/<stem>/Cargo.toml  the crate, with the Cargo.lock that pins its dependencies
//!   viper/<stem>.vpr          its Viper encoding, the whole crate in one file
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Default per-run timeout, seconds.
pub const DEFAULT_TIMEOUT_S: f64 = 300.0;

/// Default rustc arguments (before `--emit=metadata`).
pub fn default_rustc_args() -> Vec<String> {
    ["--edition", "2021", "--crate-type", "lib"]
        .map(String::from)
        .to_vec()
}

/// A generated family: stems carry knob values, read by a regex with one
/// named group per knob.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    /// Defaults to the suite name for a single `family`.
    #[serde(default)]
    pub name: Option<String>,
    pub pattern: String,
    pub knobs: Vec<String>,
}

/// `suite.json`. Every field is optional; unknown fields are an error, so a
/// typo does not silently fall back to a default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteConfig {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub rustc_args: Option<Vec<String>>,
    #[serde(default)]
    pub timeout_s: Option<f64>,
    /// Whether to run Silicon on this suite (default true).
    #[serde(default)]
    pub silicon: Option<bool>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// A generated suite: every stem must match the pattern.
    #[serde(default)]
    pub family: Option<Family>,
    /// Several families in one suite. Each must match at least one stem; with
    /// `exhaustive`, every stem must match one of them.
    #[serde(default)]
    pub families: Option<Vec<Family>>,
    #[serde(default)]
    pub exhaustive: Option<bool>,
    /// Skip `.vpr` files larger than this many megabytes (10^6 bytes); they
    /// are listed in the run file as skipped. Without it, the run-wide limit
    /// applies (`max_vpr_mb` in config.json), if any.
    #[serde(default)]
    pub max_vpr_mb: Option<f64>,
    /// At most this many timed runs per tool (the run's own count is the
    /// ceiling), for suites too slow to measure five times.
    #[serde(default)]
    pub runs: Option<usize>,
    /// At most this many untimed warm-up runs.
    #[serde(default)]
    pub warmup: Option<usize>,
    /// `--target` for checking a crate corpus with cargo: the platform the
    /// Viper encodings were generated on, so rustc compiles the same
    /// `cfg`-selected code (`rustup target add <target>` first).
    #[serde(default)]
    pub rustc_target: Option<String>,
}

/// A suite outside `benchmarks/`: its directory and its settings (the
/// `suite.json` fields, which replace a `suite.json` in the directory).
#[derive(Debug, Clone)]
pub struct External {
    pub name: String,
    pub path: PathBuf,
    pub config: Option<SuiteConfig>,
}

impl External {
    /// The `external_suites` object of config.json: `{name: {"path": ..,
    /// <suite.json fields>}}`. Relative paths are taken from `base`.
    pub fn parse_all(json: &str, base: &Path) -> Result<Vec<External>, String> {
        let map: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(json).map_err(|e| format!("external suites: {e}"))?;
        map.into_iter()
            .map(|(name, mut v)| {
                let path = v
                    .as_object_mut()
                    .and_then(|o| o.remove("path"))
                    .and_then(|p| p.as_str().map(PathBuf::from))
                    .ok_or_else(|| format!("external suite `{name}`: needs a `path`"))?;
                let rest = v.as_object().is_some_and(|o| !o.is_empty());
                let config = rest
                    .then(|| serde_json::from_value::<SuiteConfig>(v))
                    .transpose()
                    .map_err(|e| format!("external suite `{name}`: {e}"))?;
                Ok(External {
                    path: base.join(path),
                    name,
                    config,
                })
            })
            .collect()
    }
}

/// What to discover besides `benchmarks/`, and the run-wide size limit.
#[derive(Debug, Clone, Default)]
pub struct Discovery {
    pub external: Vec<External>,
    pub max_vpr_mb: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CompiledFamily {
    pub name: String,
    pub regex: Regex,
    pub knobs: Vec<String>,
}

impl CompiledFamily {
    /// Knob values of `stem`, if it belongs to this family.
    pub fn knobs_of(&self, stem: &str) -> Option<BTreeMap<String, f64>> {
        let caps = self.regex.captures(stem)?;
        self.knobs
            .iter()
            .map(|k| Some((k.clone(), caps.name(k)?.as_str().parse().ok()?)))
            .collect()
    }
}

/// One benchmark: a stem with its Rust source and/or Viper encoding.
#[derive(Debug, Clone)]
pub struct SuiteFile {
    pub stem: String,
    pub rs: Option<PathBuf>,
    pub vpr: Option<PathBuf>,
    /// A whole crate instead of one `.rs` (a crate corpus): its directory,
    /// holding `Cargo.toml`.
    pub krate: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Suite {
    pub name: String,
    pub dir: PathBuf,
    pub config: SuiteConfig,
    pub families: Vec<CompiledFamily>,
    pub files: Vec<SuiteFile>,
    /// `(stem, member)` pairs from `expected_failures.txt`.
    pub expected_failures: BTreeSet<(String, String)>,
    /// Stems whose `.vpr` is over the size limit, with their size in MB.
    pub skipped_large: Vec<(String, f64)>,
    /// Problems found while loading. Errors stop a run; warnings do not.
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Suite {
    pub fn timeout_s(&self) -> f64 {
        self.config.timeout_s.unwrap_or(DEFAULT_TIMEOUT_S)
    }

    pub fn rustc_args(&self) -> Vec<String> {
        self.config
            .rustc_args
            .clone()
            .unwrap_or_else(default_rustc_args)
    }

    pub fn silicon(&self) -> bool {
        self.config.silicon.unwrap_or(true)
    }

    /// Timed runs for this suite: the run's count, capped by its settings.
    pub fn runs(&self, run: usize) -> usize {
        self.config.runs.map_or(run, |n| n.clamp(1, run.max(1)))
    }

    /// Warm-up runs for this suite: the run's count, capped by its settings.
    pub fn warmup(&self, run: usize) -> usize {
        self.config.warmup.map_or(run, |n| n.min(run))
    }

    /// The family `stem` belongs to and its knob values.
    pub fn family_of(&self, stem: &str) -> Option<(&CompiledFamily, BTreeMap<String, f64>)> {
        self.families
            .iter()
            .find_map(|f| f.knobs_of(stem).map(|k| (f, k)))
    }

    /// The files that can be measured (they have a `.vpr`).
    pub fn measurable(&self) -> impl Iterator<Item = &SuiteFile> {
        self.files.iter().filter(|f| f.vpr.is_some())
    }
}

fn files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == ext))
        .collect();
    v.sort();
    v
}

fn stem_of(p: &Path) -> String {
    p.file_stem().unwrap().to_string_lossy().into_owned()
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
        .collect();
    v.sort();
    v
}

/// Every suite under `root` (normally `benchmarks/`), sorted by name.
pub fn discover(root: &Path) -> Vec<Suite> {
    discover_with(root, &Discovery::default())
}

/// [`discover`], plus the external suites, with the size limit applied.
pub fn discover_with(root: &Path, extra: &Discovery) -> Vec<Suite> {
    let mut suites = discover_root(root);
    for ext in &extra.external {
        let taken = suites.iter().any(|s| s.name == ext.name);
        let mut suite = load_external(ext);
        if taken {
            suite.errors.push(format!(
                "external suite `{}` has the name of a suite in {}",
                ext.name,
                root.display()
            ));
        }
        suites.push(suite);
    }
    for suite in &mut suites {
        apply_size_limit(suite, extra.max_vpr_mb);
    }
    suites.sort_by(|a, b| a.name.cmp(&b.name));
    suites
}

fn discover_root(root: &Path) -> Vec<Suite> {
    let mut suites = Vec::new();
    if !files_with_ext(root, "vpr").is_empty() {
        suites.push(load_flat("viper", root));
    }
    for dir in subdirs(root) {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let structured = dir.join("src").is_dir()
            || dir.join("vpr").is_dir()
            || dir.join("suite.json").is_file();
        if structured {
            suites.push(load_structured(&name, &dir));
            for sub in subdirs(&dir) {
                let sub_name = sub.file_name().unwrap().to_string_lossy().into_owned();
                if sub_name != "src" && sub_name != "vpr" && !files_with_ext(&sub, "vpr").is_empty()
                {
                    suites.push(load_flat(&format!("{name}/{sub_name}"), &sub));
                }
            }
        } else if !files_with_ext(&dir, "vpr").is_empty() {
            suites.push(load_flat(&name, &dir));
        }
    }
    suites
}

fn load_external(ext: &External) -> Suite {
    let dir = &ext.path;
    if !dir.is_dir() {
        let mut s = empty_suite(&ext.name, dir);
        s.errors.push(format!(
            "external suite `{}`: no directory {}",
            ext.name,
            dir.display()
        ));
        return s;
    }
    let mut suite = if dir.join("crates").is_dir() && dir.join("viper").is_dir() {
        load_crate_corpus(&ext.name, dir)
    } else if dir.join("src").is_dir() || dir.join("vpr").is_dir() {
        load_structured(&ext.name, dir)
    } else {
        load_flat(&ext.name, dir)
    };
    if let Some(config) = &ext.config {
        // The settings from config.json replace the directory's suite.json.
        suite.config = config.clone();
        suite.errors.retain(|e| !e.contains("suite.json"));
        suite.families.clear();
        check_families(&mut suite);
    }
    suite
}

/// A crate corpus: `crates/<stem>/` paired with `viper/<stem>.vpr`.
fn load_crate_corpus(name: &str, dir: &Path) -> Suite {
    let mut suite = empty_suite(name, dir);
    load_config(&mut suite);
    let crates = dir.join("crates");
    suite.files = files_with_ext(&dir.join("viper"), "vpr")
        .into_iter()
        .map(|vpr| {
            let stem = stem_of(&vpr);
            let krate = crates.join(&stem);
            SuiteFile {
                krate: krate.join("Cargo.toml").is_file().then_some(krate),
                stem,
                rs: None,
                vpr: Some(vpr),
            }
        })
        .collect();
    let have: BTreeSet<&str> = suite.files.iter().map(|f| f.stem.as_str()).collect();
    let unencoded: Vec<String> = subdirs(&crates)
        .iter()
        .map(|d| d.file_name().unwrap().to_string_lossy().into_owned())
        .filter(|c| !have.contains(c.as_str()))
        .collect();
    if !unencoded.is_empty() {
        suite.warnings.push(format!(
            "{name}: {} crates have no viper/<crate>.vpr (skipped): {}",
            unencoded.len(),
            unencoded.join(", ")
        ));
    }
    let orphans: Vec<&str> = suite
        .files
        .iter()
        .filter(|f| f.krate.is_none())
        .map(|f| f.stem.as_str())
        .collect();
    if !orphans.is_empty() {
        suite.warnings.push(format!(
            "{name}: {} encodings have no crates/<crate>/Cargo.toml (measured without rustc): {}",
            orphans.len(),
            orphans.join(", ")
        ));
    }
    load_expected_failures(&mut suite);
    check_families(&mut suite);
    suite
}

/// Drop the files whose `.vpr` is over the suite's limit (or the run-wide
/// one), remembering them so the run file can say what was left out.
fn apply_size_limit(suite: &mut Suite, default: Option<f64>) {
    let Some(limit) = suite.config.max_vpr_mb.or(default) else {
        return;
    };
    let mut kept = Vec::new();
    for f in std::mem::take(&mut suite.files) {
        let mb = f
            .vpr
            .as_ref()
            .and_then(|v| std::fs::metadata(v).ok())
            .map(|m| m.len() as f64 / 1e6);
        match mb {
            Some(mb) if mb > limit => suite.skipped_large.push((f.stem, mb)),
            _ => kept.push(f),
        }
    }
    suite.files = kept;
    if !suite.skipped_large.is_empty() {
        suite.warnings.push(format!(
            "{}: {} files over the {limit} MB .vpr limit skipped",
            suite.name,
            suite.skipped_large.len()
        ));
    }
}

fn empty_suite(name: &str, dir: &Path) -> Suite {
    Suite {
        name: name.to_string(),
        dir: dir.to_path_buf(),
        config: SuiteConfig::default(),
        families: Vec::new(),
        files: Vec::new(),
        expected_failures: BTreeSet::new(),
        skipped_large: Vec::new(),
        errors: Vec::new(),
        warnings: Vec::new(),
    }
}

fn load_flat(name: &str, dir: &Path) -> Suite {
    let mut suite = empty_suite(name, dir);
    load_config(&mut suite);
    suite.files = files_with_ext(dir, "vpr")
        .into_iter()
        .map(|vpr| SuiteFile {
            stem: stem_of(&vpr),
            rs: None,
            vpr: Some(vpr),
            krate: None,
        })
        .collect();
    check_families(&mut suite);
    suite
}

fn load_structured(name: &str, dir: &Path) -> Suite {
    let mut suite = empty_suite(name, dir);
    load_config(&mut suite);
    let mut by_stem: BTreeMap<String, SuiteFile> = BTreeMap::new();
    for rs in files_with_ext(&dir.join("src"), "rs") {
        let stem = stem_of(&rs);
        by_stem.insert(
            stem.clone(),
            SuiteFile {
                stem,
                rs: Some(rs),
                vpr: None,
                krate: None,
            },
        );
    }
    for vpr in files_with_ext(&dir.join("vpr"), "vpr") {
        let stem = stem_of(&vpr);
        let entry = by_stem.entry(stem.clone()).or_insert_with(|| SuiteFile {
            stem,
            rs: None,
            vpr: None,
            krate: None,
        });
        entry.vpr = Some(vpr);
    }
    suite.files = by_stem.into_values().collect();
    let unencoded: Vec<&str> = suite
        .files
        .iter()
        .filter(|f| f.vpr.is_none())
        .map(|f| f.stem.as_str())
        .collect();
    match unencoded.as_slice() {
        [] => {}
        [one] => suite.warnings.push(format!(
            "{name}: src/{one}.rs has no vpr/{one}.vpr (not encoded yet; skipped)"
        )),
        many => suite.warnings.push(format!(
            "{name}: {} sources have no .vpr (not encoded yet; skipped): {}",
            many.len(),
            many.join(", ")
        )),
    }

    load_expected_failures(&mut suite);
    check_families(&mut suite);
    suite
}

fn load_expected_failures(suite: &mut Suite) {
    let expected = suite.dir.join("expected_failures.txt");
    if let Ok(text) = std::fs::read_to_string(&expected) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next()) {
                (Some(stem), Some(member)) => {
                    suite
                        .expected_failures
                        .insert((stem.to_string(), member.to_string()));
                }
                _ => suite.errors.push(format!(
                    "{}: expected_failures.txt: malformed line `{line}`",
                    suite.name
                )),
            }
        }
    }
}

fn load_config(suite: &mut Suite) {
    let path = suite.dir.join("suite.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    match serde_json::from_str::<SuiteConfig>(&text) {
        Ok(c) => suite.config = c,
        Err(e) => suite
            .errors
            .push(format!("{}: suite.json: {e}", suite.name)),
    }
}

/// Compile the family patterns and hold the stems to them.
fn check_families(suite: &mut Suite) {
    let single = suite.config.family.is_some();
    let mut specs: Vec<Family> = suite.config.family.iter().cloned().collect();
    specs.extend(suite.config.families.iter().flatten().cloned());
    if single && suite.config.families.is_some() {
        suite.errors.push(format!(
            "{}: suite.json: give `family` or `families`, not both",
            suite.name
        ));
    }
    for (i, spec) in specs.into_iter().enumerate() {
        let name = match (&spec.name, single) {
            (Some(n), _) => n.clone(),
            (None, true) => suite.name.clone(),
            (None, false) => {
                suite.errors.push(format!(
                    "{}: suite.json: families[{i}] needs a `name`",
                    suite.name
                ));
                continue;
            }
        };
        let regex = match Regex::new(&format!("^(?:{})$", spec.pattern)) {
            Ok(r) => r,
            Err(e) => {
                suite
                    .errors
                    .push(format!("{}: family `{name}`: bad pattern: {e}", suite.name));
                continue;
            }
        };
        let groups: BTreeSet<&str> = regex.capture_names().flatten().collect();
        for k in &spec.knobs {
            if !groups.contains(k.as_str()) {
                suite.errors.push(format!(
                    "{}: family `{name}`: knob `{k}` is not a named group of the pattern",
                    suite.name
                ));
            }
        }
        suite.families.push(CompiledFamily {
            name,
            regex,
            knobs: spec.knobs,
        });
    }
    if suite.families.is_empty() {
        return;
    }

    let exhaustive = suite.config.exhaustive.unwrap_or(single);
    for fam in &suite.families {
        let matched = suite
            .files
            .iter()
            .filter(|f| fam.knobs_of(&f.stem).is_some())
            .count();
        if matched == 0 {
            suite.errors.push(format!(
                "{}: family `{}` matches no stem",
                suite.name, fam.name
            ));
        }
    }
    for f in &suite.files {
        let n = suite
            .families
            .iter()
            .filter(|fam| fam.knobs_of(&f.stem).is_some())
            .count();
        if n == 0 && exhaustive {
            suite.errors.push(format!(
                "{}: `{}` does not match the family pattern{}",
                suite.name,
                f.stem,
                if suite.families.len() > 1 { "s" } else { "" }
            ));
        } else if n > 1 {
            suite.errors.push(format!(
                "{}: `{}` matches more than one family",
                suite.name, f.stem
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, s: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, s).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bench-suites-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn discovers_layouts_and_pairs_by_stem() {
        let root = scratch("layout");
        write(&root.join("top.vpr"), "");
        write(&root.join("a/src/x.rs"), "");
        write(&root.join("a/vpr/x.vpr"), "");
        write(&root.join("a/src/only_rs.rs"), "");
        write(&root.join("a/vpr/only_vpr.vpr"), "");
        write(&root.join("a/expected_failures.txt"), "# c\nx m_x\n");
        write(&root.join("a/iso/l0.vpr"), "");
        write(&root.join("baseline/t.txt"), "");

        let suites = discover(&root);
        let names: Vec<&str> = suites.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a", "a/iso", "viper"]);
        let a = &suites[0];
        let stems: Vec<_> = a
            .files
            .iter()
            .map(|f| (f.stem.as_str(), f.rs.is_some(), f.vpr.is_some()))
            .collect();
        assert_eq!(
            stems,
            [
                ("only_rs", true, false),
                ("only_vpr", false, true),
                ("x", true, true)
            ]
        );
        assert_eq!(a.warnings.len(), 1);
        assert!(a.errors.is_empty());
        assert!(a.expected_failures.contains(&("x".into(), "m_x".into())));
        assert_eq!(a.measurable().count(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn crate_corpus_and_size_limit() {
        let root = scratch("crates");
        let corpus = scratch("corpus");
        write(&root.join("a/src/x.rs"), "");
        write(&root.join("a/vpr/x.vpr"), &"x".repeat(2000));
        write(
            &corpus.join("crates/small/Cargo.toml"),
            "[package]\nname = \"small\"\n",
        );
        write(&corpus.join("viper/small.vpr"), "x");
        write(
            &corpus.join("crates/big/Cargo.toml"),
            "[package]\nname = \"big\"\n",
        );
        write(&corpus.join("viper/big.vpr"), &"x".repeat(3000));
        write(&corpus.join("crates/unencoded/Cargo.toml"), "");
        write(&corpus.join("viper/orphan.vpr"), "x");

        let json = format!(
            r#"{{"crates": {{"path": {:?}, "max_vpr_mb": 0.0025, "runs": 1, "rustc_target": "aarch64-apple-darwin"}},
                "gone": {{"path": "no/such/dir"}}}}"#,
            corpus.to_string_lossy()
        );
        let discovery = Discovery {
            external: External::parse_all(&json, Path::new(".")).unwrap(),
            max_vpr_mb: Some(0.001),
        };
        let suites = discover_with(&root, &discovery);
        let names: Vec<&str> = suites.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a", "crates", "gone"]);

        // The run-wide limit applies where a suite sets none.
        assert_eq!(suites[0].files.len(), 0);
        assert_eq!(suites[0].skipped_large[0].0, "x");

        let c = &suites[1];
        assert!(c.errors.is_empty(), "{:?}", c.errors);
        let files: Vec<_> = c
            .files
            .iter()
            .map(|f| (f.stem.as_str(), f.krate.is_some(), f.rs.is_some()))
            .collect();
        assert_eq!(files, [("orphan", false, false), ("small", true, false)]);
        assert_eq!(c.skipped_large.len(), 1, "its own limit wins");
        assert_eq!(c.runs(5), 1);
        assert_eq!(c.warmup(1), 1);
        assert_eq!(
            c.config.rustc_target.as_deref(),
            Some("aarch64-apple-darwin")
        );
        assert!(c.warnings.iter().any(|w| w.contains("unencoded")));
        assert!(c.warnings.iter().any(|w| w.contains("orphan")));

        assert!(suites[2].errors[0].contains("no directory"));
        assert!(
            External::parse_all(r#"{"x": {"path": ".", "max_vpr": 1}}"#, Path::new("."))
                .unwrap_err()
                .contains("unknown field")
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&corpus);
    }

    #[test]
    fn family_patterns_are_checked() {
        let root = scratch("family");
        write(
            &root.join("g/suite.json"),
            r#"{"family": {"pattern": "loops_n(?P<nesting>\\d+)_b(?P<body>\\d+)", "knobs": ["nesting", "body"]}}"#,
        );
        write(&root.join("g/src/loops_n2_b5.rs"), "");
        write(&root.join("g/src/stray.rs"), "");
        write(&root.join("h/suite.json"), r#"{"timeuot_s": 3}"#);
        write(&root.join("h/src/a.rs"), "");

        let suites = discover(&root);
        let g = &suites[0];
        assert_eq!(g.errors.len(), 1, "{:?}", g.errors);
        assert!(g.errors[0].contains("stray"));
        let (fam, knobs) = g.family_of("loops_n2_b5").unwrap();
        assert_eq!(fam.name, "g");
        assert_eq!(knobs["nesting"], 2.0);
        assert_eq!(knobs["body"], 5.0);
        assert!(
            suites[1].errors[0].contains("unknown field"),
            "{:?}",
            suites[1].errors
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
