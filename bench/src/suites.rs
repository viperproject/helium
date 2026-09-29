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
    suites.sort_by(|a, b| a.name.cmp(&b.name));
    suites
}

fn empty_suite(name: &str, dir: &Path) -> Suite {
    Suite {
        name: name.to_string(),
        dir: dir.to_path_buf(),
        config: SuiteConfig::default(),
        families: Vec::new(),
        files: Vec::new(),
        expected_failures: BTreeSet::new(),
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
            },
        );
    }
    for vpr in files_with_ext(&dir.join("vpr"), "vpr") {
        let stem = stem_of(&vpr);
        let entry = by_stem.entry(stem.clone()).or_insert_with(|| SuiteFile {
            stem,
            rs: None,
            vpr: None,
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

    let expected = dir.join("expected_failures.txt");
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
                    "{name}: expected_failures.txt: malformed line `{line}`"
                )),
            }
        }
    }
    check_families(&mut suite);
    suite
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
