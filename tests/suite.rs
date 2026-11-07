use silver_oxide::pipeline;
use std::path::{Path, PathBuf};

fn cases_dir() -> PathBuf {
    // Env var override for CI / unusual layouts.
    if let Ok(dir) = std::env::var("SILVER_CASES_DIR") {
        return PathBuf::from(dir);
    }
    // The corpus is version-controlled with this crate under `tests/cases/`.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases")
}

fn vpr_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_vpr_files(dir, &mut files);
    files.sort();
    files
}

fn collect_vpr_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|_| panic!("cannot read dir {}", dir.display()));
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_vpr_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("vpr") {
            out.push(path);
        }
    }
}

fn file_name(p: &Path) -> &str {
    p.file_name().and_then(|s| s.to_str()).unwrap_or("?")
}

/// Passing cases: the full pipeline must succeed and every method must verify.
#[test]
fn passing_cases_all_verify() {
    let dir = cases_dir().join("passing");
    let files = vpr_files(&dir);
    assert!(!files.is_empty(), "no .vpr files found in cases/passing/");

    let mut ok = 0usize;
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let name = file_name(path);
        total += 1;
        match pipeline::run_file(path) {
            Err(e) => {
                failures.push(format!("  [PIPELINE-ERROR] {name}: {e}"));
            }
            Ok(results) if results.is_empty() => {
                failures.push(format!("  [NO-METHODS] {name}: no method bodies found"));
            }
            Ok(results) => {
                let mut file_ok = true;
                for (method, outcome) in &results {
                    if !outcome.is_ok() {
                        failures.push(format!("  [{}] {name}::{method}: {outcome}", outcome.tag()));
                        file_ok = false;
                    }
                }
                if file_ok {
                    ok += 1;
                }
            }
        }
    }

    println!("=== passing/ ({ok}/{total} OK) ===");
    for path in &files {
        let name = file_name(path);
        let msg = failures.iter().find(|f| f.contains(name));
        match msg {
            Some(f) => println!("{f}"),
            None => println!("  [OK] {name}"),
        }
    }

    assert!(
        failures.is_empty(),
        "{} passing case(s) not verified",
        failures.len()
    );
}

/// Failing cases: the pipeline must either error or at least one method must
/// fail verification. A case that unexpectedly fully verifies is a test failure.
#[test]
fn failing_cases_are_rejected() {
    let dir = cases_dir().join("failing");
    let files = vpr_files(&dir);
    assert!(!files.is_empty(), "no .vpr files found in cases/failing/");

    let mut ok = 0usize;
    let mut total = 0usize;
    let mut surprises: Vec<String> = Vec::new();

    for path in &files {
        let name = file_name(path);
        total += 1;
        let rejected = match pipeline::run_file(path) {
            Err(e) => {
                println!("  [PIPELINE-ERROR-OK] {name}: {e}");
                true
            }
            Ok(results) => results.iter().any(|(_, r)| !r.is_ok()),
        };
        if rejected {
            ok += 1;
        } else {
            surprises.push(format!(
                "  [UNEXPECTED-OK] {name}: verified but expected rejection"
            ));
        }
    }

    println!("\n=== failing/ ({ok}/{total} correctly rejected) ===");
    for s in &surprises {
        println!("{s}");
    }

    assert!(
        surprises.is_empty(),
        "{} failing case(s) unexpectedly verified:\n{}",
        surprises.len(),
        surprises.join("\n")
    );
}

/// Known-limitation canaries: cases that *should* verify but currently don't,
/// due to a characterized incompleteness (not a soundness rejection — those
/// belong in `failing/`). Each file documents its root cause in a header
/// comment. This test asserts the CURRENT (undesired) failure so a fix that
/// makes one start passing breaks the build here, prompting a promotion into
/// `passing/` and an update to whatever doc the file's header points at,
/// rather than silently going stale.
#[test]
fn known_limitations_still_fail() {
    let dir = cases_dir().join("known_limitations");
    if !dir.exists() {
        return;
    }
    let files = vpr_files(&dir);

    let mut newly_passing: Vec<String> = Vec::new();

    for path in &files {
        let name = file_name(path);
        let still_fails = match pipeline::run_file(path) {
            Err(_) => true,
            Ok(results) => results.iter().any(|(_, r)| !r.is_ok()),
        };
        if still_fails {
            println!("  [STILL-FAILING-OK] {name}");
        } else {
            newly_passing.push(name.to_string());
        }
    }

    assert!(
        newly_passing.is_empty(),
        "{} known-limitation case(s) now verify — promote to tests/cases/passing/ \
         and update the tracking doc referenced in the file's header:\n{}",
        newly_passing.len(),
        newly_passing.join("\n")
    );
}

/// Unsupported-construct cases: each file uses at least one construct this
/// verifier does not implement. The pipeline must survive it — the file still
/// parses and lowers, and its other members are still verified — and the
/// offending declaration must be reported as `Unsupported`, naming the
/// construct, never as a verified member and never as a parse error.
#[test]
fn unsupported_cases_are_named_and_survived() {
    let dir = cases_dir().join("unsupported");
    if !dir.exists() {
        return;
    }
    let files = vpr_files(&dir);
    assert!(
        !files.is_empty(),
        "no .vpr files found in cases/unsupported/"
    );

    let mut failures: Vec<String> = Vec::new();
    for path in &files {
        let name = file_name(path);
        match pipeline::run_file(path) {
            Err(e) => failures.push(format!(
                "  [PIPELINE-ERROR] {name}: {e} (an unsupported construct must not \
                 take the file down)"
            )),
            Ok(results) => {
                if !results
                    .iter()
                    .any(|(_, s)| matches!(s, pipeline::MemberStatus::Unsupported(_)))
                {
                    failures.push(format!(
                        "  [NOT-REPORTED] {name}: no member was reported as unsupported"
                    ));
                }
                for (member, status) in &results {
                    println!("  [{}] {name}::{member}: {status}", status.tag());
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} unsupported case(s) mishandled:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The two properties the reporting exists for, on one file: a rejected
/// declaration does not stop its neighbours from being verified, and a
/// declaration that depends on a rejected one is reported as skipped rather
/// than verified.
#[test]
fn rejection_is_local_but_infects_dependents() {
    let path = cases_dir().join("unsupported/dependent_is_skipped.vpr");
    let results = pipeline::run_file(&path).expect("file must still lower");
    let status = |name: &str| {
        results
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("no row for `{name}` in {results:?}"))
    };
    assert!(
        matches!(status("p"), pipeline::MemberStatus::Unsupported(_)),
        "the predicate using a magic wand must be reported unsupported"
    );
    assert!(
        matches!(status("use_p"), pipeline::MemberStatus::Skipped { .. }),
        "a method whose precondition names a rejected predicate must not verify"
    );

    let path = cases_dir().join("unsupported/rest_of_file_still_verifies.vpr");
    let results = pipeline::run_file(&path).expect("file must still lower");
    assert!(
        results.iter().any(|(n, s)| n == "fine" && s.is_ok()),
        "the untouched method must still verify: {results:?}"
    );
}
