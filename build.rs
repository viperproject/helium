//! Records the git commit the binaries are built from, for `verify --json`.
//!
//! Sets `HELIUM_GIT_COMMIT` (a full hash, or `unknown` outside a git checkout)
//! and `HELIUM_GIT_DIRTY` (`true` when tracked files differ from that commit at
//! build time). The benchmark runner records the commit it measures on its own
//! as well; this lets a lone JSON file say which build produced it.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    println!("cargo:rustc-env=HELIUM_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=HELIUM_GIT_DIRTY={dirty}");

    // Rerun when HEAD moves (checkout, commit) or the sources change (dirty).
    // `--git-path` resolves correctly inside worktrees, where `.git` is a file.
    // Only paths that exist: cargo treats a missing one as always changed.
    let mut watched = vec!["HEAD".to_string(), "index".into(), "packed-refs".into()];
    watched.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for path in &watched {
        if let Some(p) = git(&["rev-parse", "--git-path", path]) {
            if std::path::Path::new(&p).exists() {
                println!("cargo:rerun-if-changed={p}");
            }
        }
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
