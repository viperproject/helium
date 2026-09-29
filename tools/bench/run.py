#!/usr/bin/env python3
"""Measure the current commit and record it on the `benchmarks` branch.

    python3 tools/bench/run.py                       # full run, commit + push results
    python3 tools/bench/run.py --no-push             # commit locally only
    python3 tools/bench/run.py --dry-run OUT_DIR     # any machine: write results to OUT_DIR
    python3 tools/bench/run.py --dry-run OUT -- --only rust/physics_step --runs 1

Steps: build `verify` and `bench` in release mode, validate every suite with
`bench check-suites` (errors stop the run; unencoded sources only warn), run
`bench run`, store the run file and its `index.json` entry, refresh the site in
`docs/`, warn when wall time is more than 15% slower than the previous run over
the files both share, then commit (and push) the results branch.

Only the configured benchmark host may write to the results branch
(`host` in config.json): timings from any other machine are not comparable.
Arguments after `--` go to `bench run` unchanged.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R  # noqa: E402


def fmt_ratio(x) -> str:
    return f"x{x:.2f}" if x else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", metavar="DIR", type=Path, help="write results to DIR; no host check, no git")
    ap.add_argument("--no-push", action="store_true", help="commit results locally, do not push")
    ap.add_argument("--allow-dirty", action="store_true", help="record a run of uncommitted changes")
    ap.add_argument("--no-build", action="store_true", help="use the existing release binaries")
    ap.add_argument("bench_args", nargs="*", help="passed to `bench run` (after --)")
    args = ap.parse_args()

    cfg = R.load_config()
    if not args.dry_run:
        R.check_host(cfg)
        if R.git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
            R.die("the working tree has uncommitted changes: commit them, or pass --allow-dirty")

    verify = R.REPO / "target" / "release" / f"verify{R.EXE}"
    if not args.no_build:
        verify = R.build(R.REPO)
    R.check_suites(R.REPO / "benchmarks", cfg)

    store = R.Store.open(cfg, args.dry_run)
    with tempfile.TemporaryDirectory(prefix="helium-bench-") as tmp:
        out = Path(tmp) / "run.json"
        run_data = R.bench_run(
            cfg,
            out,
            ["--verify", verify, "--repo", R.REPO] + args.bench_args,
            store.results / "silicon_cache.json",
        )

    index = store.load_index()
    prev = store.previous(index, run_data)
    entry = store.save_run(index, run_data)
    store.install_site()

    s = entry["summary"]
    R.log(
        f"{run_data['commit'][:10]}: {s['files']} files, {s['members']} members, coverage {s['coverage']}, "
        f"helium {s['totals']['helium_verify']:.2f}s, overhead vs rustc {fmt_ratio(s['geomean_overhead'])}, "
        f"speedup vs Silicon {fmt_ratio(s['geomean_speedup'])}"
    )
    if s["disagreements"]:
        R.log(f"warning: {s['disagreements']} members where Helium and Silicon disagree (see the run file)")
    for w in run_data.get("warnings", []):
        R.log(f"warning: {w}")
    slower = R.soft_check(prev, run_data, float(cfg.get("slowdown_warn", 0.15)))
    if slower:
        R.log("warning: slower than the previous run:")
        for w in slower:
            R.log(f"  {w}")

    subject = run_data.get("subject") or ""
    store.commit(f"bench: {run_data['commit'][:10]} {subject}".strip(), push=not args.no_push)


if __name__ == "__main__":
    main()
