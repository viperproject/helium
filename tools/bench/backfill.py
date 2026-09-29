#!/usr/bin/env python3
"""Replay the benchmark runner over older commits.

    python3 tools/bench/backfill.py --commits 10          # last 10 first-parent commits of main
    python3 tools/bench/backfill.py --commit abc123 ...   # specific commits
    python3 tools/bench/backfill.py --suite scaling --commits 20
    python3 tools/bench/backfill.py --dry-run OUT --commits 1 -- --runs 1

Each commit is checked out into a temporary worktree and its `verify` built
there (into a shared target directory, so consecutive commits build
incrementally). The *current* `bench` then measures that build: `bench` calls
`verify` as a separate process, and reads `verify --json` or, for builds that
predate it, the old text output. Viper metrics always come from the current
`verify`, so the shape columns stay comparable across commits.

Without `--suite`, a commit is measured on its own `benchmarks/`, so a trend
starts at the commit that added the file. With `--suite`, the named suites are
taken from the current tree and measured against every older build, then
merged into that commit's existing run: a new suite gets history too. Members
an old build cannot handle come out as UNSUPPORTED or ERROR, which is itself
coverage history.

The same host rule as `run.py` applies. Arguments after `--` go to `bench run`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits", type=int, help="the last N first-parent commits of --branch")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--commit", action="append", default=[], help="a specific commit (repeatable)")
    ap.add_argument("--suite", action="append", default=[], help="only this suite, from the current tree")
    ap.add_argument("--skip-existing", action="store_true", help="skip commits that already have a run")
    ap.add_argument("--dry-run", metavar="DIR", type=Path, help="write results to DIR; no host check, no git")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("bench_args", nargs="*", help="passed to `bench run` (after --)")
    args = ap.parse_args()

    cfg = R.load_config()
    if not args.dry_run:
        R.check_host(cfg)

    commits = [R.git("rev-parse", c) for c in args.commit]
    if args.commits:
        listed = R.git("rev-list", "--first-parent", "-n", str(args.commits), args.branch).split()
        commits += list(reversed(listed))  # oldest first
    if not commits:
        R.die("nothing to do: give --commits N or --commit SHA")

    # The current tree's tools: bench itself, and verify for Viper metrics.
    current_verify = R.build(R.REPO)
    if args.suite:
        R.check_suites(R.REPO / "benchmarks", cfg)

    store = R.Store.open(cfg, args.dry_run)
    index = store.load_index()
    target_dir = (R.REPO / cfg.get("backfill_target_dir", "target/backfill")).resolve()
    done = []

    for sha in commits:
        short = sha[:10]
        existing = store.entry_for(index, sha)
        if existing and args.skip_existing and not args.suite:
            R.log(f"{short}: already measured, skipping")
            continue
        with tempfile.TemporaryDirectory(prefix="helium-backfill-") as tmp:
            wt = Path(tmp) / "wt"
            R.git("worktree", "add", "--detach", str(wt), sha)
            try:
                try:
                    verify = R.build(wt, target_dir=target_dir, bench=False)
                except SystemExit:
                    R.log(f"{short}: build failed, skipping")
                    continue
                benchmarks = R.REPO / "benchmarks" if args.suite else wt / "benchmarks"
                extra = [
                    "--verify", verify,
                    "--metrics-verify", current_verify,
                    "--repo", wt,
                    "--commit", sha,
                    "--benchmarks", benchmarks,
                ]
                for s in args.suite:
                    extra += ["--suite", s]
                run_data = R.bench_run(
                    cfg, Path(tmp) / "run.json", extra + args.bench_args, store.results
                )
            finally:
                R.git("worktree", "remove", "--force", str(wt), check=False)

        if args.suite and existing:
            old = store.load_run(existing)
            if old:
                run_data = R.merge_runs(old, run_data)
        entry = store.save_run(index, run_data)
        s = entry["summary"]
        R.log(f"{short}: {s['files']} files, coverage {s['coverage']}, helium {s['totals']['helium_verify']:.2f}s")
        done.append(short)

    if done:
        store.install_site()
        what = f" ({', '.join(args.suite)})" if args.suite else ""
        store.commit(f"bench: backfill {len(done)} commits{what}", push=not args.no_push)


if __name__ == "__main__":
    main()
