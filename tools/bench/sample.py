#!/usr/bin/env python3
"""Measure a random sample of the benchmarks, Helium and Silicon on each file,
for as long as it is left running.

    python3 tools/bench/sample.py OUT_DIR -- --suite cratesTop100Split
    python3 tools/bench/sample.py OUT_DIR -- --suite cratesTop100Split --seed 7
    python3 tools/bench/sample.py OUT_DIR --no-build -- --suite cratesTop100Split --limit 100

Builds `verify` and `bench` in release mode, then runs `bench sample` with the
tools of config.json / config.local.json (the Silicon jar, its JVM and warm-up
corpus, the timeout). The files are measured in a random order, each result is
stored in OUT_DIR as soon as it is measured, and OUT_DIR/summary.json holds the
estimates so far (with 95% confidence intervals): stop it at any time (Ctrl-C)
and run the same command again to continue. See `benchmarks/README.md`.

Timed runs per tool and file come from `sample` in config.json (default: one,
no warm-up: a sample wants many files more than repeated runs). Nothing is
written to the results branch: a sample is not a run of the whole corpus.
Arguments after `--` go to `bench sample` unchanged (and win over the config).
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the sample's directory; an existing sample there is continued")
    ap.add_argument("--no-build", action="store_true", help="use the existing release binaries")
    ap.add_argument("bench_args", nargs="*", help="passed to `bench sample` (after --)")
    args = ap.parse_args()

    cfg = R.load_config()
    verify = R.REPO / "target" / "release" / f"verify{R.EXE}"
    if not args.no_build:
        verify = R.build(R.REPO)

    sample_cfg = cfg.get("sample", {})
    cmd = [
        R.bench_exe(), "sample", "--out", args.out.resolve(), "--verify", verify, "--repo", R.REPO,
        "--warmup", sample_cfg.get("warmup", 0), "--runs", sample_cfg.get("runs", 1),
    ]
    cmd += R.tool_args(cfg, None)
    cmd += args.bench_args
    # Ctrl-C is for `bench`: it drops the file in progress and stops cleanly.
    # A handler, not SIG_IGN: an ignored signal would stay ignored in `bench`.
    signal.signal(signal.SIGINT, lambda *_: None)
    sys.exit(R.run(cmd, cwd=R.REPO, check=False))


if __name__ == "__main__":
    main()
