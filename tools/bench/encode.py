#!/usr/bin/env python3
"""Encode a suite's Rust sources to Viper with Prusti.

    python3 tools/bench/encode.py scaling               # every src/*.rs not yet encoded
    python3 tools/bench/encode.py scaling loops_n2_b5   # one stem
    python3 tools/bench/encode.py --force rust          # re-encode everything

For each `benchmarks/<suite>/src/<stem>.rs` this runs `tools/prusti_encode.sh
<src> <vpr>` (override with $PRUSTI_ENCODE), writing `vpr/<stem>.vpr`. A source
is skipped when its .vpr is newer than it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--force", action="store_true", help="re-encode even when up to date")
    args = ap.parse_args()

    suite = REPO / "benchmarks" / args.suite
    src, vpr_dir = suite / "src", suite / "vpr"
    if not src.is_dir():
        sys.exit(f"no {src}")
    vpr_dir.mkdir(exist_ok=True)
    encoder = os.environ.get("PRUSTI_ENCODE", str(REPO / "tools" / "prusti_encode.sh"))
    sources = [src / f"{s.removesuffix('.rs')}.rs" for s in args.stems] or sorted(src.glob("*.rs"))
    # On Windows a bare "bash" resolves to WSL's System32\bash.exe before PATH, so
    # resolve it via PATH; and pass forward slashes, which bash does not unescape.
    bash = shutil.which("bash") or "bash"

    failed = []
    for rs in sources:
        stem = rs.stem
        out = vpr_dir / f"{stem}.vpr"
        if not args.force and out.exists() and out.stat().st_mtime > rs.stat().st_mtime:
            print(f"[skip] {stem} (up to date)")
            continue
        print(f"[encode] {stem}", flush=True)
        r = subprocess.run([bash, Path(encoder).as_posix(), rs.as_posix(), out.as_posix()])
        if r.returncode != 0 and not out.exists():
            print(f"[FAILED] {stem}", file=sys.stderr)
            failed.append(stem)
    if failed:
        sys.exit(f"encode failures: {' '.join(failed)}")


if __name__ == "__main__":
    main()
