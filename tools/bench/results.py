"""Shared logic for the benchmark pipeline: summaries, scaling fits, the
results store on the `benchmarks` branch, and the soft slowdown check.

Used by `run.py` (one run at the current commit) and `backfill.py` (older
commits). Standard library only. See `plans/regression-pipeline.md`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SITE = HERE / "site"
RESULTS_SUBDIR = Path("benchmarks") / "results"

EXE = ".exe" if os.name == "nt" else ""


def log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[bench] error: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd, cwd=None, check=True, capture=False, env=None):
    """Run a command, echoing it. Returns stdout when `capture`."""
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
    )
    if check and r.returncode != 0:
        die(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r.stdout if capture else r.returncode


def git(*args, cwd=REPO, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, text=True, encoding="utf-8", capture_output=True)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


# ── Configuration ──


def load_config() -> dict:
    """`config.json`, overlaid with the untracked `config.local.json`."""
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    local = HERE / "config.local.json"
    if local.exists():
        cfg.update(json.loads(local.read_text(encoding="utf-8")))
    return cfg


def check_host(cfg: dict) -> str:
    """Refuse to write results from any machine but the benchmark host, so a
    stray laptop run cannot pollute the history."""
    host = os.environ.get("BENCH_HOST") or socket.gethostname()
    want = cfg.get("host")
    if not want:
        die(
            "no benchmark host configured: set \"host\" in tools/bench/config.json to the "
            f"dedicated benchmark machine's name (this machine is `{host}`), or use --dry-run"
        )
    if host != want:
        die(f"this is `{host}`, not the benchmark host `{want}`: refusing to write results (use --dry-run)")
    return host


# ── Building and running `bench` ──


def build(repo: Path, target_dir: Path | None = None, bench: bool = True) -> Path:
    """Build `verify` (and `bench`) in release mode; return the verify path."""
    env = dict(os.environ)
    if target_dir:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    run(["cargo", "build", "--release", "--bin", "verify"], cwd=repo, env=env)
    if bench:
        run(["cargo", "build", "--release", "-p", "bench"], cwd=repo, env=env)
    return (target_dir or repo / "target") / "release" / f"verify{EXE}"


def bench_exe() -> Path:
    return REPO / "target" / "release" / f"bench{EXE}"


def check_suites(benchmarks: Path, cfg: dict) -> None:
    # The external suites and `max_vpr_mb` are read by bench itself, from the
    # same config.json / config.local.json.
    cmd = [bench_exe(), "check-suites", "--benchmarks", benchmarks]
    if cfg.get("rustc_toolchain"):
        cmd += ["--rustc-toolchain", cfg["rustc_toolchain"]]
    if run(cmd, cwd=REPO, check=False) != 0:
        die("check-suites found errors; fix them before measuring")


def bench_run(cfg: dict, out: Path, extra: list, caches: Path | None) -> dict:
    """Run `bench run` with the configured tools; return the run JSON. `caches`
    is the directory holding the rustc and Silicon caches (reused across runs:
    neither tool's result depends on the commit being measured)."""
    cmd = [bench_exe(), "run", "--out", out, "--warmup", cfg.get("warmup", 1), "--runs", cfg.get("runs", 5)]
    if cfg.get("rustc_toolchain"):
        cmd += ["--rustc-toolchain", cfg["rustc_toolchain"]]
    else:
        log("warning: rustc_toolchain is not pinned in config.json; rust-toolchain.toml decides")
    if caches:
        cmd += ["--rustc-cache", caches / "rustc_cache.json"]
    if cfg.get("timeout_s"):
        cmd += ["--timeout", cfg["timeout_s"]]
    if cfg.get("silicon_jar"):
        cmd += ["--silicon-jar", cfg["silicon_jar"], "--java", cfg.get("java", "java")]
        for a in cfg.get("jvm_args", []):
            cmd += ["--jvm-arg", a]
        for a in cfg.get("silicon_args", []):
            cmd += ["--silicon-arg", a]
        if caches:
            cmd += ["--silicon-cache", caches / "silicon_cache.json"]
        warm = cfg.get("silicon_warm")
        if warm:
            for d in warm.get("corpus", []):
                cmd += ["--silicon-warm", REPO / d]
            if warm.get("warmup_s") is not None:
                cmd += ["--silicon-warmup-s", warm["warmup_s"]]
            if warm.get("file_timeout_s") is not None:
                cmd += ["--silicon-warmup-file-timeout", warm["file_timeout_s"]]
    cmd += extra
    run(cmd, cwd=REPO)
    return json.loads(out.read_text(encoding="utf-8"))


# ── Summaries ──


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def median_of(t):
    """The median of a timing column, or None unless it finished."""
    if not t or t.get("status") not in (None, "ok"):
        return None
    return t.get("median")


def summarize(run_data: dict) -> dict:
    """The summary numbers kept in `index.json`."""
    files = run_data.get("files", [])
    coverage: dict = {}
    by_suite: dict = {}
    # Self-reported times only: no total counts process or JVM startup.
    totals = {"helium_verify": 0.0, "rustc_self": 0.0, "silicon_verify": 0.0, "silicon_warm": 0.0}
    overhead, speedup, speedup_warm = [], [], []
    members = disagreements = timeouts = file_errors = 0
    for f in files:
        t = f.get("times", {})
        s = by_suite.setdefault(f["suite"], {"files": 0, "helium_verify": 0.0, "coverage": {}})
        s["files"] += 1
        for status, n in f.get("coverage", {}).items():
            coverage[status] = coverage.get(status, 0) + n
            s["coverage"][status] = s["coverage"].get(status, 0) + n
        for col in totals:
            m = median_of(t.get(col))
            if m is not None:
                totals[col] += m
        h = median_of(t.get("helium_verify"))
        if h is not None:
            s["helium_verify"] += h
        r = median_of(t.get("rustc_self"))
        sv = median_of(t.get("silicon_verify"))
        if h and r:
            overhead.append(h / r)
        if h and sv:
            speedup.append(sv / h)
        sw = median_of(t.get("silicon_warm"))
        if h and sw:
            speedup_warm.append(sw / h)
        members += len(f.get("members", []))
        disagreements += sum(1 for m in f.get("members", []) if m.get("disagreement"))
        timeouts += sum(1 for v in t.values() if v and v.get("status") == "timeout")
        file_errors += 1 if f.get("helium_error") else 0
    return {
        "files": len(files),
        "members": members,
        "coverage": coverage,
        "suites": by_suite,
        "totals": totals,
        "geomean_overhead": geomean(overhead),
        "geomean_speedup": geomean(speedup),
        "geomean_speedup_warm": geomean(speedup_warm),
        "disagreements": disagreements,
        "timeouts": timeouts,
        "file_errors": file_errors,
    }


def _fit(xs, ys):
    """Least squares y = a + b x; returns (b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys))
    return b, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def scaling_fits(run_data: dict, metric: str = "helium_verify") -> list:
    """For every generated family and knob, fit time against the knob with the
    other knobs held fixed: as a power law (log t vs log k: exponent) and as an
    exponential (log t vs k: growth factor per step). The better fit by R² is
    marked, so "went from exponential to polynomial" shows up per commit."""
    groups: dict = {}
    for f in run_data.get("files", []):
        if not f.get("family") or not f.get("knobs"):
            continue
        y = median_of(f.get("times", {}).get(metric))
        if not y or y <= 0:
            continue
        for knob, x in f["knobs"].items():
            fixed = tuple(sorted((k, v) for k, v in f["knobs"].items() if k != knob))
            groups.setdefault((f["suite"], f["family"], knob, fixed), []).append((x, y))
    fits = []
    for (suite, family, knob, fixed), pts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        pts = [(x, y) for x, y in pts if x > 0]
        if len({x for x, _ in pts}) < 3:
            continue
        xs, ys = [x for x, _ in pts], [math.log(y) for _, y in pts]
        k, r2p = _fit([math.log(x) for x in xs], ys)
        b, r2e = _fit(xs, ys)
        if k is None:
            continue
        fits.append(
            {
                "suite": suite,
                "family": family,
                "knob": knob,
                "fixed": dict(fixed),
                "n": len(pts),
                "power": {"exponent": k, "r2": r2p},
                "exp": {"factor": math.exp(b), "r2": r2e},
                "best": "power" if r2p >= r2e else "exp",
            }
        )
    return fits


def scaling_summary(fits: list) -> dict:
    """Per `suite/family/knob`: the median exponent and factor over the groups."""
    out: dict = {}
    for f in fits:
        out.setdefault(f"{f['suite']}/{f['family']}/{f['knob']}", []).append(f)
    summary = {}
    for key, fs in out.items():
        med = lambda xs: sorted(xs)[len(xs) // 2]
        summary[key] = {
            "exponent": med([f["power"]["exponent"] for f in fs]),
            "factor": med([f["exp"]["factor"] for f in fs]),
            "best": max(("power", "exp"), key=lambda b: sum(1 for f in fs if f["best"] == b)),
            "groups": len(fs),
        }
    return summary


# ── Comparisons ──


def file_key(f):
    return f"{f['suite']}/{f['stem']}"


def fingerprint(f):
    s = f.get("sha256", {})
    return (s.get("rs"), s.get("vpr"))


def fair_compare(prev: dict, cur: dict, metric: str = "helium_verify") -> dict:
    """Totals over the files both runs share with identical inputs: adding a
    suite, or re-encoding a file, is not a speed change."""
    pf = {file_key(f): f for f in prev.get("files", [])}
    cf = {file_key(f): f for f in cur.get("files", [])}
    common, changed_input = [], []
    for k in sorted(pf.keys() & cf.keys()):
        (common if fingerprint(pf[k]) == fingerprint(cf[k]) else changed_input).append(k)
    tp = tc = 0.0
    per_file = []
    for k in common:
        a = median_of(pf[k].get("times", {}).get(metric))
        b = median_of(cf[k].get("times", {}).get(metric))
        if a and b:
            tp += a
            tc += b
            per_file.append((k, a, b))
    return {
        "common": len(common),
        "added": sorted(cf.keys() - pf.keys()),
        "removed": sorted(pf.keys() - cf.keys()),
        "changed_input": changed_input,
        "prev_total": tp,
        "cur_total": tc,
        "ratio": (tc / tp) if tp > 0 else None,
        "per_file": per_file,
    }


def soft_check(prev: dict | None, cur: dict, threshold: float) -> list:
    """Warnings when this run is more than `threshold` slower than the last one
    (wall time over the shared files). Never fatal: timings are noisy, and the
    exact counter check lives in tests/perf_regression.rs."""
    if not prev:
        return []
    warnings = []
    for metric in ("helium_verify", "helium_wall"):
        c = fair_compare(prev, cur, metric)
        if c["ratio"] and c["ratio"] > 1 + threshold:
            warnings.append(
                f"{metric}: {c['cur_total']:.2f}s vs {c['prev_total']:.2f}s over {c['common']} shared files "
                f"(+{(c['ratio'] - 1) * 100:.0f}%) since {prev['commit'][:10]}"
            )
    c = fair_compare(prev, cur)
    worst = sorted(
        ((k, b / a) for k, a, b in c["per_file"] if a > 0.05), key=lambda kv: kv[1], reverse=True
    )[:5]
    for k, r in worst:
        if r > 1 + threshold:
            warnings.append(f"  {k}: x{r:.2f}")
    return warnings


# ── The results store (a worktree of the `benchmarks` branch) ──


class Store:
    """`benchmarks/results/` and `docs/` in a checkout of the results branch,
    or a plain directory for --dry-run."""

    def __init__(self, root: Path, git_backed: bool, cfg: dict):
        self.root = root
        self.git_backed = git_backed
        self.cfg = cfg
        self.results = root / RESULTS_SUBDIR

    @classmethod
    def open(cls, cfg: dict, dry_run_dir: Path | None) -> "Store":
        if dry_run_dir:
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            return cls(dry_run_dir, False, cfg)
        branch = cfg.get("results_branch", "benchmarks")
        remote = cfg.get("remote", "origin")
        wt = (REPO / cfg.get("results_worktree", "../helium-bench-results")).resolve()
        if not wt.exists():
            if git("ls-remote", "--heads", remote, branch, check=False):
                git("fetch", remote, f"{branch}:{branch}", check=False)
            if git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False):
                git("worktree", "add", str(wt), branch)
            else:
                log(f"creating orphan branch `{branch}` for results")
                git("worktree", "add", "--detach", str(wt))
                git("checkout", "--orphan", branch, cwd=wt)
                git("rm", "-rf", "--quiet", ".", cwd=wt, check=False)
        elif git("rev-parse", "--abbrev-ref", "@{upstream}", cwd=wt, check=False):
            git("pull", "--ff-only", cwd=wt)
        return cls(wt, True, cfg)

    # index.json: one entry per run, oldest commit first.
    def load_index(self) -> dict:
        p = self.results / "index.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"schema": 1, "repo_url": self.cfg.get("repo_url"), "runs": []}

    def load_run(self, entry: dict) -> dict | None:
        p = self.results / entry["file"]
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def entry_for(self, index: dict, commit: str):
        return next((e for e in index["runs"] if e["commit"] == commit), None)

    def previous(self, index: dict, run_data: dict):
        """The latest run before this one's commit (by commit date)."""
        key = run_data.get("commit_date") or run_data["date"]
        earlier = [e for e in index["runs"] if e["commit"] != run_data["commit"] and (e.get("commit_date") or e["date"]) <= key]
        return self.load_run(earlier[-1]) if earlier else None

    def save_run(self, index: dict, run_data: dict) -> dict:
        """Write the run file and its index entry (replacing an earlier run of
        the same commit)."""
        old = self.entry_for(index, run_data["commit"])
        name = f"runs/{run_data['date'][:10]}_{run_data['commit'][:12]}.json"
        fits = scaling_fits(run_data)
        run_data["scaling"] = fits
        path = self.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_data, separators=(",", ":")), encoding="utf-8")
        if old and old["file"] != name and (self.results / old["file"]).exists():
            (self.results / old["file"]).unlink()
        entry = {
            "commit": run_data["commit"],
            "commit_date": run_data.get("commit_date"),
            "date": run_data["date"],
            "subject": run_data.get("subject"),
            "host": run_data.get("host"),
            "dirty": run_data.get("dirty", False),
            "file": name,
            "tools": run_data.get("tools", {}),
            "summary": summarize(run_data),
            "scaling": scaling_summary(fits),
        }
        index["runs"] = [e for e in index["runs"] if e["commit"] != run_data["commit"]] + [entry]
        index["runs"].sort(key=lambda e: e.get("commit_date") or e["date"])
        if self.cfg.get("repo_url"):
            index["repo_url"] = self.cfg["repo_url"]
        (self.results / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        return entry

    def install_site(self) -> None:
        """Copy the static site to docs/, plus a root redirect so GitHub Pages
        can serve the branch root (the site reads ../benchmarks/results/)."""
        docs = self.root / "docs"
        if docs.exists():
            shutil.rmtree(docs)
        shutil.copytree(SITE, docs)
        (self.root / ".nojekyll").write_text("", encoding="utf-8")
        (self.root / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Helium benchmarks</title>'
            '<meta http-equiv="refresh" content="0; url=docs/">'
            '<a href="docs/">Helium benchmarks</a>\n',
            encoding="utf-8",
        )

    def commit(self, message: str, push: bool) -> None:
        if not self.git_backed:
            log(f"dry run: results left in {self.root}")
            return
        git("add", "-A", ".", cwd=self.root)
        if not git("status", "--porcelain", cwd=self.root):
            log("results unchanged; nothing to commit")
            return
        git("commit", "-q", "-m", message, cwd=self.root)
        log(f"committed results on `{self.cfg.get('results_branch', 'benchmarks')}` in {self.root}")
        if push:
            remote = self.cfg.get("remote", "origin")
            git("push", "-u", remote, self.cfg.get("results_branch", "benchmarks"), cwd=self.root)


def merge_runs(existing: dict, new: dict) -> dict:
    """Fold a run of a few suites (`backfill.py --suite`) into the existing run
    of the same commit: its files replace that suite's files, the rest stay."""
    suites = set(new.get("suites", {}))
    merged = dict(existing)
    merged["files"] = [f for f in existing.get("files", []) if f["suite"] not in suites] + new.get("files", [])
    merged["files"].sort(key=lambda f: (f["suite"], f["stem"]))
    merged["suites"] = {**existing.get("suites", {}), **new.get("suites", {})}
    merged["warnings"] = existing.get("warnings", []) + new.get("warnings", [])
    tools = dict(existing.get("tools", {}))
    for k, v in new.get("tools", {}).items():
        if v and not tools.get(k):
            tools[k] = v
    merged["tools"] = tools
    return merged
