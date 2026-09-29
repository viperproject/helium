# Benchmarks

Everything here is measured by the benchmark pipeline (`bench`, `tools/bench/`;
design in `plans/regression-pipeline.md`), which records, per commit, how long
Helium takes on each file next to rustc and Silicon, how the verifier's counters
move, and what shape each program has. Results live on the `benchmarks` branch
(`benchmarks/results/`) and are browsable on its GitHub Pages site (`docs/`).

`*.vpr` directly in this directory are also the exact-counter gate of
`tests/perf_regression.rs` (baselines in `baseline/`); the pipeline measures them
as the Viper-only suite `viper`.

## Suites

A **suite** is any directory here laid out as

```
benchmarks/<suite>/
  src/<stem>.rs             Rust source (optional: a suite may be Viper-only)
  vpr/<stem>.vpr            its Prusti encoding (committed; measuring needs no Prusti)
  suite.json                optional settings, see below
  expected_failures.txt     optional: `<stem> <member>` per line, `#` comments
```

- Suites are found by scanning `benchmarks/*/`. There is no central list.
- Files pair by stem: `src/foo.rs` goes with `vpr/foo.vpr`.
- A `.vpr` with no `.rs` is Viper-only: no rustc columns, no Rust metrics, left
  out of the rustc overhead number. `.vpr` files directly in a directory (this
  one, `panic_free/isolate/`) form Viper-only suites (`viper`,
  `panic_free/isolate`).
- A `.rs` with no `.vpr` is not encoded yet: a warning, never an error.

| suite | what |
|---|---|
| `rust/` | realistic spec-less Rust, no loops; `depth_*`, `enum_*` generated families |
| `loops/` | loops with Prusti-inferred permission invariants |
| `panic_free/` | panic-freedom tiers (`isolate/`: hand-reduced Viper) |
| `scaling/` | generated families, one knob each (`gen_*.py`) |
| `viper` (this dir) | hand-written Viper, also the `perf_regression` gate |

### `suite.json`

All fields optional; unknown fields are an error (a typo must not fall back to a
default silently).

```json
{
  "description": "Loops with nested bodies",
  "rustc_args": ["--edition", "2021", "--crate-type", "lib"],
  "timeout_s": 300,
  "silicon": true,
  "tags": ["loops"],
  "family": {
    "pattern": "loops_n(?P<nesting>\\d+)_b(?P<body>\\d+)",
    "knobs": ["nesting", "body"]
  }
}
```

`family` marks a generated suite: every stem must match the pattern, and the
knob values are read from the stem, so a new generator gets scaling charts on the
site with no site change. A suite holding several families gives
`"families": [{"name": ..., "pattern": ..., "knobs": [...]}, ...]` instead; each
must match at least one stem, and with `"exhaustive": true` every stem must match
one (see `scaling/suite.json`, and `rust/suite.json` for families mixed with
hand-written files).

## Adding a suite

1. Create `benchmarks/<suite>/src/` with the Rust sources (or a generator that
   writes them, following `rust/gen_depth.py`: one knob varies, the rest fixed).
2. Encode them: `python3 tools/bench/encode.py <suite>` (runs
   `tools/prusti_encode.sh` per source; set `PRUSTI_RUSTC` to the `prusti-rustc`
   executable). Commit `vpr/`.
3. Optionally add `suite.json` (description, tags, `family` for a generated suite).
4. `cargo run --release -p bench -- check-suites` — it reports missing `.vpr`s, a
   `suite.json` that does not parse, a family pattern that does not match, `.rs`
   files rustc rejects, and `.vpr`s older than their `.rs`.
5. Give it history: `python3 tools/bench/backfill.py --suite <suite> --commits N`
   measures just that suite against the last N builds, merged into their runs.
   Older builds that cannot handle it show up as UNSUPPORTED/ERROR, which is
   coverage history too.

Nothing in the runner, the stored data or the site needs to change.

## Running the pipeline

```bash
cargo build --release --bin verify -p bench
./target/release/bench check-suites
./target/release/bench run --only rust/physics_step --runs 1 --out /tmp/run.json   # ad hoc
python3 tools/bench/run.py --dry-run /tmp/results                                   # full run, any machine
python3 tools/bench/run.py                                                          # benchmark host only
```

`run.py` builds, runs `check-suites` (errors stop it), runs `bench run`, writes
`benchmarks/results/runs/<date>_<sha>.json` and `index.json` on the `benchmarks`
branch (a worktree at `results_worktree` in `tools/bench/config.json`), refreshes
the site in `docs/`, warns if Helium is more than 15% slower than the previous run
over the files both share, and commits and pushes. Only the host named in
`config.json` may write there; everything else uses `--dry-run DIR`.
Machine-specific settings (the Silicon jar, a pinned `rustc_toolchain`) go in the
untracked `tools/bench/config.local.json`.

What one run measures, per file: rustc (`--emit=metadata`; cached by `.rs` hash,
rustc version and arguments),
Helium (`verify --json`: total, phases, per-member times, all `VerifyStats`
counters, peak memory), Silicon (process wall time and its own reported time,
per-member verdicts; cached by `.vpr` hash and jar hash), one warm-up and five
timed runs each (median and MAD), a 300 s timeout recorded as a timeout, plus the
Rust metrics (`bench rust-metrics FILE.rs`) and Viper metrics
(`verify --viper-metrics FILE.vpr`) joined per member (`m_f` ↔ `f`).

To view results locally, serve the results checkout and open `docs/`:

```bash
python3 -m http.server -d ../helium-bench-results 8000   # then http://localhost:8000/docs/
```
