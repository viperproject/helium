# Regression and performance pipeline plan

Goal: see how the project evolves as features are added. Two parts:

- **Part A**: the time to compile a Rust file with rustc, compared with the time to check its Viper encoding with Helium and with Silicon.
- **Part B**: how verification time relates to the shape of a program.

## What's already in the repo

Most of the pieces exist. What's missing is a single runner that records results over time.

- **Corpora.** `benchmarks/rust/` has 2886 members, all of which verify. `benchmarks/loops/` and `benchmarks/panic_free/` are smaller. All three commit Prusti's `.vpr` output, so measuring never needs Prusti.
- **Generators.** `gen_depth.py` (nesting depth × statements per block) and `gen_enum.py` (variants × payload depth) already use the right design: vary one factor and hold the rest fixed.
- **Cost counters.** `pipeline::run_file_timed` returns phase timings, per-member times and `VerifyStats`. `VerifyStats` counts things like e-graph peak size, iterations, `prove_probe` and per-rule applications, and these counts are the same on every run.
- **Checks.** `tests/perf_regression.rs` compares those counters against a committed baseline, exact match. `check.sh` checks which members pass or fail.
- **Gaps.** Nothing stores results by commit. Nothing times rustc or Silicon. Nothing measures program features. The only machine output is text on stderr (`[TIMING]` / `[STATS] {:?}`), which is fragile to parse.

## Part A: Helium compared with rustc and Silicon

The comparison, per source file:

| Column | Command | Why |
|---|---|---|
| `rustc_check` | `rustc --edition 2021 --crate-type lib --emit=metadata` | Same as `cargo check`: type and borrow checking only. This is the fairest match for "checking". |
| `rustc_self` | the same runs, with `-Z time-passes` (nightly only) | rustc's own `total`, without process startup, plus each pass. The like-for-like match for `helium_verify`; the passes nest, so they do not add up to the total. |
| `helium_verify` | `verify --json` | Total time plus the phases (parse, typecheck, translate, verify) |
| `silicon_wall`, `silicon_verify` | Viper Silicon on the same `.vpr` | The reference verifier for the same input. See below. |

The headline numbers, tracked per file and as a geometric mean over the corpus:

- **overhead = helium_verify / rustc_self**: what verification costs on top of compiling.
- **speedup = silicon_verify / helium_verify**: how Helium compares with the standard Viper verifier.

Both use the time each tool reports itself, so neither counts process or JVM startup. The site and the summary totals use only these three self-reported columns; the wall-clock columns (`rustc_check`, `helium_wall`, `silicon_wall`) stay in the run files.

### Silicon

- Pin one Silicon build (a fat jar at a fixed version, recorded by its commit and SHA-256 in every result).
- Record two times per file:
  - `silicon_wall`: the whole process, including JVM startup. This is what a user waits for.
  - `silicon_verify`: the time Silicon reports for verification itself, without JVM startup. This is the fair comparison with `helium_verify`.
- Record Silicon's verdict per member as well as Helium's. A member where the two disagree is flagged in the results: a Helium FAIL where Silicon verifies is incompleteness, and a Helium OK where Silicon fails needs a soundness look.
- Silicon's result depends only on the `.vpr` and the Silicon version, not on our commit. Cache it by (`.vpr` hash, Silicon version and arguments) so the runner only reruns Silicon when an encoding, the Silicon build or its arguments change.

### How to measure

- One warm-up run, then 5 timed runs. Report the median and the spread (MAD).
- Run each command as its own process, one after another.
- Set a timeout (300 s to start). A timed-out run is recorded as a timeout, not dropped.
- Record peak memory too.
- Pin the rustc toolchain and record it with the results; Prusti already requires a specific nightly.
- All runs happen on one dedicated benchmark machine that never changes, so timings from any two commits are directly comparable. The runner still records the host name and refuses to write results from any other host, so a stray laptop run can't pollute the history.

A cargo-project form (`cargo check` against `cargo prusti` plus `verify`) can come later. The corpus is single files today.

## Part B: How program shape relates to verification time

**Measure per function, not per file.** `verify` already reports one time per member, and Prusti turns Rust function `f` into Viper method `m_f`. Joining at that level gives about 3000 data points instead of about 50. The naming rule for `impl` methods and modules still needs checking; if a name doesn't match, the fallback is a fuzzy match that gets logged.

### Rust metrics, taken from a `syn` parse

A small separate crate, so the main crate doesn't gain dependencies.

| Group | Metrics |
|---|---|
| Size | lines of code (no blanks or comments), statements, expression nodes |
| Control flow | loop count and loop nesting depth, if/else count and nesting depth, **sequential** branches (these multiply paths: 2ⁿ) versus **nested** ones, cyclomatic complexity, estimated path count, early returns |
| Pattern matching | matches, arms per match, match nesting, variants of the matched enum, payload depth |
| Signature | argument count, split into by-value, `&` and `&mut`; field count of argument types once nested structs are flattened; return type size |
| Calls | total calls, distinct callees, call-chain depth, calls inside loops or branches |
| Mutation | assignments, compound assignments, writes through `&mut`, reborrows, longest place path in a write (`a.b.c.d = …`), borrows stored in structs |
| Checks each operation adds | arithmetic ops (overflow), divisions (divide by zero), indexing (bounds), casts, `unwrap`, `panic!` |
| Types | struct nesting depth, enums touched, generic instantiations |

### Viper metrics, from our own parser

- **Size and members.** Lines, bytes, methods, functions, predicates.
- **Statements.** fold/unfold count, inhale/exhale, `acc(...)` occurrences, quantifiers, labels and gotos (the size of the control-flow graph), method calls, function applications, `assert`s.
- **Encoding blow-up.** Viper lines divided by Rust lines. Prusti's encoding size is a likely driver of cost, and this ratio makes it visible.

### Verifier counters (already collected)

These sit between program shape and time. Examples: blocks and instructions processed, obligations, peak e-graph size, `prove_probe` count, saturation iterations, and time split between the ground graph, the scratch graph and probes.

With these, a result can read like "loop nesting raises `prove_probe` count, and that drives time", not only "loop nesting correlates with time".

### Two more things worth tracking

- **Coverage.** How many members come out OK, FAIL or UNSUPPORTED at each commit. As features land, this probably moves more than speed does.
- **Scaling shape.** For each generated family, fit time against the knob as a power law or an exponential, and record the exponent per commit. A change like "loop nesting went from exponential to polynomial" is exactly the kind of evolution to watch for.

## New generated families

These go in a new `benchmarks/scaling/` directory. `rust/` has a strict no-loops rule, so loop families can't live there. Each generator follows the `gen_depth.py` pattern: one knob varies, everything else stays fixed.

| Generator | Knob | Values |
|---|---|---|
| `gen_loops.py` | loop nesting × loop body size | 1–4 × {5, 20} |
| `gen_seq_if.py` | sequential if-blocks (path explosion) | 1, 2, 4, 8, 12, 16 |
| `gen_args.py` | argument count × kind (value, `&`, `&mut`) | 1, 2, 4, 8, 16 |
| `gen_calls.py` | call-chain depth, and fan-out | 1–16 |
| `gen_mut.py` | writes through `&mut` × place path length | 1–32 × 1–4 |
| `gen_arith.py` | checked operations per block | 5–80 |
| `gen_struct.py` | struct nesting depth (fold/unfold load) | 1–6 |

The existing `gen_depth` (nested if/else) and `gen_enum` (match arms) cover the remaining axes.

**Controlled families are the main evidence.** In the hand-written corpus, lines of code correlate with nearly every other metric, so a regression fit there only backs up what the generated families show.

## Adding test suites

New suites (Rust + Viper pairs) must work without touching the runner, the stored data or the site.

### Discovery by convention

A **suite** is any directory under `benchmarks/` that follows this layout:

```
benchmarks/<suite>/
  src/<stem>.rs             Rust source (optional: a suite may be Viper-only)
  vpr/<stem>.vpr            its Viper encoding
  suite.json                optional settings, see below
  expected_failures.txt     optional, same format as benchmarks/rust/
```

- `bench` finds suites by scanning `benchmarks/*/`. There is no central list to update.
- Files are paired by stem: `src/foo.rs` goes with `vpr/foo.vpr`.
- A `.vpr` with no `.rs` is measured as Viper-only: its rustc columns and Rust metrics are `null`, and it's left out of the rustc overhead number. This covers `panic_free/isolate/` and the top-level `benchmarks/*.vpr`, which become a suite called `viper`.
- A `.rs` with no `.vpr` isn't encoded yet. It's reported as a warning and skipped, never an error.
- The existing `rust/`, `loops/` and `panic_free/` already follow the layout, so they need no changes.

### Optional `suite.json`

Defaults cover the common case; a suite only writes what it needs to change.

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

`family` marks a generated suite. The knob values are read from each stem with the regular expression, so a new generator gets scaling charts on the site with no site changes.

### Keeping history honest when the corpus changes

- **Stable keys.** Every row is keyed by `(suite, stem, member)`. The site builds its lists of suites, files and families from the data, never from a hard-coded list.
- **Input fingerprints.** Each run stores the SHA-256 of every `.rs` and `.vpr`. When a `.vpr` changes (re-encoded, new Prusti), that file's trend line shows a break marker instead of counting the jump as a Helium speed-up or regression.
- **Fair comparisons.** Adding a suite makes the total time go up without anything getting slower. So totals and geometric means in the comparison view are computed over the files the two commits have in common, with identical fingerprints. Files present in only one commit are listed separately as added or removed.
- **Trends start when a file appears.** A new suite's lines begin at the commit that added it.
- **History for new suites.** `backfill.py --suite <name>` runs just that suite against older commits, so it has history too. Older commits that can't handle the suite's features are recorded as UNSUPPORTED, which is itself useful coverage history.
- **Silicon.** The cache is keyed by `.vpr` hash, so a new suite is picked up automatically on the first run.

### Checking a new suite

`bench check-suites` validates the layout before anything is measured:

- pairs that are missing a side;
- a `suite.json` that doesn't parse, or a `family.pattern` that doesn't match every stem;
- `.rs` files rustc rejects;
- `.vpr` files that are older than their `.rs` (probably stale encodings).

`run.py` runs this first and stops on errors, but only warns about unencoded sources.

A short `benchmarks/README.md` describes the layout and the steps to add a suite: add `src/`, encode with `tools/prusti_encode.sh`, optionally add `suite.json`, run `bench check-suites`, then backfill.

## Pipeline design

```
bench/                      (new crate, a separate workspace member)
  src/rust_metrics.rs       syn visitor → per-function features
  src/bin/bench.rs          runner: discover suites, time rustc, Silicon and verify, join metrics, write JSON
tools/bench/
  run.py                    orchestration: build release, run bench, commit results to the benchmarks branch
  backfill.py               checks out old commits into temporary worktrees and replays the runner
  site/                     static GitHub Pages site (see below)
```

Changes to `verify`:

1. Add `--json`, which prints results, phases, member times, the full `VerifyStats`, peak memory and the git commit to stdout.
2. Add `--viper-metrics`, which prints the AST counts listed above.

This removes all parsing of stderr.

### Storing results

Results live in a tracked `benchmarks/results/` folder on a dedicated **`benchmarks` branch**, never on main. The benchmark machine commits and pushes to that branch after each run.

```
benchmarks branch
  benchmarks/results/
    index.json                one entry per run: commit, date, subject, totals, coverage
    runs/<date>_<sha>.json    one file per run (full data)
    silicon_cache.json        Silicon results keyed by (.vpr hash, jar hash, arguments)
    rustc_cache.json          rustc_check and rustc_self timings keyed by (.rs hash, rustc version, arguments)
  docs/                       the GitHub Pages site, served from this branch
```

All data is JSON. A run file looks like:

```json
{
  "schema": 1,
  "commit": "5c5ac69…", "dirty": false, "date": "2026-09-28T14:02:11Z",
  "host": "bench-01",
  "tools": { "rustc": "1.xx-nightly (…)", "silicon": "…@sha256:…" },
  "suites": {
    "rust": { "description": "…", "tags": [], "family": null }
  },
  "files": [
    {
      "suite": "rust", "stem": "physics_step",
      "sha256": { "rs": "…", "vpr": "…" },
      "rust_metrics": { "loc": 142, "max_loop_depth": 0, "…": "…" },
      "viper_metrics": { "loc": 18231, "folds": 311, "…": "…" },
      "times": {
        "rustc_check":   { "median": 0.081, "mad": 0.002, "runs": [0.080, 0.081, …] },
        "rustc_self":    { "median": 0.062, "mad": 0.001, "phases": { "type_check_crate": 0.02, "…": 0 } },
        "helium_verify": { "median": 3.12,  "mad": 0.04,  "phases": { "parse": 0.2, "…": 0 } },
        "silicon_wall":  { "median": 9.8 }, "silicon_verify": { "median": 6.1 }
      },
      "peak_rss_mb": { "helium": 412, "silicon": 1330 },
      "stats": { "sat_iterations": 1234, "prove_probe": 17, "…": 0 },
      "members": [
        { "name": "m_body_apply_impulse", "rust_fn": "body_apply_impulse",
          "helium": "OK", "silicon": "OK", "time": 0.21, "rust_metrics": { "…": 0 } }
      ]
    }
  ]
}
```

`index.json` holds only the summary numbers, so the site loads quickly and fetches a full run file only when it's selected.

**Backfill:** replay the runner over the last N commits on main. `bench` calls `verify` as a separate process, so older builds work. For commits older than `--json`, a best-effort text parser reads the old output. After backfill, each new commit on main adds one data point.

**Checks:** leave `perf_regression.rs` in place as the exact counter check. Add a soft check in `run.py` that warns when wall time is more than 15% slower than the previous run.

## GitHub Pages site

A static site in `docs/` on the `benchmarks` branch, published with GitHub Pages from that branch. Plain HTML and JavaScript with a charting library from a CDN (Plotly), no build step. It reads `index.json` and the run files directly. (GitHub Pages on a private repository needs a paid plan.) Every time it shows is a tool's self-reported one: `rustc_self`, `helium_verify`, `silicon_verify`.

Pages:

1. **Overview.** The latest run next to the previous one: total Helium time, overhead against rustc, speedup against Silicon, coverage (OK / FAIL / UNSUPPORTED), and the biggest per-file changes up and down.
2. **Trends.** Any metric over the commit history, as a line chart. Clicking a point opens that commit on GitHub.
3. **Compare commits.** Pick two or more commits and plot them against each other:
   - a per-file scatter of commit A against commit B (log axes, with a y = x line), so regressions stand out as points above the line;
   - a per-member table sorted by change, filterable by suite;
   - members whose verdict changed between the commits;
   - files added or removed between the commits.
4. **Scaling.** For each generated family, time against the knob on log axes, with the fitted exponent. The lines are either commits (any number overlaid, one metric) or tools (rustc, Helium and Silicon at one commit, to see whether the verifiers scale differently). A baseline option subtracts each line's value at the smallest knob, so fixed costs such as Prusti's prelude drop out and only the growth is compared; the y-axis is then linear.
5. **Explore.** Choose any Rust or Viper metric for the x-axis and any time or counter for the y-axis. One point per member, coloured by suite, at a chosen commit. This also shows the Spearman correlation for the chosen pair.

The selected commits and metrics go into the URL, so a comparison can be shared as a link.

The 20 slowest members, with their features, are listed on the overview and double as a to-do list.

## Milestones

1. **`--json` and peak memory in `verify`.** Small change, and everything else depends on it.
2. **`bench` runner with suite discovery, `check-suites`, and rustc and Silicon timing**, writing run JSON to the `benchmarks` branch. This delivers Part A for every suite, current and future.
3. **Rust and Viper metric extractors, and the per-member join.**
4. **New generators**, then encode them once with Prusti and commit the `.vpr` files.
5. **GitHub Pages site**: overview, trends and commit comparison first; scaling and explore after milestone 4.
6. **Backfill**: test the system for the last commit. It is not necessary to get the data for the whole commit history. The last commit can be the ground truth.

## Decisions

1. **Storage:** a tracked `benchmarks/results/` folder on a dedicated `benchmarks` branch.
2. **Silicon:** included as a full comparison column, with verdicts compared per member.
3. **Machine:** one dedicated benchmark machine that never changes, so no cross-machine normalisation is needed.
4. **Language:** Python for orchestration, Rust (`syn`) for the Rust metric extractor, plain HTML and JavaScript for the site.
5. **Format:** all results in JSON.
6. **Presentation:** a GitHub Pages site on the `benchmarks` branch, with commit-against-commit comparison.
7. **Extensibility:** suites are discovered by directory layout, configured by an optional `suite.json`, and compared only over files two commits share.

## Implementation status

| Milestone | State | Where |
|---|---|---|
| 1. `--json`, peak memory, `--viper-metrics` | done | `src/bin/verify.rs`, `src/json.rs`, `src/peak_memory.rs`, `src/viper/metrics.rs`, `VerifyStats::to_json` |
| 2. `bench` runner, suite discovery, `check-suites`, rustc and Silicon timing | done | `bench/` (workspace member), `tools/bench/run.py`, `results.py` |
| 3. Rust and Viper metrics, per-member join | done | `bench/src/rust_metrics.rs`, `bench/src/run.rs` (`rust_fn_for`) |
| 4. New generators | done | `benchmarks/scaling/gen_*.py`, `suite.json`, `vpr/` (encoded with `tools/bench/encode.py scaling`) |
| 5. Pages site | done (all five pages) | `tools/bench/site/`, copied to `docs/` on the `benchmarks` branch |
| 6. Backfill | done, tested on `main`'s tip (a build from before `--json`, read through the text fallback) | `tools/bench/backfill.py` |

Decisions made while implementing:

- **Several families per suite.** `suite.json` also accepts `"families": [...]` (each with a `name`), since `scaling/` holds seven generators and `rust/` has two families next to hand-written files. A single `family` still requires every stem to match; with `families`, each must match some stem, and `"exhaustive": true` restores the every-stem rule.
- **`helium_verify` is `verify`'s own pipeline total** (no process startup), the fair counterpart of `silicon_verify`; the process wall time is kept as `helium_wall`.
- **rustc is cached like Silicon.** `rustc_check` depends only on the `.rs`, the compiler and its arguments, never on the Helium commit, so each successful timing is measured once and reused (`rustc_cache.json`, marked `cached` in the run file). The Rust and Viper metrics are not cached: they take seconds for the whole corpus, and a cache would go stale whenever their extractors change.
- **rustc is timed without the rustup proxy.** The runner resolves the toolchain's own `rustc` from its sysroot once (the proxy added ~80 ms to a ~100 ms `rustc_check`). The toolchain is `rust-toolchain.toml`'s unless `rustc_toolchain` pins one in `tools/bench/config.json`; the version is recorded either way.
- **Per-member metrics go on the member for the declaration** (`m_f`), not on its `m_f#requires` / `m_f#ensures` contract checks, so each Rust function is one data point.
- **Silicon's times are parsed in all of its formats**: `12.34s` under a minute, `01m:05s` under an hour, `1h:02m:03s` above (silver's `formatMillisReadably`). Reading only the first had recorded every run over a minute as an error.
- **Silicon errors are placed by the start of their location**, which is usually a range (`@320.11--321.30`); reading only single positions (`@5.3`) had dropped those errors and reported the failing member as OK, i.e. as a Helium incompleteness. Silicon's verdict is per declaration, so a failing `m_f` leaves the verdict of its contract rows (`m_f#requires`, `m_f#ensures`) unknown rather than FAIL, and a rejection that cannot be placed in a member leaves every member unknown (with a warning) rather than OK.
- **Every Silicon error is read, or none is trusted.** From ten errors on, Silicon pads the index (`[ 0]`); reading only `[0]` had dropped all but `[10]` of `must_fail`'s eleven errors and reported ten failing members as OK (false incompleteness, and a Helium OK there would have hidden a soundness disagreement). The parser now also checks the count on Silicon's summary line: errors it could not read stand in as one error outside every member, so no member is called OK.
- **A timeout kills the whole process tree.** `java` on Windows is often the `javapath` launcher, which starts the real JVM, which starts z3s; killing only the launcher left them running, stealing CPU from later measurements and writing their summary into the next file's output (a timed-out Silicon run was recorded as verified, in more time than the timeout). Each command now runs in a job object (Windows: created suspended, assigned, then resumed, so nothing escapes) or its own process group (Unix), killed on a timeout and cleaned up after exit; a timed-out sample's output is never parsed as a verdict.
- **What the caches keep.** rustc: successful timings, reused only for runs asking no more timed runs than they hold. Silicon: finished results (same run-count rule) and timeouts with the limit they hit, reused only while the timeout is no longer; errors (a crash, an out-of-memory JVM) are measured again. The Silicon key also covers the JVM and Silicon arguments (`-Xss` decides whether Silicon crashes).
- **Pages serves the branch root**, not `/docs`: the site reads `../benchmarks/results/`, and a root `index.html` redirects to `docs/`.
- **Only `rustc_check`, Helium and Silicon are timed.** Prusti's encode time and the rustc debug/release build times were dropped: none of them is a cost of checking the file.
- **Host.** `host` in `tools/bench/config.json` is unset: `run.py` refuses to write results until it names the benchmark machine (`--dry-run DIR` works anywhere). Machine-local settings (the Silicon jar) go in the untracked `config.local.json`.
