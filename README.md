# Helium

A Rust implementation for parsing, typechecking, lowering, and verifying Silver/Viper programs.

This repository is a research and verification toolchain: it reads `.vpr` files, builds an AST, resolves identifiers and globals, typechecks declarations, lowers them to an internal VMIR representation, and then runs a custom verification pipeline.

The project is centered around the verification of permission-based logic and proof obligations, with support for a broad set of Viper features and a corpus of regression tests and performance baselines.

Helium was initially developed by Jakub Adam Trzykowski.

## What this project does

- Parses Silver/Viper source files
- Resolves names and global declarations
- Typechecks the parsed program
- Translates the typed program to VMIR
- Verifies method/function/predicate proof obligations
- Tracks acceptance, rejection, and unsupported constructs through test cases
- Maintains deterministic performance baselines for verifier regressions

## Repository layout

- `src/lib.rs` – crate entry point
- `src/pipeline.rs` – end-to-end parse → typecheck → translate → verify pipeline
- `src/viper/` – parser, AST, typechecker, and Viper-specific logic
- `src/translate/` – lowering from typed Viper to VMIR
- `src/vmir/` – intermediate representation used by the verifier
- `src/verify/` – verification engine and rule logic
- `src/bin/` – command-line tools:
    - `parse.rs` – parse a file and dump the AST
    - `typecheck.rs` – run the preprocessing/typechecking pipeline
    - `translate.rs` – lower a program to VMIR
    - `verify.rs` – verify a Viper file and print results
- `tests/cases/` – curated test corpus
    - `passing/` – cases that are expected to verify
    - `failing/` – cases that are expected to be rejected
    - `known_limitations/` – tracked incompletenesses
    - `unsupported/` – unsupported constructs that must be reported cleanly
- `benchmarks/` – benchmark suites and baseline snapshots (see `benchmarks/README.md`)
- `bench/` – the benchmark runner (a separate workspace member): times rustc, Helium and Silicon per file, extracts Rust and Viper shape metrics, writes one JSON per run
- `tools/bench/` – `run.py` / `backfill.py` record runs on the `benchmarks` branch; `site/` is the GitHub Pages site that plots them
- `Cargo.toml` – Rust project configuration
- `rust-toolchain.toml` – pins the nightly toolchain

## Getting started

### Prerequisites

This project requires the nightly Rust toolchain:

```bash
rustup toolchain install nightly
```

The repository pins that toolchain in `rust-toolchain.toml`.

### Build the project

```bash
cargo build
```

### Run the parser

```bash
cargo run --bin parse -- tests/cases/passing/field.vpr
```

### Run the typechecker

```bash
cargo run --bin typecheck -- tests/cases/passing/field.vpr
```

### Verify a file

```bash
cargo run --bin verify -- tests/cases/passing/field.vpr
```

Useful option:

```bash
cargo run --bin verify -- --breakdown tests/cases/passing/field.vpr
```

This prints the verification status of each member and, with `--breakdown`, the slowest verification steps and rule timings.

For tools, `--json` prints one JSON object instead (statuses, phase timings, per-member times, all verifier counters, peak memory, the build's git commit), and `--viper-metrics` prints shape metrics of the parsed program without verifying:

```bash
cargo run --release --bin verify -- --json tests/cases/passing/field.vpr
cargo run --release --bin verify -- --viper-metrics tests/cases/passing/field.vpr
```

## Benchmark pipeline

Per commit, `tools/bench/run.py` records how long Helium takes on every benchmark file next to `rustc` and Viper's Silicon, the verifier's cost counters, and each program's shape, on the `benchmarks` branch; its GitHub Pages site shows trends, commit-against-commit comparisons, scaling fits and metric correlations. See `benchmarks/README.md`.

## Typical workflow

1. Add or update a `.vpr` case in `tests/cases/`.
2. Run the relevant test set:

```bash
cargo test --test suite -- --nocapture
```

3. If a verification change intentionally affects performance, refresh the baselines:

```bash
UPDATE_PERF_BASELINE=1 cargo test --test perf_regression
```

4. Review the generated diff before committing the updated baseline files.

## Tests and quality checks

The project includes a focused test harness for correctness and cost regression:

```bash
cargo test --test suite
cargo test --test perf_regression
```

These cover:

- accepted programs verifying successfully
- rejected programs being rejected consistently
- known limitations staying tracked
- unsupported constructs being surfaced as unsupported instead of crashing the pipeline
- deterministic verification-cost changes being caught by benchmark baselines

## Notes

- This is not a standard end-user product crate; it is a verifier implementation and research tool.
- The codebase intentionally uses nightly-only Rust features and is built around a custom verification pipeline rather than a stable public API.
- For support and design context, the source comments and the regression corpus are the most useful reference points.

## Example verification results

The verifier reports one result per unit in the file, with statuses such as:

- `OK` – verified successfully
- `FAIL` – proof obligation not discharged
- `UNSUPPORTED` – construct is not implemented
- `ERROR` – typecheck/translation rejected the unit
- `SKIP` – dependent on a rejected or failed unit

This makes it easier to diagnose whether failures are semantic, unsupported-feature-related, or caused by a dependency chain.