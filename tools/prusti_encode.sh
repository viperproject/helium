#!/usr/bin/env bash
# Encode one Rust source to Viper with Prusti (prusti-next, the task-encoder rewrite).
#
# Usage: prusti_encode.sh <source_file.rs> <output_file.vpr>
#
# Environment:
#   PRUSTI_RUSTC             prusti-rustc executable (required; there is deliberately no
#                            PATH fallback, which could silently pick up classic Prusti)
#   PRUSTI_CHECK_OVERFLOWS   encode overflow checks (default: false; panic_free/ sets true)
#   PRUSTI_ENCODE_VERIFY     set to true to let Prusti verify the program with Silicon
#                            (default: false, encode only)
#
# Prusti runs in a scratch directory and dumps the whole crate as a single program to
# log/viper_program/program-check.vpr, which is moved to <output_file.vpr>.
#
# The .vpr starts with `// prusti-encode:` comment lines recording which Prusti made it:
# its `--version` line, `git describe` of the checkout holding the executable, and the
# sha256 of prusti-driver (the binary that encodes). The first two can be stale (the
# version is baked in at build time; the checkout may have moved since), the hash
# cannot: two encodings are from the same encoder only if their driver hashes match.
#
# Prusti has no encode-only mode: no_verify / skip_verification skip the encoder too,
# and the dump is written just before the program is handed to Silicon. So unless
# PRUSTI_ENCODE_VERIFY=true, Silicon gets a 1s global `--timeout`: the dump is
# unaffected (byte-identical), but Prusti then exits non-zero. A non-zero exit is
# therefore not fatal as long as the program was dumped.
#
# Prusti reads every PRUSTI_* environment variable as a config flag and panics on
# unknown ones, so our own tooling variables (PRUSTI_RUSTC, PRUSTI_ENCODE,
# PRUSTI_ENCODE_VERIFY) are removed from its environment.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_file.rs> <output_file.vpr>" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: source file '$1' does not exist." >&2
    exit 1
fi

SRC_FILE="$(realpath "$1")"
OUT_FILE="$(realpath -m "$2")"
mkdir -p "$(dirname "$OUT_FILE")"

if [ -z "${PRUSTI_RUSTC:-}" ]; then
    echo "Error: PRUSTI_RUSTC is not set." >&2
    echo "Set it to the prusti-next prusti-rustc executable (not one from PATH)." >&2
    exit 1
fi
if ! command -v "$PRUSTI_RUSTC" &> /dev/null; then
    echo "Error: prusti-rustc not found ('$PRUSTI_RUSTC')." >&2
    exit 1
fi

PRUSTI_BIN_DIR="$(dirname "$(command -v "$PRUSTI_RUSTC")")"
PRUSTI_DRIVER="$PRUSTI_BIN_DIR/prusti-driver"
[ -f "$PRUSTI_DRIVER" ] || PRUSTI_DRIVER="$PRUSTI_DRIVER.exe"
if [ ! -f "$PRUSTI_DRIVER" ]; then
    echo "Error: no prusti-driver next to '$PRUSTI_RUSTC'." >&2
    exit 1
fi
PROV_VERSION="$(env -u PRUSTI_RUSTC -u PRUSTI_ENCODE -u PRUSTI_ENCODE_VERIFY \
    "$PRUSTI_RUSTC" --version 2>&1 | grep -m1 '^Prusti version' || echo unknown)"
PROV_CHECKOUT="$(git -C "$PRUSTI_BIN_DIR" describe --always --dirty --abbrev=11 2>/dev/null || echo unknown)"
PROV_DRIVER="$(sha256sum "$PRUSTI_DRIVER" | cut -c1-16)"

echo "Encoding: $1 -> $2"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
LOG_FILE="$TMP_DIR/prusti.log"

VERIFY="${PRUSTI_ENCODE_VERIFY:-false}"
if [ "$VERIFY" = "true" ]; then
    SILICON_ARGS="${PRUSTI_EXTRA_VERIFIER_ARGS:-}"
else
    SILICON_ARGS="--timeout 1"
fi

status=0
(
    cd "$TMP_DIR"
    env -u PRUSTI_RUSTC -u PRUSTI_ENCODE -u PRUSTI_ENCODE_VERIFY \
        PRUSTI_DUMP_VIPER_PROGRAM=true \
        PRUSTI_EXTRA_VERIFIER_ARGS="$SILICON_ARGS" \
        PRUSTI_CHECK_OVERFLOWS="${PRUSTI_CHECK_OVERFLOWS:-false}" \
        "$PRUSTI_RUSTC" --edition=2021 --crate-type=lib "$SRC_FILE"
) > "$LOG_FILE" 2>&1 || status=$?

VPR_FILE="$TMP_DIR/log/viper_program/program-check.vpr"
if [ -f "$VPR_FILE" ]; then
    {
        echo "// prusti-encode: $PROV_VERSION"
        echo "// prusti-encode: checkout $PROV_CHECKOUT, prusti-driver sha256 $PROV_DRIVER"
        echo "// prusti-encode: PRUSTI_CHECK_OVERFLOWS=${PRUSTI_CHECK_OVERFLOWS:-false}"
        cat "$VPR_FILE"
    } > "$OUT_FILE"
    if [ "$status" -ne 0 ] && [ "$VERIFY" = "true" ]; then
        echo "Note: prusti-rustc exited with $status (program rejected), but the encoding was dumped."
    fi
    echo "Successfully generated: $2"
else
    echo "Error: Prusti did not dump a .vpr for $1 (exit $status). Prusti output:" >&2
    if [ -d "$TMP_DIR/log/viper_program" ]; then
        echo "(it dumped $(ls "$TMP_DIR/log/viper_program" | head -1) instead: classic Prusti, not prusti-next?)" >&2
    fi
    tail -n 40 "$LOG_FILE" >&2
    exit 1
fi
