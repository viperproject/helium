#!/usr/bin/env bash
# Run the verifier over vpr/ and compare each member against expected.txt.
#
# Usage: check.sh [stem ...]        # no args = every stem named in expected.txt
#
# Exit code is 1 if any UNSOUND row is present, 0 otherwise: incompleteness (we reject a
# panic-free program) is the expected state of several tiers and must not fail the run,
# while unsoundness (we accept a program that can panic) always must.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
verify="$repo/target/release/verify"

[[ -x "$verify" ]] || { echo "build first: cargo build --release --bin verify" >&2; exit 2; }

stems=("$@")
if [[ ${#stems[@]} -eq 0 ]]; then
    mapfile -t stems < <(grep -v '^\s*#' "$here/expected.txt" | awk 'NF {print $1}' | sort -u)
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

unsound=0
incomplete=0
matched=0
missing=0

for stem in "${stems[@]}"; do
    vpr="$here/vpr/$stem.vpr"
    if [[ ! -f "$vpr" ]]; then
        echo "[no-vpr] $stem (run ./encode_all.sh)" >&2
        continue
    fi
    "$verify" "$vpr" > "$work/$stem.out" 2>/dev/null

    while read -r s member want; do
        [[ "$s" == "$stem" ]] || continue
        line="$(grep -E "\[(OK|FAIL)\] $member(:|\$)" "$work/$stem.out" | head -1)"
        if [[ -z "$line" ]]; then
            echo "MISSING   $stem::$member (not in verifier output)"
            missing=$((missing + 1))
            continue
        fi
        got=FAIL
        [[ "$line" == *"[OK]"* ]] && got=OK

        if [[ "$got" == "$want" ]]; then
            matched=$((matched + 1))
        elif [[ "$want" == FAIL ]]; then
            echo "UNSOUND   $stem::$member (accepted a program that can panic)"
            unsound=$((unsound + 1))
        else
            reason="${line#*: }"
            echo "INCOMPLETE $stem::$member ($reason)"
            incomplete=$((incomplete + 1))
        fi
    done < <(grep -v '^\s*#' "$here/expected.txt" | awk 'NF')
done

echo
echo "as expected: $matched   incomplete: $incomplete   unsound: $unsound   missing: $missing"
[[ $unsound -eq 0 ]] || exit 1
