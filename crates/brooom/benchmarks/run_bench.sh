#!/usr/bin/env bash
# Run brooom and vroom on the same set of instances and compare.
#
# Usage:
#   ./run_bench.sh                    # default instances
#   ./run_bench.sh r1_0100 r1_0500    # specific instances
#
# Reports cost (lower is better) and wall-clock time per solver.

set -u
export LC_ALL=C

DIR="$(cd "$(dirname "$0")" && pwd)"
INST_DIR="$DIR/instances"
OUT_DIR="$DIR/results"
mkdir -p "$OUT_DIR"

BROOOM="${BROOOM:-$DIR/../target/release/brooom}"
VROOM="${VROOM:-$(command -v vroom)}"

if [[ ! -x "$BROOOM" ]]; then
    echo "brooom binary not found; run: cargo build --release" >&2
    exit 1
fi
if [[ -z "$VROOM" || ! -x "$VROOM" ]]; then
    echo "vroom not found in PATH; install with: brew install vroom" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    files=( "$INST_DIR"/*.json )
else
    files=()
    for n in "$@"; do
        files+=( "$INST_DIR/${n}.json" )
    done
fi

printf "%-12s %-10s %-12s %-10s %-12s %-10s %-8s %-8s\n" \
    "instance" "vroom_cost" "vroom_t(s)" "brooom_cost" "brooom_t(s)" "cost_ratio" "speed_x" "verdict"
printf "%s\n" "$(printf '=%.0s' {1..96})"

for f in "${files[@]}"; do
    name="$(basename "$f" .json)"

    # Vroom — matrix is embedded in the JSON, so no routing engine.
    t0=$(python3 -c 'import time;print(time.perf_counter())')
    "$VROOM" -i "$f" -o "$OUT_DIR/${name}.vroom.json" 2>"$OUT_DIR/${name}.vroom.err"
    t1=$(python3 -c 'import time;print(time.perf_counter())')
    vroom_t=$(python3 -c "print($t1 - $t0)")
    if ! python3 -c "import json,sys;d=json.load(open('$OUT_DIR/${name}.vroom.json'));sys.exit(0 if 'summary' in d else 1)" 2>/dev/null; then
        printf "%-12s %-10s %-12s %-10s %-12s %-10s %-8s %-8s\n" \
            "$name" "ERR" "$vroom_t" "?" "?" "—" "—" "VROOMERR"
        cat "$OUT_DIR/${name}.vroom.err" | head -3
        continue
    fi
    vroom_cost=$(python3 -c "import json;print(json.load(open('$OUT_DIR/${name}.vroom.json'))['summary']['cost'])")
    vroom_unassigned=$(python3 -c "import json;print(json.load(open('$OUT_DIR/${name}.vroom.json'))['summary']['unassigned'])")

    # brooom
    t0=$(python3 -c 'import time;print(time.perf_counter())')
    "$BROOOM" -i "$f" -o "$OUT_DIR/${name}.brooom.json" >/dev/null 2>&1
    t1=$(python3 -c 'import time;print(time.perf_counter())')
    brooom_t=$(python3 -c "print($t1 - $t0)")
    brooom_cost=$(python3 -c "import json;print(json.load(open('$OUT_DIR/${name}.brooom.json'))['summary']['cost'])")
    brooom_unassigned=$(python3 -c "import json;print(json.load(open('$OUT_DIR/${name}.brooom.json'))['summary']['unassigned'])")

    # Ratios
    cost_ratio=$(python3 -c "v=$vroom_cost;b=$brooom_cost;print(f'{b/v:.2f}' if v>0 else 'NA')")
    speed_x=$(python3 -c "v=$vroom_t;b=$brooom_t;print(f'{v/b:.1f}x' if b>0 else 'NA')")

    verdict="—"
    if (( $(python3 -c "print(1 if $brooom_unassigned > $vroom_unassigned else 0)") )); then
        verdict="UNASSIGNED+"
    elif (( $(python3 -c "print(1 if $brooom_t < $vroom_t and $brooom_cost <= $vroom_cost*1.05 else 0)") )); then
        verdict="WIN"
    elif (( $(python3 -c "print(1 if $brooom_t < $vroom_t else 0)") )); then
        verdict="FAST"
    else
        verdict="SLOW"
    fi

    printf "%-12s %-10.2f %-12.3f %-10.2f %-12.3f %-10s %-8s %-8s\n" \
        "$name" "$vroom_cost" "$vroom_t" "$brooom_cost" "$brooom_t" "$cost_ratio" "$speed_x" "$verdict"
done
