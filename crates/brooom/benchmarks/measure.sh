#!/usr/bin/env bash
# Measure brooom on each instance N times, report best (min) time and peak RAM.
#
# Usage:
#   ./measure.sh                 # default 3 runs per instance
#   ./measure.sh 5 r1_0250       # 5 runs, single instance
#
# Output is one line per instance:
#   <name> time_s=<min_t> rss_mb=<peak_rss> cost=<cost> unassigned=<u>

set -u
export LC_ALL=C

DIR="$(cd "$(dirname "$0")" && pwd)"
INST_DIR="$DIR/instances"
BROOOM="${BROOOM:-$DIR/../target/release/brooom}"

N="${1:-3}"
shift || true
if [[ $# -eq 0 ]]; then
    files=( "$INST_DIR"/*.json )
else
    files=()
    for n in "$@"; do files+=( "$INST_DIR/${n}.json" ); done
fi

for f in "${files[@]}"; do
    name="$(basename "$f" .json)"
    best_t="999999"
    best_rss="0"
    cost=""
    unas=""
    for ((i=0; i<N; i++)); do
        out=$(/usr/bin/time -l "$BROOOM" -i "$f" -o /tmp/brooom_meas.json 2>&1)
        # Real time in seconds (last line "%e" not available with -l on macOS;
        # parse "user" + "sys" or use time.perf_counter via python)
        t=$(python3 -c "
import json, time, subprocess, sys
t0 = time.perf_counter()
subprocess.run(['$BROOOM', '-i', '$f', '-o', '/tmp/brooom_meas.json'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f'{time.perf_counter() - t0:.3f}')
")
        rss=$(echo "$out" | awk '/peak memory/{print int($1/1024/1024*10)/10}')
        cost=$(python3 -c "import json;print(json.load(open('/tmp/brooom_meas.json'))['summary']['cost'])")
        unas=$(python3 -c "import json;print(json.load(open('/tmp/brooom_meas.json'))['summary']['unassigned'])")
        if (( $(echo "$t < $best_t" | bc -l) )); then best_t="$t"; fi
        if (( $(echo "$rss > $best_rss" | bc -l) )); then best_rss="$rss"; fi
    done
    printf "%-12s time_s=%-7s rss_mb=%-6s cost=%-10s unassigned=%s\n" \
        "$name" "$best_t" "$best_rss" "$cost" "$unas"
done
