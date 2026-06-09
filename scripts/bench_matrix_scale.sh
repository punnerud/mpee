#!/usr/bin/env bash
# Scale bench_matrix with continuous RSS monitoring.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH=./target/release/bench_matrix
DATASET=london
PROFILE=car
BUDGET_MB=500
DEADLINE_SEC=1800
LOGDIR="${LOGDIR:-$(pwd)/benchmarks/london-scale}"
mkdir -p "$LOGDIR"

START_EPOCH=$(date +%s)
DEADLINE_FILE="$LOGDIR/deadline_epoch"

deadline() {
  if [[ -f "$DEADLINE_FILE" ]]; then
    cat "$DEADLINE_FILE"
  else
    echo $(( START_EPOCH + DEADLINE_SEC ))
  fi
}

extend_deadline_minutes() {
  local minutes=$1
  echo $(( $(date +%s) + minutes * 60 )) > "$DEADLINE_FILE"
  echo "Deadline extended to $(date -r "$(cat "$DEADLINE_FILE")" -Iseconds) (+${minutes} min from now)"
}

# Avoid bc/printf locale issues (nb_NO: bc emits "1.1", printf expects "1,1").
kb_to_mb() {
  awk -v kb="${1:-0}" 'BEGIN { printf "%.1f", kb / 1024 }'
}

monitor_rss() {
  local pid=$1
  local log=$2
  local peak_kb=0
  echo "# mem monitor pid=$pid started $(date -Iseconds)" > "$log"
  while kill -0 "$pid" 2>/dev/null; do
    local rss_kb=0
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [[ "$rss_kb" =~ ^[0-9]+$ ]] || rss_kb=0
    if (( rss_kb > peak_kb )); then
      peak_kb=$rss_kb
    fi
    printf '%s rss_mb=%s peak_mb=%s\n' "$(date +%H:%M:%S)" \
      "$(kb_to_mb "$rss_kb")" \
      "$(kb_to_mb "$peak_kb")" >> "$log"
    sleep 1
  done
  echo "FINAL_PEAK_MB=$(kb_to_mb "$peak_kb")" >> "$log"
  echo "$peak_kb"
}

run_size() {
  local n=$1
  local budget_mb="${2:-$BUDGET_MB}"
  local tag=""
  if [[ "$budget_mb" != "$BUDGET_MB" ]] || [[ -n "${FORCE_BUDGET_TAG:-}" ]]; then
    tag="_b${budget_mb}MB"
  fi
  local out="$LOGDIR/n${n}${tag}.log"
  local mem="$LOGDIR/n${n}${tag}_mem.log"
  local summary="$LOGDIR/summary.txt"

  if (( $(date +%s) >= $(deadline) )); then
    echo "[skip] n=$n budget=${budget_mb}MB — deadline reached ($(date -r "$(deadline)" -Iseconds))" | tee -a "$summary"
    return 1
  fi

  local run_start
  run_start=$(date +%s)
  echo "=== bench_matrix ${n}×${n} budget=${budget_mb}MB $(date -Iseconds) ===" | tee "$out"
  # Run bench_matrix directly (not via pipe) so $! is the real process for RSS.
  "$BENCH" "$DATASET" "$PROFILE" "$n" 0 random "" f32 , "$budget_mb" \
    >> "$out" 2>&1 &
  local pid=$!
  local peak_kb
  peak_kb=$(monitor_rss "$pid" "$mem")
  wait "$pid" || true
  local exit=$?
  local peak_mb
  peak_mb=$(kb_to_mb "$peak_kb")
  local run_elapsed=$(( $(date +%s) - run_start ))
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  local bench_line
  bench_line=$(grep -E '^[0-9]+x[0-9]+' "$out" | tail -1 || true)
  echo "RESULT n=$n budget_mb=$budget_mb exit=$exit peak_rss_mb=$peak_mb run_elapsed_s=$run_elapsed elapsed_total_s=$elapsed ${bench_line:+$bench_line}" | tee -a "$summary"
  echo "---" >> "$summary"
  return 0
}

run_budget_compare() {
  local n=$1
  shift
  local budgets=("$@")
  FORCE_BUDGET_TAG=1
  for b in "${budgets[@]}"; do
    run_size "$n" "$b" || break
  done
  unset FORCE_BUDGET_TAG
}

if [[ "${1:-}" == "--extend-minutes" ]]; then
  extend_deadline_minutes "${2:?minutes required}"
  exit 0
fi

finalize_from_mem() {
  local n=$1
  local mem="$LOGDIR/n${n}_mem.log"
  local summary="$LOGDIR/summary.txt"
  local peak_kb=0
  if [[ -f "$mem" ]]; then
    local line peak_mb
    line=$(grep '^FINAL_PEAK_MB=' "$mem" | tail -1 || true)
    if [[ -n "$line" ]]; then
      peak_mb=${line#FINAL_PEAK_MB=}
      peak_kb=$(awk -v mb="$peak_mb" 'BEGIN { printf "%.0f", mb * 1024 }')
    else
      peak_kb=$(awk -F'[= ]' '/peak_mb=/ { gsub(/,/, ".", $3); if ($3+0 > max) max=$3+0 } END { printf "%.0f", max * 1024 }' "$mem")
    fi
  fi
  local peak_mb
  peak_mb=$(kb_to_mb "$peak_kb")
  local bench_line
  bench_line=$(grep -E '^[0-9]+x[0-9]+' "$LOGDIR/n${n}.log" 2>/dev/null | tail -1 || true)
  echo "RESULT n=$n exit=0 peak_rss_mb=$peak_mb (finalized after handoff) ${bench_line:+$bench_line}" | tee -a "$summary"
  echo "---" >> "$summary"
}

if [[ "${1:-}" == "--budget-compare" ]]; then
  n="${2:?size required}"
  shift 2
  budgets=("$@")
  if ((${#budgets[@]} == 0)); then
    echo "usage: $0 --budget-compare SIZE BUDGET_MB [BUDGET_MB ...]" >&2
    exit 1
  fi
  echo "Budget compare n=$n: ${budgets[*]} MB"
  run_budget_compare "$n" "${budgets[@]}"
  echo "=== DONE $(date -Iseconds) ===" | tee -a "$LOGDIR/summary.txt"
  cat "$LOGDIR/summary.txt"
  exit 0
fi

if [[ "${1:-}" == "--continue" ]]; then
  WAIT_PID="${2:?pid required}"
  WAIT_N="${3:?completed size required}"
  shift 3
  echo "Waiting for bench_matrix pid=$WAIT_PID (n=$WAIT_N)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 2
  done
  wait "$WAIT_PID" 2>/dev/null || true
  finalize_from_mem "$WAIT_N"

  if [[ "${1:-}" == "--budget-compare" ]]; then
    n="${2:?size required}"
    shift 2
    budgets=("$@")
    echo "Budget compare n=$n: ${budgets[*]} MB"
    run_budget_compare "$n" "${budgets[@]}"
  else
    SIZES=("$@")
    if ((${#SIZES[@]} == 0)); then
      SIZES=(400000)
    fi
    echo "Continuing sizes: ${SIZES[*]}"
    for n in "${SIZES[@]}"; do
      run_size "$n" || break
    done
  fi
  echo "=== DONE $(date -Iseconds) ===" | tee -a "$LOGDIR/summary.txt"
  cat "$LOGDIR/summary.txt"
  exit 0
fi

: > "$LOGDIR/summary.txt"
echo $(( START_EPOCH + DEADLINE_SEC )) > "$DEADLINE_FILE"
for n in 100000 200000 400000 600000 800000 1000000; do
  run_size "$n" || break
done

echo "=== DONE $(date -Iseconds) ===" | tee -a "$LOGDIR/summary.txt"
cat "$LOGDIR/summary.txt"