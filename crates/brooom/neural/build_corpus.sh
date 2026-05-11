#!/usr/bin/env bash
# Build training corpus: 30 synthetic instances x (Vroom + brooom solution).
set -e
DIR="$(cd "$(dirname "$0")"/.. && pwd)"
INST_DIR="$DIR/benchmarks/instances"
RES_DIR="$DIR/benchmarks/results"
BROOOM="$DIR/target/release/brooom"
VROOM="${VROOM:-$(command -v vroom)}"

mkdir -p "$RES_DIR"
N_SEEDS=10
SIZES=(50 100 250)

count=0
for size in "${SIZES[@]}"; do
    for seed in $(seq 1 $N_SEEDS); do
        tag="s${seed}"
        name="r1_$(printf '%04d' $size)_${tag}"

        # 1. Generate instance
        python3 "$DIR/benchmarks/gen_solomon_like.py" --seed "$seed" --tag "$tag" $size >/dev/null

        # 2. Run Vroom
        "$VROOM" -i "$INST_DIR/${name}.json" -o "$RES_DIR/${name}.vroom.json" 2>/dev/null

        # 3. Run brooom
        "$BROOOM" -i "$INST_DIR/${name}.json" -o "$RES_DIR/${name}.brooom.json" >/dev/null 2>&1

        count=$((count + 1))
        printf "[%2d/%d] %s -- vroom=%s brooom=%s\n" \
            "$count" "$((${#SIZES[@]} * N_SEEDS))" "$name" \
            "$(python3 -c "import json;print(int(json.load(open('$RES_DIR/${name}.vroom.json'))['summary']['cost']))")" \
            "$(python3 -c "import json;print(int(json.load(open('$RES_DIR/${name}.brooom.json'))['summary']['cost']))")"
    done
done

echo ""
echo "Corpus ready: $count pairs saved in $RES_DIR/"
