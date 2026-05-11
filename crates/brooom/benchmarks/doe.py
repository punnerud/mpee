#!/usr/bin/env python3
"""DoE-runner for brooom hyperparameter-eksperimenter.

Definer faktorer (CLI-flagg) med flere nivåer, kjør full eller fractional
factorial design, og samle (kost, tid) per kombinasjon. Viser main effects og
finner beste kombinasjon.

Bruk:
    ./doe.py --instance r1_0250 --factors factors.json --design full
    ./doe.py --instance r1_1000 --factors factors.json --design plackett-burman

factors.json-format:
    {
        "ils_iters":      [10, 30, 100],
        "granular_k":     [10, 20, 40],
        "ils_kick_size":  [0.2, 0.4, 0.6]
    }

Output: csv på stdout med kolonnene faktor1,faktor2,...,cost,time_s.
Etter alle kjøringene: main-effects-tabell og beste konfigurasjon.

Plackett-Burman: bare 2-nivå-faktorer (laveste og høyeste verdi). Brukes som
screening for å identifisere viktigste faktorer før full factorial.
"""

import argparse
import csv
import itertools
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="DoE for brooom")
    p.add_argument("--instance", required=True, help="Instance name (e.g. r1_0250)")
    p.add_argument("--factors", required=True, help="JSON with factor levels")
    p.add_argument("--design", choices=["full", "plackett-burman", "one-at-a-time"],
                   default="full", help="Experimental design")
    p.add_argument("--brooom", default=None, help="Path to brooom binary")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repeats per config (median taken)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-run timeout in seconds")
    p.add_argument("--out", default=None, help="CSV output path")
    return p.parse_args()


def design_full(factors):
    """Cartesian product of all factor levels."""
    keys = list(factors.keys())
    values = [factors[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def design_one_at_a_time(factors):
    """Each factor varied alone with others held at their median value."""
    keys = list(factors.keys())
    median_idx = lambda lvls: lvls[len(lvls) // 2]
    base = {k: median_idx(factors[k]) for k in keys}
    yield dict(base)
    for k in keys:
        for v in factors[k]:
            if v == base[k]:
                continue
            cfg = dict(base)
            cfg[k] = v
            yield cfg


def design_plackett_burman(factors):
    """2-level Plackett-Burman screening for up to 7 factors via the
    standard PB-12 matrix (12 runs covers up to 11 factors). Only the
    extreme levels of each factor are used."""
    if len(factors) > 11:
        print("PB-12 supports max 11 factors; truncating.", file=sys.stderr)
    keys = list(factors.keys())[:11]
    # PB-12 design matrix (rows = runs, cols = factors).
    pb12 = [
        "+-+---+++-+",
        "++-+---+++-",
        "-++-+---+++",
        "+-++-+---++",
        "++-++-+---+",
        "+++-++-+---",
        "-+++-++-+--",
        "--+++-++-+-",
        "---+++-++-+",
        "+---+++-++-",
        "-+---+++-++",
        "-----------",
    ]
    for row in pb12:
        cfg = {}
        for i, k in enumerate(keys):
            lo, hi = factors[k][0], factors[k][-1]
            cfg[k] = hi if row[i] == "+" else lo
        yield cfg


DESIGNS = {
    "full": design_full,
    "plackett-burman": design_plackett_burman,
    "one-at-a-time": design_one_at_a_time,
}


# Map factor name → CLI flag template. Add entries here when a new flag is
# introduced. Default: --<name>=<value> with hyphen substitution.
FLAG_MAP = {
    "ils_iters": "--ils-iters",
    "ils_kick_size": "--ils-kick-size",
    "granular_k": "--granular-k",
    "multi_start": "--multi-start",
    "max_passes": "--max-passes",
    "time_limit_s": "--time-limit-s",
}


def build_cmd(brooom, instance_path, cfg):
    args = [brooom, "-i", instance_path]
    for k, v in cfg.items():
        flag = FLAG_MAP.get(k, "--" + k.replace("_", "-"))
        args.append(f"{flag}={v}")
    return args


def parse_solution(path):
    with open(path) as f:
        data = json.load(f)
    return data["summary"]["cost"]


def run_one(brooom, instance_path, cfg, repeats, timeout):
    costs, times = [], []
    out = "/tmp/brooom_doe_out.json"
    for _ in range(repeats):
        cmd = build_cmd(brooom, instance_path, cfg) + ["-o", out]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return None, None
        t1 = time.perf_counter()
        costs.append(parse_solution(out))
        times.append(t1 - t0)
    return statistics.median(costs), statistics.median(times)


def main_effects(rows, factors):
    """For each factor, compute mean cost across all rows where that factor
    sits at each of its levels. Difference between best and worst level is
    the main effect."""
    out = {}
    for k, levels in factors.items():
        per_level = {v: [] for v in levels}
        for r in rows:
            v = r.get(k)
            if v in per_level and r["cost"] is not None:
                per_level[v].append(r["cost"])
        means = {v: statistics.mean(c) if c else float("nan")
                 for v, c in per_level.items()}
        valid = [m for m in means.values() if m == m]  # filter NaN
        effect = (max(valid) - min(valid)) if valid else 0.0
        out[k] = (means, effect)
    return out


def main():
    args = parse_args()
    factors = json.loads(Path(args.factors).read_text())

    brooom = args.brooom or shutil.which("brooom")
    if not brooom:
        # Fall back to ../target/release/brooom relative to this script.
        local = Path(__file__).resolve().parent.parent / "target" / "release" / "brooom"
        if local.exists():
            brooom = str(local)
        else:
            print("brooom binary not found; run cargo build --release", file=sys.stderr)
            sys.exit(1)

    inst_dir = Path(__file__).resolve().parent / "instances"
    inst_path = inst_dir / f"{args.instance}.json"
    if not inst_path.exists():
        print(f"instance not found: {inst_path}", file=sys.stderr)
        sys.exit(1)

    designer = DESIGNS[args.design]
    configs = list(designer(factors))
    print(f"# {len(configs)} configurations × {args.repeats} repeats", file=sys.stderr)

    rows = []
    keys = list(factors.keys())
    for i, cfg in enumerate(configs):
        cost, t = run_one(brooom, str(inst_path), cfg, args.repeats, args.timeout)
        row = dict(cfg)
        row["cost"] = cost
        row["time_s"] = t
        rows.append(row)
        cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"[{i+1:3d}/{len(configs)}] {cfg_str}  →  cost={cost} t={t:.2f}s"
              if cost is not None else f"[{i+1:3d}/{len(configs)}] {cfg_str}  →  FAILED",
              file=sys.stderr)

    # CSV output.
    out_path = args.out or sys.stdout
    if isinstance(out_path, str):
        f = open(out_path, "w", newline="")
    else:
        f = out_path
    writer = csv.DictWriter(f, fieldnames=keys + ["cost", "time_s"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    if isinstance(out_path, str):
        f.close()

    # Main effects + best config.
    print("\n# Main effects (range = max(level-mean) - min(level-mean))",
          file=sys.stderr)
    effects = main_effects(rows, factors)
    sorted_effects = sorted(effects.items(), key=lambda kv: -kv[1][1])
    for k, (means, eff) in sorted_effects:
        levels_str = "  ".join(f"{v}: {m:.1f}" for v, m in means.items()
                               if m == m)
        print(f"  {k:18s} effect={eff:7.1f}    {levels_str}", file=sys.stderr)

    valid_rows = [r for r in rows if r["cost"] is not None]
    if valid_rows:
        best = min(valid_rows, key=lambda r: r["cost"])
        print("\n# Best configuration", file=sys.stderr)
        for k in keys:
            print(f"  {k} = {best[k]}", file=sys.stderr)
        print(f"  cost = {best['cost']}", file=sys.stderr)
        print(f"  time_s = {best['time_s']:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
