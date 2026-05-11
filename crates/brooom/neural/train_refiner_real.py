"""Refiner-NN trent på REELLE brooom + Vroom-par.

Treningssignal: for hver edge i brooom-løsningen, label = 1 hvis edgen
også finnes i Vroom-løsningen (good), 0 hvis kun i brooom (suboptimal).
NN lærer å predikere "er-i-vroom" = "edge-er-good".

Bruk: i Rust-LS, identifiser brooom-edges med lav score → de er kandidater
for relocate/swap. Speed-gevinst (færre probes) + cost-gevinst (LS finner
suboptimale edges som ellers var "settled").
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(os.path.dirname(OUT_DIR), "benchmarks", "results")
INST_DIR = os.path.join(os.path.dirname(OUT_DIR), "benchmarks", "instances")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 200
BATCH = 512
EMBED = 32
LR = 1e-3


def load_pair(name):
    """Returnér (brooom_solution, vroom_solution, problem) for `name`."""
    with open(os.path.join(RES_DIR, f"{name}.brooom.json")) as f:
        b = json.load(f)
    with open(os.path.join(RES_DIR, f"{name}.vroom.json")) as f:
        v = json.load(f)
    with open(os.path.join(INST_DIR, f"{name}.json")) as f:
        p = json.load(f)
    return b, v, p


def edge_set(sol):
    """Set of (loc_idx_a, loc_idx_b) edges in a solution."""
    edges = set()
    for r in sol["routes"]:
        steps = r["steps"]
        for i in range(len(steps) - 1):
            a = steps[i].get("location_index", steps[i].get("id", -1))
            bv = steps[i + 1].get("location_index", steps[i + 1].get("id", -1))
            edges.add((a, bv))
    return edges


def build_dataset(names):
    """For hver brooom-edge, bygg features + label.
    Features:
      - edge-length (matrix-distanse, normalized per problem)
      - position-in-route (0..1)
      - degree of node a in matrix (sum of distances)
      - degree of node b
      - tw-stramhet for a og b
      - rute-lengde (antall stops)
      - er-til-eller-fra-depot (binær)
    Label: 1 hvis (a,b) ∈ Vroom-edges, else 0.
    """
    feats_all = []
    labels_all = []
    for name in names:
        b, v, p = load_pair(name)
        e_v = edge_set(v)
        # Hent matrise-distanser fra problem (første profil).
        prof = list(p["matrices"].keys())[0]
        D = p["matrices"][prof]["durations"]
        n = len(D)
        # Per-node summer (degree).
        deg = [sum(row) / max(n - 1, 1) for row in D]
        max_d = max(max(row) for row in D)
        max_d = max(max_d, 1)
        # Per-job TW-stramhet.
        tw_widths = {}
        max_tw_end = max((j["time_windows"][0][1] if j.get("time_windows") else 0
                          for j in p["jobs"]), default=86400)
        for j in p["jobs"]:
            li = j["location_index"]
            if j.get("time_windows"):
                w = j["time_windows"][0][1] - j["time_windows"][0][0]
                tw_widths[li] = w / max_tw_end
            else:
                tw_widths[li] = 1.0

        for r in b["routes"]:
            steps = r["steps"]
            route_len = len(steps)
            for i in range(len(steps) - 1):
                a = steps[i].get("location_index", -1)
                bn = steps[i + 1].get("location_index", -1)
                if a < 0 or bn < 0 or a >= n or bn >= n: continue
                edge_d = D[a][bn] / max_d
                pos_norm = i / max(route_len - 1, 1)
                deg_a = deg[a] / max_d
                deg_b = deg[bn] / max_d
                tw_a = tw_widths.get(a, 1.0)
                tw_b = tw_widths.get(bn, 1.0)
                len_norm = route_len / 30.0  # typical route length
                is_depot = 1.0 if (steps[i]["type"] in ("start", "end")
                                    or steps[i + 1]["type"] in ("start", "end")) else 0.0
                feats = [edge_d, pos_norm, deg_a, deg_b, tw_a, tw_b, len_norm, is_depot]
                label = 1 if (a, bn) in e_v else 0
                feats_all.append(feats)
                labels_all.append(label)
    return torch.tensor(feats_all, dtype=torch.float32), \
           torch.tensor(labels_all, dtype=torch.float32)


class EdgeClassifier(nn.Module):
    def __init__(self, in_dim=8, embed=EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
            nn.Linear(embed, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    # Auto-discover all (brooom, vroom)-par in benchmarks/results/.
    import glob
    pairs = []
    for bp in sorted(glob.glob(os.path.join(RES_DIR, "*.brooom.json"))):
        name = os.path.basename(bp).replace(".brooom.json", "")
        vp = os.path.join(RES_DIR, f"{name}.vroom.json")
        if os.path.exists(vp):
            pairs.append(name)
    names = pairs
    print(f"Bygger treningsdata fra {len(names)} brooom+vroom-par")
    feats, labels = build_dataset(names)
    print(f"  Total edges: {len(feats)}")
    print(f"  Positive (good edges): {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    print(f"  Negative (only-brooom): {int((1-labels).sum())} ({(1-labels.mean())*100:.1f}%)")

    # 80/20 train/val split.
    n = len(feats)
    perm = torch.randperm(n)
    cut = int(n * 0.8)
    train_x, train_y = feats[perm[:cut]].to(DEVICE), labels[perm[:cut]].to(DEVICE)
    val_x, val_y = feats[perm[cut:]].to(DEVICE), labels[perm[cut:]].to(DEVICE)

    model = EdgeClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor((1 - labels.mean()) / labels.mean()).to(DEVICE)

    t0 = time.perf_counter()
    log = open(os.path.join(OUT_DIR, "training_log_refiner_real.txt"), "w")
    for epoch in range(EPOCHS):
        # Mini-batch.
        idx = torch.randperm(len(train_x), device=DEVICE)[:BATCH]
        x, y = train_x[idx], train_y[idx]
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()
                val_acc = ((val_logits > 0).float() == val_y).float().mean().item()
                # AUC via simple rank-corr proxy.
                pred_pos = val_logits[val_y == 1].mean().item()
                pred_neg = val_logits[val_y == 0].mean().item()
                margin = pred_pos - pred_neg
            msg = (f"epoch {epoch:4d}  train_loss={loss.item():.4f}  "
                   f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                   f"margin={margin:+.3f}")
            print(msg); log.write(msg + "\n"); log.flush()

    print(f"Trent på {time.perf_counter()-t0:.1f}s")

    # Final eval.
    with torch.no_grad():
        all_logits = model(feats.to(DEVICE))
        scores = torch.sigmoid(all_logits).cpu()
    # Mean score for good vs bad edges.
    good_mean = scores[labels == 1].mean().item()
    bad_mean = scores[labels == 0].mean().item()
    print(f"\nFinal score-statistikk:")
    print(f"  Good edges (in-vroom)     : mean score {good_mean:.3f}")
    print(f"  Bad edges (only-in-brooom): mean score {bad_mean:.3f}")
    print(f"  Skille (margin)           : {good_mean - bad_mean:+.3f}")
    log.write(f"\nFinal: good={good_mean:.3f}, bad={bad_mean:.3f}, margin={good_mean-bad_mean:+.3f}\n")
    log.close()

    # Export ONNX.
    model.eval().cpu()
    ex = torch.randn(1, 8)
    torch.onnx.export(
        model, (ex,),
        os.path.join(OUT_DIR, "edge_refiner.onnx"),
        input_names=["features"], output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nEksportert til neural/edge_refiner.onnx")


if __name__ == "__main__":
    main()
