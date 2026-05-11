"""Refiner v3: cost-delta target.

Instead of binary "edge-is-in-vroom", we learn to predict the COST DELTA
of a 2-opt swap that removes the edge. Stronger continuous signal,
directly tied to the cost function.

For each brooom edge (a, b) in a route:
  - Randomly sample another edge (c, d) further along in the same route
  - Compute 2-opt gain: new edges (a, c) + (b, d) vs old (a, b) + (c, d)
  - Target = -gain (positive if swap improves)

The model learns to predict "how much can we save by removing this edge
in favor of a better alternative".
"""

import os
import json
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(os.path.dirname(OUT_DIR), "benchmarks", "results")
INST_DIR = os.path.join(os.path.dirname(OUT_DIR), "benchmarks", "instances")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 300
BATCH_INSTANCES = 8
EMBED = 64
LR = 5e-4


def load_instance(name):
    bp = os.path.join(RES_DIR, f"{name}.brooom.json")
    ip = os.path.join(INST_DIR, f"{name}.json")
    if not (os.path.exists(bp) and os.path.exists(ip)):
        return None
    with open(bp) as f: b = json.load(f)
    with open(ip) as f: p = json.load(f)
    prof = list(p["matrices"].keys())[0]
    D = p["matrices"][prof]["durations"]
    n = len(D)
    if n > 1500: return None

    max_d = max(max(row) for row in D) or 1
    rows = torch.tensor(D, dtype=torch.float32) / max_d

    deg_mean = rows.mean(dim=1)
    deg_max = rows.max(dim=1).values
    deg_min_nonzero = rows.where(rows > 0, torch.tensor(1.0)).min(dim=1).values
    deg_std = rows.std(dim=1)

    tw_start = torch.zeros(n)
    tw_end = torch.ones(n)
    demand = torch.zeros(n)
    max_tw_end = max((j["time_windows"][0][1] for j in p["jobs"] if j.get("time_windows")), default=86400)
    cap = p["vehicles"][0]["capacity"][0] if p["vehicles"][0].get("capacity") else 1
    for j in p["jobs"]:
        li = j["location_index"]
        if 0 <= li < n:
            if j.get("time_windows"):
                tw_start[li] = j["time_windows"][0][0] / max_tw_end
                tw_end[li] = j["time_windows"][0][1] / max_tw_end
            if j.get("delivery"):
                demand[li] = j["delivery"][0] / max(cap, 1)

    node_feats = torch.stack([
        deg_mean, deg_max, deg_min_nonzero, deg_std,
        tw_start, tw_end, demand,
        torch.zeros(n).index_fill_(0, torch.tensor([0]), 1.0),
    ], dim=-1)

    # For each route in brooom, gather (route_steps, edges with positions).
    routes_data = []
    for r in b["routes"]:
        steps = [s.get("location_index", -1) for s in r["steps"]]
        steps = [s for s in steps if 0 <= s < n]
        if len(steps) < 4: continue  # need ≥4 for 2-opt
        routes_data.append(steps)

    return {
        "node_feats": node_feats,
        "dist": rows,
        "routes": routes_data,
        "n": n,
    }


def sample_2opt_targets(inst, n_samples=200):
    """For random 2-opt-swap-kandidater, beregn target = -gain.
    Returns (edge_a, edge_b, edge_extra, target)."""
    a_idx = []; b_idx = []; extras = []; targets = []
    rows = inst["dist"]
    n = inst["n"]
    for route in inst["routes"]:
        rl = len(route)
        n_per_route = min(n_samples // max(len(inst["routes"]), 1), rl * 2)
        for _ in range(n_per_route):
            i = torch.randint(0, rl - 1, (1,)).item()
            j_min = i + 1
            j = torch.randint(j_min, rl, (1,)).item()
            if j == i: continue
            a = route[i]; b = route[i + 1] if i + 1 < rl else route[0]
            c = route[j]; d = route[(j + 1) % rl]
            old = rows[a, b].item() + rows[c, d].item()
            new = rows[a, c].item() + rows[b, d].item()
            gain = old - new  # positive = improvement
            a_idx.append(a); b_idx.append(b)
            extras.append([
                rows[a, b].item(),
                i / max(rl - 1, 1),
                rl / 30.0,
            ])
            targets.append(gain)
    return (
        torch.tensor(a_idx, dtype=torch.long, device=DEVICE),
        torch.tensor(b_idx, dtype=torch.long, device=DEVICE),
        torch.tensor(extras, dtype=torch.float32, device=DEVICE),
        torch.tensor(targets, dtype=torch.float32, device=DEVICE),
    )


class RefinerV3(nn.Module):
    def __init__(self, node_in=8, embed=EMBED):
        super().__init__()
        self.node_proj = nn.Linear(node_in, embed)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed, nhead=4, dim_feedforward=embed * 2,
                batch_first=True, dropout=0.0,
            ), num_layers=2,
        )
        self.head = nn.Sequential(
            nn.Linear(embed * 2 + 3, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
            nn.Linear(embed, 1),
        )

    def forward(self, node_feats, a_idx, b_idx, extras):
        h = self.node_proj(node_feats)
        h = self.encoder(h).squeeze(0)
        h_a = h[a_idx]; h_b = h[b_idx]
        feats = torch.cat([h_a, h_b, extras], dim=-1)
        return self.head(feats).squeeze(-1)


def main():
    pairs = []
    for bp in sorted(glob.glob(os.path.join(RES_DIR, "*.brooom.json"))):
        name = os.path.basename(bp).replace(".brooom.json", "")
        if os.path.exists(os.path.join(INST_DIR, f"{name}.json")):
            pairs.append(name)

    print(f"Loading {len(pairs)} brooom-instanser...")
    instances = []
    for name in pairs:
        d = load_instance(name)
        if d is not None: instances.append(d)
    print(f"  Lastet {len(instances)} instanser med "
          f"{sum(len(d['routes']) for d in instances)} ruter")

    n_train = int(len(instances) * 0.8)
    perm = torch.randperm(len(instances))
    train_inst = [instances[i] for i in perm[:n_train]]
    val_inst = [instances[i] for i in perm[n_train:]]

    model = RefinerV3().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def forward_inst(inst):
        node_feats = inst["node_feats"].unsqueeze(0).to(DEVICE)
        a, b, ex, tgt = sample_2opt_targets(inst)
        if len(tgt) == 0: return None
        pred = model(node_feats, a, b, ex)
        return pred, tgt

    t0 = time.perf_counter()
    log = open(os.path.join(OUT_DIR, "training_log_refiner_v3.txt"), "w")
    for epoch in range(EPOCHS):
        idx = torch.randperm(len(train_inst))[:BATCH_INSTANCES]
        loss_sum = 0.0; n_batches = 0
        opt.zero_grad()
        for i in idx:
            res = forward_inst(train_inst[i])
            if res is None: continue
            pred, tgt = res
            loss = F.mse_loss(pred, tgt)
            loss.backward()
            loss_sum += loss.item(); n_batches += 1
        opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                preds_all = []; tgts_all = []
                for inst in val_inst:
                    res = forward_inst(inst)
                    if res is None: continue
                    preds_all.append(res[0]); tgts_all.append(res[1])
                if preds_all:
                    preds = torch.cat(preds_all); tgts = torch.cat(tgts_all)
                    val_mse = F.mse_loss(preds, tgts).item()
                    # Rank-correlation: are highest-gain edges predicted highest?
                    pred_rank = preds.argsort(descending=True)
                    tgt_rank = tgts.argsort(descending=True)
                    # Top-10 hit rate.
                    top10_pred = set(pred_rank[:10].tolist())
                    top10_tgt = set(tgt_rank[:10].tolist())
                    hit = len(top10_pred & top10_tgt) / 10.0
                else:
                    val_mse = 0.0; hit = 0.0
            msg = (f"epoch {epoch:4d}  train_mse={loss_sum/max(n_batches,1):.5f}  "
                   f"val_mse={val_mse:.5f}  top10_hit={hit:.0%}")
            print(msg); log.write(msg + "\n"); log.flush()

    print(f"Trained in {time.perf_counter()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
