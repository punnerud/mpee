"""Refiner v2: transformer-encoder over node-features → per-edge classifier.

I stedet for flat MLP per edge, encoder vi NODENE først (transformer over
6-d node-features med matrise-statistikk), så per-edge klassifisering på
(h_a, h_b, edge-features). Dette fanger opp lokal kontekst som node-degree
og clustering rundt hver kant.

Treningskorpus: alle (brooom, vroom)-par i benchmarks/results/.
"""

import os
import json
import glob
import time
import math
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


def edge_set(sol):
    edges = set()
    for r in sol["routes"]:
        steps = r["steps"]
        for i in range(len(steps) - 1):
            a = steps[i].get("location_index", -1)
            b = steps[i + 1].get("location_index", -1)
            edges.add((a, b))
    return edges


def load_instance(name):
    """Load and prep one (brooom, vroom, problem) tuple. Returns:
        - node_feats: (n_loc, 8) tensor of node-level features
        - brooom_edges: list of (a, b, pos_in_route, route_len)
        - vroom_edge_set: set of (a, b)
    """
    bp = os.path.join(RES_DIR, f"{name}.brooom.json")
    vp = os.path.join(RES_DIR, f"{name}.vroom.json")
    ip = os.path.join(INST_DIR, f"{name}.json")
    if not (os.path.exists(bp) and os.path.exists(vp) and os.path.exists(ip)):
        return None
    with open(bp) as f: b = json.load(f)
    with open(vp) as f: v = json.load(f)
    with open(ip) as f: p = json.load(f)
    prof = list(p["matrices"].keys())[0]
    D = p["matrices"][prof]["durations"]
    n = len(D)
    if n > 1500: return None  # skip very large for memory

    # Per-node features. Index 0 is depot.
    max_d = max(max(row) for row in D) or 1
    rows = torch.tensor(D, dtype=torch.float32) / max_d  # (n, n)
    deg_mean = rows.mean(dim=1)
    deg_max = rows.max(dim=1).values
    deg_min_nonzero = rows.where(rows > 0, torch.tensor(1.0)).min(dim=1).values
    deg_std = rows.std(dim=1)

    # TW + demand from problem.jobs (depot has zeros).
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
        torch.zeros(n).index_fill_(0, torch.tensor([0]), 1.0),  # depot indicator
    ], dim=-1)  # (n, 8)

    # brooom edges with route-context.
    brooom_edges = []
    for r in b["routes"]:
        steps = r["steps"]
        rl = len(steps)
        for i in range(len(steps) - 1):
            a = steps[i].get("location_index", -1)
            bn = steps[i + 1].get("location_index", -1)
            if 0 <= a < n and 0 <= bn < n:
                brooom_edges.append((a, bn, i, rl))

    e_v = edge_set(v)

    return {
        "node_feats": node_feats,
        "dist": rows,
        "edges": brooom_edges,
        "vroom_edges": e_v,
        "n": n,
    }


class RefinerV2(nn.Module):
    """Transformer-encoder over noder + per-edge classifier."""
    def __init__(self, node_in=8, embed=EMBED):
        super().__init__()
        self.node_proj = nn.Linear(node_in, embed)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed, nhead=4, dim_feedforward=embed * 2,
                batch_first=True, dropout=0.0,
            ), num_layers=2,
        )
        # Per-edge: (h_a, h_b, edge_dist, pos_in_route_norm, route_len_norm)
        self.edge_head = nn.Sequential(
            nn.Linear(embed * 2 + 3, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
            nn.Linear(embed, 1),
        )

    def forward(self, node_feats, edge_idx_a, edge_idx_b, edge_extra):
        """node_feats: (1, n, 8); edge_idx_*: (E,); edge_extra: (E, 3)."""
        h = self.node_proj(node_feats)
        h = self.encoder(h).squeeze(0)  # (n, E)
        h_a = h[edge_idx_a]
        h_b = h[edge_idx_b]
        feats = torch.cat([h_a, h_b, edge_extra], dim=-1)
        return self.edge_head(feats).squeeze(-1)


def main():
    pairs = []
    for bp in sorted(glob.glob(os.path.join(RES_DIR, "*.brooom.json"))):
        name = os.path.basename(bp).replace(".brooom.json", "")
        if os.path.exists(os.path.join(RES_DIR, f"{name}.vroom.json")):
            pairs.append(name)

    print(f"Loading {len(pairs)} (brooom, vroom)-par...")
    instances = []
    for name in pairs:
        d = load_instance(name)
        if d is not None:
            instances.append(d)
    print(f"Lastet {len(instances)} instanser. Total brooom-edges: "
          f"{sum(len(d['edges']) for d in instances)}")
    pos_count = sum(sum(1 for (a, b, _, _) in d["edges"] if (a, b) in d["vroom_edges"])
                    for d in instances)
    total_count = sum(len(d["edges"]) for d in instances)
    print(f"  Positive (in-vroom): {pos_count} ({pos_count/total_count*100:.1f}%)")

    # Train/val split per instans (80/20).
    n_train = int(len(instances) * 0.8)
    perm = torch.randperm(len(instances))
    train_inst = [instances[i] for i in perm[:n_train]]
    val_inst = [instances[i] for i in perm[n_train:]]
    print(f"  Train: {len(train_inst)} instanser, Val: {len(val_inst)} instanser")

    model = RefinerV2().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor((1 - pos_count / total_count) / (pos_count / total_count + 1e-9)).to(DEVICE)

    def forward_inst(inst):
        node_feats = inst["node_feats"].unsqueeze(0).to(DEVICE)
        edges = inst["edges"]
        a_idx = torch.tensor([e[0] for e in edges], device=DEVICE, dtype=torch.long)
        b_idx = torch.tensor([e[1] for e in edges], device=DEVICE, dtype=torch.long)
        # extra: (edge_dist, pos_in_route, route_len_norm)
        extras = []
        for (a, b, pos, rl) in edges:
            extras.append([
                inst["dist"][a, b].item(),
                pos / max(rl - 1, 1),
                rl / 30.0,
            ])
        edge_extra = torch.tensor(extras, device=DEVICE, dtype=torch.float32)
        labels = torch.tensor(
            [1.0 if (a, b) in inst["vroom_edges"] else 0.0 for (a, b, _, _) in edges],
            device=DEVICE, dtype=torch.float32,
        )
        logits = model(node_feats, a_idx, b_idx, edge_extra)
        return logits, labels

    t0 = time.perf_counter()
    log = open(os.path.join(OUT_DIR, "training_log_refiner_v2.txt"), "w")
    for epoch in range(EPOCHS):
        # Sample BATCH_INSTANCES instances.
        idx = torch.randperm(len(train_inst))[:BATCH_INSTANCES]
        loss_sum = 0.0
        opt.zero_grad()
        for i in idx:
            logits, labels = forward_inst(train_inst[i])
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            loss_sum += loss.item()
        opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                val_correct = 0; val_total = 0
                pos_logits = []; neg_logits = []
                for inst in val_inst:
                    logits, labels = forward_inst(inst)
                    val_correct += ((logits > 0).float() == labels).sum().item()
                    val_total += len(labels)
                    pos_logits.extend(logits[labels == 1].tolist())
                    neg_logits.extend(logits[labels == 0].tolist())
                val_acc = val_correct / max(val_total, 1)
                pos_mean = sum(pos_logits) / max(len(pos_logits), 1)
                neg_mean = sum(neg_logits) / max(len(neg_logits), 1)
            msg = (f"epoch {epoch:4d}  train_loss={loss_sum/BATCH_INSTANCES:.4f}  "
                   f"val_acc={val_acc:.3f}  pos_mean={pos_mean:+.3f}  "
                   f"neg_mean={neg_mean:+.3f}  margin={pos_mean-neg_mean:+.3f}")
            print(msg); log.write(msg + "\n"); log.flush()

    print(f"Trent på {time.perf_counter()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
