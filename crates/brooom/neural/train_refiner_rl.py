"""Refiner v4: REINFORCE -- NN policy drives a mini-LS.

Concept:
  - Episode: NN scores all brooom edges -> LS prioritizes top-K -> final cost
  - Reward = (baseline_cost - final_cost). Positive if NN-guided LS beats
    random LS.
  - REINFORCE update: log_prob(picked edges) * advantage

The loss function IS the solver itself -- we measure cost AFTER the full LS
pipeline has been allowed to converge. Stronger signal than cost-delta-per-swap.
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
EPOCHS = 200
BATCH = 4
EMBED = 64
LR = 5e-4
LS_PASSES = 3   # mini-LS budget — small for fast iteration


def load_brooom_routes(name):
    bp = os.path.join(RES_DIR, f"{name}.brooom.json")
    ip = os.path.join(INST_DIR, f"{name}.json")
    if not (os.path.exists(bp) and os.path.exists(ip)): return None
    with open(bp) as f: b = json.load(f)
    with open(ip) as f: p = json.load(f)
    prof = list(p["matrices"].keys())[0]
    D = p["matrices"][prof]["durations"]
    n = len(D)
    if n > 500: return None  # cap at 500 for fast RL training
    routes = []
    for r in b["routes"]:
        steps = [s.get("location_index", -1) for s in r["steps"]]
        steps = [s for s in steps if 0 <= s < n]
        if len(steps) >= 4: routes.append(steps)
    if not routes: return None
    rows = torch.tensor(D, dtype=torch.float32) / max(max(max(row) for row in D), 1)
    return {"routes": routes, "dist": rows, "n": n}


def route_cost(route, dist):
    return sum(dist[route[i], route[i + 1]].item() for i in range(len(route) - 1)) \
         + dist[route[-1], route[0]].item()


def two_opt_swap(route, i, j):
    return route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]


def random_ls(route, dist, n_passes=LS_PASSES):
    """Baseline 2-opt LS: pass-by-pass, accept first improvement."""
    cur = list(route)
    for _ in range(n_passes):
        improved = False
        n = len(cur)
        for i in range(n - 2):
            for j in range(i + 2, n):
                old = dist[cur[i], cur[i+1]].item() + dist[cur[j], cur[(j+1) % n]].item()
                new = dist[cur[i], cur[j]].item() + dist[cur[i+1], cur[(j+1) % n]].item()
                if new < old - 1e-9:
                    cur = two_opt_swap(cur, i, j)
                    improved = True; break
            if improved: break
        if not improved: break
    return cur


def guided_ls(route, dist, edge_scores, n_passes=LS_PASSES):
    """Score-prioritized 2-opt LS: tries moves ranked on (s_i + s_j) first."""
    cur = list(route)
    for _ in range(n_passes):
        n = len(cur)
        # Build scored move candidates.
        cands = []
        for i in range(n - 2):
            for j in range(i + 2, n):
                # Score: lower (more "must-attempt") for edges with low edge_score
                s = edge_scores[cur[i]] + edge_scores[cur[i+1]] + edge_scores[cur[j]] + edge_scores[cur[(j+1) % n]]
                cands.append((s, i, j))
        cands.sort()  # lowest score first
        improved = False
        for _, i, j in cands:
            old = dist[cur[i], cur[i+1]].item() + dist[cur[j], cur[(j+1) % n]].item()
            new = dist[cur[i], cur[j]].item() + dist[cur[i+1], cur[(j+1) % n]].item()
            if new < old - 1e-9:
                cur = two_opt_swap(cur, i, j)
                improved = True; break
        if not improved: break
    return cur


class NodeScorer(nn.Module):
    """Per-node "is-this-edge-suboptimal" scorer."""
    def __init__(self, embed=EMBED):
        super().__init__()
        # Input per node: degree-statistics from dist matrix.
        self.encoder = nn.Sequential(
            nn.Linear(4, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
            nn.Linear(embed, 1),
        )

    def forward(self, dist):
        n = dist.shape[0]
        deg_mean = dist.mean(dim=1, keepdim=True)
        deg_max = dist.max(dim=1, keepdim=True).values
        deg_std = dist.std(dim=1, keepdim=True)
        depot_ind = torch.zeros(n, 1, device=dist.device)
        depot_ind[0] = 1.0
        feats = torch.cat([deg_mean, deg_max, deg_std, depot_ind], dim=-1)
        return self.encoder(feats).squeeze(-1)


def main():
    pairs = []
    for bp in sorted(glob.glob(os.path.join(RES_DIR, "*.brooom.json"))):
        name = os.path.basename(bp).replace(".brooom.json", "")
        d = load_brooom_routes(name)
        if d is not None: pairs.append((name, d))
    print(f"Loaded {len(pairs)} usable instances (n ≤ 500).")

    n_train = int(len(pairs) * 0.8)
    train_p = pairs[:n_train]; val_p = pairs[n_train:]

    model = NodeScorer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.perf_counter()
    log = open(os.path.join(OUT_DIR, "training_log_refiner_rl.txt"), "w")
    for epoch in range(EPOCHS):
        # Sample BATCH instances.
        idx = torch.randperm(len(train_p))[:BATCH]
        rewards = []; logp_sum = 0.0
        opt.zero_grad()

        for i in idx:
            _, inst = train_p[i]
            dist = inst["dist"].to(DEVICE)

            # Sample stochastic node-scores via NN + Gumbel noise.
            mu = model(dist)  # (n,)
            std = 0.5
            noise = torch.randn_like(mu) * std
            scores = mu + noise

            # Apply guided-LS to each route, sum costs.
            scores_cpu = scores.detach().cpu().tolist()
            mu_cpu = mu.detach().cpu().tolist()  # for log_prob

            guided_total = 0.0
            random_total = 0.0
            for route in inst["routes"]:
                guided = guided_ls(route, inst["dist"], scores_cpu)
                rand = random_ls(route, inst["dist"])
                guided_total += route_cost(guided, inst["dist"])
                random_total += route_cost(rand, inst["dist"])

            advantage = random_total - guided_total  # positive = NN beats baseline
            rewards.append(advantage)

            # log_prob of the noise sample (approx via -0.5*((x-mu)/std)²).
            logp = -((scores - mu) ** 2).sum() / (2 * std ** 2)
            (-(advantage * logp / BATCH)).backward()

        opt.step()
        avg_reward = sum(rewards) / max(len(rewards), 1)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                val_advs = []
                for _, inst in val_p:
                    dist = inst["dist"].to(DEVICE)
                    mu = model(dist)
                    scores = mu.detach().cpu().tolist()
                    g_tot = sum(route_cost(guided_ls(r, inst["dist"], scores), inst["dist"]) for r in inst["routes"])
                    r_tot = sum(route_cost(random_ls(r, inst["dist"]), inst["dist"]) for r in inst["routes"])
                    val_advs.append(r_tot - g_tot)
                val_adv = sum(val_advs) / max(len(val_advs), 1)
            msg = f"epoch {epoch:4d}  train_reward={avg_reward:+.5f}  val_advantage={val_adv:+.5f}"
            print(msg); log.write(msg + "\n"); log.flush()

    print(f"Trained in {time.perf_counter()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
