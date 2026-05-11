"""Train a refiner NN: given an existing route, predict improvement of
2-opt swap (i, j) -- i.e. reverse segment route[i..j].

Concept: the NN learns to PRIORITIZE 2-opt moves so LS tries the most
promising ones first. If the NN can rank correctly in top-K, LS can do fewer
probes per pass -- speed gain.

Training data is generated synthetically: random TSP instances, random
non-optimal routes (e.g. nearest-neighbor with noise), and for each
(route, swap_pair) we compute the gain of the swap. The NN learns to
predict that gain.

Output: ONNX export of the model that can be called from Rust.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N = 20  # nodes per TSP instance
EMBED = 32
EPOCHS = 300
BATCH = 128
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def gen_instance(batch=BATCH, n=N):
    """Generate batch of (coords, suboptimal_route, distmatrix). Sub-optimal
    route is built via nearest-neighbor + random noise (swap 2 random tasks)."""
    coords = torch.rand(batch, n, 2, device=DEVICE)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = diff.norm(dim=-1)  # (B, N, N)

    # Build NN-tour for each batch element.
    routes = torch.zeros(batch, n, dtype=torch.long, device=DEVICE)
    for b in range(batch):
        visited = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        cur = 0
        routes[b, 0] = cur
        visited[cur] = True
        for step in range(1, n):
            d = dist[b, cur].clone()
            d[visited] = float('inf')
            nxt = d.argmin().item()
            routes[b, step] = nxt
            visited[nxt] = True
            cur = nxt
    # Add noise: random pairwise swap on 50% of instances.
    swap_mask = torch.rand(batch) < 0.5
    for b in range(batch):
        if swap_mask[b]:
            i = torch.randint(0, n, (1,)).item()
            j = torch.randint(0, n, (1,)).item()
            if i != j:
                tmp = routes[b, i].clone()
                routes[b, i] = routes[b, j]
                routes[b, j] = tmp
    return coords, routes, dist


def tour_length_batch(dist, route):
    """dist: (B, N, N), route: (B, N) → (B,)"""
    b, n = route.shape
    nxt = route.roll(-1, dims=1)
    idx = torch.arange(b, device=route.device).unsqueeze(-1).expand(-1, n)
    return dist[idx, route, nxt].sum(dim=-1)


def two_opt_gain_batch(dist, route, i_idx, j_idx):
    """Compute the cost CHANGE if we reverse route[i..j] (inclusive).
    Negative = improvement. Returns (B,) for per-batch (i, j) pairs."""
    bs, n = route.shape
    bidx = torch.arange(bs, device=route.device)
    a = route[bidx, i_idx]
    b_node = route[bidx, (i_idx + 1) % n]
    c = route[bidx, j_idx]
    d = route[bidx, (j_idx + 1) % n]
    old = dist[bidx, a, b_node] + dist[bidx, c, d]
    new = dist[bidx, a, c] + dist[bidx, b_node, d]
    return new - old


class RefinerNet(nn.Module):
    """Tar (current_route_embedding, edge-pair-features) → predikert gain.

    Per-node features lages fra (coord, position-in-route, prev-edge-len,
    next-edge-len). Modellen tar to nodepar (i, j) og returnerer score for
    2-opt-swap mellom dem.
    """
    def __init__(self, embed=EMBED):
        super().__init__()
        # Per-node feature: (x, y, prev_len, next_len, normalized_pos)
        self.node_mlp = nn.Sequential(
            nn.Linear(5, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
        )
        # Pair head: takes 2 node embeddings → score.
        self.pair_mlp = nn.Sequential(
            nn.Linear(embed * 2, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
            nn.Linear(embed, 1),
        )

    def encode_nodes(self, coords, route, dist):
        """Compute per-node feature tensor (B, N, 5)."""
        b, n, _ = coords.shape
        # Node coords in route order.
        idx = route.unsqueeze(-1).expand(-1, -1, 2)
        ordered = coords.gather(1, idx)  # (B, N, 2)
        # Prev-edge length per node in route order.
        bidx = torch.arange(b, device=coords.device).unsqueeze(-1).expand(-1, n)
        prev = route.roll(1, dims=1)
        nxt = route.roll(-1, dims=1)
        prev_len = dist[bidx, prev, route]
        next_len = dist[bidx, route, nxt]
        pos_norm = torch.linspace(0, 1, n, device=coords.device).unsqueeze(0).expand(b, -1)
        feats = torch.cat([
            ordered, prev_len.unsqueeze(-1), next_len.unsqueeze(-1), pos_norm.unsqueeze(-1),
        ], dim=-1)  # (B, N, 5)
        return self.node_mlp(feats)  # (B, N, E)

    def forward(self, coords, route, dist, i_idx, j_idx):
        """Predict swap-gain for batch of (i, j) pairs.
        i_idx, j_idx: (B,) — indices into the route order.
        Returns (B,) — predicted improvement (negative = better).
        """
        h = self.encode_nodes(coords, route, dist)  # (B, N, E)
        b = h.shape[0]
        bidx = torch.arange(b, device=h.device)
        h_i = h[bidx, i_idx]
        h_j = h[bidx, j_idx]
        pair = torch.cat([h_i, h_j], dim=-1)
        return self.pair_mlp(pair).squeeze(-1)


def main():
    print(f"device={DEVICE}, N={N}, epochs={EPOCHS}")
    log = open(os.path.join(OUT_DIR, "training_log_refiner.txt"), "w")

    model = RefinerNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        coords, route, dist = gen_instance()
        # Sample random (i, j) pairs (i < j).
        i_idx = torch.randint(0, N - 1, (BATCH,), device=DEVICE)
        j_idx = torch.randint(0, N, (BATCH,), device=DEVICE)
        j_idx = torch.maximum(j_idx, i_idx + 1).clamp(max=N - 1)

        # Ground truth: actual 2-opt gain.
        target = two_opt_gain_batch(dist, route, i_idx, j_idx)
        pred = model(coords, route, dist, i_idx, j_idx)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                # Eval: rank-correlation between predicted and actual gains
                # for full N×N candidate set on a fresh batch.
                eval_coords, eval_route, eval_dist = gen_instance(batch=8)
                ii = torch.arange(N - 1, device=DEVICE).repeat_interleave(N - 1)
                jj = torch.arange(1, N, device=DEVICE).repeat(N - 1)
                valid = ii < jj
                ii = ii[valid]; jj = jj[valid]
                # Replicate to batch.
                actual_gains = []
                pred_gains = []
                for b_idx in range(8):
                    c = eval_coords[b_idx:b_idx+1]
                    r = eval_route[b_idx:b_idx+1]
                    d = eval_dist[b_idx:b_idx+1]
                    ag = two_opt_gain_batch(d.expand(len(ii), -1, -1),
                                            r.expand(len(ii), -1), ii, jj)
                    pg = model(c.expand(len(ii), -1, -1), r.expand(len(ii), -1),
                               d.expand(len(ii), -1, -1), ii, jj)
                    actual_gains.append(ag); pred_gains.append(pg)
                actual_gains = torch.stack(actual_gains)
                pred_gains = torch.stack(pred_gains)
                # Top-5 hit rate: of the 5 truly-best swaps, how many appear in the model's top-5?
                top5_pred = pred_gains.topk(5, dim=-1, largest=False).indices
                top5_true = actual_gains.topk(5, dim=-1, largest=False).indices
                hit = sum(len(set(top5_pred[b].tolist()) & set(top5_true[b].tolist()))
                          for b in range(8)) / 40.0
            msg = f"epoch {epoch:4d}  loss={loss.item():.5f}  top-5-hit={hit:.2%}"
            print(msg); log.write(msg + "\n"); log.flush()

    print(f"Trened i {time.perf_counter()-t0:.1f}s")
    log.close()

    # Export to ONNX. The model is awkward because both encoder and pair-head
    # are needed; export the full forward as one graph.
    model.eval().cpu()
    coords_ex = torch.rand(1, N, 2)
    route_ex = torch.arange(N).unsqueeze(0)
    diff = coords_ex.unsqueeze(2) - coords_ex.unsqueeze(1)
    dist_ex = diff.norm(dim=-1)
    i_ex = torch.tensor([0])
    j_ex = torch.tensor([5])
    torch.onnx.export(
        model, (coords_ex, route_ex, dist_ex, i_ex, j_ex),
        os.path.join(OUT_DIR, "refiner.onnx"),
        input_names=["coords", "route", "dist", "i_idx", "j_idx"],
        output_names=["gain"],
        dynamic_axes={"coords": {0: "batch"},
                      "route": {0: "batch"},
                      "dist": {0: "batch"},
                      "i_idx": {0: "batch"},
                      "j_idx": {0: "batch"},
                      "gain": {0: "batch"}},
        opset_version=17,
    )
    print(f"Refiner eksportert til refiner.onnx (statisk N={N})")


if __name__ == "__main__":
    main()
