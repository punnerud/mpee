"""Tren Pointer Network på N=20 CVRPTW (random Solomon-lignende).

Utvider TSP-versjonen med:
  - 6-d node-features: (x, y, demand, tw_start, tw_end, service)
  - Depot-node (index 0) som åpner og lukker hver rute
  - Multi-vehicle output via depot-retur (modellen kan velge depot
    hvilket som helst tidspunkt → starter ny rute)
  - State per step: (current_node, current_load, current_time)
  - REINFORCE-loss inkluderer travel-cost + TW-violation-penalty

Bruk:
    /Users/punnerud/Downloads/ainmt/venv/bin/python3 neural/train_pointer_cvrptw.py
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_CUSTOMERS = 20  # not counting depot
EMBED = 64
EPOCHS = 400
BATCH = 64
LR = 5e-4
CAP = 30.0  # vehicle capacity
HORIZON = 4.0  # max planning time
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def gen_batch(batch=BATCH, n=N_CUSTOMERS):
    """Generate random CVRPTW instances. Returns:
      coords: (B, N+1, 2)   — index 0 = depot
      demand: (B, N+1)      — depot demand=0
      tw_start, tw_end: (B, N+1)
      service: (B, N+1)
    """
    # Coords: depot at (0.5, 0.5); customers uniform.
    cust = torch.rand(batch, n, 2, device=DEVICE)
    depot = torch.full((batch, 1, 2), 0.5, device=DEVICE)
    coords = torch.cat([depot, cust], dim=1)

    # Demand: random uniform [1, 9] per customer; depot=0.
    demand = torch.randint(1, 10, (batch, n), device=DEVICE).float()
    demand = torch.cat([torch.zeros(batch, 1, device=DEVICE), demand], dim=1)

    # TW: each customer has tw centered around 2 * d(depot,i) so it's
    # reachable. tw_width random uniform in [0.3, 1.0].
    d_depot = (cust - 0.5).norm(dim=-1)  # (B, N)
    tw_center = 2.0 * d_depot + torch.rand(batch, n, device=DEVICE) * 1.5
    tw_width = 0.3 + 0.7 * torch.rand(batch, n, device=DEVICE)
    tw_start_c = (tw_center - tw_width / 2).clamp(min=0.0)
    tw_end_c = (tw_center + tw_width / 2).clamp(max=HORIZON - 0.5)
    tw_start = torch.cat([torch.zeros(batch, 1, device=DEVICE), tw_start_c], dim=1)
    tw_end = torch.cat([torch.full((batch, 1), HORIZON, device=DEVICE), tw_end_c], dim=1)

    # Service time: 0.1 per customer; depot 0.
    service = torch.cat([
        torch.zeros(batch, 1, device=DEVICE),
        torch.full((batch, n), 0.1, device=DEVICE),
    ], dim=1)

    return coords, demand, tw_start, tw_end, service


def build_node_features(coords, demand, tw_start, tw_end, service):
    """Concatenate node features into (B, N+1, 6)."""
    return torch.cat([
        coords,
        demand.unsqueeze(-1),
        tw_start.unsqueeze(-1),
        tw_end.unsqueeze(-1),
        service.unsqueeze(-1),
    ], dim=-1)


class PointerNetCVRPTW(nn.Module):
    def __init__(self, embed=EMBED):
        super().__init__()
        self.node_embed = nn.Linear(6, embed)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed, nhead=4, dim_feedforward=embed * 2,
            batch_first=True, dropout=0.0,
        )
        # Decoder context: graph emb + last_emb + current_load + current_time.
        # Encode load+time as small projection to embed-dim.
        self.state_proj = nn.Linear(2, embed)
        self.decoder_w = nn.Linear(embed * 3, embed)
        self.attn_q = nn.Linear(embed, embed)
        self.attn_k = nn.Linear(embed, embed)
        self.scale = math.sqrt(embed)

    def encode(self, node_feats):
        h = self.node_embed(node_feats)
        return self.encoder(h)

    def decode_step(self, h_nodes, h_graph, last_emb, state, mask):
        """state: (B, 2) = (current_load_norm, current_time_norm)
        mask: (B, N+1) — True where node is selectable (not visited
              customer; depot always selectable).
        """
        b, n, e = h_nodes.shape
        state_emb = self.state_proj(state)  # (B, E)
        ctx = torch.cat([h_graph, last_emb, state_emb], dim=-1)
        q = self.attn_q(self.decoder_w(ctx)).unsqueeze(1)
        k = self.attn_k(h_nodes)
        logits = (q * k).sum(-1) / self.scale
        logits = logits.masked_fill(~mask, -1e9)
        return logits

    def forward(self, coords, demand, tw_start, tw_end, service, greedy=False):
        """Generate a multi-route CVRPTW solution. Returns:
          tour: (B, T) — sequence of node indices including depot returns
          logp: (B,)  — sum of log-probs for actions taken
          cost: (B,)  — total travel time + TW violation penalty
        """
        b, n_total, _ = coords.shape
        feats = build_node_features(coords, demand, tw_start, tw_end, service)
        h = self.encode(feats)
        h_graph = h.mean(dim=1)

        visited = torch.zeros(b, n_total, dtype=torch.bool, device=coords.device)
        visited[:, 0] = False  # depot can be revisited
        last = torch.zeros(b, dtype=torch.long, device=coords.device)  # start at depot
        load = torch.zeros(b, device=coords.device)
        time_now = torch.zeros(b, device=coords.device)
        total_cost = torch.zeros(b, device=coords.device)
        logp_sum = torch.zeros(b, device=coords.device)

        # Generate up to T = 2N steps (with depot returns).
        max_steps = 2 * n_total
        tour = []
        for step in range(max_steps):
            # Mask: node selectable if NOT yet visited (depot always selectable
            # but penalize selecting it when at depot, to avoid infinite loops).
            unvisited_customers = ~visited.clone()
            # Also: customer is infeasible if demand > capacity-load OR
            # arrival > tw_end. Mask these out for stability.
            current_coord = coords.gather(1, last.view(b, 1, 1).expand(-1, 1, 2)).squeeze(1)
            cust_coords = coords  # (B, N+1, 2)
            travel = (cust_coords - current_coord.unsqueeze(1)).norm(dim=-1)  # (B, N+1)
            arrival = time_now.unsqueeze(-1) + travel
            tw_ok = arrival <= tw_end
            cap_ok = (load.unsqueeze(-1) + demand) <= CAP
            feasible = unvisited_customers & tw_ok & cap_ok
            feasible[:, 0] = (last != 0)  # depot only if not already there

            # If nothing feasible: forced to depot (effectively end of episode
            # for that batch element). Mark depot feasible.
            no_feasible = ~feasible.any(dim=-1)
            feasible[no_feasible, 0] = True

            state = torch.stack([load / CAP, time_now / HORIZON], dim=-1)
            last_emb = h.gather(1, last.view(b, 1, 1).expand(-1, 1, EMBED)).squeeze(1)
            logits = self.decode_step(h, h_graph, last_emb, state, feasible)

            if greedy:
                idx = logits.argmax(dim=-1)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            logp = F.log_softmax(logits, dim=-1).gather(1, idx.unsqueeze(-1)).squeeze(-1)
            logp_sum = logp_sum + logp

            # Apply transition.
            travel_taken = travel.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            arrival_taken = time_now + travel_taken
            tw_start_taken = tw_start.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            tw_end_taken = tw_end.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            wait = (tw_start_taken - arrival_taken).clamp(min=0.0)
            tw_violation = (arrival_taken - tw_end_taken).clamp(min=0.0)
            service_taken = service.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            time_now = arrival_taken + wait + service_taken
            demand_taken = demand.gather(1, idx.unsqueeze(-1)).squeeze(-1)

            # If returning to depot: reset load+time.
            at_depot = idx == 0
            load = torch.where(at_depot, torch.zeros_like(load), load + demand_taken)
            time_now = torch.where(at_depot, torch.zeros_like(time_now), time_now)

            # Mark visited (only customers; depot is reusable).
            visited.scatter_(1, idx.unsqueeze(-1), idx.unsqueeze(-1) > 0)

            total_cost = total_cost + travel_taken + 5.0 * tw_violation
            tour.append(idx)
            last = idx

            # Done: all customers visited AND we're at depot.
            done = (visited[:, 1:].all(dim=-1)) & at_depot
            if done.all(): break

        tour = torch.stack(tour, dim=1)
        return tour, logp_sum, total_cost


def main():
    print(f"device={DEVICE}, N={N_CUSTOMERS}, epochs={EPOCHS}")
    log = open(os.path.join(OUT_DIR, "training_log_cvrptw.txt"), "w")

    model = PointerNetCVRPTW().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    baseline = None

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        coords, demand, ts, te, svc = gen_batch()
        tour, logp, cost = model(coords, demand, ts, te, svc, greedy=False)

        with torch.no_grad():
            _, _, cost_g = model(coords, demand, ts, te, svc, greedy=True)
        if baseline is None:
            baseline = cost_g.mean()
        else:
            baseline = 0.95 * baseline + 0.05 * cost_g.mean()

        advantage = (cost - cost_g).detach()
        loss = (advantage * logp).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            msg = (f"epoch {epoch:4d}  sample={cost.mean().item():.4f}  "
                   f"greedy={cost_g.mean().item():.4f}  loss={loss.item():.4f}")
            print(msg)
            log.write(msg + "\n")
            log.flush()

    print(f"Trened i {time.perf_counter()-t0:.1f}s")
    log.write(f"Trened i {time.perf_counter()-t0:.1f}s\n")

    # Eval.
    coords, demand, ts, te, svc = gen_batch(batch=128)
    with torch.no_grad():
        _, _, cost_g = model(coords, demand, ts, te, svc, greedy=True)
    print(f"\nN={N_CUSTOMERS} CVRPTW greedy mean cost: {cost_g.mean().item():.4f}")
    log.write(f"Final greedy: {cost_g.mean().item():.4f}\n")
    log.close()

    # Export ONNX (encoder + decoder-step separately).
    model.eval().cpu()

    class EncWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, feats): return self.m.encode(feats)

    class DecWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, h_nodes, h_graph, last_emb, state, mask):
            state_emb = self.m.state_proj(state)
            ctx = torch.cat([h_graph, last_emb, state_emb], dim=-1)
            q = self.m.attn_q(self.m.decoder_w(ctx)).unsqueeze(1)
            k = self.m.attn_k(h_nodes)
            logits = (q * k).sum(-1) / self.m.scale
            mask_neg = (~mask).float() * 1e9
            return logits - mask_neg

    enc = EncWrap(model).eval()
    dec = DecWrap(model).eval()

    feats_ex = torch.randn(1, N_CUSTOMERS + 1, 6)
    torch.onnx.export(
        enc, (feats_ex,),
        os.path.join(OUT_DIR, "pointer_cvrptw_encoder.onnx"),
        input_names=["node_features"], output_names=["h_nodes"],
        dynamic_axes={"node_features": {0: "batch", 1: "n_nodes"},
                      "h_nodes": {0: "batch", 1: "n_nodes"}},
        opset_version=17,
    )
    print(f"Encoder eksportert.")

    h_nodes_ex = torch.randn(1, N_CUSTOMERS + 1, EMBED)
    h_graph_ex = torch.randn(1, EMBED)
    last_emb_ex = torch.randn(1, EMBED)
    state_ex = torch.randn(1, 2)
    mask_ex = torch.ones(1, N_CUSTOMERS + 1, dtype=torch.bool)
    torch.onnx.export(
        dec, (h_nodes_ex, h_graph_ex, last_emb_ex, state_ex, mask_ex),
        os.path.join(OUT_DIR, "pointer_cvrptw_decoder.onnx"),
        input_names=["h_nodes", "h_graph", "last_emb", "state", "mask"],
        output_names=["logits"],
        dynamic_axes={"h_nodes": {0: "batch", 1: "n_nodes"},
                      "h_graph": {0: "batch"},
                      "last_emb": {0: "batch"},
                      "state": {0: "batch"},
                      "mask": {0: "batch", 1: "n_nodes"},
                      "logits": {0: "batch", 1: "n_nodes"}},
        opset_version=17,
    )
    print(f"Decoder eksportert.")


if __name__ == "__main__":
    main()
