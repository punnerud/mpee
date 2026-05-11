"""Train a small Pointer Network on N=20 TSP via REINFORCE.

Proof-of-concept for Neural Combinatorial Optimization integration with
brooom. TSP is the simplest VRP variant (one vehicle, no TW, no capacity)
-- if this works, the architecture scales naturally to CVRPTW with
more features per node.

Usage:
    /Users/punnerud/Downloads/ainmt/venv/bin/python3 neural/train_pointer_tsp.py

Output:
    - neural/pointer_tsp.onnx -- the trained model
    - neural/training_log.txt -- loss per epoch
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
N_NODES = 20
EMBED = 64
EPOCHS = 200
BATCH = 64
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def gen_batch(batch=BATCH, n=N_NODES):
    """Random uniform points in [0,1]^2."""
    return torch.rand(batch, n, 2, device=DEVICE)


def tour_length(coords, tour):
    """Total Euclidean length of a tour (closed loop). coords: (B,N,2),
    tour: (B,N) sequence of node indices."""
    b, n, _ = coords.shape
    idx = tour.unsqueeze(-1).expand(-1, -1, 2)
    ordered = coords.gather(1, idx)  # (B,N,2)
    diff = ordered - ordered.roll(-1, dims=1)
    return diff.norm(dim=-1).sum(dim=-1)


class PointerNet(nn.Module):
    """Encoder-decoder med attention. Encoder lager node-embeddings;
    decoder genererer tur token-for-token via attention over remaining-mask."""

    def __init__(self, embed=EMBED):
        super().__init__()
        self.node_embed = nn.Linear(2, embed)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed, nhead=4, dim_feedforward=embed * 2,
            batch_first=True, dropout=0.0,
        )
        # Decoder: query is the embedding of last-visited node + initial graph emb.
        self.decoder_w = nn.Linear(embed * 2, embed)
        # Pointer attention.
        self.attn_q = nn.Linear(embed, embed)
        self.attn_k = nn.Linear(embed, embed)
        self.scale = math.sqrt(embed)

    def encode(self, coords):
        h = self.node_embed(coords)
        h = self.encoder(h)
        return h  # (B, N, E)

    def decode_step(self, h_nodes, h_graph, last_node_idx, mask):
        """Compute logits over candidates for the next step.
        h_nodes: (B, N, E)
        h_graph: (B, E)  — global embedding (mean of h_nodes)
        last_node_idx: (B,) or None for the start token
        mask: (B, N) — True where node is still selectable
        """
        b, n, e = h_nodes.shape
        if last_node_idx is None:
            last_emb = torch.zeros(b, e, device=h_nodes.device)
        else:
            last_emb = h_nodes.gather(
                1, last_node_idx.view(b, 1, 1).expand(-1, 1, e)
            ).squeeze(1)
        ctx = torch.cat([h_graph, last_emb], dim=-1)
        q = self.attn_q(self.decoder_w(ctx)).unsqueeze(1)  # (B,1,E)
        k = self.attn_k(h_nodes)  # (B,N,E)
        logits = (q * k).sum(-1) / self.scale  # (B,N)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

    def forward(self, coords, greedy=False):
        """Generate a complete tour autoregressively. Returns
        (tour, log_probs) where tour is (B, N) and log_probs is (B,) summed."""
        h = self.encode(coords)
        h_graph = h.mean(dim=1)
        b, n, _ = coords.shape
        mask = torch.ones(b, n, dtype=torch.bool, device=coords.device)
        last = None
        tour = torch.zeros(b, n, dtype=torch.long, device=coords.device)
        logp_sum = torch.zeros(b, device=coords.device)
        for step in range(n):
            logits = self.decode_step(h, h_graph, last, mask)
            logp = F.log_softmax(logits, dim=-1)
            if greedy:
                idx = logp.argmax(dim=-1)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            tour[:, step] = idx
            logp_sum = logp_sum + logp.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            mask = mask.scatter(1, idx.unsqueeze(-1), False)
            last = idx
        return tour, logp_sum


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_file = open(os.path.join(OUT_DIR, "training_log.txt"), "w")
    print(f"device={DEVICE}, N={N_NODES}, epochs={EPOCHS}, batch={BATCH}")
    log_file.write(f"device={DEVICE}, N={N_NODES}, epochs={EPOCHS}, batch={BATCH}\n")

    model = PointerNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    baseline = None

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        coords = gen_batch()
        # Sample tours via stochastic policy.
        tour, logp = model(coords, greedy=False)
        cost = tour_length(coords, tour)

        # Greedy rollout as baseline (variance reduction).
        with torch.no_grad():
            tour_g, _ = model(coords, greedy=True)
            cost_g = tour_length(coords, tour_g)
        if baseline is None:
            baseline = cost_g.mean()
        else:
            baseline = 0.95 * baseline + 0.05 * cost_g.mean()

        # REINFORCE loss with greedy rollout as critic.
        advantage = (cost - cost_g).detach()
        loss = (advantage * logp).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            msg = (f"epoch {epoch:4d}  sample_cost={cost.mean().item():.4f}  "
                   f"greedy_cost={cost_g.mean().item():.4f}  loss={loss.item():.4f}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    t1 = time.perf_counter()
    print(f"Trened i {t1-t0:.1f}s")
    log_file.write(f"Trened i {t1-t0:.1f}s\n")

    # Quick eval: compare greedy NN vs random tour vs nearest-neighbor.
    coords = gen_batch(batch=256)
    with torch.no_grad():
        tour_nn, _ = model(coords, greedy=True)
        nn_cost = tour_length(coords, tour_nn).mean().item()
    rand_tour = torch.stack([torch.randperm(N_NODES, device=DEVICE) for _ in range(256)])
    rand_cost = tour_length(coords, rand_tour).mean().item()
    print(f"\nEvaluation on 256 random N=20 TSP instances:")
    print(f"  Random tour       : {rand_cost:.4f}")
    print(f"  Trained pointer-NN: {nn_cost:.4f}  ({(rand_cost - nn_cost)/rand_cost*100:.1f}% bedre)")
    log_file.write(f"\nFinal: rand={rand_cost:.4f}, NN={nn_cost:.4f}\n")
    log_file.close()

    # Export to ONNX with example input.
    model.eval()
    example_coords = torch.rand(1, N_NODES, 2, device=DEVICE)
    onnx_path = os.path.join(OUT_DIR, "pointer_tsp.onnx")

    # ONNX-export of an autoregressive loop is awkward; we export the encoder
    # + a single decode-step, and let the Rust side run the loop.
    class EncoderWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, coords): return self.m.encode(coords)

    class DecoderWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, h_nodes, h_graph, last_emb, mask):
            # last_emb: (B, E) — caller pre-extracts; mask: (B, N) bool.
            ctx = torch.cat([h_graph, last_emb], dim=-1)
            q = self.m.attn_q(self.m.decoder_w(ctx)).unsqueeze(1)
            k = self.m.attn_k(h_nodes)
            logits = (q * k).sum(-1) / self.m.scale
            mask_neg = mask.float() * 1e9
            return logits - mask_neg  # (B, N)

    enc = EncoderWrap(model).to("cpu").eval()
    dec = DecoderWrap(model).to("cpu").eval()

    torch.onnx.export(
        enc, (example_coords.cpu(),),
        os.path.join(OUT_DIR, "pointer_tsp_encoder.onnx"),
        input_names=["coords"], output_names=["h_nodes"],
        dynamic_axes={"coords": {0: "batch", 1: "n_nodes"},
                      "h_nodes": {0: "batch", 1: "n_nodes"}},
        opset_version=17,
    )
    print(f"Encoder eksportert til pointer_tsp_encoder.onnx")

    # Decoder dummy inputs.
    h_nodes_ex = torch.randn(1, N_NODES, EMBED)
    h_graph_ex = torch.randn(1, EMBED)
    last_emb_ex = torch.randn(1, EMBED)
    mask_ex = torch.zeros(1, N_NODES, dtype=torch.bool)
    torch.onnx.export(
        dec, (h_nodes_ex, h_graph_ex, last_emb_ex, mask_ex),
        os.path.join(OUT_DIR, "pointer_tsp_decoder.onnx"),
        input_names=["h_nodes", "h_graph", "last_emb", "mask"],
        output_names=["logits"],
        dynamic_axes={"h_nodes": {0: "batch", 1: "n_nodes"},
                      "h_graph": {0: "batch"},
                      "last_emb": {0: "batch"},
                      "mask": {0: "batch", 1: "n_nodes"},
                      "logits": {0: "batch", 1: "n_nodes"}},
        opset_version=17,
    )
    print(f"Decoder eksportert til pointer_tsp_decoder.onnx")
    print(f"Filer i {OUT_DIR}/")


if __name__ == "__main__":
    main()
