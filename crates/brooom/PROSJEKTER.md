# brooom NN development — large projects

Living plan for three parallel/sequential ML projects. Each phase has a
goal, dependencies, a checklist, and success criteria. Tracking is by
flipping `[ ]` → `[x]` per task.

**Current baseline (ML disabled):**

| N | Vroom | brooom | Δ |
|---|---|---|---|
| 50 | 1554 | 1563 | +0.6% |
| 100 | 2466 | 2463 | −0.1% (tied) |
| 250 | 4472 | 4373 | **−2.2% WIN** |
| 500 | 7132 | 7091 | **−0.6% WIN** |
| 1000 | 11748 | 12162 | +3.5% |

**What is done:**
- LS stack (regret-3, SwapStar, polish-pass)
- Embedding (22-d) + cache + ANN search
- DoE runner
- Distillation pipeline (PyTorch → ONNX → Rust): TSP/CVRPTW pointer-NN,
  edge-refiner v1/v2/v3/v4-RL

**What did not turn out useful:**
- Distillation v1 (flat MLP, 5 params): margin 0.05
- Distillation v2 (transformer, 35 params): margin 0.37 but val_acc 60 %
- v3 (cost-delta MSE): top10_hit 0 %
- v4 (REINFORCE): val_advantage constant negative

---

## Project A: Diagnostic regime mapping (Level 1)

**Goal:** Identify which parameter regimes brooom beats Vroom in and
which regimes Vroom beats brooom in. Empirical foundation for ML work.

**Dependencies:** none (all tools exist).
**Time estimate:** 1-2 working days.

### Checklist

**A1. Parameter space (4 h)**
- [ ] Define the axes: clustering (R/C/RC), TW tightness, demand
      density, capacity utilisation, N (50/100/250/500/1000)
- [ ] Extend `gen_solomon_like.py` with a `--cluster-type R|C|RC` flag
- [ ] Extend with `--tw-tightness 0.1..1.0`
- [ ] Extend with `--demand-spread uniform|skewed`
- [ ] Save as JSON with metadata field `meta: {cluster, tw, demand, n, seed}`

**A2. Generate corpus (2 h)**
- [ ] 100 instances per size (50/100/250) = 300 instances total
- [ ] Vary all 4 axes with Latin Hypercube Sampling
- [ ] Store under `benchmarks/regime_corpus/`

**A3. Run solvers (4-6 h)**
- [ ] Run Vroom on all 300 instances
- [ ] Run brooom on all 300 instances
- [ ] Save results + timings to `regime_results.csv`

**A4. Analysis (4 h)**
- [ ] Plot: brooom-cost / Vroom-cost against each axis
- [ ] Identify top-10 regimes where brooom > Vroom (more than +5 %)
- [ ] Identify top-10 regimes where brooom < Vroom (more than −5 %)
- [ ] Correlation matrix between features and cost ratio
- [ ] Write `REGIME_REPORT.md` with findings

**A5. ML-relevant features (4 h)**
- [ ] Per-instance embedding (22-d via embedding.rs)
- [ ] Logistic regression: predict win/loss from embedding
- [ ] Identify 5-10 features with the highest predictive power
- [ ] These become the input features for Projects B+C

### Success criteria

- ≥10 distinct regimes identified
- Quantitative explanation of when the brooom stack is competitive
- Empirical basis for deciding whether Project B/C is worth the work

---

## Project B: Generator search for hard cases (Level 2)

**Goal:** Build a library of 100-500 instances where brooom beats Vroom
by ≥3 %. Training data for adversarial NN.

**Dependencies:** Project A done (knowledge of regimes).
**Time estimate:** 5-7 working days.

### Checklist

**B1. Evolutionary-search framework (1 day)**
- [ ] Define the instance parameter vector (8-12 floats)
- [ ] Mutation operators: Gaussian on each axis ± uniform flip
- [ ] Crossover: per-axis swap between two parents
- [ ] Selection: top-K by reward = brooom_cost / vroom_cost
- [ ] Implement in `neural/evo_search.py`

**B2. Reward pipeline (1 day)**
- [ ] Single functor that takes a parameter vector → instance → runs
      Vroom + brooom → cost ratio
- [ ] Cache prior runs (same params = don't re-run)
- [ ] Parallelise across cores

**B3. Run generations (2-3 days)**
- [ ] Initial pop: 100 random samples from the whole space
- [ ] 20 generations × 100 individuals = 2000 evals
- [ ] Track convergence: best/avg reward per generation
- [ ] Save top-500 unique instances under `benchmarks/hard_corpus/`

**B4. Validation (1 day)**
- [ ] Re-run Vroom + brooom on all 500 with a different seed
- [ ] Verify that brooom wins are reproducible
- [ ] Write `HARD_CORPUS_REPORT.md` with the summary

### Success criteria

- ≥100 instances where brooom beats Vroom by ≥3 % (deterministic)
- Diversity: the instances cover at least 3 distinct regimes
- Ready as training input for Project C

---

## Project C: Adversarial NN architecture (Level 3)

**Goal:** Co-evolved generator-NN + solver-NN that improve each other
over time.

**Dependencies:** Projects A + B done. PyTorch Geometric installed.
**Time estimate:** 30-60 working days (research-grade).

### Checklist

**C1. Infrastructure (3-5 days)**
- [ ] Install PyTorch Geometric + DGL
- [ ] Set up a PyG DataLoader for instance batches
- [ ] Build a GNN encoder (3-layer GAT or GIN) over the distance matrix
- [ ] Build a node decoder (per-node prediction)
- [ ] Validate on Project B's hard_corpus

**C2. Solver-NN baseline (5-10 days)**
- [ ] Pointer network with GNN encoder (extends the existing
      `train_pointer_cvrptw.py`)
- [ ] Train supervised: NN output predicts the Vroom solution
- [ ] Train REINFORCE: reward = −final_cost
- [ ] Export as ONNX, integrate into Rust as warm-start or refiner

**C3. Generator-NN (5-10 days)**
- [ ] Conditional generator: takes a parameter distribution → emits
      instance features
- [ ] Training: reward = brooom_cost − solver_NN_cost (positive =
      "found a weakness")
- [ ] REINFORCE or GAN-style
- [ ] Generate "hard-for-solver-NN" instances

**C4. Co-evolution loop (10-20 days)**
- [ ] Curriculum: solver-NN trains on generator-NN's output
- [ ] Generator-NN trains to find instances solver-NN fails on
- [ ] Track Elo rating between generator and solver
- [ ] When one stops improving, increase diversity pressure
- [ ] Evaluate against the Vroom baseline at every epoch

**C5. Integration into the brooom solver (5-10 days)**
- [ ] ONNX export of the final solver-NN
- [ ] Rust-side warm-start or refiner
- [ ] Bench on Solomon r1_0050 to r1_1000
- [ ] Goal: beat Vroom consistently

### Success criteria

- Solver-NN beats Vroom on ≥2 of 5 Solomon instances
- Generator-NN finds instances where the brooom stack actually fails
- Reproducible training pipeline
- Publication-worthy artefact

---

## Tracking table

| Project | Status | Progress |
|---|---|---|
| A — Diagnostic regime mapping | Not started | 0/22 tasks |
| B — Generator search | Blocked (depends on A) | 0/12 tasks |
| C — Adversarial NN | Blocked (depends on A+B) | 0/29 tasks |

## Decision points

**After Project A:**
- If ≥10 distinct regimes are found → continue to B.
- If <5 regimes → redefine the parameter space or stop ML work.
- If it turns out brooom is competitive on most regimes → ML may not be
  worth it.

**After Project B:**
- If the hard_corpus has ≥100 reproducible instances → continue to C.
- If <50 → consider whether the problem distribution is too narrow, or
  whether our stack is too strong.

**After Project C:**
- If solver-NN beats Vroom consistently → integrate as the default
  warm-start.
- If not → document as a negative-result research artefact; consider
  hybrid approaches.

---

## Risks

- **Distribution shift:** ML models trained on synthetic instances may
  generalise poorly to real OSM data.
- **Compute:** training needs GPU time; if that becomes prohibitive,
  consider Apple MPS (free, but 5-10× slower than CUDA).
- **Interpretability:** ML models are black-box; debugging failures is
  hard.
- **Maintenance:** ONNX models are tied to architecture — every new
  training iteration is a new export.

## Dependencies on the existing codebase

| Module | Used in | To be modified? |
|---|---|---|
| `src/embedding.rs` | A5, C2-C5 | No (may be extended) |
| `src/cache.rs` | A4, B2 | No |
| `src/regression.rs` | A5 | Possibly (logistic regression) |
| `src/neural.rs` | C5 | Yes (extend for GNN) |
| `benchmarks/doe.py` | A1-A4 | Possibly (extend the flag set) |
| `benchmarks/gen_solomon_like.py` | A1, B2 | Yes (parameter flags) |
| `neural/train_*.py` | C2-C4 | Yes (add GNN) |
