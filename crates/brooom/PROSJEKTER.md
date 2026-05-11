# brooom NN-utvikling — store prosjekter

Levende plan for tre parallelle/sekvensielle ML-prosjekter. Hver fase har
mål, avhengigheter, sjekkliste og suksesskriterier. Sporing skjer ved å
hake av `[ ]` → `[x]` per oppgave.

**Nåværende baseline (uten ML-aktivert):**

| N | Vroom | brooom | Δ |
|---|---|---|---|
| 50 | 1554 | 1563 | +0.6% |
| 100 | 2466 | 2463 | −0.1% (nær) |
| 250 | 4472 | 4373 | **−2.2% WIN** |
| 500 | 7132 | 7091 | **−0.6% WIN** |
| 1000 | 11748 | 12162 | +3.5% |

**Hva som er ferdig:**
- LS-stack (regret-3, SwapStar, polish-pass)
- Embedding (22-d) + cache + ANN-search
- DoE-runner
- Distillation-pipeline (PyTorch → ONNX → Rust): TSP/CVRPTW pointer-NN, edge-refiner v1/v2/v3/v4-RL

**Hva som ikke ble nyttig:**
- Distillation v1 (flat MLP, 5 par): margin 0.05
- Distillation v2 (transformer, 35 par): margin 0.37 men val_acc 60%
- v3 (cost-delta MSE): top10_hit 0%
- v4 (REINFORCE): val_advantage konstant negativ

---

## Prosjekt A: Diagnostisk regime-mapping (Nivå 1)

**Mål:** Identifisere i hvilke parameter-regimer brooom slår Vroom og
hvor Vroom slår brooom. Empirisk grunnlag for ML-arbeid.

**Avhengigheter:** ingen (alle verktøy finnes).
**Tidsestimat:** 1-2 arbeidsdager.

### Sjekkliste

**A1. Parameter-rom (4 timer)**
- [ ] Definer aksene: clustering (R/C/RC), TW-stramhet, demand-tetthet,
      capacity-utnyttelse, N (50/100/250/500/1000)
- [ ] Utvide `gen_solomon_like.py` med `--cluster-type R|C|RC`-flagg
- [ ] Utvide med `--tw-tightness 0.1..1.0`
- [ ] Utvide med `--demand-spread uniform|skewed`
- [ ] Lagre som JSON med metadata-felt `meta: {cluster, tw, demand, n, seed}`

**A2. Generere korpus (2 timer)**
- [ ] 100 instanser per størrelse (50/100/250) = 300 instanser totalt
- [ ] Variere alle 4 akser med Latin Hypercube Sampling
- [ ] Lagre i `benchmarks/regime_corpus/`

**A3. Kjøre solvere (4-6 timer)**
- [ ] Kjør Vroom på alle 300 instanser
- [ ] Kjør brooom på alle 300 instanser
- [ ] Lagre resultater + tider i `regime_results.csv`

**A4. Analyse (4 timer)**
- [ ] Plot: brooom-cost / Vroom-cost vs hver akse
- [ ] Identifiser top-10 regimer hvor brooom > Vroom (mer enn +5%)
- [ ] Identifiser top-10 regimer hvor brooom < Vroom (mer enn −5%)
- [ ] Korrelasjons-matrise mellom features og kost-ratio
- [ ] Skriv `REGIME_REPORT.md` med funn

**A5. ML-relevante features (4 timer)**
- [ ] Per-instans embedding (22-d via embedding.rs)
- [ ] Logistic regression: predict win/loss fra embedding
- [ ] Identifiser 5-10 features med høyest predictive power
- [ ] Disse blir input-features til Prosjekt B+C

### Suksesskriterier

- ≥10 distinkte regimer identifisert
- Kvantitativ forklaring av når brooom-stacken er konkurransedyktig
- Empirisk grunnlag for å avgjøre om Prosjekt B/C er verdt arbeidet

---

## Prosjekt B: Generator-search for hard cases (Nivå 2)

**Mål:** Bygge bibliotek av 100-500 instanser hvor brooom vinner ≥3% mot
Vroom. Treningsdata for adversarial-NN.

**Avhengigheter:** Prosjekt A ferdig (kunnskap om regimer).
**Tidsestimat:** 5-7 arbeidsdager.

### Sjekkliste

**B1. Evolutionary search rammeverk (1 dag)**
- [ ] Definer instans-parameter-vektor (8-12 floats)
- [ ] Mutation-operatorer: gaussian on each axis ± uniform-flip
- [ ] Crossover: per-axis swap mellom to parents
- [ ] Selection: top-K basert på reward = brooom_cost / vroom_cost
- [ ] Implementer i `neural/evo_search.py`

**B2. Reward-pipeline (1 dag)**
- [ ] Solo-functor som tar parameter-vektor → instans → kjør Vroom + brooom → kost-ratio
- [ ] Cache-baserte tidligere kjøringer (samme parametre = ikke re-kjør)
- [ ] Parallellisere på multi-core

**B3. Kjør generasjoner (2-3 dager)**
- [ ] Initial pop: 100 random fra hele space
- [ ] 20 generasjoner × 100 individer = 2000 evals
- [ ] Track convergence: best/avg reward per generasjon
- [ ] Lagre top-500 unike instanser i `benchmarks/hard_corpus/`

**B4. Validering (1 dag)**
- [ ] Re-kjør Vroom + brooom på alle 500 med ulik seed
- [ ] Verifiser at brooom-vinn er reproducer-bar
- [ ] Skriv `HARD_CORPUS_REPORT.md` med oppsummering

### Suksesskriterier

- ≥100 instanser hvor brooom vinner ≥3% mot Vroom (deterministisk)
- Diversitet: instansene dekker minst 3 distinkte regimer
- Klart for input til Prosjekt C som treningsdata

---

## Prosjekt C: Adversarial NN-arkitektur (Nivå 3)

**Mål:** Co-evolved generator-NN + solver-NN som over tid forbedrer
hverandre.

**Avhengigheter:** Prosjekt A + B ferdig. PyTorch Geometric installert.
**Tidsestimat:** 30-60 arbeidsdager (forskningsnivå).

### Sjekkliste

**C1. Infrastruktur (3-5 dager)**
- [ ] Installere PyTorch Geometric + DGL
- [ ] Sette opp PyG-DataLoader for instans-batches
- [ ] Bygge GNN-encoder (3-lag GAT eller GIN) over distanse-matrise
- [ ] Bygge node-decoder (per-node prediksjon)
- [ ] Validere på Prosjekt B's hard_corpus

**C2. Solver-NN baseline (5-10 dager)**
- [ ] Pointer-network med GNN-encoder (utvider eksisterende `train_pointer_cvrptw.py`)
- [ ] Trene supervised: NN-output predict Vroom-løsning
- [ ] Trene REINFORCE: reward = -final_cost
- [ ] Eksportere som ONNX, integrere i Rust som warm-start eller refiner

**C3. Generator-NN (5-10 dager)**
- [ ] Conditional generator: tar parameter-distribusjon → genererer instans-features
- [ ] Trening: reward = brooom_cost - solver_NN_cost (positiv = "fant en svakhet")
- [ ] REINFORCE eller GAN-stil
- [ ] Generere "hard-for-solver-NN" instanser

**C4. Co-evolution loop (10-20 dager)**
- [ ] Curriculum: solver-NN trenes på generator-NNs output
- [ ] Generator-NN trenes på å finne instanser solver-NN feiler på
- [ ] Track Elo-rating mellom generator og solver
- [ ] Når en av de slutter å forbedre, øk diversitets-pressure
- [ ] Evaluere mot Vroom-baseline ved hver epoch

**C5. Integrasjon i brooom-solver (5-10 dager)**
- [ ] ONNX-eksport av final solver-NN
- [ ] Rust-side warm-start eller refiner
- [ ] Bench på Solomon r1_0050 til r1_1000
- [ ] Mål om vi slår Vroom konsekvent

### Suksesskriterier

- Solver-NN slår Vroom på ≥2 av 5 Solomon-instanser
- Generator-NN finner instanser hvor brooom-stack faktisk feiler
- Reproduserbar trening-pipeline
- Forskningspublikasjon-verdig artefakt

---

## Sporings-tabell

| Prosjekt | Status | Fremgang |
|---|---|---|
| A — Diagnostisk regime-mapping | Ikke startet | 0/22 oppgaver |
| B — Generator-search | Blokkert (avh. A) | 0/12 oppgaver |
| C — Adversarial NN | Blokkert (avh. A+B) | 0/29 oppgaver |

## Beslutningspunkter

**Etter Prosjekt A:**
- Hvis ≥10 distinkte regimer funnet → fortsett til B
- Hvis <5 regimer → omdefinere parameter-rommet eller stoppe ML-arbeid
- Hvis det viser seg at brooom-stack er konkurransedyktig på de fleste regimer → ML kanskje ikke verdt det

**Etter Prosjekt B:**
- Hvis hard_corpus har ≥100 reproducerbare instanser → fortsett til C
- Hvis <50 → tenk på om problem-distribusjonen er begrenset, eller om vår stack er for sterk

**Etter Prosjekt C:**
- Hvis solver-NN slår Vroom konsekvent → integrere som default warm-start
- Hvis ikke → dokumentere som negative-result research, vurder hybride tilnærminger

---

## Risikoer

- **Distribusjons-shift:** ML-modell trent på syntetiske instanser kan generalisere dårlig til ekte OSM-data
- **Compute:** trening krever GPU-tid; hvis det blir uoverkommelig, vurder Apple MPS (gratis, men 5-10× tregere enn CUDA)
- **Fortolkning:** ML-modeller er black-box; hvis de feiler, vanskelig å feilsøke
- **Vedlikehold:** ONNX-modeller er bundet til arkitektur — ny treningsiterasjon = ny eksport

## Avhengigheter på eksisterende kodebase

| Modul | Brukes i | Skal endres? |
|---|---|---|
| `src/embedding.rs` | A5, C2-C5 | Nei (kan utvides) |
| `src/cache.rs` | A4, B2 | Nei |
| `src/regression.rs` | A5 | Mulig (logistic regression) |
| `src/neural.rs` | C5 | Ja (utvide for GNN) |
| `benchmarks/doe.py` | A1-A4 | Mulig (utvide flag-set) |
| `benchmarks/gen_solomon_like.py` | A1, B2 | Ja (parameter-flagg) |
| `neural/train_*.py` | C2-C4 | Ja (legge til GNN) |
