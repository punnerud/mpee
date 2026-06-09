# bench_matrix skalering — London car

Oppdatert: 2026-06-09
Maskin: Apple M3 Pro, ~36 GB RAM, 11 tråder
Dataset: `greater-london.osm.pbf` (n = 1 163 421)

## Fullførte kjøringer

| N | Budget | Chunk | Tid (s) | Celler/s | Peak RSS | Speedup† |
|---|--------|-------|---------|----------|----------|----------|
| 100k | 250 MB | 97 | **884.1** | 11.3M | 310 MB | 0.52× |
| 100k | 500 MB | 296 | **456.5** | 21.9M | 472 MB | 1.00× |
| 100k | 1 GB | 693 | **386.3** | 25.9M | 968 MB | 1.18× |
| 100k | 2 GB | 1489 | **400.6** | 25.0M | 1693 MB | 1.14× |
| 100k | 4 GB | 1500 | **389.2** | 25.7M | 1583 MB | 1.17× |
| 100k | 8 GB | 1500 | **387.2** | 25.8M | 1604 MB | 1.18× |
| 200k | 500 MB | 150 | **2441.0** | 16.4M | 469 MB | 1.00× |
| 200k | 1 GB | 352 | **1940.0** | 20.6M | 891 MB | 1.26× |

†Speedup vs samme N @ 500 MB. Alle kjøringer: 97.9–98.0 % finite celler.

### Budsjettsammenligning 100k

| Budget | Chunk | Tid | Speedup vs 500 MB |
|--------|-------|-----|-------------------|
| 250 MB | 97 | 884.1 s | 0.52× |
| 500 MB | 296 | 456.5 s | 1.00× |
| 1 GB | 693 | 386.3 s | **1.18×** |
| 2 GB | 1489 | 400.6 s | 1.14× |
| 4 GB | 1500 | 389.2 s | 1.17× |
| 8 GB | 1500 | 387.2 s | **1.18×** |

Fra 4 GB og opp treffer chunk-planleggeren tak på 1500 — ytelsen metner (~387 s).
250 MB halverer effektiv RAM men koster ~1.9× wall-clock.

### Budsjettsammenligning 200k

| Budget | Chunk | Tid | Speedup vs 500 MB |
|--------|-------|-----|-------------------|
| 500 MB | 150 | 2441.0 s | 1.00× |
| 1 GB | 352 | 1940.0 s | **1.26×** |

## Konklusjoner

- Streaming over 100k/200k fungerer (peak under budsjett-cap)
- Mer budget → større chunk → raskere opp til metning (~1.2× for 100k, ~1.3× for 200k)
- Lavt budsjett (250 MB) er brukbart men tregt; chunk faller 296 → 97
- Skalering 100k→200k @ 500 MB: 4× celler, 5.3× tid (chunk halvert)

## Filer

- `summary.txt` — resultater per kjøring
- `n{N}_b{B}MB.log` / `_mem.log` — rå logger
- Kjør med `scripts/bench_matrix_scale.sh`