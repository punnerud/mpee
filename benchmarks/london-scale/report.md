# bench_matrix skalering — London car

Oppdatert: 2026-06-09
Maskin: Apple M3 Pro, ~36 GB RAM, 11 tråder
Dataset: `greater-london.osm.pbf` (n = 1 163 421)

## Fullførte kjøringer

| N | Budget | Chunk | Tid (s) | Celler/s | Peak RSS |
|---|--------|-------|---------|----------|----------|
| 100k | 500 MB | 296 | **456.5** | 21.9M | 472 MB |
| 100k | 1 GB | 693 | **386.3** | 25.9M | 968 MB |
| 100k | 2 GB | 1489 | **400.6** | 25.0M | 1693 MB |
| 200k | 500 MB | 150 | **2441.0** | 16.4M | 469 MB |

### Budsjettsammenligning 100k

| Budget | Chunk | Tid | Speedup vs 500 MB |
|--------|-------|-----|-------------------|
| 500 MB | 296 | 456.5 s | 1.00× |
| 1 GB | 693 | 386.3 s | **1.18×** |
| 2 GB | 1489 | 400.6 s | 1.14× |

### Detaljer

**100k @ 500 MB** — output 76 GB, 98.0 % finite, avg dur=1785s  
**100k @ 1 GB** — output 76 GB, 98.0 % finite, avg dur=1785s  
**100k @ 2 GB** — output 76 GB, 98.0 % finite, avg dur=1785s  
**200k @ 500 MB** — output 305 GB, 97.9 % finite, avg dur=1786s

## Konklusjoner

- Streaming over 100k/200k fungerer (peak < 1 GB RAM @ 500 MB budget)
- Mer budget → større chunk → raskere opp til metning (100k: 18 % speedup 500 MB → 1 GB)
- Skalering 100k→200k @ 500 MB: 4× celler, 5.3× tid (chunk halvert)

## Filer

- `summary.txt` — resultater per kjøring
- `n{N}_b{B}MB.log` / `_mem.log` — rå logger
- Kjør med `scripts/bench_matrix_scale.sh`