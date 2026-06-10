# Gateway-indeksen: den komprimerte matrisen som oppslagsindeks (MTZT)

> *"Hvis det kun er 3 veier som forbinder gitte områder, bør alt innenfor
> områdene ha disse tre som en indeks i minnet, slik at oppslag går raskere?"*

Ja — og det er nå formatets oppslagslag. Dette dokumentet beskriver `MTZT`
(stream-containeren i [`crates/matcodec/`](../crates/matcodec/)), som gjør den
komprimerte matrisen til en **random-access-indeks i minnet**, ikke bare et
lagringsformat.

## Idéen

Veinett mellom adskilte regioner henger sammen gjennom få "gateways" (broer,
fjelloverganger, motorveier). For punkter `i` i region A og `j` i region B er

```
d(i,j) = min over gateways g av  d(i,g) + d(g,j)
```

Bridge-modellen i matcodec utnytter dette til *komprimering* (residualen mot
min-plus-basen har nær null entropi). MTZT utnytter det samme til *oppslag*:
hvis hvert punkt har avstanden til de få relevante gateway-punktene resident i
minnet, kan kryss-region-celler besvares i O(L) — uten å røre de komprimerte
dataene i det hele tatt.

## Formatet

```
MTZT
  n, L, landemerke-id-er
  dlj-blob          L×n   landemerke→punkt-avstander (resident)
  n rammer          kun residual-raden, deflatert per rad
  footer:
    dil-blob        n×L   punkt→landemerke-avstander — "gateway-indeksen" (resident)
    blockmax-blob   n×L   max|residual| per (rad × landemerke-Voronoi-celle), u8 (resident)
```

- Kolonne `j` tilhører Voronoi-cellen til sitt nærmeste landemerke (`cell_of`,
  avledes deterministisk fra `dlj` — lagres aldri).
- `blockmax[i][c] == 0` betyr: *hele* blokken (rad `i` × celle `c`) er
  reprodusert eksakt av min-plus-basen. Da er `cell(i,j)` = basen, O(L), null
  dekomprimering.
- Resident minne: **2 heltall + 1 byte per (punkt × landemerke)** + 2n bytes.
  0,9 MB ved n=3000/L=32; ~14 MB ved n=50 000/L=32.
- Tapsfritt som før: `decompress(compress(D)) == D` byte for byte. Gamle
  `MTZS`-blobber dekodes fortsatt av `decompress_rows`; `MtzReader` krever MTZT.

## API-et (MtzReader)

| Kall | Kostnad | Garanti |
|---|---|---|
| `cell(i,j)` | O(L) når blokken er indeks-eksakt, ellers én liten inflate + O(L) | alltid eksakt |
| `cell_within(i,j,tol)` | O(L) når `blockmax ≤ tol` | overestimat ≤ tol, underestimat ≤ avrundingsstøy; verdibasert → trygt også på asymmetriske matriser |
| `cell_bounds(i,j)` | O(L) alltid | bro-øvre + ALT-nedre grense; `lo == up` ⇒ eksakt; krever metrisk matrise (`metric_ok`) |
| `row(i)` | hopper over rammen helt når raden er indeks-eksakt | alltid eksakt |

Residual-rader (ikke rekonstruerte rader) LRU-caches, så et kaldt eksakt
oppslag er én liten inflate + O(L) — aldri en full radrekonstruksjon.

**Unreachable-celler** (snapping-feil, døde punkter) håndteres verdibasert:
encoderen verifiserer celle for celle at `base ≥ UNREACHABLE ⇒ verdi ==
UNREACHABLE`, og forgifter blokken (255) hvis regelen ikke holder. Én død
kolonne ødelegger dermed ikke blokkene sine.

## Pivot-utvalg av landemerker

`pick_landmarks` velger ikke lenger farthest-point-punkter, men **pivoter**:
grådig fasilitetslokalisering som minimerer min-plus-residualen over et
deterministisk utvalg av (i,j)-par. På gateway-strukturerte data konvergerer
dette mot de faktiske gateway-punktene ("de 3 veiene"); på strukturløse data
degraderer det til en k-medians-aktig spredning. Bonus: residualene får lavere
entropi, så **komprimeringen ble også bedre** på alt vi har målt.

## Målt

Alle tall: Apple M3 Pro, 200 000 tilfeldige `(i,j)`-oppslag, LRU-cache 64
rader, `cargo run --release -p matcodec --example cell_bench`.
"Gammel kode" = forrige committede versjon (MTZS, FPS-landemerker,
radrekonstruksjon per oppslag).

### Syntetisk gateway-verden (8 regioner × 3 veier, eksakt L1-metrikk)

| n | Komprimering | Indeks-eksakte blokker | `cell()` med indeks | uten indeks† | gammel kode‡ |
|--:|--:|--:|--:|--:|--:|
| 3000 | 2,7x → **17,6x** | 88 % | **1,5 µs** | 11,0 µs (7,5x) | 56,6 µs (**39x**) |
| 6000 | **19,4x** | 86 % | **2,4 µs** | 16,1 µs (6,6x) | ~108 µs (**44x**) |

† samme kode med hurtigstien skrudd av (`set_index_fast_path(false)`) — isolerer
indeksens bidrag. ‡ forrige committede versjon, som i tillegg rekonstruerte hele
raden per kaldt oppslag. `cell_bounds`: ~55 ns, eksakt på 86–87 % av oppslagene.
Glatt euklidsk verden (verste fall, ingen gateway-struktur): ingen regresjon —
34,5 µs både med og uten indeks (gammel kode: ~80 µs, residual-cachen alene gir
2,3x), komprimering 1,85x → 2,06x.

### Ekte veidata — London 2000×2000 (delivery_van, dijeng CH)

| L | Blob | Blokker innen 0/2/5/15/60 s | `cell()` eksakt | `cell_within(5s)` | `cell_bounds` |
|--:|--:|---|--:|--:|--:|
| 32 | 4,16 MB (3,85x) | 2/13/21/34/70 % | 18,6 µs | 14,9 µs | 73 ns |
| 64 | 3,86 MB (4,15x) | 7/33/44/59/84 % | 17,1 µs | 10,3 µs | 128 ns |
| 128 | 3,82 MB (4,18x) | 14/52/63/75/91 % | **14,9 µs** | **6,5 µs** | 252 ns |

Gammel kode: 48,9 µs og 5,49 MB blob (jevnt spredte landemerker, 2,92x). Dvs.
eksakt oppslag **3,3x raskere** og blob **31 % mindre** — og med 5 s toleranse
(irrelevant for VRP-nabolagssøk) **7,5x raskere**.

Ærlig funn: på ekte veidata er residualene ±sekunder (snapping, enveiskjøring),
ikke eksakt null — så den *eksakte* O(L)-stien dekker mindre enn på strukturert
data. Gevinsten der ligger i toleranse- og grense-API-ene, og den vokser med L.

### Skala — streamet komprimering rett fra CH-grafen (aldri n² i minne)

`cargo run --release -p dijeng --example stream_compress` (én CH
one-to-many-spørring per rad, jevnt spredte landemerker):

| Matrise | Tid | Peak RAM | Blob | Rå | open | Resident | `cell()` | `cell_bounds` |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10k × 10k | 490 s | 140 MB | 121,8 MB (3,28x) | 400 MB | 0,01 s | **2,9 MB** | 82 µs | **63 ns** |

En 400 MB-matrise besvarer tilfeldige oppslag fra 2,9 MB varmt minne pluss den
komprimerte blobben (RAM eller mmap), og åpner på 10 ms. På denne skalaen — med
jevnt spredte landemerker (pivot-mining trenger full matrise) og storby-
asymmetri — er det grensene (63 ns) som er O(L)-gevinsten; eksakte oppslag
koster én ramme-inflate (~80 µs ved n=10k), og toleranse-andelene er lave
(5–7 %). Pivot-utvalg på et streamet utvalg ville løftet dette (se nedenfor).

Per-rad-streaming (én CH-spørring + én deflate per rad) er praktisk opp til
~10–20k; over det er flaskehalsen zlib-kostnaden per rad, og riktig vei er å
mate kodeken fra dijengs *chunked* many-to-many (som genererer 100k × 100k på
under 90 s, se README-tabellen) — formatet er klart for det, adapteren er ikke
skrevet.

## Lab: to kandidat-forbedringer målt (2026-06-10)

`cargo run --release -p matcodec --example structure_lab -- <varint|pyramid> <matrix.json|gateway:N>`
— måling uten formatendring.

**1. Polyline-stil koding (delta + zigzag + varint før deflate): KLAR GEVINST.**
Blob-størrelse mot dagens deflate-over-rå-i32, samme innhold, fortsatt tapsfritt:

| Verden | I dag | Beste variant | Mindre |
|---|--:|--:|--:|
| Ekte London 2000² | 4,12 MB | 2,77 MB | **33 %** |
| Ekte London 4000² | 15,08 MB | 9,71 MB | **36 %** |
| Gateway 3000 | 2,04 MB | 1,46 MB | 28 % |
| Gateway 6000 | 7,41 MB | 5,47 MB | 26 % |

Det som virker: zigzag-varint **+ deflate** på residual-rammene (varint alene er
*verre* enn deflate på ekte data), celle-gruppert kolonnerekkefølge med delta
innen gruppen (best på ekte data; krever at dekoderen kjenner `cell_of`, som
allerede avledes resident), og delta-langs-punkt for `dil` / celle-gruppert
delta for `dlj` (halverer dlj). Oppslagsstiene påvirkes ikke — tabellene
dekodes til flate arrays ved `open`. Gevinsten vokser med n på ekte data.

**2. Naiv 2-nivå pyramide (k nærmeste huber + tett hub-matrise): NEGATIVT på
ekte data.** Per byte taper den mot dagens flate pivot-tabell: på London 4000²
gir pyr H=128/k=8 (386 KB resident) samme kvalitet som flat L=8 (296 KB), mens
flat L=32 (1,2 MB) er langt foran begge. Årsaken er hub-*utvalget*: punktets k
*nærmeste* huber er ikke hubene som ligger *på korteste vei* til målet — riktig
"exit" avhenger av retningen.

**3. Sti-bevisst hub-utvalg (`hubpath`): GJENNOMBRUDD — slår den flate
tabellen på ekte data med en brøkdel av minnet.** Samme residente layout som
pyramiden, men hvert punkts huber velges ved *best-via-mining*: sample par
(i,j), finn huben som minimerer `d(i,a)+d(a,j)`, kreditér den til ut-settet til
`i` og inn-settet til `j`; behold de k mest krediterte per punkt (retningsdelt:
k ut-huber og k inn-huber). Ekte London 4000² (delivery_van):

| Variant | Resident | Eksakt | innen 2 s | innen 5 s |
|---|--:|--:|--:|--:|
| flat L=32 | 1160 KB | 27,3 % | 42,4 % | 53,3 % |
| flat L=64 | 2312 KB | 42,4 % | 61,7 % | 72,2 % |
| naiv H=128 k=8 | 386 KB | 8,9 % | 15,8 % | 23,5 % |
| sti H=128 k=8 | 386 KB | 35,5 % | 56,0 % | 64,5 % |
| **sti H=128 k=16** | **706 KB** | **43,1 %** | **66,0 %** | **74,7 %** |

Sti-utvalget er 4x bedre enn naivt utvalg på identisk minne, og k=16-varianten
slår flat L=64 på *alle* metrikker med **3,3x mindre resident minne** —
oppslagskostnaden er sammenlignbar (k² min-plus-ledd ≈ L). I tillegg dekkes
82 % av cellene av ren 2-hop hub-labeling (felles hub i ut/inn-settene), som er
enda billigere per oppslag. Gevinsten vokser med n (på 2000² slår k=8-varianten
flat L=32; på 4000² er den nær L=64). Konklusjonen fra forsøk 2 snudd til
oppskrift: pyramiden virker — når hubene velges etter *stier*, ikke nærhet.
Naturlig fortsettelse: bytt `dil`-tabellen i MTZT med retningsdelte
sti-labels (n×2k i stedet for n×L), og hent hub-kandidater fra veiklassene i
dijeng-grafen for enda bedre dekning.

## Hva dette IKKE gjør (ennå)

- brooms local search bruker fortsatt en tett in-RAM-matrise — matcodec sitter
  på ingest/lagringssiden. Solver-hastigheten per iterasjon er uendret av dette
  arbeidet. Spaken som konverterer oppslagsfart til søkefart er å la LS-pruning
  forkaste trekk via `cell_bounds`/`cell_within` (de fleste trekk forkastes, og
  et trekk som kan forkastes på en ~250 ns-grense trenger aldri eksakt verdi).
- Pivot-utvalget trenger hele matrisen i RAM; streaming-stien bruker fortsatt
  jevnt spredte landemerker. Pivot-mining over et streamet utvalg er mulig,
  ikke gjort.
- Blokk-eksakthet krever at gateway-punktene finnes blant matrisepunktene.
  Graf-noder fra dijeng (ekte veikryss) som ekstra landemerker ville gjort
  ekte veidata like indeks-eksakte som syntetisk — formatet støtter det
  allerede (landemerker er bare rader/kolonner).
