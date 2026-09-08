# MPEE Planet — hele verdens veinett, offline

Et komplett offline rutedatasett for hele planeten: kjøreruter fra adresse
eller koordinat, eksakte avstander, geokoding begge veier, bompenger, og et
kart å tegne den blå streken på — alt fra én lokal mappe, uten nett.

Bygger på [MPEE](https://github.com/punnerud/mpee), som er bygget for **ett
nedlastet område**. Denne pakken bryter den begrensningen: den bytter ut
MPEE-motorens minnemodell med en out-of-core pipeline som håndterer hele
`planet.osm.pbf` (94,6 GB) på en laptop med 36 GB RAM.

## Nøyaktighetskontrakten

Dette er hele designpremisset, og det er verdt å være presis om det:

| | Nøyaktighet | Hvorfor |
|---|---|---|
| **Avstand for hele ruten** | **Eksakt** | Hvert segment måles én gang på *original* OSM-geometri med WGS84-metrikken i `f64`, lagres som heltall centimeter, og summeres i `u64`. Ingen flyttallsdrift, ingen avhengighet av hvor grovt linjen tegnes. |
| **Endepunktene** | ~1 m | Startpunktet snappes til nærmeste punkt på nærmeste *segment* — ikke nærmeste kryss, som kan ligge en kilometer unna. |
| **Den tegnede linjen** | 1 m | Douglas–Peucker med 1 m toleranse, lagret som rå `i32`-koordinatpar. |

Bare de to delvise endesegmentene er forholdsmessig fordelt. Alt imellom
bidrar med sin nøyaktige lagrede lengde.

Sfærisk haversine — som de fleste enkle rutemotorer bruker — bommer med opptil
0,5 %, altså **2,5 km på en 500 km rute**. Det ville gjort hele øvelsen
poengløs, så vi integrerer WGS84-metrikken i stedet (`geo.rs`, med Vincenty
som fasit for lange hopp).

## Hvorfor det får plass

Naiv tilnærming: behold hver OSM-node som en grafnode. Planeten har
**1,51 milliarder** noder på kjørbare veier — det sprenger både minne og
`u32`.

Nøkkelgrepet er **krysskontraksjon**: en grafnode er et kryss eller en
blindvei, en kant er strekningen mellom to slike. Målt på planeten
(planet-260831, 94 612 383 571 byte):

```
1 506 515 243 vei-noder   →   270 173 819 kryss        (17,8 %)
                              337 598 888 segmenter
                              638 184 250 rettede kanter
                                9 298 243 gatenavn
```

Formpunktene forsvinner fra *grafen*, men lengden er allerede målt på dem, så
avstanden er upåvirket. Av 1 225 933 240 formpunkter beholdes 754 062 123
(61,5 %) til tegning ved 1 m toleranse.

## Hvordan det bygges uten å sprenge minnet

MPEE-motoren legger alle OSM-noder i `HashMap<i64,(f32,f32)>`. På planeten er
det ~9,6 milliarder oppføringer — flere hundre GB. Løsningen er å aldri nøkle
noe på OSM-id:

1. **Pass 1** leser bare *way-halvdelen* av fila (funnet med binærsøk i
   blob-katalogen) og markerer i bitmap over rå id-rom hvilke noder en kjørbar
   vei berører, og hvilke som er kryss. Et kryss er nettopp en node den *andre*
   veien treffer — noe `AtomicU64::fetch_or` rapporterer gratis.
2. **Pass 2** leser bare node-halvdelen og skriver koordinater i et **tett**
   array indeksert av `rank1(id)`. Nodene kommer i stigende id-rekkefølge, så
   det er sekvensiell fylling, og alle senere oppslag er O(1) uten hashing.

Topp-RAM er de tre bitmapene (~1,8 GB hver), ikke dataene.

**Planet-scan: 113 sekunder.** Kontraksjonen som følger tar 19 minutter.

### To ting som bare dukker opp på planet-skala

Begge ble målt, ikke gjettet, og begge handler om *hva slags* minne noe er
— ikke hvor mye:

**Anonymt minne kan bare swappes ut.** Adressepasset holdt records i en
`Vec` på heapen. Den er anonym: OS-et kan ikke bare forkaste den, det må
skrive den til swap. Og det den fortrengte var nettopp koordinattabellen
passet trengte residente. Resultat: 22 % CPU-utnyttelse, resten venting.
Å skrive de samme radene til en fil og mmap-e dem tilbake er samme mengde
data, men *forkastbar* — utnyttelsen gikk til 75 %.

**En kald tabell koster en disk-lesing per oppslag.** Bygningssentroider
slår opp tilfeldig i de 16,5 GB med koordinater som pass 2 skrev en time
tidligere, og som ingenting hadde rørt siden. Hvert oppslag ble en
NVMe-lesing. Løsningen er ikke å lese smartere, men å lese *sekvensielt
først*: tabellen får plass i 36 GB, så én strømmet gjennomgang gjør alle de
etterfølgende tilfeldige oppslagene til cache-treff.

## Målt resultat — hele planeten

Bygget fra `planet-260831.osm.pbf` (94 612 383 571 byte) på en Apple M3 Pro
med 36 GB RAM og en ekstern SSD.

| | |
|---|---:|
| Kryss (grafnoder) | 270 173 819 |
| Rettede kanter | 638 184 250 |
| Segmenter | 337 598 888 |
| Adressepunkter | 238 961 343 |
| Distinkte gater (adresser) | 3 853 523 |
| Bomstasjoner | 62 611 |
| **Ferdig datasett** | **38,32 GB** |
| Byggeskrap (slettbart) | 44,4 GB |

Fordelingen:

| lag | GB |
|---|---:|
| graf (CSR + koordinater) | 14,53 |
| geometri (tegnet linje, 1 m) | 7,38 |
| segmenter (lengde, attributter, navn) | 6,75 |
| adresser + geokoding | 6,23 |
| romlig indeks | 3,06 |
| gatenavn | 0,27 |
| bomtakster (mpedb) | 0,02 |

Byggetid, ende til ende: **~55 minutter** (scan 113 s, kontraksjon 19 min,
graf 2,7 min, adresser 5,5 min, bom sekunder).

### Ruting, målt

Alle kald cache — første forespørsel i regionen. Avstandene er innen et par
prosent av virkeligheten.

| rute | avstand | noder besøkt | kald | varm |
|---|---:|---:|---:|---:|
| Oslo → Trondheim | 487,5 km | 1,4 M | 6,1 s | 0,20 s |
| Paris → Marseille | 774,7 km | 6,2 M | 4,9 s | 0,89 s |
| Berlin → Wien | 677,7 km | 6,5 M | 2,0 s | |
| New York → Boston | 344,5 km | 3,6 M | 1,0 s | |
| Tokyo → Osaka | 494,0 km | 5,4 M | 1,2 s | |
| Sydney → Melbourne | 858,4 km | 1,2 M | 0,4 s | |
| Nairobi → Kampala | 655,7 km | 1,1 M | 0,3 s | |
| Lisboa → Warszawa | 3 326,7 km | 46,0 M | 10,4 s | |
| Oslo sentrum (3,3 km) | 3,327 km | 2 374 | 7 ms | 2 ms |

En 140× større graf koster nesten ingenting varm: søket berører bare
korridoren mellom endepunktene, og Hilbert-ordningen gjør den korridoren til
sammenhengende sider.

## Bygging: enten helt, eller ikke i det hele tatt

Et datasett er ~45 løse filer i en katalog. Den ordningen har to feilmoduser,
og begge ble truffet under byggingen av dette:

* **En revet bygging ser ut som en ferdig.** Adressepasset ble avbrutt to
  ganger midt i. Ingenting på disk visste det, så katalogen inneholdt pass
  4-utdata ved siden av pass 5-utdata som ennå ikke fantes — og en server
  startet mot den ville svart selvsikkert fra et halvt datasett.
* **Proveniensen bodde i filtidsstempler**, som er en dårlig database. Hvilken
  planet-fil ga dette? Ved hvilken forenkling? Hvilke passeringer ble faktisk
  ferdige? Det ærlige svaret var «grep byggeloggen».

Derfor bygger en kjøring nå inn i sin egen katalog under `builds/`, fører hver
passering og hvert artefakt inn i en mpedb-katalog underveis, og blir live
først når siste passering har committet — ved å gi nytt navn til en `HEAD`-fil,
som er atomisk.

```
work/
  HEAD                 →  b1788793158
  catalog.mpedb           hva som skjedde
  builds/
    b1788793158/          det som er live
    b1788793061/          en bygging som døde, beholdt som bevis
```

`HEAD` er en tekstfil og ikke en rad i katalogen, fordi en leser må finne
datasettet *før* den kan åpne noen database, og fordi `rename(2)` er
primitivet som gjør byttet udelelig. Katalogen skrives først og `HEAD` sist:
et krasj mellom dem etterlater en ferdig, upublisert bygging og lesere på
forrige datasett — den retningen det ikke koster noe å feile i.

```bash
mpee-planet build  planet-latest.osm.pbf work 1.0 nvdb.json   # stag, verifiser, publiser
mpee-planet builds work        # hva som er bygget, og hva som er live
mpee-planet verify work        # er datasettet internt konsistent?
mpee-planet prune  work 1      # slett gamle bygginger (aldri den live)
mpee-planet adopt  work src.pbf  # ta et eldre datasett under forvaltning
```

### Datasettet beskriver seg selv

Hver array er en tett tabell av faste poster, og lengdene henger sammen:
`csr.to` må holde nøyaktig én `u32` per kant, `geom.off` én per segment pluss
en terminator, `addr.coord` ett koordinatpar per adresse. `verify` regner
etter, og en bygging som ble revet midt i en passering gir lengder som ikke
kan være sanne samtidig:

```
  csr.rto: is 1000000 bytes, but the rest of the dataset implies 16251876
```

Det koster ingenting — ikke én byte innhold leses. `--deep` legger på
BLAKE3-digester fra katalogen, som fanger det lengdene ikke kan se: en flippet
bit i en fil som fortsatt har riktig størrelse.

Fordi lengdene *bestemmer* tallene, kan et datasett som er eldre enn katalogen
adopteres med **målt** statistikk i stedet for tall kopiert fra en logg. Da
planeten ble adoptert ga utregningen 270 173 819 noder, 638 184 250 kanter og
238 961 343 adresser — nøyaktig det passeringene hadde rapportert timer før.

## Lokale endringer, uten ombygging

En vei stenges for veiarbeid, en fartsgrense er feil, en bru er ute. Datasettet
tar 55 minutter å bygge om og grafen det produserer er en uforanderlig mmap, så
verken å vente eller å redigere på stedet er mulig. Overstyringer er derfor et
tynt, foranderlig lag ruteren konsulterer.

### Hva en overstyring er nøklet på — og hvorfor det avgjør alt

**Ikke segment-id.** Segment-id-er er *bygge-lokale*: neste ombygging
renummererer hver eneste én, så en overstyring lagret mot `seg 41209883` ville
stille begynt å gjelde en annen vei. Det er samme feilklasse som en priskolonne
som parser til null — en bug som ser ut som data.

En overstyring er derfor nøklet på det som overlever en ombygging: **et sted på
bakken**, eventuelt avgrenset med gatenavn. Den bindes til de segmentene som
ligger der i datasettet som er live *nå*, og bindingen rapporteres:

```
$ mpee-planet override add work closed 59.9232,10.7295 \
      --radius 300 --street Hegdehaugsveien --note "bridge out"
override 1 added — matches 9 segment(s)
```

En regel som ikke treffer noe sier fra med én gang, i stedet for å gjøre
ingenting i stillhet — og forteller hva som *faktisk* ligger der:

```
override 1 added — matches 0 segment(s)
  nothing matched. Roads within 600 m:
    (unnamed)  (19 segment(s))
    Jernsætervegen  (5 segment(s))
```

Det var slik en ekte mangel ble funnet: motorveier bærer `ref=E 6` og som
oftest ingen `name` i det hele tatt, så de lengste strekningene av en rute sto
blanke — både i steg-listen og som mål for en overstyring. Segmenter uten navn
arver nå sin `ref`, som ikke koster noe (samme pool) og er slik folk omtaler
disse veiene uansett.

**Og det holder over en ombygging.** Målt: samme rute, 3,554 km med stengingen;
hele datasettet bygget om fra PBF-en (nytt bygge-id, alle segment-id-er
renummerert); fortsatt 3,554 km, fortsatt de samme 9 segmentene stengt.

### Tre slag

| slag | virkning |
|---|---|
| `closed` | Kanten slappes ikke i det hele tatt |
| `speed` | Kjør den i denne farten i stedet for den tagget |
| `penalty` | Gang kjøretiden — `> 1` fraråder, `< 1` oppfordrer |

`speed 10` og `penalty 5` på en 50-vei gir samme svar, som de skal.

### Kostnaden i den varme løkken

En planet-rute slapper ~11 millioner kanter, så «er dette segmentet
overstyrt?» må besvares med en bit-test, ikke et oppslag. Oppløsningen
produserer derfor et *tilstedeværelses-bitmap* over segment-id-er (42 MB for
planeten) med en liten sortert tabell bak. Målt på Oslo → Trondheim,
1,1 M noder besøkt, median av 5:

```
ingen overstyringsdatabase  81 ms
tom database                81 ms
én overstyring (9 segmenter) 82 ms
```

### Anvendt uten omstart

Overstyringene ligger i `overrides.mpedb` i **roten**, ikke i byggekatalogen —
de skal overleve datasettet de ble skrevet mot. En kjørende server leser en
generasjonsteller én gang per forespørsel (støy ved siden av en rute) og
løser opp på nytt bare når den faktisk har endret seg. Fordi mpedbs lesere
aldri blokkeres av skriveren, lander en endring i *neste spørring*, ikke i
neste utrulling:

```
1. server kjører, baseline       : 3.327 km  4.9 min   overrides_active=0
2. en annen prosess skriver en stenging
3. samme server, neste request   : 3.554 km  5.6 min   overrides_active=9
```

Ingen omstart, ingen ombygging. `/api/overrides` forteller hva serveren
faktisk honorerer, og kartet tegner stengingen.

## Overlay: samle en region til veiene som går inn i den

En lang rute krysser tusenvis av områder den aldri kjører inn i. Uten hjelp
besøker søket hver eneste node i dem likevel, for det har ingen måte å vite at
et boligfelt med tre innkjørsler ikke er verdt å gå inn i. Overlay-en gir den
den kunnskapen: hver region erstattes, for gjennomgangstrafikk, av en tabell
over korteste vei mellom **grensenodene** — de en kant krysser inn i.
Interiøret røres aldri.

Dette er Customizable Route Planning, og det er matcodecs gateway-struktur
brukt på grafen i stedet for på en matrise: prediker regionen ved sine få
innganger, og behold den eksakte kostnaden mellom dem.

### Partisjonen er hele spillet

Målt på Norge, samme målstørrelse på 4 096 noder:

| | grensenoder | overlay | regioner med ≤4 porter |
|---|---:|---:|---:|
| geografiske blokker | 2,25 % | 18,4 MB | **0** |
| **grafavstand** | **2,00 %** | **5,4 MB** | **902** |

Å skjære langs lengde- og breddegrader går tvers gjennom tette områder; å la
regionene vokse langs veiene finner halvøyene, dalene og feltene bak ett kryss.
Samme målstørrelse, en tredjedel av overlay-en.

### Hva den faktisk kjøper

Oslo → Trondheim på planeten, hver ruter i sin egen prosess:

| | vanlig | overlay |
|---|---:|---:|
| klokketid | 1,02 s | **0,03 s** |
| topp-RSS | **4,39 GB** | **58 MB** |
| sidehentinger | 268 407 | **3 725** |
| noder besøkt | 1 373 698 | 38 569 |

De 4,39 GB er søkets egne avstands- og forelder-arrays, én oppføring per node
i grafen. Overlay-søket berører bare grensenoder, så det holder dem i hash-kart
— og datasettet blir brukbart på en maskin med 512 MB.

**Men det er ikke en CPU-optimalisering.** Hver grensenode slapper ~139
tabelloppføringer der en vanlig node slapper 2,4 kanter, så overlay-en gjør
*mer* arbeid og berører *mindre* minne. Målt over seks ruter på fire
kontinenter: mellom 3,4× raskere (Oslo) og 0,5× — altså tregere — (Tokyo), på
en maskin med RAM til overs. På 512 MB er det forskjellen mellom mulig og
umulig.

### Eksakthet

Alle seks rutene ga **+0 ds** mot det vanlige søket, og 40 av 40 tilfeldige
nodepar ga nøyaktig samme svar som en referanse-Dijkstra. En celletabell holder
korteste vei mellom to grensenoder **med bare kanter inne i regionen**; en
ekte korteste vei som går ut og inn igjen er ikke tapt, for overlay-søket
finner den ved å gå ut gjennom grensen og inn igjen — akkurat som veien gjør.

Sammenligningen avslørte også en feil i den eksisterende ruteren:
`build_route` valgte blant parallelle kanter etter korteste *lengde* mens søket
minimerte *tid*, så avstanden var alltid riktig, men varigheten kunne
overrapporteres.

### Tabellene er smalnet per region, ikke komprimert

Regiontabellene er den største enkeltdelen av datasettet, og de er fulle av
tall som ikke trenger plassen de får. Vi målte tre ting før vi skrev noe:

- matcodecs rang-1-modell traff **5,19 %** av oppføringene eksakt. Den modellen
  beskriver *gjennomgangsstruktur mellom* regioner; våre tabeller er *innenfor*
  én. Feil modell, og residualene ble større enn tallene.
- rad-minimum som prediktor kjøpte ingenting.
- men **97,69 %** av alle verdier får plass i 16 bits rått, og de brede
  verdiene klumper seg: **67 av 19 208** regioner (0,35 %) trenger 32 bits.

Så vi lagrer bredden per region — ett byte i `ov.width` — og lar `cost()`
gruble over den. Grenen er konstant gjennom hele regionen, så prosessoren
gjetter den riktig hver gang, og oppslaget er fortsatt én indeksert lesning
uten dekoding.

| | 32 bits overalt | bredde per region |
|---|---:|---:|
| Norge | 1,00× | **1,83× mindre** |
| planeten | 2,79 GB | **1,61 GB** (1,73×) |

### Partisjonen strandet en tredjedel av grensegrafen

BFS-en fyller en region til 4096 noder, stopper, og forlater det som lå igjen
på fronten. De nodene ble plukket opp senere som egne regioner — og målt på
planeten var **1 756 486 regioner nøyaktig én node**, hvorav **1 685 456 hadde
kanter** (snittgrad 1,10, altså blindveitupper). En blindvei som ligger alene
er tvunget til å være grensenode: alle kantene dens forlater regionen.
Nærmere 30 % av grensegrafen var slike fragmenter.

Å absorbere en tupp i nabo­regionen er gratis i tabellstørrelse. Den bringer
ingen ny grensenode med seg — den har ingen andre steder å gå — så `b` er
uendret og `b²` med den. Passet kan bare fjerne grensenoder.

| | før | etter |
|---|---:|---:|
| regioner | 4 129 904 | 1 254 558 |
| grensenoder | 12 650 695 (4,68 %) | **6 503 968 (2,41 %)** |
| tabell | 3,03 GB | **1,61 GB** |

Norge falt tilsvarende, fra 2,28 % til **0,73 %** grensenoder.

### Et andre nivå — og hva det faktisk var verdt

Nivå 2 gjør det samme én etasje opp: det kollapser en *gruppe* regioner til
veiene som går inn i gruppen, så å krysse et kontinent leser én tabelloppføring
per gruppe i stedet for å gå gjennom hver grensenode underveis. Konstruksjonen
er selvlik — en nivå-2-tabell bygges av *samme* restrikterte Dijkstra, kjørt
over nivå-1-overlaygrafen i stedet for veigrafen — og partisjonen er nøstet, så
en nivå-2-region er en union av hele nivå-1-regioner.

Planeten: 1 140 914 nivå-2-regioner, 1 763 604 grensenoder (27,1 % av nivå 1).

| rute | nivå 1 | + nivå 2 |
|---|---:|---:|
| Oslo → Trondheim | 12 663 besøkt, 136 ms | **5 163, 14 ms** |
| Tokyo → Osaka | 221 848, 1 093 ms | **74 299, 889 ms** |
| Lisboa → Warszawa | 896 383, 6 911 ms | **261 605, 6 839 ms** |

Det tredjedeler antall besøkte noder, men **på den lange ruten kjøpte det ingen
tid** når cachen er romslig: hver nivå-2-node slapper ~856 naboer der en
nivå-1-node slapper ~5, så arbeidet per node vokste like mye som antallet falt.
Gevinsten kommer først når minnet er trangt, fordi færre noder er færre sider:

| rute | 512 MB | 256 MB | 128 MB | 64 MB |
|---|---:|---:|---:|---:|
| Oslo → Trondheim | 0,01 s | 0,11 s | 0,01 s | **0,008 s** |
| Tokyo → Osaka | 0,9 s | 1,6 s | 2,0 s | **3,1 s** |
| Lisboa → Warszawa | 6,8 s | 7,8 s | 12,9 s | **16,9 s** |

Mot ett nivå uten absorpsjon var Lisboa → Warszawa **115,5 s** på 64 MB, og
Oslo → Trondheim 0,15 s. Alle kostnader er fortsatt identiske.

Prisen er ærlig: `l2.mat` er 4,64 GB — den nest største filen i datasettet, og
den smalner bare 1,09×, fordi nivå-2-regioner er store nok til at interne
avstander sprenger 16 bits. Den fortjener plassen sin på 64 MB og knapt ellers.

### Hvor mye grovere et nivå bør være — målt, ikke antatt

Første forsøk tilpasset en rett linje gjennom to punkter: grensen så ut til å
krympe som `A^-0,20`, og den eksponenten ble brukt til å argumentere for at
grovere nivåer ikke lønner seg. **Et tredje punkt viste at den slutningen var
feil.** Eksponenten er ikke konstant — grensen faller *raskere* jo grovere
nivået blir:

| steg | grensenoder | eksponent |
|---|---:|---:|
| 215 → 16 384 (76×) | 6,50M → 4,01M | `A^-0,111` |
| 16 384 → 131 072 (8×) | 4,01M → 1,76M | **`A^-0,396`** |

Og spørringen sier det samme. Lisboa → Warszawa på 64 MB:

| nivå-2-steg | grensenoder | tabell | besøkt | tid |
|---|---:|---:|---:|---:|
| 4× (16 384) | 4,01M | 2,86 GB | 570 843 | 30,5 s |
| 32× (131 072) | 1,76M | 4,64 GB | 261 605 | **16,9 s** |

Grovere er bedre, og det peker på hva formen egentlig skal være: ikke ett
ekstra nivå med en gjettet størrelse, men en **rekursjon** — øyer inne i øyer.
Grensenodene er selv et veinett, og nivå-2-tabellen bygges allerede av samme
restrikterte Dijkstra kjørt over nivå-1-overlaygrafen. Så koden bør være en
løkke, og den bør rekursere til toppnivåets grensegraf er liten nok til at man
rett og slett **søker direkte i den**. Med `-0,396` videre fra 131 072 lander
et toppnivå grovt på 150–350 k noder, og en Dijkstra over 160 k noder er
millisekunder.

Det fjerner samtidig behovet for et eget transittnode-bord: man trenger ikke
`T×T` når toppen er søkbar. Det er verdt å si tydelig, fordi et tidligere
utkast av dette dokumentet påsto det motsatte på grunnlag av
to-punkts-tilpasningen.

### Stigen

Så nivåene er ikke to, de er en rekursjon som stopper av seg selv. Hver rung
er `STEP` (32) ganger grovere enn den under, og løkken gir seg på det første
av tre: toppen er liten nok til at en vanlig Dijkstra over den er triviell, en
rung kastet mindre enn 10 % av grensen (så neste ville koste mer enn den
sparer), eller taket på antall nivåer.

Alle nivåer har samme form, fordi et nivå *er* et veinett: nodene er nivået
under sine grensenoder, kantene er det nivåets tabellsnarveier pluss
snittkantene mellom regionene. Derfor bygges hver rung av **samme** restrikterte
Dijkstra som nivå 0, bare kjørt over nivået under i stedet for veigrafen — og
en region som holder fire millioner veinoder søkes som en graf på noen tusen.
Partisjonen er nøstet, så hver rungs grense er en *delmengde* av rungen under.

Norge, tvunget lavt for å vise formen (`overlay2 no-managed 32768 1500`):

| nivå | regioner | grensenoder |
|---|---:|---:|
| 0 | 10 772 | 14 174 |
| 1 | 9 953 | 6 122 |
| 2 | 9 610 | **1 473** |

Deretter stopper den: 1 473 noder søker man rett i.

Filene heter `cell.*`/`ov.*` for nivå 0 og `l2.*`, `l3.* …` oppover, så et
datasett bygget før stigen fantes åpner uendret. `verify` krever at hver rung
er komplett *og* at stigen ikke har hull — en manglende rung under en som
finnes ville latt en spørring klatre til et nivå den ikke kan komme ned fra.

### Målt på et lite minne, ikke argumentert for det

Påstanden om at dette kjører på 512 MB hvilte på at hvert array er en ren,
skrivebeskyttet filmapping: minnet et søk berører er en *cache*, ikke en
allokering. Det er sant, men på en maskin med 39 GB blir det aldri satt på
prøve — ingenting tar sidene fra deg.

Så prosessen tar dem fra seg selv. `MPEE_CACHE_MB` setter et tak på residens;
når søket krysser det, leveres de mappede sidene tilbake til kjernen og må
hentes fra disk igjen. Det som blir igjen er søketilstanden, som er anonym og
ikke kan gjenvinnes.

| rute | søketilstand | 512 MB | 256 MB | 128 MB | 64 MB |
|---|---:|---:|---:|---:|---:|
| Oslo → Trondheim | 1,3 MB | 0,13 s | 0,12 s | 0,13 s | **0,15 s** |
| Tokyo → Osaka | 12,8 MB | 2,4 s | 2,3 s | 2,3 s | **3,1 s** |
| Lisboa → Warszawa | 84,2 MB | 9,2 s | 12,1 s | 63,3 s | **115,5 s** |

(Tallene over er ett nivå og strandet partisjon. Se lenger opp for hva
absorpsjonen og nivå 2 gjorde med dem.)

**Hver eneste kostnad er identisk med den ucappede — helt ned til 64 MB.**
Det er hele poenget: å kaste en ren side endrer ingenting, den kommer tilbake
lik. En test fester det (`capping_the_page_cache_changes_timing_but_never_the_answer`),
med taket satt til én byte så sidene slippes ved hver eneste sjekk.

Oslo → Trondheim treffer aldri taket i det hele tatt — den topper på 54 MB og
går like fort på 64 MB som på 1 GB. Lisboa → Warszawa, 2 900 km tvers over
Europa, er brukbar ned til 256 MB og faller så av en klippe. Klippen har en
enkel forklaring: søketilstanden er 84 MB anonymt minne. På 128 MB er den
alene to tredjedeler av budsjettet, og de resterende ~44 MB er for lite cache
til å holde regionene den arbeider i. Det er den grensen, ikke datasettets
størrelse, som bestemmer hvor lite minne en gitt rute kan kjøre på.

## Kom i gang

```bash
cargo build --release

# Hele verden (94,6 GB nedlasting, ~3,4 TB ledig anbefalt under bygging)
wget -c https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf

mpee-planet scan     planet-latest.osm.pbf work   # pass 1+2
mpee-planet contract work 1.0                     # pass 3 — 1 m tegnetoleranse
mpee-planet graph    work                         # pass 4 — Hilbert + CSR
mpee-planet addr     work                         # pass 5 — geokoding
mpee-planet toll     work                         # pass 6 — bomstasjoner

mpee-planet stats work        # størrelsesoversikt
mpee-planet serve work        # http://127.0.0.1:8099
```

Et land går på sekunder — Norge bygger komplett på under 5 s og blir 0,30 GB.

### Kommandolinje

```bash
mpee-planet route work 59.9109,10.7527 63.4366,10.3986
mpee-planet find  work "Karl Johans gate" 22 Oslo
mpee-planet rev   work 59.9109,10.7527

# Kjør ruten slik den ville kjørt på en liten maskin: MPEE_ROUTER=overlay
# velger overlay-søket, MPEE_CACHE_MB setter tak på residente sider.
MPEE_ROUTER=overlay MPEE_CACHE_MB=512 mpee-planet route work 59.9109,10.7527 63.4366,10.3986
```

## Kartet er datasettet

Det er ingen kartflis-avhengighet. `/api/roads` streamer veigeometri for
utsnittet rett fra de samme mmap-ede arrayene ruteren bruker, så bakgrunnen i
kartet *er* datasettet. Trekk ut nettverkskabelen og alt virker likt.

To indekser gjør det mulig: et finmasket rutenett på 0,01° for snapping og
bynær tegning, og et grovt på 1° med bare hovedveier for kontinent-zoom.

## Bompenger — og hvor grensen mot databasen går

En bomstasjon er ikke en egenskap ved en vei — den er en egenskap ved å
**passere et punkt i en retning**. Oslo-ringen tar betalt inn, ikke ut. Derfor
tvinges hver `barrier=toll_booth` / `highway=toll_gantry` til å bli en
grafnode, og ruteren rapporterer portalene den passerer *gjennom*.

Datasettet er delt ett bestemt sted, og bare der:

| | hvor | hvorfor |
|---|---|---|
| «er node 41 209 883 en bomstasjon?» | **array** (`toll.vertex`) | Spørres én gang per node i et søk som besøker millioner. Binærsøk, ingen spørring. |
| «hva koster den, for hvem, når?» | **mpedb** (`toll.mpedb`) | Lite, foranderlig, kommer fra flere kilder, og spørsmålene er relasjonelle. |

Det er samme grense som går ved grafen selv: en datastruktur du traverserer på
den ene siden, et datasett du spør på den andre.

### Hva flat fil ikke kunne

`toll.tariff.tsv` hadde ett tall per bomstasjon. NVDB publiserer langt mer, og
alt sammen er nå adresserbart:

```
                     flat fil        mpedb
takster per portal   1               inntil 4 (liten/stor bil × standard/rush)
tidsdimensjon        ingen           346 rushtidsvinduer, som `time`-kolonner
timesregel           per rute        per anlegg, med varighet og passeringsgruppe
manglende pris       ble 0.0         står som ukjent, og telles
```

Det siste er ikke en detalj. Den gamle parseren endte i `.unwrap_or(0.0)`, så
en skrivefeil i prisfila ble stille til en **gratis bomstasjon**. Nå er prisen
en `numeric` — eksakt desimal, aldri flyttall — og en pris som ikke lar seg
tolke fører til at portalen står som upriset i stedet for som gratis.

### Spørsmålet som ikke var tilgjengelig

Oslo S → Trondheim S, samme rute, fire svar:

| | når som helst | 07:15 (rush) |
|---|---:|---:|
| liten bil | 145 NOK | **175 NOK** |
| stor bil | 351 NOK | **444 NOK** |

Oppdelingen er internt konsistent: 223 (Kværnerveien, rushtakst for stor bil)
+ 47 + 104 + 70 = 444. E6 Lodalen passeres, men står som dekket av
Fjellinjens timesregel; E6 Vindåsliene faller tilbake til standardtakst fordi
den ikke har noen rushtakst — den ligger utenfor en byring.

### Og spørsmål vi ikke visste at vi hadde

Å legge det i en database gjorde datakvaliteten synlig. Et sveip over
planetens `charge`-tagger fant `4.50 BRL/motorcar`, en naken `€`, prisen
`25000` uten valuta, og `50 RUR @ (class:1 AND 7:00-0:00)` — betinget
prising som denne modellen ikke kan innfri. En løs parser gjør hver av dem til
et selvsikkert galt tall. Parseren avviser dem nå, og 282 falske takster
forsvant; portalene står i stedet som upriset, som er svaret en disponent kan
handle på.

```bash
mpee-planet tolldb work nvdb-bomstasjoner.json   # bygg takstlaget
mpee-planet tollq  work "SELECT ..."             # spør det om hva som helst
mpee-planet route  work <fra> <til> hgv 07:15    # pris ruten
```

Norge henter takstene fra **NVDB** (Statens vegvesen, objekttype 45) — se
`scripts/nvdb-toll.py`. 433 av 463 stasjoner matcher OSM-portalene innen 300 m.
Resten av verden får det OSM har.

## Hvorfor rå lagring, ikke komprimert

Geometrien kunne vært delta-varint-kodet til ~1/3 av størrelsen. Den er det
ikke: hele datasettet får plass ukomprimert, og rå `i32`-par gir O(1)
oppslag og null dekoding når linjen tegnes. Enkelhet og hastighet foran noen
GB vi ikke trenger.

## Verifisering

Ingenting her hviler på øyemål:

* `tests/pbf_conformance.rs` — den håndskrevne PBF-leseren kjøres mot
  `osmpbf`-crate-en på et ekte utdrag og må stemme på node-/way-antall,
  id-sjekksummer, koordinatsjekksummer, ref-sjekksummer og tag-bytes.
* `tests/routing_correctness.rs` — toveis-søket sammenlignes med en ren
  Dijkstra på tilfeldige par (toveis-søk er nettopp der rutemotorer stille
  går galt); rutens lengde må være heltallssummen av delene; og snapping
  kontrolleres mot et brute-force-sveip av alle segmenter i nabocellene.
* Enhetstester for rank/select, geodesi, forenkling, Hilbert, profil og
  adressenormalisering (æøå bevares — `Bogata` og `Bøgata` er ikke samme gate).

```bash
cargo test --release
MPEE_TEST_PBF=norway-latest.osm.pbf cargo test --release --test pbf_conformance
MPEE_TEST_DATA=work-no             cargo test --release --test routing_correctness
```
