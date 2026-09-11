# Plan: adaptiv partisjonering

Partisjonen skal forbedre seg *mens datasettet betjenes*, styrt av trafikken og
en kalibrert kostnadsmodell, innenfor harde grenser for disk, minne og latens.
Ingen ombygging. Samme grep som flyttet byggingen fra 27 minutter til 34
sekunder: arbeidet flyttes fra «alt på forhånd» til «det som trengs, når det
trengs».

Databaseverdenen kaller det *adaptive indexing*. Analogien til MPEE-planleggeren
i MPEdb er strukturell, ikke løs: samplet statistikk → kostnadsmodell → harde
grenser → valg, og statistikken samles inkrementelt under kjøring, ikke i et
analysepass.

---

## Hvorfor: det som er målt

Alt under er målt på planeten i dag, 9. september 2026.

**Problemet er ekstremt konsentrert.** På nivå 2 ligger 100 % av
fyllearbeidet i 0,1 % av regionene — tusen av 1,12 millioner. Median-regionen
har 7 porter; de verste har 9 883. Å gjøre de 99,9 % raskere endrer ingenting.

**Gode øyer finnes, men blir funnet ved uhell.** 3 245 regioner på nivå 0 har
i snitt 8 porter og 2 400 noder — nøyaktig «få veier inn, mange internt». De
oppstår fordi BFS-en tilfeldigvis stoppet på et heldig sted. Ingen leter etter
dem. Fra 75-persentilen og opp står nodetallet stille på ~4 300 mens portene
vokser fra 76 til 219: det er regioner som traff størrelsestaket og ble kappet
der, uansett hvordan snittet så ut.

**Disk *er* en avveiningsakse — men det tok en måling på planeten å se det.**
På Norge var `Σb²` flat, 1,7 til 2,0 MB over et 64× spenn i regionstørrelse,
og det ble tolket som empirisk bekreftelse av at tabellstørrelsen er `c²·n`
uavhengig av regionstørrelse. **Fase 0 motbeviste det.** På planeten vokser
`Σb²` fra 2,32 til 3,87 GB over samme spenn — 67 %. Norge er for lite og for
dominert av frakoblede komponenter til å oppføre seg som et veinett.

| target | grensenoder | disk `Σb²` | b median | b p99 |
|---:|---:|---:|---:|---:|
| 1 024 | 11 073 033 | 2,32 GB | 26 | 108 |
| 4 096 | 6 503 968 | 2,79 GB | 23 | 219 |
| 16 384 | 3 845 485 | 3,48 GB | 13 | 432 |
| 32 768 | 2 940 543 | 3,87 GB | 10 | 595 |

**Grovere snitt: 3,8× færre grensenoder, 1,67× mer disk, og patologien
vokser.** Median-porttallet faller fra 26 til 10 mens 99-persentilen
femdobles, 108 → 595. Grovere gjør de fleste regionene bedre og de verste
mye verre — konsentrasjonsproblemet, forsterket.

Grensegrafen er både det spørringen går gjennom og det som må være resident,
så tre akser peker i to retninger. Det er ingen fri gevinst; det er et valg
som krever en policy — og det er nettopp derfor fase 2 må lande først.

**Hastigheten så flat ut på Norge, og det var målegulvet.** 9 630 besøkt og
5 025 besøkt ga begge 7 ms. På planeten, der de samme konfigurasjonene gir
1,5 millioner mot 419 000 besøkte noder, er forskjellen 2 252 mot 922 ms. En
måling som ikke kan skille to konfigurasjoner sier ingenting om at de er like.

**Port-tak ville gitt 22× på nivå 2**, i både disk og fyllearbeid. Anslaget er
optimistisk: det teller ikke portene et snitt selv skaper.

**Trafikken har korridorstruktur, men ikke ekstremt.** 5 % av segmentene
bærer 45 % av passeringene; 0,1 % bærer 1,5 %. Nok til at *hvor* et snitt
legges betyr noe; ikke nok til et transittnode-bord.

**`TARGET = 4096` var den siste ubegrunnede konstanten** da dette ble skrevet.
Det stemmer ikke lenger, og to av påstandene rundt den var feil: stoppregelen
på 200 000 var *ikke* erstattet — den lå i `build_ladder` og kappet stigen for
tidlig helt til 10. september — og nivå 3 ble ikke avvist på kost, det ble
bygget. Se «Natten som målte partisjonen» nedenfor.

---

## Hva som finnes

Alt dette er bygget og testet (117 tester, 0 clippy):

- Late nivåer med fylling per rad og atomiske tilstedeværelsesbiter
- Invalidering per region (`invalidate`, `invalidate_for`), trygg under
  betjening
- Gjenopptakbar oppvarming med QoS-styring og to pooler (`MPEE_WARM_FAST`,
  `MPEE_WARM_SLOW`)
- Kalibrert fyllekostnadsmodell (`fill_estimate`), treffer på 8 % på nivå 2
- Diff mot ny planetfil → berørte regioner, 60 s og 0,17 GB
- `MPEE_MAX_LEVEL` for å måle stigehøyder uten ombygging
- `betweenness`-verktøy som sampler ruter og teller segmentbruk
- Nøstet partisjon: en region treffer nøyaktig én region per nivå

## Hva som mangler

Én operasjon og to estimatorer:

1. **Splitt en region in place**, langs et snitt gitt av dens egne rader.
2. **Hastighetsmodell**: `Σ b(region)` langs samplede ruter.
3. **Minnemodell**: besøkte noder × bytes per rad, samme datakilde.

Og en policy som binder dem sammen.

---

## Faser

Hver fase har et målbart utfall og kan stoppes etter.

### Fase 0 — Bekreft premisset på planeten ✅ GJORT

Kjørt 9. september 2026, seks punkter à 30 s i en APFS-klone (0,114 s å
klone 74 GB; ingen disk forbrukt før noe skrives).

**Utfallet var negativt, og det var poenget.** «Grovere er bedre for minne,
nøytralt for disk» holdt på Norge og holder *ikke* på planeten: disk vokser
67 % over spennet. Se tabellen over.

Planen stopper ikke — adaptiviteten er like riktig — men fase 4 må balansere
tre akser i stedet for to, og valget av hvor på kurven man skal ligge kan ikke
tas uten fase 2.

For dagens begrensninger er byttet fortsatt gunstig: datasettet er 40,27 GB av
50, og 4 096 → 32 768 koster ~1,1 GB for 2,2× mindre grensegraf. **Minne er
den bindende grensen, ikke disk.** Men det er et valg nå, ikke en gratis
gevinst.

### Fase 1 — Statistikk som biprodukt ✅ GJORT

Implementert 9. september 2026. `l{n}.use`: én `u32` per medlem, parallell med
`vlist`. Valgfri på filnivå — mangler den, koster den én null-sjekk, og et
eksisterende datasett virker uendret.

| | |
|---|---|
| kostnad | **0,7 %** på oppvarming (3,02 s mot 3,00 s, tre kjøringer hver) |
| lagring | 4 byte per grensenode på nivået under — **7 MB** på planetens rung 1 |
| signal | **8,1×** spredning mellom travleste 5 % og median-medlem, nivå 1 |

**Første forsøk telte feil ting, og signalet var matematisk umulig.** Å telle
*besøk* ga nøyaktig 1,0× spredning — ikke fordi tellingen var ødelagt, men
fordi full oppvarming regner ut hver rad, og hver rad besøker hvert medlem én
gang. Telleren målte antall rader.

Betweenness er hvor mange korteste veier som går *gjennom* et medlem, og det
krever treet. Rettelsen: `par`-array per rad, skrevet ved forbedrende
relaksering, og `b` tilbakevandringer etter at raden er ferdig — altså utenfor
den innerste løkken, der et binærsøk per relaksering allerede dominerer.
Kostnaden ble den samme 0,7 %.

**Signalet er sterkest lavt i stigen.** Nivå 1 gir 8,1×; nivå 2 gir 1,8×. Jo
grovere nivå, jo flatere betweenness — på toppen er alt en gjennomfartsåre.
Snittstyring er mest verdt der nede.

### Fase 2 — De manglende estimatorene ✅ GJORT

Kjørt 9. september 2026. Fire granulariteter på planeten, samme rute
(Lisboa → Warszawa), nivå 0 alene, korridoren varmet av rutens egen
førstekjøring.

| target | grensenoder | besøkt | `k` | varm | kald |
|---:|---:|---:|---:|---:|---:|
| 1 024 | 11 073 033 | 1 503 315 | 0,1358 | 2 252 ms | 26 s |
| 4 096 | 6 503 968 | 895 851 | 0,1377 | 1 118 ms | 52 s |
| 16 384 | 3 845 485 | 539 233 | 0,1402 | 1 013 ms | 135 s |
| 32 768 | 2 940 543 | 419 072 | 0,1425 | 922 ms | 219 s |

**Det som holder:**

```
besøkt(rute, partisjon) = k(rute) × grensegraf(partisjon)
```

`k` varierer 5 % over et 32× spenn i granularitet — den tilhører **ruten**, ikke
partisjonen. Bekreftet uavhengig på Norge, der tre ruter ga 0,655/0,679/0,655,
0,639/0,674/0,664 og 0,042/0,034/0,039. Merk den siste: `k` er 16 ganger
mindre for en kort rute, så den må samples per trafikkprofil. En universell
konstant finnes ikke — det var den første antakelsen målingen felte.

**Det som ble falsifisert:** at arbeidet er `k × Σb²`. Modellen forutsa at
32 768 skulle være 1,67× *tregere* enn 1 024 fordi `Σb²` vokser. Tiden falt
2,44×. Forklaringen er sidelesning: hver besøkt node leser én sammenhengende
rad, og 52 oppføringer og 329 oppføringer er begge under én side. Kostnaden
følger antall besøk, ikke besøk ganger portantall.

```
tid  ≈ besøk^0,66     (log-log r = 0,92)
kald ≈ besøk^-1,68
```

**Men eksponenten er ikke konstant over kurven.** 1 024 → 4 096 halverer tiden;
4 096 → 32 768 gir bare 1 118 → 922 ms. Nesten hele gevinsten ligger i første
halvdel. En to-punkts tilpasning ville lest 2,4× som en rett linje — samme
feil som ble gjort med grenseeksponenten `A^-0,20` tidligere.

**Konsekvens for policyen.** Fra 4 096 til 32 768:

| | |
|---|---|
| varm spørretid | 1,21× raskere |
| minne (grensegraf) | 2,21× mindre |
| disk | 1,39× mer |
| **kald fyllekost** | **4,2× dyrere** |

`TARGET = 4096` ligger nær et fornuftig punkt. Å gå grovere kjøper hovedsakelig
minne og betaler i fyllekost. Hvilken vei det faller avhenger av driftsmodellen:
**varmes alt på forhånd, lønner grovere seg; varmer trafikken, lønner finere.**

*Gjenstår i denne fasen:* fyllekostnadsmodellen underestimerer — den anslo
5,0 min for nivå 1 der vi målte 13. For en regel som *avviser* er det den
farlige retningen. `besøk^-1,68` er nå et bedre grunnlag enn `Σ(b·m)·b_under`
og bør erstatte den.

### Fase 3 — Splitt-operasjonen ✅ GJORT

`split_level(paths, ds, ov, k, plan)` flytter deler av en region ut i en ny,
og beholder hver annen regions rader.

Målt på Norge: 24 regioner delt, **13 370 rader beholdt**, 5 127 regnet om,
30 ruter uendret.

**Radene overlever fordi `head` ikke er monoton.** En region hvis grense er
uendret beholder sin forskyvning og sine rader; bare de som faktisk flyttet
får ny plass bakerst. Den gamle plassen blir avfall til en komprimering —
log-strukturert lagring. Indeksarrayene (`bnd`, `bhead`, `vhead`, `vlist`)
bygges om i sin helhet, men de er 31 MB mot 5,6 GB tabell på planetens nivå 0.

**Layoutet måtte endres først, og det var ikke forutsett.** Bakoverradene lå
forskjøvet med hele nivåets oppføringstall, så en voksende fil flyttet hver
eneste bakoverrad i hver eneste region. Første splitt ga 4 926 der svaret er
214 867 — en rute som så ut som gratis reise. Hver region eier nå én
sammenhengende blokk på `2b²`, forover og bakover samlet, og da spiller det
ingen rolle at naboene endrer størrelse.

*Prisen:* formatet er endret, så et datasett bygget før dette må varmes om.

*Verifisert:* eksakthetstesten mot vanlig Dijkstra fanger layoutfeilen om den
gjeninnføres (mutasjonstestet). En egen test kjører splitten på en klone og
krever både at svarene er uendret og at flere rader ble beholdt enn forkastet
— det andre fordi en implementasjon som stille regner om alt ville bestått det
første.

*Ikke gjort:* publisering gjennom byggekatalogen. `split_level` skriver en ny
struktur; å bytte den inn under lesere er fase 5.

### Fase 4 — Policy ✅ GJORT

`plan_splits_by_cost` velger *hvilke* regioner og *hvor* snittet går.

**Hvilke:** en regions bidrag er `b²`, både på disk og i arbeidet en spørring
gjør, så de verste deles først. På planetens nivå 2 bærer tusen regioner av
1,12 millioner alt arbeidet.

**Hvor:** grådig konduktans — ta alltid medlemmet som legger til færrest nye
snittkanter — med betweenness fra fase 1 som tiebreaker nedover, så grensen
legger seg der trafikken ikke er.

Målt på Norge:

| nivå | BFS-halvering | styrt snitt |
|---|---:|---:|
| 0 | +18,9 % porter, 3 770 rader om | **+7,0 %, 2 092 rader** |
| 1 | +44,3 % porter | **+34,5 %** |

**2,7× færre nye porter på nivå 0**, og 44 % færre rader å regne om fordi
snittet treffer færre naboer. Det er nettopp posten som gjorde alle
port-tak-anslagene optimistiske — 22× på nivå 2 ble regnet uten å telle nye
porter i det hele tatt.

Gevinsten er mye mindre høyt i stigen (+34,5 % mot +44,3 %), og det følger av
fase 1: betweenness er 8,1× spredt på nivå 1 og 1,8× på nivå 2. Jo grovere,
jo mer er alt en gjennomfartsåre, og jo mindre stille grunn finnes det å legge
et snitt på.

**To feil i nøstingen som planen kalte gratis.** Jeg skrev at «`c′` arver `c`
sin forelder» og implementerte det ikke: nivået over adresserer sine enheter
med region-id, så dets `of` må vokse. Og da det var rettet, bygde nivået over
seg fra `ov.level(k-1).boundary()` — de **gamle** portene — og ga 245 441 der
svaret er 214 867. Hvert nivås medlemmer er nivået under sin grense, og det er
nettopp grensen en splitt endrer. Nå føres de nye portene oppover.

*Testen splitter nå midt i stigen*, ikke på nivå 0, fordi et nivå-0-snitt bare
øver det nivået det rører. Begge feilene over ville sluppet gjennom.

### Fase 5 — Publiseringsløkke ✅ GJORT

`adapt <root> <level> <max-gates> [limit] [warm-seconds]` kjører én syklus:
planlegg, klon, splitt, varm, verifiser, bytt.

Målt på Norge, nivå 1:

```
49 region(s) to split at level 1
cloned to b1788956930 in 0.01 s
  9966 -> 10015 regions, gates 6189 -> 8325 (+34.5 %), 2858 rows kept
  warmed 13628 rows in 0.4 s
published b1788956930
```

Rutene var uendret gjennom byttet, og HEAD gikk fra én id til den neste med
`rename` — atomisk, så en leser ser den ene eller den andre, aldri en halv.

**Rekkefølgen er poenget.** Oppvarmingen skjer *før* byttet, ikke etter, så
ingen møter en kald tabell. Det er den samme lærdommen som fra planet-målingen
tidligere: en helt kald lat stige gir 322 sekunder på en rute som tar 92 ms
varm, fordi rekursjonen drar hele kjeden under seg med seg.

Planleggingen skjer mot den *levende* strukturen før klonen, så en syklus uten
noe å gjøre koster en lesning og ingen kopi.

*Ikke gjort:* å koble syklusen til diff-veien, slik at en ny planetfil både
invaliderer og utløser en ny vurdering av de berørte regionene.

### Fase 6 — Konvergens

Kjør trafikk (ekte eller `betweenness`-samplet), og mål per syklus: `Σb²`,
grensegraf, `Σb` langs ruter, faktisk latens. Stopp når marginalgevinsten per
syklus faller under en terskel.

*Utfall:* en kurve som viser hva adaptiviteten faktisk kjøpte, mot den
statiske partisjonen. Det er det eneste som avgjør om dette var verdt det.

---

## Natten som målte partisjonen — 9.–10. september 2026

Alt under er målt på planeten, og hver konklusjon har et tall bak seg. Tre
ideer ble bygget og forkastet; de står her fordi det er dyrere å finne dem
igjen enn å lese om dem.

### Det som viste seg å binde: fyllekostnad, ikke disk

Grensegrafen krymper omtrent lineært med grovere snitt. Fyllekostnaden vokser
som `rader × medlemmer × kanter-per-node`, og **kanter-per-node er `b` på
nivået under** — fordi en tabell gjør en region til en *komplett graf* på
portene sine. Veigrafen har 2,4 kanter per node; planetens fjerde rung hadde
1 657.

Det ene forholdet forklarer alt som ble forkastet:

| forsøk | hva det gjorde | hvorfor det falt |
|---|---|---|
| `GATE_CAP = 4096` | toppnivå 30 850 porter, best i sveipet | 3 898 porter per region, 121 MB per rad — én korridor ufylt etter 2 timer |
| femte rung | 264 ms mot 287 | fire ganger oppvarmingen for 8 % |
| aggregeringstak 32 | `b` fra 1 793 til 8, akkurat som tenkt | 3,3× tregere å fylle, 2,8× tregere å spørre |

Aggregeringstaket er den mest lærerike. Det gjorde nøyaktig det det skulle —
klikkene forsvant — men klikkene kostet i **fyllingen**, ikke i spørringen. I
spørringen settles de fleste noder på nivå 0, der `b` er 47 uansett partisjon.
Og i fyllingen ble klikkene erstattet av 22 millioner små rader, som hver har
en oppsettskostnad `fill_estimate` ikke kjenner.

**`fill_estimate` har en blind flekk.** Den teller relakseringer og er
kalibrert på store rader. På agg=32 spådde den 8,2× raskere; virkeligheten ble
3,3× tregere. En bom på faktor 27. Modellen mangler kostnaden ved å sette opp,
skrive og markere en rad — konstant per rad, og dominerende når radene er små.

### Portstyrt vekst — det som faktisk virket

Veksten stoppet på veinodetall og reparerte etterpå det som ble for stort. Da
bandt taket ikke: regionene lå på 68 porter mot et tak på 1 024. Å gjøre porter
til *vekstkriteriet*, med eksakt inkrementell telling, ga:

| | reparasjonssnitt | portstyrt |
|---|---:|---:|
| nivå 1 | 2 103 147 porter | 1 226 805 |
| nivå 2 | 1 600 094 | 348 515 |
| tabell | 25,58 GB | 16,71 GB |
| Lisboa–Warszawa | 213 445 noder | **62 690** |

Eksakt telling framfor anslag var nødvendig: `Σb` over de absorberte regionene
bommer 6,6× fordi den ikke ser begravingen.

### Konstantene, med sveipene bak seg

`GATE_CAP = 2048` er sveipet fram over 256–4096. Under 1024 klarer ikke
regionene å begrave porter og stigen slutter å kollapse; over 2048 vokser
radene raskere enn grensen krymper. Se doc-kommentaren i `overlay.rs` for hele
tabellen.

**Taket er datasettavhengig.** Norge vil ha 128 og planeten 2 048 — 2,2×
forskjell i besøkte noder mellom dem. Det bør lagres per datasett, ikke være
en global konstant.

`MAX_LEVELS` er hevet fra 8 til 24, og den absolutte stoppterskelen på 200 000
grensenoder er fjernet. Den kappet Norge til **ett** nivå fordi 14 174 porter
lå under en grense satt for planeten; uten den bygger Norge fire, og rutene ble
4,7–6,1× raskere. `MPEE_LADDER_SHED` styrer nå høyden, med 10 % som standard —
men målingen sier at 50 % er nærmere riktig: rungene som halverte tiden kastet
over 80 % av grensen, de som kastet under halvparten kjøpte 8 % til sammen for
2,32 GB og fire ganger oppvarmingen.

### Feil som ble funnet, og hva de har til felles

Tre feil i én natt, alle av samme form: **en regel som avviser noe, men ikke
rydder opp etter seg.**

- Layout-stempelet manglet, så planeten svarte Paris–Berlin på 52 sekunder —
  en tredel av Paris–Lyon på dobbel avstand. Feillagte tabeller feiler ikke;
  de gir en annen regions tall, som er gyldige kostnader.
- Tomt-nivå-vakten lå bare i den late byggeveien. Den ivrige skrev syv
  degenererte nivåer før noen så det.
- 10 %-regelen avviste et trinn *etter* at det var skrevet, og lot det ligge.
  Planeten bar et sjette nivå bit for bit identisk med det femte.

### Hva som er verifisert

`ovcheck` på 40 tilfeldige nodepar mot full Dijkstra på det ekte planetbygget:
**40/40 eksakt, verste avvik 0 ds.** Referansen settlet 7,53 milliarder noder,
overlayet 2,37 millioner — 3 176× færre, 19,8× raskere.

Varm planet, 52 GB på disk: Paris sentrum 1 ms, Paris–Lyon 48 ms, Oslo–Roma
166 ms, Lisboa–Warszawa 317 ms. Tolv tilfeldige europeiske ruter på urørte
korridorer: 30–285 ms med sidene i cache, 0,5–2,5 s ved første treff. Under
`MPEE_CACHE_MB=128` svarer den fortsatt, på 4,3 s, med identisk nodetall.

A\* er bygget og eksakt, men gir bare 7 % her — stigen har allerede tatt
gevinsten den ville hentet. Den står av som standard. Lagrede `xyz` ble bygget
og målt til å ikke bidra (797 mot 799 ms); det som kostet var å regne målets
posisjon *inne* i løkka, en løkkeinvariant.

### Matrise for flåteruting

En flåteoppgave trenger en matrise, ikke enkeltruter. Målt: 3,2 ms ren søketid
per rute, mot 22 ms med prosessoppstart — **oppstarten koster 7× mer enn
søket.** 500 stopp er 249 500 par: 13 min ren søketid, 91 min med oppstart,
**1,6 s** med én-til-mange.

Naiv én-til-mange virker ikke: med 500 mål inneholder hver region på øvre
nivåer et mål, `home`-sjekken slår inn, og søket klatrer aldri. Riktig form er
bøtter over grensenodene — ett kort bakoversøk per mål ut til grensen, ett
foroversøk per kilde, og kombiner. Det er strømmende N×N på spørrelaget.

Og et helt lands toppnivå er lite nok til å caches helt: **Norge har 338
porter på toppen, altså 0,9 MB full matrise** (0,6 MB med frame-of-reference).
Planeten er 41 GB, men ingen flåte kjører på planeten.

### Kompresjon med direkte oppslag

Målt på ekte tabellrader: frame-of-reference bit-pakking gir **1,6–1,8× og
slår deflate**, som er 1,50–1,71× — og FoR har `O(1)` oppslag der deflate må
pakkes ut. Taket er lavt fordi dataene er ekte høy-entropi: spennet i en rad er
800 000 tideler, altså 20 bits reell informasjon per verdi.

Innvendingen i koden om at «a lazy table cannot narrow» gjelder ikke FoR:
narrowingen velger bredde per *region* og trenger alle verdiene samtidig, mens
FoR er per *rad* — og `ensure_row` regner nettopp ut en hel rad om gangen.

---

## Oppdateringen, målt ende til ende — 11. september 2026

Diffen fant at 3,1 % av regionene er berørt, og det høres billig ut. Det er
det ikke, og grunnen er verdt å skrive ned.

### Å oppdage er billig, å utføre er dyrt

| ledd | kostnad |
|---|---|
| 9 dagers endringer, OSM-replikasjon | 960 MB, 60 s |
| les dem og finn berørte veier | 32 s, **én kjerne, 241 MB** |
| berørte regioner | 3,1 % på nivå 0 |
| **rader som må regnes om** | **8 543 176 av 16 189 660 — 52,8 %** |
| **refyll** | **2 t 17 min** |
| verifisert etterpå | kostnad og nodetall identiske |

Tre prosent av regionene drar med seg over halve stigen. Refyllen tar lenger
tid enn den opprinnelige oppvarmingen gjorde, fordi den startet på 25 % fylt
mens refyllen startet på null på de dyre nivåene.

### Hvorfor: invalideringskornet er grovt der det koster

| nivå | regioner **med porter** | berørt |
|---:|---:|---:|
| 0 | 137 507 | 29 % |
| 1 | 39 632 | 21 % |
| 2 | 2 608 | **100 %** |
| 3 | **55** | **100 %** |
| 4 | **40** | **100 %** |

Over nivå 1 finnes det bare noen titalls regioner med porter, hver av dem
kontinentstor. Ni dagers globale redigeringer treffer samtlige. «Inkrementell»
oppdatering er derfor bare inkrementell på nivå 0 og 1 — de billige — og sparer
ingenting der all tiden ligger.

### Finere korn gjør det verre, målt

Samme endringer mot tre partisjoner:

| partisjon | nivåer | rader totalt | droppet | andel |
|---|---:|---:|---:|---:|
| uten agg-tak | 5 | 16 189 660 | 8 543 176 | **52,8 %** |
| agg=32 | 12 | 22 213 306 | 12 359 960 | 55,6 % |
| agg=8 | 6 | 26 924 570 | 15 143 614 | 56,2 % |

Andelen er nær konstant, mens radantallet vokser. Endringene er globalt
spredt, så jo mindre regionene er, jo flere av dem skjærer det samme arealet.
Aggregeringstaket taper dermed på **alle tre** akser — fylling, spørring og
oppdatering — og den tredje, som var den eneste uprøvde da det ble forkastet,
er den klareste taperen.

### Det som kan hjelpe, og hva det krever

Invalideringen er i dag strukturell: berører noe en region, faller alt over
den. Den burde være **verdibasert** — regn om, sammenlign, og spre bare der et
tall faktisk flyttet seg. I målingen over var grafen uendret, så en verdibasert
spredning ville stoppet umiddelbart og spart alle 2 t 17 min.

Tre betingelser må holde før spredningen kan stoppes, og den tredje er den som
kan gi et stille galt svar:

1. **Radene er uendret.** Den opplagte.
2. **Snittkantene er uendret.** En fartsendring på en veg *mellom* to regioner
   flytter nivået overs graf uten å røre en rad. Lett å glemme, usynlig når den
   er glemt.
3. **Partisjonen er uendret.** En ny veg kan skape et kryss, en slettet kan
   dele en region. Da er det strukturen som er utdatert, og ingen
   verdisammenligning kan se det.

Dagens kode kan ikke gjøre dette: `invalidate` sletter biten før noe regnes, så
den gamle verdien finnes ikke å sammenligne mot. Rekkefølgen må snus.

**Og den bør bygges med verifisering som del av mekanismen.** Fire stille feil
ble funnet på ett døgn, alle av samme form — noe som ikke lykkes og heller ikke
stopper: et layout-stempel som manglet, en tomt-nivå-vakt som bare lå i én av to
byggeveier, et cache-tak som var lagt på men aldri håndhevet, og et måleskript
der en mislykket byggekommando lot skriptet måle forrige partisjon tre ganger
med plausible tall. Et generasjonsnummer per region som *må* stemme er tryggere
enn en sammenligning man håper ble gjort, og `ovcheck` hører i byggeporten og
ikke i ettertanken.

---

## Risikoer, ærlig

- **Snittet skaper porter.** Alle anslag i dag ignorerer det. Fase 1s
  brukskart er det som begrenser skaden; fase 6 er det som måler den.
- **Overtilpasning til trafikken.** Regioner ingen ruter gjennom forbedres
  aldri. Det er akseptabelt — de spørres ikke — men en ny trafikkprofil
  starter kald der.
- **`fill_estimate` undervurderer små rader grovt.** Se avsnittet over: en
  bom på faktor 27 på agg=32. Enhver policy som velger partisjon på anslått
  fyllekostnad må kalibreres på radstørrelsen den faktisk vil produsere.
- **Hastighetsmodellen er nå validert, men bare på én rute og ett nivå.**
  `k(rute)` er målt på fire granulariteter for Lisboa → Warszawa og tre ruter
  på Norge. Den er ikke testet på flere nivåer i stigen, og eksponenten 0,66 er
  et snitt over en kurve som ikke er en rett linje. Policyen bør bruke
  interpolasjon i den målte tabellen, ikke eksponenten.
- **Klone-og-bytt forutsetter APFS eller tilsvarende.** På et filsystem uten
  `clonefile` er klonen en ekte kopi av radcachen — 33 GB på planeten i dag.
- **Rekursjonen i `ensure_row`.** Å splitte på nivå 0 invaliderer tre nivåer
  over. På en kald stige er det dyrt; på en varm er det fire regioner.

## Avgjørelser som er dine

1. **Grensene.** Disk ≤ 50 GB er sagt. Resident ≤ 64 MB? Latensmål for en
   kontinentrute?
2. **Trafikkilden.** Bare ekte spørringer, eller `betweenness`-sampling for
   å bootstrappe før det finnes trafikk?
3. **Nivåer.** Start med nivå 0 alene, eller alle? Nivå 0 er billigst å
   verifisere; effekten er størst på nivå 2.
4. **Fase 0 først, eller hopp til fase 1?** Fase 0 koster en time og kan
   stoppe hele planen. Anbefalt.

## Rekkefølge

Fase 0 ✅ → 2 ✅ → 1 ✅ → 3 ✅ → 4 ✅ → 5 ✅ → **6**.

Fase 0 og 2 var rene målinger, begge med negative utfall som endret planen —
disken er ikke flat på planeten, og `Σb²` er ikke hastighetsmodellen. Siden er
det skrevet mye kode: layout-stempel, portstyrt vekst, aggregeringstak,
chord-basert geometrisk grense, `xyz`- og `traffic`-kommandoene, og A\*.

Fase 6 står fortsatt, men den er blitt mindre presserende enn den så ut. Den
skulle måle hva adaptiviteten kjøpte mot en statisk partisjon — men natten
9.–10. september målte den *statiske* partisjonen grundig, og fant at
gevinsten der var stor og fyllekostnaden er det bindende. En adaptiv syklus
som forbedrer partisjonen må betale den samme fyllekostnaden på nytt for hver
region den rører. Det er fase 6s egentlige spørsmål nå.

**Rekkefølgen jeg ville tatt herfra:**

1. Lagre porttaket per datasett. Det er målt til 128 for Norge og 2 048 for
   planeten, og en global konstant kan ikke tjene begge.
2. Bøtte-basert én-til-mange. Det er den ene funksjonen som mangler for at
   flåtebruk skal gå fra 91 minutter til sekunder.
3. `MPEE_LADDER_SHED = 0.5` som standard, om målingen bekreftes på flere ruter.
4. Fase 6, med fyllekostnaden som hovedstørrelse i stedet for `Σb²`.

Og én ting som dukket opp underveis og ikke står i noen fase: `bounds_of` —
billigste og dyreste gjennomfart per region — er bygget og **aldri kalt**. Den
kan hoppe over en hel tabellrad når regionens billigste kryssing ikke kan slå
den beste kjente veien, og på et lat nivå er det å hoppe over en utregning på
fem sekunder, ikke bare en lesning.
