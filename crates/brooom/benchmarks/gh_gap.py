#!/usr/bin/env python3
"""Gap mot BKS for G&H-kjøringen.

Instansene regenereres slik (committes ikke — 15 MB):
  curl -sLo /tmp/gh200.zip https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/homberger_200_customer_instances.zip
  unzip /tmp/gh200.zip -d /tmp/gh200 && (cd /tmp/gh200 && for f in *.TXT; do mv "$f" "$(echo $f | tr A-Z a-z)"; done)
  python3 solomon_to_vroom.py --all /tmp/gh200 instances_gh200
BKS: gh200_bks.json (skrapet fra sintef.no/projectweb/top/vrptw/200-customers): per klasse og totalt, begge solvere.
Kost-skala: solver = BKS-distanse × 100 (solomon_to_vroom SCALE).
BKS er hierarkisk (kjøretøy først, så distanse) — vi rapporterer distansegap
og flagger når solverens rutetall avviker fra BKS-rutetallet."""
import csv, json, sys
from collections import defaultdict

import pathlib
bks = json.load(open(pathlib.Path(__file__).parent / 'gh200_bks.json'))
rows = list(csv.DictReader(open(sys.argv[1])))

per = defaultdict(dict)
for r in rows:
    if r['cost']:
        per[r['instance']][r['solver']] = (float(r['cost']) / 100.0, int(r['routes']), int(r['unassigned'] or 0))

cls = lambda name: name.split('_')[0].upper() + ('1' if name.split('_')[0][-1] not in '12' else '')
classes = defaultdict(lambda: defaultdict(list))
head2head = {'brooom': 0, 'pyvrp': 0, 'tie': 0}
veh_mismatch = {'brooom': 0, 'pyvrp': 0}
unassigned_ct = {'brooom': 0, 'pyvrp': 0}

for inst, sv in sorted(per.items()):
    b = bks.get(inst)
    if not b or 'brooom' not in sv or 'pyvrp' not in sv:
        continue
    c = inst.split('_')[0].upper()
    for s in ('brooom', 'pyvrp'):
        dist, routes, un = sv[s]
        gap = 100.0 * (dist - b['distance']) / b['distance']
        classes[c][s].append(gap)
        if routes != b['vehicles']:
            veh_mismatch[s] += 1
        if un:
            unassigned_ct[s] += 1
    db, dp = sv['brooom'][0], sv['pyvrp'][0]
    if abs(db - dp) < 0.005:
        head2head['tie'] += 1
    elif db < dp:
        head2head['brooom'] += 1
    else:
        head2head['pyvrp'] += 1

print(f"{'klasse':8} {'n':>3} {'brooom gap':>12} {'pyvrp gap':>12}")
allb, allp = [], []
for c in sorted(classes):
    gb, gp = classes[c]['brooom'], classes[c]['pyvrp']
    allb += gb; allp += gp
    print(f"{c:8} {len(gb):>3} {sum(gb)/len(gb):>11.2f}% {sum(gp)/len(gp):>11.2f}%")
print(f"{'ALLE':8} {len(allb):>3} {sum(allb)/len(allb):>11.2f}% {sum(allp)/len(allp):>11.2f}%")
print(f"\nhead-to-head: brooom {head2head['brooom']} / uavgjort {head2head['tie']} / pyvrp {head2head['pyvrp']}")
print(f"rutetall != BKS: brooom {veh_mismatch['brooom']}/60, pyvrp {veh_mismatch['pyvrp']}/60")
print(f"instanser med unassigned: brooom {unassigned_ct['brooom']}, pyvrp {unassigned_ct['pyvrp']}")
