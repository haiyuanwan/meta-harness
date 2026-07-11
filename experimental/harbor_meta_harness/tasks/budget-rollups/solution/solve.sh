#!/bin/sh
python - <<'PY'
import csv
import json
from collections import defaultdict

totals = defaultdict(int)
with open('/app/spend.csv', newline='') as source:
    for row in csv.DictReader(source):
        if row['state'] != 'approved':
            continue
        totals[row['cost_center']] += int(row['amount_cents'])

payload = {key: value for key, value in sorted(totals.items()) if value}
with open('/app/rollups.json', 'w') as target:
    json.dump(payload, target)
PY
