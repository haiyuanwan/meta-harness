#!/bin/sh
python - <<'PY'
import json
from collections import defaultdict

latest = {}
with open('/app/ledger.jsonl') as source:
    for line in source:
        record = json.loads(line)
        if record['invoice'] not in latest or record['revision'] > latest[record['invoice']]['revision']:
            latest[record['invoice']] = record

balances = defaultdict(int)
for record in latest.values():
    if record['status'] != 'posted':
        continue
    balances[record['account']] += record['amount_cents'] if record['kind'] == 'charge' else -record['amount_cents']

with open('/app/report.json', 'w') as target:
    json.dump(dict(sorted(balances.items())), target, sort_keys=True)
PY
