#!/bin/sh
python - <<'PY'
import json

latest = {}
with open('/app/events.jsonl') as source:
    for line in source:
        record = json.loads(line)
        if record['id'] not in latest or record['seq'] > latest[record['id']]['seq']:
            latest[record['id']] = record

kept = [
    {
        'id': record['id'],
        'seq': record['seq'],
        'status': record['status'],
        'note': record['note'],
    }
    for record in latest.values()
    if record['status'] != 'void'
]
kept.sort(key=lambda item: item['id'])
with open('/app/unique.json', 'w') as target:
    json.dump(kept, target)
PY
