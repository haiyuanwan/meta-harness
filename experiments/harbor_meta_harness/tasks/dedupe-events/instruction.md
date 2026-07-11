# Dedupe events

Read `/app/events.jsonl`.
Each line is a JSON object with `id`, `seq`, `status`, and `note`.

For each `id`, keep only the record with the highest `seq`.
If the selected record has `status` equal to `void`, drop that id entirely.

Write `/app/unique.json` as a JSON list of the remaining records.
Sort the list by `id` ascending.
Each object must contain exactly the keys `id`, `seq`, `status`, and `note` with the selected values.
