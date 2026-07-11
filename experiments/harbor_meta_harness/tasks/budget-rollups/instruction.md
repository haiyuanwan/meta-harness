# Budget rollups

Read `/app/spend.csv`.
It has a header row and columns `txn_id`, `cost_center`, `state`, and `amount_cents`.

Keep only rows whose `state` is `approved`.
Sum `amount_cents` per `cost_center`.
Omit any cost center whose approved total is zero.

Write `/app/rollups.json` as a JSON object mapping cost center to integer total cents.
Sort keys lexicographically.
Do not include any other fields.
