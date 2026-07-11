# Reconcile the ledger

Read `/app/ledger.jsonl` and write `/app/report.json`.

Each line is a JSON object with `invoice`, `account`, `revision`, `status`, `kind`, and `amount_cents`.

For each invoice, use only its highest `revision`.
Ignore a selected invoice unless its `status` is `posted`.
For every remaining invoice, add `amount_cents` for `kind == "charge"` and subtract it for `kind == "credit"`.

`report.json` must be a JSON object whose keys are accounts and whose values are integer balances in cents.
Include every account that has at least one selected invoice, sort keys lexicographically, and do not include any other fields.

You may create helper files, but `/app/report.json` is the required artifact.
