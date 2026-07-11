#!/usr/bin/env bash
# Download the Harbor datasets used by active suites/*.toml (legacy registry).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p datasets
harbor download 'terminal-bench@2.0' -o datasets/terminal-bench-2-tmp --export --overwrite
rm -rf datasets/terminal-bench-2
mv datasets/terminal-bench-2-tmp/terminal-bench datasets/terminal-bench-2
rm -rf datasets/terminal-bench-2-tmp
for spec in \
  'codepde@1.0' \
  'humanevalfix@1.0'
do
  name="${spec%@*}"
  harbor download "$spec" -o "datasets/${name}-tmp" --export --overwrite
  rm -rf "datasets/$name"
  mv "datasets/${name}-tmp/$name" "datasets/$name"
  rm -rf "datasets/${name}-tmp"
done
python scripts/prepare_collections.py
