#!/usr/bin/env bash
# Download Metrica's public sample games (other matches, allowed by RULES.md) into data/external/.
# data/ is gitignored, so nothing downloaded here is ever committed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/external
if [ ! -d data/external/sample-data/.git ]; then
  git clone --depth 1 https://github.com/metrica-sports/sample-data data/external/sample-data
fi
