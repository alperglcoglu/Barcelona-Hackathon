#!/usr/bin/env bash
# Download SkillCorner's open A-League matches (other matches, allowed by RULES.md) into
# data/external/skillcorner/. data/ is gitignored, so nothing downloaded here is ever committed.
# The tracking files are Git LFS objects, about 90 MB each; they are fetched from GitHub's media host.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/external/skillcorner
BASE=https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches
MEDIA=https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches
mkdir -p "$DEST"
for id in $(python -c "from tuning import skillcorner; print(' '.join(map(str, skillcorner.MATCHES)))"); do
  [ -s "$DEST/${id}_match.json" ] || curl -sSfL -o "$DEST/${id}_match.json" "$BASE/$id/${id}_match.json"
  [ -s "$DEST/${id}.jsonl" ] || curl -sSfL -o "$DEST/${id}.jsonl" "$MEDIA/$id/${id}_tracking_extrapolated.jsonl"
  echo "$id"
done
