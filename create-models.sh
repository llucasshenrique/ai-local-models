#!/usr/bin/env bash
# Idempotently (re)create the tuned ollama tags from the Modelfiles next to this script.
# Only runs `ollama create`; never deletes or overwrites the base models.
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
for f in "$dir"/*.Modelfile; do
  base="$(basename "$f" .Modelfile)"   # e.g. gemma4-agent.12b
  tag="${base/./:}"                    # gemma4-agent:12b
  echo "creating $tag from $(basename "$f")"
  ollama create "$tag" -f "$f"
done
