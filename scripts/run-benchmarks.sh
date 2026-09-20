#!/usr/bin/env bash
# Pull/create models, then run the opencode smoke matrix. Logs to results/opencode.jsonl (also on screen).
set -u
cd "$(dirname "$0")/.."; mkdir -p results
ollama list | grep -q '^qwen3:8b' || ollama pull qwen3:8b
scripts/create-models.sh
for m in gemma4-agent:12b gemma4:12b qwen3-agent:8b qwen3-agent:14b; do
  echo "=== $m ==="; python3 bench/smoke.py opencode "$m" | tee -a results/opencode.jsonl; ollama stop "$m" >/dev/null 2>&1
done
echo "=== ALL DONE ==="
