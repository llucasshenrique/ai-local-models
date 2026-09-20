#!/usr/bin/env bash
# Repeat the finalists N times (default 3) x 3 tasks to separate real differences from noise.
# Appends {"rep": i, ...} JSON lines to results/repeats.jsonl. Skips model/rep pairs already recorded.
cd "$(dirname "$0")/.."
N=${N:-3}
MODELS=${MODELS:-"gemma4-agent:12b sw-ornith-9b-q4-k-m sw-granite4-1-8b-q4-k-m qwen3-agent:8b"}
touch results/repeats.jsonl
for rep in $(seq 1 "$N"); do
  for m in $MODELS; do
    grep -q "\"model\": \"$m\".*\"rep\": $rep\b" results/repeats.jsonl && continue
    echo "=== $m rep $rep ==="
    python3 bench/smoke.py opencode "$m" | while read -r l; do echo "${l%\}}, \"rep\": $rep}" | tee -a results/repeats.jsonl; done
    ollama stop "$m" >/dev/null 2>&1
  done
done
echo "=== REPEATS DONE ==="
