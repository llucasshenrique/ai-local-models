#!/usr/bin/env bash
# Serialize GPU use: only one model-loading script at a time (shared lock).
[ -z "${GPU_LOCKED:-}" ] && GPU_LOCKED=1 exec flock /tmp/local-llm-gpu.lock "$0" "$@"
# Same 3 tasks through different harnesses; all but opencode are measured through bench/proxy.py
# (requests + peak prompt tokens). Appends to results/harness.jsonl. HARNESSES/MODELS overridable.
cd "$(dirname "$0")/.."
HARNESSES=${HARNESSES:-"minimal mini aider pi"}
MODELS=${MODELS:-"sw-ornith-9b-q4-k-m gemma4-agent:12b"}
python3 bench/proxy.py & proxy=$!
trap 'kill $proxy 2>/dev/null' EXIT
sleep 2
for m in $MODELS; do
  for h in $HARNESSES; do
    echo "=== $h / $m ==="
    python3 bench/smoke.py "$h" "$m" | tee -a results/harness.jsonl
    ollama stop "$m" >/dev/null 2>&1
  done
done
echo "=== HARNESS COMPARE DONE ==="
