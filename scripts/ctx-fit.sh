#!/usr/bin/env bash
# Serialize GPU use: only one model-loading script at a time (shared lock).
[ -z "${GPU_LOCKED:-}" ] && GPU_LOCKED=1 exec flock /tmp/local-llm-gpu.lock "$0" "$@"
# How much context fits on the 11 GB GPU per finalist? Loads each model at increasing num_ctx and records
# `ollama ps` (size, CPU/GPU split) and generation speed. Appends to results/ctx-fit.jsonl.
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"gemma4-agent:12b sw-ornith-9b-q4-k-m sw-granite4-1-8b-q4-k-m qwen3-agent:8b"}
CTXS=${CTXS:-"16384 32768 49152 65536"}
for m in $MODELS; do
  for c in $CTXS; do
    out=$(python3 bench/bench.py "$m" "$c" none 2>&1 | tail -1)
    fit=$(ollama ps | awk 'NR==2{print $3,$4,$5,$6}')
    echo "{\"model\": \"$m\", \"ctx\": $c, \"fit\": \"$fit\", \"bench\": ${out:-null}}" | tee -a results/ctx-fit.jsonl
    ollama stop "$m" >/dev/null 2>&1; sleep 3
  done
done
echo "=== CTX FIT DONE ==="
