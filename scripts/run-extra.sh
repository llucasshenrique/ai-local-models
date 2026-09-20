#!/usr/bin/env bash
# Serialize GPU use: only one model-loading script at a time (shared lock).
[ -z "${GPU_LOCKED:-}" ] && GPU_LOCKED=1 exec flock /tmp/local-llm-gpu.lock "$0" "$@"
# Re-run specific model/task pairs: ./run-extra.sh MODEL [task ...]  (appends to results/opencode.jsonl)
cd "$(dirname "$0")/.."; m=$1; shift
python3 bench/smoke.py opencode "$m" "$@" | tee -a results/opencode.jsonl; ollama stop "$m" >/dev/null 2>&1
