#!/usr/bin/env bash
# Serialize GPU use: only one model-loading script at a time (shared lock).
[ -z "${GPU_LOCKED:-}" ] && GPU_LOCKED=1 exec flock /tmp/local-llm-gpu.lock "$0" "$@"
# Controlled same-host A/B of OLLAMA_FLASH_ATTENTION + OLLAMA_KV_CACHE_TYPE=q8_0 (no sudo, temp server on :11435).
# Uses the ~/.ollama store (needs gemma4:12b there). Reports tok/s and GPU fit at 16k/32k/48k ctx, 2 runs each.
cd "$(dirname "$0")/.."
# Unload anything the main server (:11434) still holds so the temp server has the whole GPU.
for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done
sleep 3
export OLLAMA_HOST=127.0.0.1:11435
run_variant() {  # $1 label, rest = env assignments
  label=$1; shift
  env "$@" ollama serve >/dev/null 2>&1 & pid=$!
  sleep 6
  for ctx in 16384 32768 49152; do
    for i in 1 2; do
      out=$(python3 bench/bench.py gemma4:12b $ctx false 127.0.0.1:11435 2>&1 | tail -1)
      fit=$(ollama ps | tail -1 | awk '{print $3,$4,$5,$6}')
      echo "{\"variant\": \"$label\", \"ctx\": $ctx, \"run\": $i, \"bench\": $out, \"fit\": \"$fit\"}" | tee -a results/fa-ab.jsonl
    done
    ollama stop gemma4:12b >/dev/null 2>&1; sleep 2
  done
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
run_variant default OLLAMA_FLASH_ATTENTION=0 OLLAMA_KV_CACHE_TYPE=f16
run_variant fa-q8   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
run_variant fa-only OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=f16
echo "=== FA A/B DONE ==="
