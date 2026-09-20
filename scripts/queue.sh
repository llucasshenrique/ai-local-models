#!/usr/bin/env bash
# Run the GPU jobs strictly one after another: wait for any running sweep, then ablation, repeats, FA A/B.
# Each script also takes /tmp/local-llm-gpu.lock, so nothing can overlap even if started by hand.
cd "$(dirname "$0")/.."
while pgrep -f "bench/sweep.py" | grep -qv "^$$\$"; do sleep 15; done
# qwen3:8b-q8_0 overlapped with a stray concurrent job: drop its record and re-run it alone.
python3 - <<'P'
import json
for f, key in (("results/sweep.jsonl", "base"), ("results/sweep-tasks.jsonl", "model")):
    want = "qwen3:8b-q8_0" if key == "base" else "sw-qwen3-8b-q8-0"
    rows = [l for l in open(f) if l.strip()]
    keep = [l for l in rows if json.loads(l)[key] != want]
    open("results/invalid/contaminated-" + f.split("/")[1], "a").writelines(l for l in rows if l not in keep)
    open(f, "w").writelines(keep)
P
flock /tmp/local-llm-gpu.lock python3 bench/sweep.py qwen3:8b-q8_0
scripts/ablation.sh
scripts/repeats.sh
scripts/fa-ab.sh
echo "=== QUEUE DONE ==="
