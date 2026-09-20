#!/usr/bin/env bash
# Run the GPU jobs strictly one after another: wait for any running sweep, then ablation, repeats, FA A/B.
# Each script also takes /tmp/local-llm-gpu.lock, so nothing can overlap even if started by hand.
cd "$(dirname "$0")/.."
while pgrep -f "bench/sweep.py" | grep -qv "^$$\$"; do sleep 15; done
scripts/ablation.sh
scripts/repeats.sh
scripts/fa-ab.sh
echo "=== QUEUE DONE ==="
