#!/usr/bin/env bash
# Second stage, strictly after queue.sh: context headroom first, then the harness comparison.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/queue.sh" >/dev/null; do sleep 20; done
scripts/ctx-fit.sh
scripts/harness-compare.sh
echo "=== QUEUE2 DONE ==="
