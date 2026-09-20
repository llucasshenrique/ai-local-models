#!/usr/bin/env bash
# Separate the effect of context length vs sampling on gemma4:12b (opencode smoke, 3 tasks each).
# Also records the context the untuned model really loads with (ollama ps CONTEXT column).
cd "$(dirname "$0")/.."
python3 - <<'P'
import json, os, subprocess
cfg = os.path.expanduser("~/.config/opencode/opencode.json"); c = json.load(open(cfg)); m = c["provider"]["ollama"]["models"]
for n in ("abl-gemma4-ctxonly", "abl-gemma4-samponly"):
    m.setdefault(n, {"name": n, "tool_call": True, "limit": {"context": 32768, "output": 4096}})
json.dump(c, open(cfg, "w"), indent=2)
P
for n in abl-gemma4-ctxonly abl-gemma4-samponly; do ollama create $n -f modelfiles/ablation/$n.Modelfile >/dev/null; done
echo "=== untuned gemma4:12b real context ==="
curl -s localhost:11434/api/generate -d '{"model":"gemma4:12b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null; ollama ps | tee results/ablation-ps.txt; ollama stop gemma4:12b
for n in abl-gemma4-ctxonly abl-gemma4-samponly; do echo "=== $n ==="; python3 bench/smoke.py opencode $n | tee -a results/ablation.jsonl; ollama stop $n >/dev/null 2>&1; done
echo "=== ABLATION DONE ==="
