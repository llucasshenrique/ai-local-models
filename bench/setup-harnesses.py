#!/usr/bin/env python3
"""One-time harness config for the comparison (idempotent, adds only; existing entries are kept).
- pi: adds provider `ollama-proxy` (-> bench/proxy.py on :11436) to ~/.pi/agent/models.json
- mini-swe-agent: writes ~/.config/mini-swe-agent/.env (skips the first-run wizard; no keys involved)
"""
import json, os
MODELS = {"gemma4-agent:12b": 32768, "sw-ornith-9b-q4-k-m": 16384, "sw-granite4-1-8b-q4-k-m": 16384, "qwen3-agent:8b": 16384}
p = os.path.expanduser("~/.pi/agent/models.json")
c = json.load(open(p)) if os.path.exists(p) else {"providers": {}}
c["providers"]["ollama-proxy"] = {"api": "openai-completions", "apiKey": "ollama", "baseUrl": "http://127.0.0.1:11436/v1",
    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
    "models": [{"id": m, "contextWindow": w, "maxTokens": 4096, "input": ["text"]} for m, w in MODELS.items()]}
json.dump(c, open(p, "w"), indent=2)
d = os.path.expanduser("~/.config/mini-swe-agent"); os.makedirs(d, exist_ok=True)
e = os.path.join(d, ".env")
if not os.path.exists(e):
    open(e, "w").write('MSWEA_CONFIGURED="true"\nMSWEA_COST_TRACKING="ignore_errors"\n')
print("ok")
