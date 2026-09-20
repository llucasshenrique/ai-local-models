#!/usr/bin/env python3
"""Quick ollama bench: python3 bench.py MODEL [num_ctx] [think:true|false|none] [host]"""
import json, sys, time, urllib.request, subprocess
model = sys.argv[1]; ctx = int(sys.argv[2]) if len(sys.argv) > 2 else None
think = sys.argv[3] if len(sys.argv) > 3 else "none"
host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1:11434"
body = {"model": model, "stream": False, "messages": [{"role": "user", "content":
  "Write a Python function is_prime(n) with a docstring and three asserts. Code only."}],
  "options": {}}
if ctx: body["options"]["num_ctx"] = ctx
if think in ("true", "false"): body["think"] = think == "true"
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{host}/api/chat",
  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=170))
m = r["message"]; ev = r.get("eval_count", 0)
print(json.dumps({"model": model, "ctx": ctx, "think": think, "wall_s": round(time.time()-t, 1),
  "gen_tok": ev, "tok_s": round(ev/(r["eval_duration"]/1e9), 1),
  "thinking_chars": len(m.get("thinking") or ""), "answer_chars": len(m["content"])}))
