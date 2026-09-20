#!/usr/bin/env python3
"""Model/quantization sweep: for each base tag, pull it, create a tuned `sw-*` tag, register it in
the opencode config, measure GPU fit + tok/s, then run the 3-task smoke test.

usage: sweep.py [base_tag ...]      (default: the SWEEP list below)
Appends one JSON line per model to results/sweep.jsonl and one per task to results/sweep-tasks.jsonl.
Only adds new tags; never deletes or overwrites existing models. Unloads each model when done.
"""
import json, os, re, subprocess, sys, time, urllib.request
import smoke

SWEEP = [
    "ornith:9b-q4_K_M", "ornith:9b-q8_0",
    "granite4.1:8b-q3_K_M", "granite4.1:8b-q4_K_M", "granite4.1:8b-q6_K",
    "granite4.1:3b-q4_K_M", "granite4.1:3b-q8_0",
    "qwen3:8b-q8_0",
]
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.expanduser("~/.config/opencode/opencode.json")
CTX, PRED = 16384, 4096
HOST = "http://127.0.0.1:11434"

def tuned_name(base):
    return "sw-" + re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")

def modelfile(base):
    # Same sampling as the hand-tuned tags; template/stop tokens are inherited from the base model.
    return (f"FROM {base}\nPARAMETER temperature 0.25\nPARAMETER top_k 20\nPARAMETER top_p 0.9\n"
            f"PARAMETER repeat_penalty 1.05\nPARAMETER repeat_last_n 256\nPARAMETER num_ctx {CTX}\nPARAMETER num_predict {PRED}\n")

def register_opencode(name):
    c = json.load(open(CFG))
    m = c["provider"]["ollama"]["models"]
    if name not in m:
        m[name] = {"name": name, "tool_call": True, "limit": {"context": CTX, "output": PRED},
                   "options": {"temperature": 0.25, "top_p": 0.9, "presence_penalty": 0.3, "frequency_penalty": 0.1}}
        json.dump(c, open(CFG, "w"), indent=2); open(CFG, "a").write("\n")

def bench(name):
    body = {"model": name, "stream": False, "options": {"num_ctx": CTX}, "think": False,
            "messages": [{"role": "user", "content": "Write a Python function is_prime(n) with a docstring and three asserts. Code only."}]}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(HOST + "/api/chat", json.dumps(body).encode(),
                      {"Content-Type": "application/json"}), timeout=170))
    except Exception as e:
        # some models reject the think flag; retry without it
        body.pop("think")
        r = json.load(urllib.request.urlopen(urllib.request.Request(HOST + "/api/chat", json.dumps(body).encode(),
                      {"Content-Type": "application/json"}), timeout=170))
    tok_s = round(r["eval_count"] / (r["eval_duration"] / 1e9), 1)
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return tok_s, " ".join(ps.split()[2:6])   # "<size> GB <cpu%/gpu%> ..." or "<size> GB 100% GPU"

def main(bases):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    done = set()
    p = os.path.join(HERE, "results", "sweep.jsonl")
    if os.path.exists(p):
        done = {json.loads(l)["base"] for l in open(p) if l.strip()}
    for base in bases:
        name = tuned_name(base)
        if base in done:
            print(f"skip {base}: already measured", flush=True); continue
        print(f"=== {base} -> {name} ===", flush=True)
        if subprocess.run(["ollama", "pull", base]).returncode != 0:
            print("pull failed, skipping", flush=True); continue
        subprocess.run(["ollama", "create", name, "-f", "-"], input=modelfile(base), text=True, capture_output=True)
        register_opencode(name)
        rec = dict(base=base, tag=name)
        try:
            rec["tok_s"], rec["fit"] = bench(name)
        except Exception as e:
            rec["tok_s"], rec["fit"] = None, f"bench failed: {e}"
        print(json.dumps(rec), flush=True)
        results = [smoke.run("opencode", name, t) for t in (1, 2, 3)]
        for r in results:
            print(json.dumps(r), flush=True)
            open(os.path.join(HERE, "results", "sweep-tasks.jsonl"), "a").write(json.dumps(r) + "\n")
        rec["passed"] = sum(r["done"] for r in results)
        rec["wall_s"] = round(sum(r["wall_s"] for r in results), 1)
        rec["loops"] = sum(r["loop"] for r in results)
        open(os.path.join(HERE, "results", "sweep.jsonl"), "a").write(json.dumps(rec) + "\n")
        subprocess.run(["ollama", "stop", name], capture_output=True)

if __name__ == "__main__":
    main(sys.argv[1:] or SWEEP)
