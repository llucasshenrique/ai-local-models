"""Advisor: ask the best available local model to study the results and propose experiments.

The advisor is an LLM, so its output is treated as *hypotheses*: it is parsed, validated against safe parameter ranges and
known models, saved to results/advice.{json,md}, and can seed `tune --from-advice`, where measurement (train + held-out
tasks) decides. Nothing it says is executed or written into your configs."""
import json, os, re, time
from . import families as fam, ollama, report, store
from .lock import gpu_lock

ADVICE = os.path.join(store.RESULTS, "advice.json")
ALLOWED = {"temperature": (0.0, 1.0), "top_k": (1, 100), "top_p": (0.5, 1.0), "repeat_penalty": (1.0, 1.5), "num_ctx": (4096, 65536)}
INTS = {"top_k", "num_ctx"}

def pick_advisor(cfg, explicit=None):
    """(tag, why). 'auto' = best pass rate in the results, larger parameter count wins ties; must be installed."""
    want = explicit or cfg.get("advisor", {}).get("model", "auto")
    if want != "auto": return want, "chosen in the config/CLI"
    inst = ollama.installed(); best = {}
    for r in report.summary():
        tag = r["model"] if ":" in r["model"] else r["model"] + ":latest"
        if tag in inst:
            size = fam.params_b(fam.parse_variant(r["model"])[0]) or 0
            key = ((r["passed"] + 1) / (r["n"] + 2), size, -(r["wall"] or 1e9))     # Laplace-smoothed: 3/3 must not beat 9/9 by default
            if tag not in best or key > best[tag][0]: best[tag] = (key, r["model"])
    if best:
        tag = max(best.values(), key=lambda v: v[0])[1]
        return tag, "best sample-size-adjusted pass rate in your results (ties: more parameters, then faster)"
    for m in cfg["models"]:
        if ollama.has(m["tag"]): return m["tag"], "no results yet: first installed model in the config"
    raise SystemExit("no advisor model available: run some trials or set [advisor] model in the config")

def failures(limit=3, tail=12):
    """Tails of a few failed trials' traces, so the advisor sees *how* things fail, not just that they do."""
    out = []
    for r in reversed(store.rows()):
        if r["done"] or len(out) >= limit: continue
        p = os.path.join(store.RESULTS, "traces", r["harness"], f"{r['model'].replace(':', '_')}.{r['task']}.{r['rep']}.log")
        if os.path.exists(p):
            out.append(f"[{r['harness']} / {r['model']} / {r['task']} #{r['rep']}" + (" LOOP" if r.get("loop") else "") + (" TIMEOUT" if r.get("timeout") else "") + "]\n"
                       + "\n".join(open(p, errors="ignore").read().splitlines()[-tail:])[:900])
    return out

def digest(cfg, max_rows=14):
    """Plain-text summary of everything the advisor is allowed to see."""
    parts = ["RESULTS (harness, model, pass, median wall s, median LLM requests, peak prompt tokens, loops, timeouts):"]
    for x in report.summary()[:max_rows]:
        parts.append(f'- {x["harness"]}, {x["model"]}, {x["passed"]}/{x["n"]}, {x["wall"]}, {x["reqs"]}, {x["peak"]}, {x["loops"]}, {x["timeouts"]}')
    for name, rows, pick, why in report.family_table():
        parts.append(f"FAMILY {name}: recommended {pick['tag'] if pick else '-'} ({why})")
        parts += [f'- {r["tag"]}: {100 * r["rate"]:.0f}% n={r["n"]}, {r["tok_s"]} tok/s, {r["size_gb"]} GB, {r["gpu_pct"]}% GPU, ctx {r["ctx"]}' for r in rows]
    parts.append("CURRENT MODELS AND PARAMS: " + json.dumps({m["tag"]: m.get("params", {}) for m in cfg["models"]}))
    tune = store.rows(os.path.join(store.RESULTS, "tune.jsonl"))[-8:]
    if tune: parts.append("RECENT TUNE STEPS: " + json.dumps([{k: t.get(k) for k in ("phase", "changed", "score", "confirmed")} for t in tune]))
    parts += ["FAILURE EXAMPLES:"] + (failures() or ["(none recorded)"])
    return "\n".join(parts)

PROMPT = """You are helping tune local LLM coding agents that run small, test-verified code tasks on an 11 GB GPU.
Study the data and propose the most promising next experiments. Prefer few, well-justified changes. Sample sizes (n) differ a lot: never call a difference real when n is small (under 6) or when the gap is one trial.
Only these Modelfile parameters may change: {allowed}. Answer with JSON only:
{{"analysis": "2-4 sentences on what limits success or speed",
  "experiments": [{{"model": "<tag from the data>", "changes": {{"temperature": 0.2}}, "why": "<one sentence>"}}],
  "ideas": ["<other suggestions a human should consider: tasks, harness settings, model families to add>"]}}

DATA
{digest}"""

def sanitize(advice, known_models):
    """Keep only experiments on known models with allowed keys and in-range numbers. Returns (clean, dropped notes)."""
    dropped, clean = [], []
    for e in (advice.get("experiments") or [])[:8]:
        if not isinstance(e, dict) or e.get("model") not in known_models: dropped.append(f"unknown model: {e.get('model') if isinstance(e, dict) else e}"); continue
        ch = {}
        for k, v in (e.get("changes") or {}).items():
            lo, hi = ALLOWED.get(k, (None, None))
            if lo is None: dropped.append(f"{e['model']}: parameter {k} not allowed"); continue
            try: v = int(v) if k in INTS else round(float(v), 3)
            except (TypeError, ValueError): dropped.append(f"{e['model']}: {k}={v!r} not a number"); continue
            if not lo <= v <= hi: dropped.append(f"{e['model']}: {k}={v} outside [{lo}, {hi}]"); continue
            ch[k] = v
        if ch: clean.append({"model": e["model"], "changes": ch, "why": str(e.get("why", ""))[:300]})
    return {"analysis": str(advice.get("analysis", ""))[:1200], "experiments": clean, "ideas": [str(i)[:300] for i in (advice.get("ideas") or [])[:8]]}, dropped

def parse_json(text):
    try: return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m: return json.loads(m.group(0))
        raise

def ask(model, prompt, think=False, timeout=600):
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0.3, "num_ctx": 16384}, "think": think,
            "messages": [{"role": "user", "content": prompt}]}
    try: r = ollama.api("/api/chat", body, timeout)
    except Exception:
        body.pop("think"); r = ollama.api("/api/chat", body, timeout)
    return r["message"]["content"]

def advise(cfg, model=None, think=False, dry_run=False, log=print):
    tag, why = pick_advisor(cfg, model)
    prompt = PROMPT.format(allowed=", ".join(f"{k} in [{a}, {b}]" for k, (a, b) in ALLOWED.items()), digest=digest(cfg))
    if dry_run: return {"advisor": tag, "why": why, "prompt": prompt}
    log(f"advisor: {tag} ({why}); prompt {len(prompt)} chars")
    with gpu_lock():
        ollama.unload_all()
        try: raw = ask(tag, prompt, think)
        finally: ollama.stop(tag)
    clean, dropped = sanitize(parse_json(raw), {m["tag"] for m in cfg["models"]})
    out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "advisor": tag, "why": why, "think": think, "advice": clean, "dropped": dropped}
    os.makedirs(store.RESULTS, exist_ok=True); json.dump(out, open(ADVICE, "w"), indent=2)
    with open(os.path.join(store.RESULTS, "advice.md"), "w") as f:
        f.write(f"# Advice from {tag}\n\n_Hypotheses from a local model, not facts. Test them with `llmeval tune MODEL --from-advice`._\n\n{clean['analysis']}\n\n## Experiments\n")
        f.writelines(f"- `{e['model']}`: {e['changes']}: {e['why']}\n" for e in clean["experiments"])
        f.write("\n## Ideas\n" + "".join(f"- {i}\n" for i in clean["ideas"]) + ("\n## Dropped (unsafe or invalid)\n" + "".join(f"- {d}\n" for d in dropped) if dropped else ""))
    return out

def space_from_advice(tag):
    """{param: [values]} for `tune --from-advice`, from the latest saved advice for this model."""
    if not os.path.exists(ADVICE): return None
    space = {}
    for e in json.load(open(ADVICE))["advice"]["experiments"]:
        if e["model"] == tag:
            for k, v in e["changes"].items(): space.setdefault(k, [])[len(space.get(k, [])):] = [v] if v not in space.get(k, []) else []
    return space or None
