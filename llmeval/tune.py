"""Bounded self-improvement loop: the eval is the fitness function for a coordinate search over Modelfile parameters.

Guardrails: candidates are throw-away `tune-*` tags (removed afterwards; nothing else is ever deleted or overwritten);
the search scores on TRAIN tasks and the winner must beat the baseline again on held-out tasks before it is proposed;
the result is only a proposal file (results/tune-best.toml). Configs are never modified automatically."""
import hashlib, json, os, statistics as st, subprocess
from . import ollama, store, tasks as T
from .harnesses import REGISTRY
from .lock import gpu_lock
from .proxy import Proxy
from .runner import run_trial

SPACE = {"temperature": [0.1, 0.25, 0.5], "top_k": [10, 20, 40], "top_p": [0.8, 0.9, 0.95],
         "repeat_penalty": [1.0, 1.05, 1.15], "num_ctx": [16384, 32768]}
TUNE_LOG = os.path.join(store.RESULTS, "tune.jsonl")

def score(recs):
    """Pass rate dominates; median wall time breaks ties (lower is better)."""
    if not recs: return -1
    rate = sum(r["done"] for r in recs) / len(recs)
    return round(rate * 1000 - st.median(r["wall_s"] for r in recs), 1)

def _evaluate(harness, tag, ts, reps, timeout, num_ctx, proxy):
    REGISTRY[harness].prepare(tag, num_ctx)
    return [run_trial(harness, tag, t, r, timeout, proxy, num_ctx) for t in ts for r in range(1, reps + 1)]

def tune(model, holdout, harness="pi", reps=2, timeout=180, space=None, keep_tags=False):
    """Generator of progress events. `model` is a config model dict (tag, base, params)."""
    space = space or SPACE; base = model["base"]; ts = T.load()
    train = [t for t in ts if t["id"] not in holdout]; held = [t for t in ts if t["id"] in holdout]
    if not train or not held: raise SystemExit("need both train and held-out tasks (set [tune] holdout in the config)")
    made = []
    proxy = Proxy() if REGISTRY[harness].needs_proxy else None
    def cand(params):
        tag = "tune-" + hashlib.sha1(json.dumps([base, params], sort_keys=True).encode()).hexdigest()[:8]
        if not ollama.has(tag): ollama.create(tag, base, params); made.append(tag)
        return tag
    try:
        with gpu_lock():
            ollama.unload_all()
            best = dict(model["params"]); tag = cand(best)
            best_recs = _evaluate(harness, tag, train, reps, timeout, int(best.get("num_ctx", 16384)), proxy); best_score = score(best_recs)
            store.append({"phase": "baseline", "params": best, "score": best_score}, TUNE_LOG)
            yield {"type": "baseline", "params": best, "score": best_score}
            for key, values in space.items():
                for v in values:
                    if v == best.get(key): continue
                    trial = {**best, key: v}; ctag = cand(trial)
                    recs = _evaluate(harness, ctag, train, reps, timeout, int(trial.get("num_ctx", 16384)), proxy); s = score(recs)
                    store.append({"phase": "candidate", "changed": {key: v}, "params": trial, "score": s}, TUNE_LOG)
                    win = s > best_score + 20        # require a real gain (~1 more pass or >=20 s faster), not noise
                    yield {"type": "candidate", "changed": {key: v}, "score": s, "accepted": win}
                    if win: best, best_score = trial, s
                    ollama.stop(ctag)
            # confirmation on held-out tasks: winner vs baseline
            base_params = model["params"]
            if best != base_params:
                b = score(_evaluate(harness, cand(base_params), held, 3, timeout, int(base_params.get("num_ctx", 16384)), proxy))
                w = score(_evaluate(harness, cand(best), held, 3, timeout, int(best.get("num_ctx", 16384)), proxy))
                confirmed = w >= b
            else:
                b = w = best_score; confirmed = False
            store.append({"phase": "holdout", "baseline": b, "winner": w, "confirmed": confirmed, "params": best}, TUNE_LOG)
            out = os.path.join(store.RESULTS, "tune-best.toml")
            if confirmed:
                with open(out, "w") as f:
                    f.write(f'# Proposed by `llmeval tune` (held-out score {w} vs baseline {b}). Review before applying.\n[[models]]\ntag = "{model["tag"]}"\nbase = "{base}"\n[models.params]\n')
                    for k, val in best.items(): f.write(f"{k} = {val}\n")
            yield {"type": "done", "confirmed": confirmed, "baseline": b, "winner": w, "params": best, "proposal": out if confirmed else None}
    finally:
        if proxy: proxy.close()
        if not keep_tags:
            for t in made:                       # only tags created by this run, all prefixed `tune-`
                if t.startswith("tune-"): ollama.stop(t); subprocess.run(["ollama", "rm", t], capture_output=True)   # unload first: a loaded tag cannot be removed


def tune_family(cfg, name, holdout, harness="pi", reps=2, timeout=180, space=None):
    """Family-level tuning: (1) tune the family's recommended variant (else its best), (2) check whether the winning parameter
    changes also help every sibling variant on the held-out tasks. The proposal is written only if the tuned variant is
    confirmed AND no sibling gets worse. Generator of progress events; configs are never modified."""
    from . import report
    variants = [m for m in cfg["models"] if m.get("family") == name and m.get("base")]
    if not variants: raise SystemExit(f"family '{name}' has no buildable variants in the config")
    table = {f: (rows, pick) for f, rows, pick, _ in report.family_table()}
    rows, pick = table.get(name, ([], None))
    lead = next((v for v in variants if pick and v["tag"] == pick["tag"]), None) or max(
        variants, key=lambda v: next((r["rate"] for r in rows if r["tag"] == v["tag"]), -1))
    yield {"type": "family", "family": name, "lead": lead["tag"], "siblings": [v["tag"] for v in variants if v is not lead]}
    result = None
    for ev in tune(lead, holdout, harness, reps, timeout, space=space):
        yield ev
        if ev["type"] == "done": result = ev
    if not result or not result["confirmed"]:
        yield {"type": "family_done", "confirmed": False, "reason": "no confirmed improvement on the lead variant", "proposal": None}; return
    changed = {k: v for k, v in result["params"].items() if lead["params"].get(k) != v}
    held = [t for t in T.load() if t["id"] in holdout]; proxy = Proxy() if REGISTRY[harness].needs_proxy else None; made, sib = [], []
    try:
        with gpu_lock():
            ollama.unload_all()
            for v in variants:
                if v is lead: continue
                def sc(params):
                    tag = "tune-" + hashlib.sha1(json.dumps([v["base"], params], sort_keys=True).encode()).hexdigest()[:8]
                    if not ollama.has(tag): ollama.create(tag, v["base"], params); made.append(tag)
                    s = score(_evaluate(harness, tag, held, 3, timeout, int(params.get("num_ctx", 16384)), proxy)); ollama.stop(tag); return s
                b, w = sc(v["params"]), sc({**v["params"], **changed})
                sib.append({"variant": v["tag"], "baseline": b, "tuned": w, "improves": w >= b})
                store.append({"phase": "family_transfer", "family": name, **sib[-1], "changed": changed}, TUNE_LOG)
                yield {"type": "transfer", **sib[-1]}
    finally:
        if proxy: proxy.close()
        for t in made:
            if t.startswith("tune-"): ollama.stop(t); subprocess.run(["ollama", "rm", t], capture_output=True)
    ok = all(x["improves"] for x in sib)
    out = os.path.join(store.RESULTS, "tune-family-best.toml")
    if ok:
        with open(out, "w") as f:
            f.write(f'# Proposed by `llmeval tune --family {name}`: lead {lead["tag"]} confirmed; no sibling variant got worse. Review before applying.\n[[families]]\nname = "{name}"\n[families.params]\n')
            for k, val in changed.items(): f.write(f"{k} = {val}\n")
    yield {"type": "family_done", "confirmed": ok, "changed": changed, "transfer": sib, "proposal": out if ok else None,
           "reason": None if ok else "the change helped the lead but hurt at least one sibling variant"}
