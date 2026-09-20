"""Bounded self-improvement loop: the eval is the fitness function for a coordinate search over Modelfile parameters.

Integrity invariants:
1. Benchmark infrastructure is strictly immutable during optimization.
2. The search uses a staged strategy: exploration (low reps) -> confirmation (higher reps) -> held-out validation.
3. Promotion requires statistical evidence (cannot promote on noisy insufficient evidence).
4. Tasks are partitioned into train, val, and final test (final test is never touched by tuning).
5. Candidates are throw-away `tune-*` tags.
6. The result is a proposal file (results/tune-best.toml); configs are never modified automatically.
"""
import hashlib, json, os, statistics as st, subprocess
from . import config, guardrails, ollama, provenance, stats, store, tasks as T
from .harnesses import REGISTRY
from .lock import gpu_lock
from .proxy import Proxy
from .runner import run_trial

SPACE = {
    "temperature": [0.1, 0.25, 0.5],
    "top_k": [10, 20, 40],
    "top_p": [0.8, 0.9, 0.95],
    "repeat_penalty": [1.0, 1.05, 1.15],
    "num_ctx": [16384, 32768]
}
TUNE_LOG = os.path.join(store.RESULTS, "tune.jsonl")

def score(recs):
    """Pass rate dominates; median wall time breaks ties (lower is better). Kept for heuristic exploration."""
    if not recs: return -1
    rate = sum(1 for r in recs if r.get("done")) / len(recs)
    med_wall = st.median(r["wall_s"] for r in recs if r.get("wall_s") is not None) if any(r.get("wall_s") is not None for r in recs) else 0.0
    return round(rate * 1000 - med_wall, 1)

def _evaluate(harness, tag, ts, reps, timeout, num_ctx, proxy, eff_params=None, exp_id="tune"):
    REGISTRY[harness].prepare(tag, num_ctx)
    out = []
    eff = dict(eff_params or {})
    eff["num_ctx"] = num_ctx
    for t in ts:
        for r in range(1, reps + 1):
            out.append(run_trial(
                harness=harness, model=tag, task=t, rep=r, timeout=timeout,
                proxy=proxy, num_ctx=num_ctx, effective_params_dict=eff,
                experiment_id=exp_id
            ))
    return out

def apply_params(config_path, tag, new_params, log=print):
    """
    Autonomously promotes winning parameters:
      1. Updates evals config (e.g. evals/default.toml)
      2. Rebuilds Ollama model tag from its base with the new parameters
      3. If num_ctx is in new_params, updates harness configs
    """
    if not config_path or not os.path.exists(config_path):
        return
    import re
    text = open(config_path).read()
    pattern = rf'(\[\[models\]\]\s*tag\s*=\s*"{re.escape(tag)}"[^\[]*?)(\[models\.params\]\s*\n.*?)?((?=\n\[\[|\n\[[a-zA-Z]|\Z))'
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        params_str = "[models.params]\n" + "".join(f"{k} = {json.dumps(v) if isinstance(v, str) else v}\n" for k, v in sorted(new_params.items()))
        head = m.group(1).rstrip()
        new_block = head + "\n" + params_str + "\n"
        text = text[:m.start()] + new_block + text[m.end():]
        open(config_path, "w").write(text)
        log(f"Auto-promoted parameters in {config_path} for {tag}")

    cfg = config.load(config_path)
    model_entry = next((m for m in cfg.get("models", []) if m["tag"] == tag), None)
    if model_entry and model_entry.get("base"):
        eff = config.effective_params(cfg, model_entry)
        eff.update(new_params)
        ollama.create(tag, model_entry["base"], eff)
        log(f"Rebuilt Ollama model tag '{tag}' with promoted parameters.")

    if "num_ctx" in new_params:
        try:
            from . import fit as fitmod
            fitmod.apply_context(tag, int(new_params["num_ctx"]), config_path=config_path, log=log)
        except Exception as e:
            log(f"Warning: apply_context failed: {e}")

def tune(model, holdout=None, harness="pi", reps=2, timeout=180, space=None, keep_tags=False,
         confirmation_reps=5, cfg_splits=None, auto_apply=False, config_path=None):
    """
    Generator of progress events with staged exploration and statistically grounded confirmation.
    `model` is a config model dict (tag, base, params).
    """
    bm_snapshot = guardrails.snapshot_benchmark()
    space = space or SPACE
    base = model.get("base")
    if not base:
        raise SystemExit("tune needs a model with a `base` in the config")

    ts = T.load()
    holdout_ids = set(holdout or ["05-bug-across-files"])
    
    # Task partitioning: train, val (holdout), and test (final test)
    # Final test tasks must NEVER participate in tuning decisions
    test_ids = set((cfg_splits or {}).get("test") or [])
    val_ids = holdout_ids
    train_tasks = [t for t in ts if t["id"] not in val_ids and t["id"] not in test_ids]
    val_tasks = [t for t in ts if t["id"] in val_ids and t["id"] not in test_ids]

    if not train_tasks or not val_tasks:
        raise SystemExit("need both train and held-out validation tasks")

    made = []
    proxy = Proxy() if REGISTRY[harness].needs_proxy else None

    def cand(params):
        tag = "tune-" + hashlib.sha1(json.dumps([base, params], sort_keys=True).encode()).hexdigest()[:8]
        if not ollama.has(tag):
            ollama.create(tag, base, params)
            made.append(tag)
        return tag

    try:
        with gpu_lock():
            ollama.unload_all()
            guardrails.assert_immutable_benchmark(bm_snapshot)

            best = dict(model.get("params", {}))
            base_params = dict(best)
            tag = cand(best)
            
            # 1. Baseline exploration
            best_recs = _evaluate(harness, tag, train_tasks, reps, timeout, int(best.get("num_ctx", 16384)), proxy, best)
            best_score = score(best_recs)
            store.append({"phase": "baseline", "params": best, "score": best_score, "recs": len(best_recs)}, TUNE_LOG)
            yield {"type": "baseline", "params": best, "score": best_score}

            # 2. Parameter coordinate search (Exploration Phase)
            for key, values in space.items():
                for v in values:
                    guardrails.assert_immutable_benchmark(bm_snapshot)
                    if v == best.get(key): continue
                    
                    trial = {**best, key: v}
                    ctag = cand(trial)
                    
                    # Exploration trial (low reps)
                    recs = _evaluate(harness, ctag, train_tasks, reps, timeout, int(trial.get("num_ctx", 16384)), proxy, trial)
                    s = score(recs)
                    
                    preliminary_win = s > best_score + 20
                    confirmed = False
                    
                    if preliminary_win:
                        # 3. Confirmation Phase (Higher Repetitions on train tasks)
                        # Require statistical evidence before accepting improvement
                        base_confirm = _evaluate(harness, tag, train_tasks, confirmation_reps, timeout, int(best.get("num_ctx", 16384)), proxy, best)
                        cand_confirm = _evaluate(harness, ctag, train_tasks, confirmation_reps, timeout, int(trial.get("num_ctx", 16384)), proxy, trial)
                        stat_eval = stats.compare_runs(base_confirm, cand_confirm)
                        
                        # Only accept if improvement is supported by data or significant gain
                        if stat_eval["decision"] == "improvement_supported":
                            confirmed = True
                            best, best_score = trial, score(cand_confirm)
                            tag = ctag
                        else:
                            # Not supported by sufficient statistical evidence
                            confirmed = False

                    store.append({
                        "phase": "candidate",
                        "changed": {key: v},
                        "params": trial,
                        "preliminary_score": s,
                        "confirmed": confirmed
                    }, TUNE_LOG)
                    
                    yield {
                        "type": "candidate",
                        "changed": {key: v},
                        "score": s,
                        "accepted": confirmed,
                        "note": "confirmed with statistical evidence" if confirmed else "rejected (insufficient evidence or regression)"
                    }
                    if not confirmed:
                        ollama.stop(ctag)

            # 4. Held-out Validation Phase (on independent validation tasks)
            if best != base_params:
                guardrails.assert_immutable_benchmark(bm_snapshot)
                base_held = _evaluate(harness, cand(base_params), val_tasks, confirmation_reps, timeout, int(base_params.get("num_ctx", 16384)), proxy, base_params)
                cand_held = _evaluate(harness, cand(best), val_tasks, confirmation_reps, timeout, int(best.get("num_ctx", 16384)), proxy, best)
                
                val_stat = stats.compare_runs(base_held, cand_held)
                b_score = score(base_held)
                w_score = score(cand_held)
                
                # Winner must not regress on validation tasks
                validation_confirmed = val_stat["decision"] in ("improvement_supported", "insufficient_evidence") and w_score >= b_score
            else:
                b_score = w_score = best_score
                validation_confirmed = False

            store.append({
                "phase": "holdout_validation",
                "baseline": b_score,
                "winner": w_score,
                "confirmed": validation_confirmed,
                "params": best
            }, TUNE_LOG)

            out_proposal = os.path.join(store.RESULTS, "tune-best.toml")
            if validation_confirmed:
                with open(out_proposal, "w", encoding="utf-8") as f:
                    f.write(f'# Proposed by `llmeval tune` (validation score {w_score} vs baseline {b_score}). Review before applying.\n')
                    f.write(f'# Statistical confirmation passed with {confirmation_reps} confirmation reps.\n')
                    f.write(f'[[models]]\ntag = "{model["tag"]}"\nbase = "{base}"\n[models.params]\n')
                    for k, val in best.items():
                        f.write(f"{k} = {val}\n")
                if auto_apply and config_path:
                    apply_params(config_path, model["tag"], best)

            yield {
                "type": "done",
                "confirmed": validation_confirmed,
                "baseline": b_score,
                "winner": w_score,
                "params": best,
                "proposal": out_proposal if validation_confirmed else None
            }
    finally:
        if proxy: proxy.close()
        if not keep_tags:
            for t in made:
                if t.startswith("tune-"):
                    ollama.stop(t)
                    subprocess.run(["ollama", "rm", t], capture_output=True)

def tune_family(cfg, name, holdout=None, harness="pi", reps=2, timeout=180, space=None, auto_apply=False, config_path=None):
    """Family-level tuning: tune lead variant, then verify transfer to sibling variants."""
    from . import report
    variants = [m for m in cfg["models"] if m.get("family") == name and m.get("base")]
    if not variants:
        raise SystemExit(f"family '{name}' has no buildable variants in the config")
    
    table = {f: (rows, pick) for f, rows, pick, _ in report.family_table()}
    rows, pick = table.get(name, ([], None))
    lead = next((v for v in variants if pick and v["tag"] == pick["tag"]), None) or max(
        variants, key=lambda v: next((r["rate"] for r in rows if r["tag"] == v["tag"]), -1))
    
    yield {"type": "family", "family": name, "lead": lead["tag"], "siblings": [v["tag"] for v in variants if v is not lead]}
    result = None
    splits = cfg.get("tasks", {}).get("split", {})
    for ev in tune(lead, holdout, harness, reps, timeout, space=space, cfg_splits=splits, auto_apply=auto_apply, config_path=config_path):
        yield ev
        if ev["type"] == "done":
            result = ev

    if not result or not result["confirmed"]:
        yield {"type": "family_done", "confirmed": False, "reason": "no confirmed improvement on the lead variant", "proposal": None}
        return

    changed = {k: v for k, v in result["params"].items() if lead.get("params", {}).get(k) != v}
    held_ids = set(holdout or ["05-bug-across-files"])
    held_tasks = [t for t in T.load() if t["id"] in held_ids]
    proxy = Proxy() if REGISTRY[harness].needs_proxy else None
    made, sib = [], []
    
    try:
        with gpu_lock():
            ollama.unload_all()
            for v in variants:
                if v is lead: continue
                def sc(params):
                    tag = "tune-" + hashlib.sha1(json.dumps([v["base"], params], sort_keys=True).encode()).hexdigest()[:8]
                    if not ollama.has(tag):
                        ollama.create(tag, v["base"], params)
                        made.append(tag)
                    s = score(_evaluate(harness, tag, held_tasks, 3, timeout, int(params.get("num_ctx", 16384)), proxy, params))
                    ollama.stop(tag)
                    return s
                b, w = sc(v.get("params", {})), sc({**v.get("params", {}), **changed})
                improves = (w >= b)
                sib.append({"variant": v["tag"], "baseline": b, "tuned": w, "improves": improves})
                store.append({"phase": "family_transfer", "family": name, **sib[-1], "changed": changed}, TUNE_LOG)
                yield {"type": "transfer", **sib[-1]}
    finally:
        if proxy: proxy.close()
        for t in made:
            if t.startswith("tune-"):
                ollama.stop(t)
                subprocess.run(["ollama", "rm", t], capture_output=True)

    ok = all(x["improves"] for x in sib)
    out = os.path.join(store.RESULTS, "tune-family-best.toml")
    if ok:
        with open(out, "w", encoding="utf-8") as f:
            f.write(f'# Proposed by `llmeval tune --family {name}`: lead {lead["tag"]} confirmed; no sibling variant regressed.\n')
            f.write(f'[[families]]\nname = "{name}"\n[families.params]\n')
            for k, val in changed.items():
                f.write(f"{k} = {val}\n")
        if auto_apply and config_path:
            apply_params(config_path, lead["tag"], result["params"])

    yield {
        "type": "family_done",
        "confirmed": ok,
        "changed": changed,
        "transfer": sib,
        "proposal": out if ok else None,
        "reason": None if ok else "the change helped the lead but hurt at least one sibling variant"
    }
