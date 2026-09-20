import argparse, json, os, sys
from . import config, discover as discovermod, fit as fitmod, prune as prunemod, ollama, report, runner, store, tasks as T
from .harnesses import REGISTRY

DEFAULT_CFG = os.path.join(store.ROOT, "evals", "default.toml")

def cmd_list(a):
    print("harnesses:")
    for h in REGISTRY.values(): print(f"  {h.name:9} {'yes' if h.available() else 'NO '}  {'(experimental) ' if h.experimental else ''}{h.note}")
    print("tasks:")
    for t in T.load(): print(f"  {t['id']:22} {t['title']}")
    try: print("ollama:", ollama.version(), "|", len(ollama.installed()), "models installed")
    except Exception as e: print("ollama: not reachable:", e)

def cmd_selfcheck(a):
    bad = 0
    for t in T.load():
        ok = T.selfcheck(t); bad += not ok; print(("OK     " if ok else "BROKEN "), t["id"])
    sys.exit(1 if bad else 0)

def cmd_prepare(a): config.prepare(config.load(a.config), family=a.family, dry_run=a.dry_run)

def cmd_run(a):
    ctx_sweep = [int(x.strip()) for x in a.ctx_sweep.split(",")] if getattr(a, "ctx_sweep", None) else None
    task_ids = a.tasks.split(',') if a.tasks else None
    for ev in runner.run_matrix(
        config.load(a.config), force=a.force, keep=a.keep, family=a.family,
        variant=a.variant, task_ids=task_ids, ctx_sweep=ctx_sweep,
        experiment_id=getattr(a, "experiment_id", None),
        reps_override=getattr(a, "reps", None),
        split_name=getattr(a, "split", None)
    ):
        t = ev["type"]
        if t == "plan": print(f"{ev['total']} trials to run ({ev['skipped']} already done)")
        elif t == "model": print(f"== {ev['model']}")
        elif t == "start":
            ctx_str = f" ctx {ev.get('num_ctx')}" if ev.get("num_ctx") else ""
            print(f"[{ev['n']}] {ev['harness']:9} {ev['task']}{ctx_str} rep {ev['rep']} ...", end=" ", flush=True)
        elif t == "trial": print(("PASS" if ev["done"] else "FAIL") + f" {ev['wall_s']}s" + (" LOOP" if ev.get("loop") else "") + (" TIMEOUT" if ev["timeout"] else "") + (" TAMPERED" if ev["tampered"] else ""))

def cmd_families(a):
    """Configured families, their variants and whether each is ready; then the recommendation from existing results."""
    cfg = config.load(a.config)
    for m in cfg["models"]:
        if m.get("family"): print(f"{m['family']:14} {m['size'] or '-':6} {m['quant'] or '-':8} {m['tag']:34} {'ready' if ollama.has(m['tag']) else 'not built'}")
    from . import report
    for name, rows, pick, why in report.family_table(): print(f"\n{name}: recommended {pick['tag'] if pick else '-'} ({why})")

def cmd_add_family(a):
    from . import models
    models.add_family(a.config, a.name, a.sizes.split(","), a.quants.split(","), a.ctx, a.as_is)

def cmd_advise(a):
    from . import advisor
    out = advisor.advise(config.load(a.config), a.model, a.think, a.dry_run)
    if a.dry_run: print(f"advisor would be: {out['advisor']} ({out['why']})\n\n{out['prompt']}"); return
    print(open(os.path.join(store.RESULTS, "advice.md")).read())

def cmd_compare(a):
    from . import report
    out = report.compare(a.before, a.after, a.harness, a.model, a.tasks.split(",") if a.tasks else None)
    print(json.dumps(out, indent=1)); sys.exit(0 if out["verdict"] in ("better", "same") else 1)

def cmd_fit(a):
    cfg = config.load(a.config)
    if getattr(a, "optimize", False):
        tags = [a.model] if getattr(a, "model", None) else [m["tag"] for m in cfg["models"] if ollama.has(m["tag"])]
        if not tags:
            raise SystemExit("No installed models found to optimize. Specify --model <tag>.")
        for tag in tags:
            res = fitmod.optimize(tag, min_ctx=a.min_ctx, max_ctx=a.max_ctx, prompt_ratio=a.stress)
            if getattr(a, "apply", False) and res.get("recommended_ctx"):
                fitmod.apply_context(tag, res["recommended_ctx"], config_path=a.config)
        return

    for r in fitmod.measure([m["tag"] for m in cfg["models"]], [int(x) for x in a.ctx.split(",")]):
        print(json.dumps({k: r.get(k) for k in ("model", "num_ctx", "tok_s", "fit", "error")}))

def cmd_apply_ctx(a):
    fitmod.apply_context(a.model, a.ctx, config_path=a.config)

def cmd_report(a):
    md = report.render()
    if a.out: open(a.out, "w").write(md); print("wrote", a.out)
    else: print(md)

def cmd_tune(a):
    from . import tune
    cfg = config.load(a.config); holdout = cfg.get("tune", {}).get("holdout", ["05-bug-across-files"])
    if not a.model and not a.family: raise SystemExit("give a MODEL tag or --family NAME")
    space = None
    if a.from_advice:
        from . import advisor
        space = advisor.space_from_advice(a.model or next(v["tag"] for v in cfg["models"] if v.get("family") == a.family))
        if not space: raise SystemExit("no saved advice for this model: run `llmeval advise` first")
    if a.family:
        events = tune.tune_family(cfg, a.family, holdout, a.harness, a.reps, cfg["run"]["timeout"], space)
    else:
        model = next(m for m in cfg["models"] if m["tag"] == a.model)
        if not model.get("base"): raise SystemExit("tune needs a model with a `base` in the config")
        events = tune.tune(
            model, holdout, a.harness, a.reps, cfg["run"]["timeout"], space=space,
            confirmation_reps=getattr(a, "confirm_reps", 5),
            cfg_splits=cfg.get("tasks", {}).get("split", {})
        )
    for ev in events: print(json.dumps(ev))

def cmd_discover(a):
    cfg = config.load(a.config)
    plan = discovermod.discover(cfg, a.name, a.ctx, a.max, a.hf, a.advisor)
    print("\n".join(discovermod.format_plan(plan)))
    if a.apply:
        discovermod.apply(a.config, plan)
        if a.pull: config.prepare(config.load(a.config), family=a.name)
        else: print(f"next: python3 -m llmeval prepare --family {a.name}   (downloads {plan['download_gb']} GB)")
    else: print("\n(dry run: nothing changed. Add --apply to write the family to the config, --pull to also download.)")

def cmd_prune(a):
    cfg = config.load(a.config)
    plan = prunemod.plan(cfg, a.keep_top, a.margin, a.min_n, a.include_bases)
    print("\n".join(prunemod.format_plan(plan)))
    if a.verbose: print("\nprotected:", json.dumps(plan["protected"], indent=1))
    if not plan["delete"]: return
    if not a.yes: print("\n(dry run: nothing deleted. Re-run with --yes to delete, --drop-config to also remove their config entries.)"); return
    prunemod.execute(plan, a.config, a.drop_config)

def cmd_loop(a):
    """Automated end-to-end evaluation & self-improvement loop for a family."""
    cfg = config.load(a.config)
    family = a.family
    if not family:
        cfg_families = cfg.get("families", [])
        if cfg_families and cfg_families[0].get("name"):
            family = cfg_families[0]["name"]
        else:
            f_cands = [m.get("family") for m in cfg.get("models", []) if m.get("family")]
            family = f_cands[0] if f_cands else "granite4.1"

    print(f"=== [1/5] DISCOVERY: Searching models for family '{family}' ===")
    init_ctx = a.ctx if a.ctx > 0 else 32768
    plan = discovermod.discover(cfg, family, init_ctx)
    print("\n".join(discovermod.format_plan(plan)))
    if plan and plan.get("models"):
        discovermod.apply(a.config, plan)
        cfg = config.load(a.config)

    print(f"\n=== [2/5] HARDWARE FIT: Probing context feasibility & limits ===")
    family_models = [m for m in cfg.get("models", []) if m.get("family") == family or family in m.get("tag", "")]
    hardware_max_safe = 32768
    for m in (family_models or cfg.get("models", [])):
        if ollama.has(m["tag"]):
            for r in fitmod.measure([m["tag"]], ctxs=[16384, 32768, 49152, 65536]):
                print(f"  {r['model']} {r['num_ctx'] // 1024}k: {r.get('fit', r.get('error'))} {r.get('tok_s', '')} tok/s")
                if not r.get("error") and (r.get("fit") or {}).get("gpu_pct", 0) >= 99.0:
                    hardware_max_safe = max(hardware_max_safe, r["num_ctx"])

    print(f"\n=== [3/5] BENCHMARK MATRIX & AUTONOMOUS CONTEXT DISCOVERY ===")
    if a.ctx > 0:
        candidates = [16384, a.ctx] if a.ctx > 16384 else [a.ctx]
    elif hardware_max_safe >= 65536:
        candidates = [16384, 32768, 65536]
    elif hardware_max_safe >= 32768:
        candidates = [16384, 32768]
    else:
        candidates = [8192, 16384]

    print(f"Sweeping context candidates: {[c // 1024 for c in candidates]}k...")
    has_fam = any(m.get("family") == family for m in cfg.get("models", []))
    for ev in runner.run_matrix(cfg, family=family if has_fam else None, ctx_sweep=candidates):
        if ev["type"] == "trial":
            ctx_str = f" ctx {ev.get('num_ctx')}" if ev.get("num_ctx") else ""
            print(f"  {'PASS' if ev['done'] else 'FAIL'} {ev['harness']:9} {ev['task']}{ctx_str} #{ev['rep']} {ev['wall_s']}s")

    # Select best context size
    curve = report.context_curve()
    fam_curve = [c for c in curve if (any(m["tag"] == c["model"] for m in family_models) or family in c["model"]) and c["n"] > 0]
    if not fam_curve:
        fam_curve = [c for c in curve if c["num_ctx"] in candidates and c["n"] > 0]

    if fam_curve:
        import collections
        ctx_scores = collections.defaultdict(list)
        for c in fam_curve: ctx_scores[c["num_ctx"]].append(c)
        ranked = []
        for c_val, c_list in ctx_scores.items():
            avg_rate = sum(x["rate"] for x in c_list) / len(c_list)
            tot_fails = sum(x["failures"] for x in c_list)
            med_wall = sum((x["median_wall"] or 0) for x in c_list) / len(c_list)
            ranked.append((c_val, avg_rate, tot_fails, med_wall))
        ranked.sort(key=lambda x: (-x[1], x[2], x[3], -x[0]))
        best_c, b_rate, b_fails, b_wall = ranked[0]
        print(f"\n--> BEST CONTEXT DISCOVERED: {best_c // 1024}k (Pass Rate: {b_rate*100:.0f}%, Wall: {b_wall:.1f}s, Fails: {b_fails})")
        for m in family_models:
            fitmod.apply_context(m["tag"], best_c, config_path=a.config)
        cfg = config.load(a.config)
    else:
        best_c = candidates[-1]
        print(f"\n--> Defaulting to hardware-safe context: {best_c // 1024}k")

    print(f"\n=== [4/5] AUTONOMOUS RECURSIVE SELF-IMPROVEMENT (RSI) ===")
    from . import tune
    holdout = cfg.get("tune", {}).get("holdout", ["05-bug-across-files"])
    harness = cfg["run"]["harnesses"][0]
    timeout = cfg["run"]["timeout"]
    max_cycles = getattr(a, "cycles", 3) or 3

    for cycle in range(1, max_cycles + 1):
        print(f"\n--- Recursive Self-Improvement Cycle {cycle}/{max_cycles} ---")
        improved = False
        if has_fam:
            for ev in tune.tune_family(cfg, family, holdout, harness, a.reps, timeout, auto_apply=True, config_path=a.config):
                if ev["type"] == "candidate":
                    print(f"  Variant {ev.get('candidate')}: score={ev.get('score')} accepted={ev.get('accepted')} ({ev.get('note', '')})")
                elif ev["type"] == "family_done":
                    if ev.get("confirmed"):
                        improved = True
                        print(f"  Cycle {cycle} Winner Promoted: {ev.get('winner')}")
                    else:
                        print(f"  Cycle {cycle}: No sibling improvement confirmed ({ev.get('reason')}).")
        else:
            lead = next((m for m in family_models if m.get("base")), None)
            if lead:
                for ev in tune.tune(lead, holdout, harness, a.reps, timeout, confirmation_reps=a.confirm_reps, cfg_splits=cfg.get("tasks", {}).get("split", {}), auto_apply=True, config_path=a.config):
                    if ev["type"] == "candidate":
                        print(f"  Candidate {ev.get('changed')}: score={ev.get('score')} accepted={ev.get('accepted')} ({ev.get('note', '')})")
                    elif ev["type"] == "done":
                        if ev.get("confirmed"):
                            improved = True
                            print(f"  Cycle {cycle} Winner Promoted: winner={ev.get('winner')}")
                        else:
                            print(f"  Cycle {cycle}: No candidate achieved statistical confirmation.")
            else:
                print("  No tunable base models; skipping parameter mutation.")
                break

        cfg = config.load(a.config)
        if not improved:
            print("  RSI Convergence reached: System stabilized; no further statistically supported improvements.")
            break

    print(f"\n=== [5/5] MULTI-OBJECTIVE PARETO RECOMMENDATIONS ===")
    for fam_name, rows, pick, why in report.family_table(policy=a.policy):
        if fam_name == family:
            ci = f"[{int(100*pick.get('ci_low', 0))}%, {int(100*pick.get('ci_high', 0))}%]" if pick else ""
            print(f"Family {fam_name} Winner ({a.policy}): {pick['tag'] if pick else 'None'} {ci} ({why})")

def cmd_import_legacy(a):
    """Import the pre-llmeval repeats/harness results (marked legacy) so they show up in reports."""
    have = store.done_keys(); n = 0
    for f, default_rep in (("repeats.jsonl", None), ("harness.jsonl", 1)):
        p = os.path.join(store.RESULTS, f)
        for r in store.rows(p):
            r.setdefault("rep", default_rep or 1); r["task"] = {1: "01-add-function", 2: "02-rename-symbol", 3: "03-fix-bug"}.get(r["task"], r["task"])
            if store.key(r) in have: continue
            store.append({**r, "legacy": True, "tampered": False}); have.add(store.key(r)); n += 1
    print("imported", n, "rows")

def cmd_demo_data(a):
    from . import demo
    demo.seed(); print("seeded sample results in", store.RESULTS)

def cmd_models(a):
    """Installed ollama models and whether each is already in the config."""
    cfg = config.load(a.config); inlist = {m["tag"] if ":" in m["tag"] else m["tag"] + ":latest" for m in cfg["models"]}
    inst = ollama.installed()
    print(f"{'in config':10} {'model':36} ctx")
    for tag in sorted(inst): print(f"{'yes' if tag in inlist else '-':10} {tag:36} {config._num_ctx(tag, 0) or ''}")
    missing = sorted(inlist - set(inst))
    if missing: print("\nin config but not installed (run `prepare`, or `ollama pull`):", ", ".join(missing))

def cmd_add(a):
    """Append models to the config. Existing installed tags are used as-is; --base builds a tuned tag from a base model."""
    from . import models
    models.add(a.config, a.tags, a.base, a.ctx, a.pull)

def cmd_demo_data(a):
    from . import demo
    demo.seed(); print("seeded sample results in", store.RESULTS)

def cmd_models(a):
    """Installed ollama models and whether each is already in the config."""
    cfg = config.load(a.config); inlist = {m["tag"] if ":" in m["tag"] else m["tag"] + ":latest" for m in cfg["models"]}
    inst = ollama.installed()
    print(f"{'in config':10} {'model':36} ctx")
    for tag in sorted(inst): print(f"{'yes' if tag in inlist else '-':10} {tag:36} {config._num_ctx(tag, 0) or ''}")
    missing = sorted(inlist - set(inst))
    if missing: print("\nin config but not installed (run `prepare`, or `ollama pull`):", ", ".join(missing))

def cmd_add(a):
    """Append models to the config. Existing installed tags are used as-is; --base builds a tuned tag from a base model."""
    cfg = config.load(a.config); have = {m["tag"] for m in cfg["models"]}; blocks = []
    for tag in a.tags:                       # validate everything first so a bad tag never leaves a half-written config
        if tag in have: print(f"skip {tag}: already in the config"); continue
        if a.base:
            block = f'\n[[models]]\ntag  = "{tag}"\nbase = "{a.base}"\n' + (f"[models.params]\nnum_ctx = {a.ctx}\n" if a.ctx else "")
        else:
            if a.pull: ollama.pull(tag)
            if not ollama.has(tag): raise SystemExit(f"{tag} is not installed: use --pull to download it, or --base to build it from another model")
            block = f'\n[[models]]\ntag = "{tag}"\n'
        blocks.append((tag, block))
    if blocks: open(a.config, "a").write("".join(b for _, b in blocks))
    print("added:", ", ".join(t for t, _ in blocks) if blocks else "nothing")

def cmd_tui(a):
    from . import tui
    tui.main(a.config)

def main(argv=None):
    p = argparse.ArgumentParser(prog="llmeval", description=__doc__ or "Evaluate local models and coding-agent harnesses")
    p.add_argument("-c", "--config", default=DEFAULT_CFG)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(f=cmd_list)
    sub.add_parser("selfcheck", help="prove every task is solvable (no GPU)").set_defaults(f=cmd_selfcheck)
    sub.add_parser("prepare", help="pull base models and create tuned tags").set_defaults(f=cmd_prepare)
    sub.choices["prepare"].add_argument("--family"); sub.choices["prepare"].add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--variant", help="label stored with each trial so before/after runs of the same trials can be compared"); r.add_argument("--tasks", help="comma list of task ids (overrides the config)"); r.add_argument("--family", help="only the variants of one family"); r.add_argument("--force", action="store_true"); r.add_argument("--keep", action="store_true")
    r.add_argument("--ctx-sweep", help="comma-separated context sizes to benchmark, e.g. 16384,32768,65536")
    r.add_argument("--experiment-id", help="experiment identity identifier")
    r.add_argument("--split", help="evaluate only tasks from this split: train, val, or test")
    r.add_argument("--reps", type=int, help="override repetition count")
    r.set_defaults(f=cmd_run)
    f = sub.add_parser("fit", help="GPU fit measurement or empirical context optimization")
    f.add_argument("--ctx", default="16384,32768,49152,65536", help="comma-separated context sizes to probe")
    f.add_argument("--optimize", action="store_true", help="run empirical optimization loop to find max practical context")
    f.add_argument("--model", help="model tag to optimize")
    f.add_argument("--min-ctx", type=int, default=16384, help="minimum context baseline (default: 16384)")
    f.add_argument("--max-ctx", type=int, default=262144, help="maximum context ceiling (default: 262144)")
    f.add_argument("--stress", type=float, default=0.75, help="context load ratio for stress testing (default: 0.75)")
    f.add_argument("--apply", action="store_true", help="recreate model tag with recommended safe num_ctx")
    f.set_defaults(f=cmd_fit)
    o = sub.add_parser("report"); o.add_argument("--out")
    o.add_argument("--policy", default="balanced", choices=["balanced", "max_quality", "fastest", "pareto"], help="Pareto recommendation policy")
    o.set_defaults(f=cmd_report)
    t = sub.add_parser("tune", help="bounded self-improvement search over Modelfile params"); t.add_argument("model", nargs="?"); t.add_argument("--family", help="tune a whole family: lead variant, then check the siblings"); t.add_argument("--harness", default="pi"); t.add_argument("--reps", type=int, default=2); t.add_argument("--from-advice", action="store_true", help="search only the values the advisor proposed")
    t.add_argument("--confirm-reps", type=int, default=5, help="repetitions for confirmation phase before promotion")
    t.set_defaults(f=cmd_tune)
    sub.add_parser("import-legacy").set_defaults(f=cmd_import_legacy)
    ad = sub.add_parser("advise", help="ask the best local model to analyse results and propose experiments")
    ad.add_argument("--model", help="advisor tag (default: [advisor] model in the config, else the best model in your results)")
    ad.add_argument("--think", action="store_true", help="enable thinking mode if the model supports it"); ad.add_argument("--dry-run", action="store_true", help="show the prompt, call nothing"); ad.set_defaults(f=cmd_advise)
    dv = sub.add_parser("discover", help="find a family on ollama/HuggingFace and let the advisor pick sizes and quantizations for this machine")
    dv.add_argument("name"); dv.add_argument("--ctx", type=int, default=32768); dv.add_argument("--max", type=int, default=3, help="variants to choose")
    dv.add_argument("--hf", action="store_true", help="also search HuggingFace (automatic when ollama lists fewer than 2 variants)"); dv.add_argument("--advisor")
    dv.add_argument("--apply", action="store_true", help="write the family to the config"); dv.add_argument("--pull", action="store_true", help="with --apply: also download"); dv.set_defaults(f=cmd_discover)
    pr = sub.add_parser("prune", help="delete models that lost the ranking (dry run unless --yes)")
    pr.add_argument("--keep-top", type=int, default=3); pr.add_argument("--margin", type=float, default=0.15); pr.add_argument("--min-n", type=int, default=6)
    pr.add_argument("--include-bases", action="store_true", help="also delete base models whose tuned tags are all deleted (this frees the disk)")
    pr.add_argument("--drop-config", action="store_true"); pr.add_argument("--verbose", action="store_true"); pr.add_argument("--yes", action="store_true"); pr.set_defaults(f=cmd_prune)
    cp = sub.add_parser("compare", help="paired before/after comparison of two --variant labels (exit 1 if 'worse')")
    cp.add_argument("before"); cp.add_argument("after"); cp.add_argument("--harness"); cp.add_argument("--model"); cp.add_argument("--tasks"); cp.set_defaults(f=cmd_compare)
    sub.add_parser("families", help="configured families, variant status and the current recommendation").set_defaults(f=cmd_families)
    af = sub.add_parser("add-family", help="add a model family: every size x quantization becomes a variant"); af.add_argument("name")
    af.add_argument("--sizes", required=True, help="comma list, e.g. 3b,8b"); af.add_argument("--quants", required=True, help="comma list, e.g. q4_K_M,q6_K,q8_0")
    af.add_argument("--ctx", type=int); af.add_argument("--as-is", action="store_true"); af.set_defaults(f=cmd_add_family)
    sub.add_parser("models", help="installed ollama models and whether they are in the config").set_defaults(f=cmd_models)
    d = sub.add_parser("add", help="add models to the config (installed tags as-is, or built from --base)")
    d.add_argument("tags", nargs="+"); d.add_argument("--base"); d.add_argument("--ctx", type=int); d.add_argument("--pull", action="store_true"); d.set_defaults(f=cmd_add)
    ap = sub.add_parser("apply-ctx", help="apply a context size across Modelfiles, evals config, Pi agent, Opencode, and Ollama")
    ap.add_argument("model"); ap.add_argument("ctx", type=int); ap.set_defaults(f=cmd_apply_ctx)
    sub.add_parser("demo-data", help="write synthetic results (use with LLMEVAL_RESULTS=/tmp/dir)").set_defaults(f=cmd_demo_data)
    lp = sub.add_parser("loop", help="automated end-to-end evaluation & self-improvement loop for a family")
    lp.add_argument("family", nargs="?", default="", help="family name (optional, defaults to primary configured family)")
    lp.add_argument("--ctx", type=int, default=0, help="target context size (0 = auto-discover optimal context)")
    lp.add_argument("--cycles", type=int, default=3, help="maximum recursive self-improvement cycles (default: 3)")
    lp.add_argument("--reps", type=int, default=2)
    lp.add_argument("--confirm-reps", type=int, default=5)
    lp.add_argument("--policy", default="balanced", choices=["balanced", "max_quality", "fastest", "pareto"])
    lp.set_defaults(f=cmd_loop)
    sub.add_parser("tui").set_defaults(f=cmd_tui)
    a = p.parse_args(argv); a.f(a)
