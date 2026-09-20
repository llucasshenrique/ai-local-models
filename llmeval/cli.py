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
    for ev in runner.run_matrix(config.load(a.config), force=a.force, keep=a.keep, family=a.family, variant=a.variant, task_ids=a.tasks.split(',') if a.tasks else None):
        t = ev["type"]
        if t == "plan": print(f"{ev['total']} trials to run ({ev['skipped']} already done)")
        elif t == "model": print(f"== {ev['model']}")
        elif t == "start": print(f"[{ev['n']}] {ev['harness']:9} {ev['task']} rep {ev['rep']} ...", end=" ", flush=True)
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
    for r in fitmod.measure([m["tag"] for m in cfg["models"]], [int(x) for x in a.ctx.split(",")]):
        print(json.dumps({k: r.get(k) for k in ("model", "num_ctx", "tok_s", "fit", "error")}))

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
        events = tune.tune(model, holdout, a.harness, a.reps, cfg["run"]["timeout"], space=space)
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
    r = sub.add_parser("run"); r.add_argument("--variant", help="label stored with each trial so before/after runs of the same trials can be compared"); r.add_argument("--tasks", help="comma list of task ids (overrides the config)"); r.add_argument("--family", help="only the variants of one family"); r.add_argument("--force", action="store_true"); r.add_argument("--keep", action="store_true"); r.set_defaults(f=cmd_run)
    f = sub.add_parser("fit"); f.add_argument("--ctx", default="16384,32768,49152,65536"); f.set_defaults(f=cmd_fit)
    o = sub.add_parser("report"); o.add_argument("--out"); o.set_defaults(f=cmd_report)
    t = sub.add_parser("tune", help="bounded self-improvement search over Modelfile params"); t.add_argument("model", nargs="?"); t.add_argument("--family", help="tune a whole family: lead variant, then check the siblings"); t.add_argument("--harness", default="pi"); t.add_argument("--reps", type=int, default=2); t.add_argument("--from-advice", action="store_true", help="search only the values the advisor proposed"); t.set_defaults(f=cmd_tune)
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
    sub.add_parser("demo-data", help="write synthetic results (use with LLMEVAL_RESULTS=/tmp/dir)").set_defaults(f=cmd_demo_data)
    sub.add_parser("tui").set_defaults(f=cmd_tui)
    a = p.parse_args(argv); a.f(a)
