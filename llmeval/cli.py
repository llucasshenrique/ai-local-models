import argparse, json, os, sys
from . import config, fit as fitmod, ollama, report, runner, store, tasks as T
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

def cmd_prepare(a): config.prepare(config.load(a.config))

def cmd_run(a):
    for ev in runner.run_matrix(config.load(a.config), force=a.force, keep=a.keep):
        t = ev["type"]
        if t == "plan": print(f"{ev['total']} trials to run ({ev['skipped']} already done)")
        elif t == "model": print(f"== {ev['model']}")
        elif t == "start": print(f"[{ev['n']}] {ev['harness']:9} {ev['task']} rep {ev['rep']} ...", end=" ", flush=True)
        elif t == "trial": print(("PASS" if ev["done"] else "FAIL") + f" {ev['wall_s']}s" + (" LOOP" if ev.get("loop") else "") + (" TIMEOUT" if ev["timeout"] else "") + (" TAMPERED" if ev["tampered"] else ""))

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
    cfg = config.load(a.config); model = next(m for m in cfg["models"] if m["tag"] == a.model)
    if not model.get("base"): raise SystemExit("tune needs a model with a `base` in the config")
    holdout = cfg.get("tune", {}).get("holdout", ["05-bug-across-files"])
    for ev in tune.tune(model, holdout, a.harness, a.reps, cfg["run"]["timeout"]):
        print(json.dumps(ev))

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
    r = sub.add_parser("run"); r.add_argument("--force", action="store_true"); r.add_argument("--keep", action="store_true"); r.set_defaults(f=cmd_run)
    f = sub.add_parser("fit"); f.add_argument("--ctx", default="16384,32768,49152,65536"); f.set_defaults(f=cmd_fit)
    o = sub.add_parser("report"); o.add_argument("--out"); o.set_defaults(f=cmd_report)
    t = sub.add_parser("tune", help="bounded self-improvement search over Modelfile params"); t.add_argument("model"); t.add_argument("--harness", default="pi"); t.add_argument("--reps", type=int, default=2); t.set_defaults(f=cmd_tune)
    sub.add_parser("import-legacy").set_defaults(f=cmd_import_legacy)
    sub.add_parser("demo-data", help="write synthetic results (use with LLMEVAL_RESULTS=/tmp/dir)").set_defaults(f=cmd_demo_data)
    sub.add_parser("tui").set_defaults(f=cmd_tui)
    a = p.parse_args(argv); a.f(a)
