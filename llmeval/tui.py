"""Terminal UI (stdlib curses). Tabs: 1 Results  2 Run  3 Context fit  4 Setup  5 Tune.
Keys: 1-5 switch tab | up/down/PgUp/PgDn scroll | r start run (tab 2) | m measure fit (tab 3) | s selfcheck, p prepare (tab 4)
      t tune the first model with a `base` (tab 5) | x stop after the current trial | q quit.
Long jobs run in one worker thread; the GPU lock keeps them serial and the UI never loads a model itself."""
import collections, curses, threading, time
from . import config, fit as fitmod, ollama, report, runner, store, tasks as T
from .harnesses import REGISTRY

TABS = ["Results", "Run", "Context fit", "Setup", "Tune"]

# ---- content (pure functions returning lines; unit-testable without a terminal) ----
def results_lines():
    s = report.summary()
    if not s: return ["No results yet. Press 2, then r to start a run (or `llmeval import-legacy`)."]
    out = [f'{"Harness":9} {"Model":26} {"Pass":>7} {"Wall s":>7} {"Reqs":>5} {"PeakTok":>8} {"Loop":>4} {"T/O":>4}', "-" * 78]
    for x in s:
        out.append(f'{x["harness"]:9} {x["model"][:26]:26} {x["passed"]:>3}/{x["n"]:<3} {x["wall"] or 0:>7.1f} {x["reqs"] or 0:>5.0f} {x["peak"] or 0:>8} {x["loops"]:>4} {x["timeouts"]:>4}')
    pt = report.per_task(); tasks = sorted({t for d in pt.values() for t in d})
    out += ["", "Pass count per task", "-" * 78, " " * 37 + " ".join(t[:2] for t in tasks)]
    for (h, m), d in sorted(pt.items()):
        out.append(f"{h:9} {m[:26]:26} " + " ".join(f"{sum(d[t])}/{len(d[t])}" if t in d else " - " for t in tasks))
    return out

def fit_lines():
    fit = store.rows(fitmod.FIT)
    if not fit: return ["No context-fit data. Press m to measure every configured model (16k..64k)."]
    latest = {(r["model"], r["num_ctx"]): r for r in fit}; ctxs = sorted({k[1] for k in latest})
    out = [f'{"Model":26} ' + " ".join(f"{c // 1024:>3}k(GB/GPU%/tok/s)".rjust(19) for c in ctxs), "-" * 100]
    for m in sorted({k[0] for k in latest}):
        cells = []
        for c in ctxs:
            r = latest.get((m, c))
            cells.append("error" if not r or "error" in r or not r.get("fit") else f'{r["fit"]["size_gb"]}/{r["fit"]["gpu_pct"]}%/{r["tok_s"]}')
        out.append(f"{m[:26]:26} " + " ".join(c.rjust(19) for c in cells))
    return out

def setup_lines(cfg):
    out = ["Harnesses", "-" * 60]
    out += [f"  {h.name:9} {'installed' if h.available() else 'MISSING  '} {'(experimental) ' if h.experimental else ''}{h.note[:40]}" for h in REGISTRY.values()]
    out += ["", "Tasks", "-" * 60] + [f"  {t['id']:22} {t['title']}" for t in T.load()]
    out += ["", "Models (from config)", "-" * 60]
    try: have = ollama.installed()
    except Exception: have = {}
    for m in cfg["models"]:
        tag = m["tag"] if ":" in m["tag"] else m["tag"] + ":latest"
        out.append(f"  {m['tag']:26} {'installed' if tag in have else 'not built  '} ctx {m['num_ctx']:>6} {('base ' + m['base']) if m.get('base') else ''}")
    out += ["", f"ollama {'reachable' if have else 'NOT reachable'}; runs.jsonl has {len(store.rows())} trials"]
    return out

# ---- worker ----
class Worker:
    def __init__(self): self.log = collections.deque(maxlen=500); self.busy = ""; self.progress = (0, 0); self.stop = False; self.thread = None
    def start(self, name, fn):
        if self.thread and self.thread.is_alive(): self.log.append("busy: wait for the current job"); return
        self.stop = False; self.busy = name; self.progress = (0, 0)
        def go():
            try: fn(self)
            except SystemExit as e: self.log.append(f"stopped: {e}")
            except Exception as e: self.log.append(f"ERROR: {e}")
            self.busy = ""
        self.thread = threading.Thread(target=go, daemon=True); self.thread.start()

def job_run(cfg):
    def fn(w):
        for ev in runner.run_matrix(cfg):
            if ev["type"] == "plan": w.progress = (0, ev["total"]); w.log.append(f'{ev["total"]} trials ({ev["skipped"]} already done)')
            elif ev["type"] == "model": w.log.append(f'== {ev["model"]}')
            elif ev["type"] == "trial":
                w.progress = (ev["n"], w.progress[1])
                w.log.append(f'{"PASS" if ev["done"] else "FAIL"} {ev["harness"]:9} {ev["task"]} #{ev["rep"]} {ev["wall_s"]}s' + (" LOOP" if ev.get("loop") else ""))
            if w.stop: w.log.append("stop requested: finishing after this trial"); return
    return fn

def job_fit(cfg):
    def fn(w):
        for r in fitmod.measure([m["tag"] for m in cfg["models"]]): w.log.append(f'{r["model"]} {r["num_ctx"]}: ' + (r["error"] if "error" in r else f'{r["fit"]} {r["tok_s"]} tok/s'))
    return fn

def job_setup(cfg, what):
    def fn(w):
        if what == "selfcheck":
            for t in T.load(): w.log.append(("OK   " if T.selfcheck(t) else "BROKEN ") + t["id"])
        else: config.prepare(cfg, w.log.append)
    return fn

def job_tune(cfg):
    def fn(w):
        from . import tune
        m = next((m for m in cfg["models"] if m.get("base")), None)
        if not m: raise SystemExit("no model with a `base` in the config")
        for ev in tune.tune(m, cfg.get("tune", {}).get("holdout", ["05-bug-across-files"]), cfg["run"]["harnesses"][0], 2, cfg["run"]["timeout"]):
            w.log.append(str({k: v for k, v in ev.items() if k != "type"}) if ev["type"] != "done" else f'DONE confirmed={ev["confirmed"]} baseline={ev["baseline"]} winner={ev["winner"]}')
    return fn

# ---- curses loop ----
def main(config_path):
    cfg = config.load(config_path); w = Worker(); curses.wrapper(lambda scr: loop(scr, cfg, w))

def loop(scr, cfg, w):
    curses.curs_set(0); scr.timeout(400); tab, off = 0, 0
    while True:
        h, wd = scr.getmaxyx(); scr.erase()
        bar = " ".join(f"[{i + 1}]{n}" if i == tab else f" {i + 1} {n}" for i, n in enumerate(TABS))
        scr.addnstr(0, 0, f" llmeval  {bar}", wd - 1, curses.A_REVERSE)
        if tab == 0: lines = results_lines()
        elif tab == 2: lines = fit_lines()
        elif tab == 3: lines = setup_lines(cfg) + ["", "s selfcheck   p prepare (pull + create tuned tags)"] + list(w.log)[-8:]
        else:
            done, total = w.progress
            head = [("RUNNING: " + w.busy + (f"  {done}/{total}" if total else "")) if w.busy else ("idle - " + ("r start run, x stop" if tab == 1 else "t start tune")), ""]
            lines = head + list(w.log)
        off = max(0, min(off, max(0, len(lines) - (h - 2))))
        for i, l in enumerate(lines[off:off + h - 2]): scr.addnstr(1 + i, 1, l, wd - 2)
        scr.addnstr(h - 1, 0, f' q quit  1-5 tabs  arrows scroll  {"job: " + w.busy if w.busy else ""}', wd - 1, curses.A_DIM)
        scr.refresh()
        k = scr.getch()
        if k in (ord("q"), 27): return
        elif k in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")): tab, off = k - ord("1"), 0
        elif k == curses.KEY_DOWN: off += 1
        elif k == curses.KEY_UP: off -= 1
        elif k == curses.KEY_NPAGE: off += h - 3
        elif k == curses.KEY_PPAGE: off -= h - 3
        elif k == ord("x"): w.stop = True
        elif k == ord("r") and tab == 1: w.start("run", job_run(cfg))
        elif k == ord("m") and tab == 2: w.start("context fit", job_fit(cfg))
        elif k == ord("s") and tab == 3: w.start("selfcheck", job_setup(cfg, "selfcheck"))
        elif k == ord("p") and tab == 3: w.start("prepare", job_setup(cfg, "prepare"))
        elif k == ord("t") and tab == 4: w.start("tune", job_tune(cfg))
