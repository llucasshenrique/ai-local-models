"""Terminal UI (stdlib curses). Tabs: 1 Results  2 Run  3 Context fit  4 Models  5 Setup  6 Tune.
Mouse: click tabs, buttons and model rows (click a selected row again to add it); wheel scrolls.
Keys: 1-6 or left/right switch tab | up/down/PgUp/PgDn scroll | r start run (tab 2) | m measure fit (tab 3)
      tab 4: up/down select, Enter add the selected installed model, n new model, f new family, R refresh
      tab 5: s selfcheck, p prepare | tab 6: t tune the first model with a `base`, a ask the advisor for ideas | x stop after the current trial | q quit.
Long jobs run in one worker thread; the GPU lock keeps them serial and the UI never loads a model itself."""
import collections, curses, threading, time
from . import advisor, config, fit as fitmod, models as modelmgmt, ollama, report, runner, store, tasks as T
from .harnesses import REGISTRY

TABS = ["Results", "Run", "Context fit", "Models", "Setup", "Tune"]

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
    for name, rows, pick, why in report.family_table():
        out += ["", f"Family {name}: recommended {pick['tag'] if pick else '-'} ({why})", "-" * 78]
        out += [f'{"*" if pick and r["tag"] == pick["tag"] else " "} {r["tag"][:34]:34} {str(r["size"] or "-"):>5} {str(r["quant"] or "-"):>7} {100 * r["rate"]:>4.0f}% n={r["n"]:<3} {r["tok_s"] or "-"} tok/s {r["size_gb"] or "-"} GB {r["gpu_pct"] or "-"}% GPU' for r in sorted(rows, key=lambda r: (r["size"] or "", r["quant"] or ""))]
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

def models_lines(rows, sel):
    if rows is None: return ["ollama is not reachable, so installed models cannot be listed."]
    out = ["Installed ollama models  ([x] = in the config, tested by `run`)", f'{"":3}{"":4}{"Model":38} {"GB":>5}', "-" * 60]
    out += [f'{">" if i == sel else " "}  [{"x" if inc else " "}] {tag[:36]:36} {gb:>5}' for i, (tag, gb, inc) in enumerate(rows)]
    return out + ["", "Enter: add selected   n: new model (pull or build from a base)   f: new family (sizes x quantizations)   R: refresh"]

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

def job_add(path, cfg, tags, base=None, ctx=None, pull=False):
    def fn(w):
        modelmgmt.add(path, tags, base, ctx, pull, w.log.append)
        cfg.clear(); cfg.update(config.load(path))
    return fn

def job_advise(cfg):
    def fn(w):
        out = advisor.advise(cfg, log=w.log.append)
        a = out["advice"]; w.log.append("ANALYSIS: " + a["analysis"])
        w.log.extend(f'TRY {e["model"]} {e["changes"]}: {e["why"]}' for e in a["experiments"])
        w.log.extend("IDEA: " + i for i in a["ideas"]); w.log.extend("DROPPED: " + d for d in out["dropped"])
    return fn

def job_family(path, cfg, name, sizes, quants, ctx):
    def fn(w):
        if modelmgmt.add_family(path, name, sizes, quants, ctx, log=w.log.append): cfg.clear(); cfg.update(config.load(path))
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
MODELS_HEADER = 3                      # lines above the first model row in models_lines()
BUTTONS = {                            # tab index -> [(label, key it triggers)]
    1: [("Run (r)", "r"), ("Stop (x)", "x")], 2: [("Measure fit (m)", "m")],
    3: [("Add selected (Enter)", "\n"), ("New model (n)", "n"), ("New family (f)", "f"), ("Refresh (R)", "R")],
    4: [("Selfcheck (s)", "s"), ("Prepare (p)", "p")], 5: [("Tune (t)", "t"), ("Advise (a)", "a"), ("Stop (x)", "x")]}

def tab_spans():
    """[(x0, x1)] of each tab label in the title bar; shared by drawing and click handling."""
    spans, x = [], len(" llmeval  ")
    for i, n in enumerate(TABS):
        w = len(f" {i + 1} {n}"); spans.append((x, x + w)); x += w + 1
    return spans

def main(config_path):
    cfg = config.load(config_path); w = Worker(); modelmgmt.bind(config_path); curses.wrapper(lambda scr: loop(scr, cfg, w, config_path))

def prompt(scr, label, default=""):
    """One-line text input on the bottom row; Esc cancels (returns None)."""
    h, wd = scr.getmaxyx(); buf = default; scr.timeout(-1); curses.curs_set(1)
    try:
        while True:
            scr.move(h - 1, 0); scr.clrtoeol(); scr.addnstr(h - 1, 0, f"{label}: {buf}", wd - 1, curses.A_BOLD); scr.refresh()
            k = scr.getch()
            if k in (10, 13, curses.KEY_ENTER): return buf.strip()
            if k == 27: return None
            if k in (curses.KEY_BACKSPACE, 127, 8): buf = buf[:-1]
            elif 32 <= k < 127: buf += chr(k)
    finally: scr.timeout(400); curses.curs_set(0)

def loop(scr, cfg, w, path):
    curses.curs_set(0); scr.timeout(400); scr.keypad(True)
    curses.mouseinterval(0); curses.mousemask(curses.ALL_MOUSE_EVENTS)   # press/release reported separately; we act on presses
    up, down = getattr(curses, "BUTTON4_PRESSED", 0), getattr(curses, "BUTTON5_PRESSED", 0)
    st = {"tab": 0, "off": 0, "sel": 0, "rows": None, "rows_at": 0}; hit = []      # hit = clickable buttons of the last frame

    def act(k):
        """One handler for keyboard and mouse: clicks on buttons/tabs are translated into the same keys."""
        tab, h = st["tab"], scr.getmaxyx()[0]; rows = st["rows"]
        if k == ord("q"): return True            # Esc must not quit: a stray escape sequence would kill the UI
        if k in (ord(c) for c in "123456"): st.update(tab=k - ord("1"), off=0, rows_at=0)
        elif k == curses.KEY_DOWN and tab == 3 and rows: st["sel"] = min(st["sel"] + 1, len(rows) - 1)
        elif k == curses.KEY_UP and tab == 3 and rows: st["sel"] = max(st["sel"] - 1, 0)
        elif k == curses.KEY_DOWN: st["off"] += 1
        elif k == curses.KEY_UP: st["off"] -= 1
        elif k == curses.KEY_NPAGE: st["off"] += h - 3
        elif k == curses.KEY_PPAGE: st["off"] -= h - 3
        elif k == curses.KEY_HOME: st["off"] = 0
        elif k == curses.KEY_LEFT: st.update(tab=(tab - 1) % len(TABS), off=0, rows_at=0)
        elif k == curses.KEY_RIGHT: st.update(tab=(tab + 1) % len(TABS), off=0, rows_at=0)
        elif k == ord("x"): w.stop = True
        elif k == ord("r") and tab == 1: w.start("run", job_run(cfg))
        elif k == ord("m") and tab == 2: w.start("context fit", job_fit(cfg))
        elif k in (10, 13, curses.KEY_ENTER) and tab == 3 and rows:
            w.start("add model", job_add(path, cfg, [rows[st["sel"]][0]])); st["rows_at"] = 0
        elif k == ord("R") and tab == 3: st["rows_at"] = 0
        elif k == ord("n") and tab == 3:
            tag = prompt(scr, "new model tag (e.g. qwen3:8b or my-agent:9b)")
            if tag:
                base = prompt(scr, "base model to build it from (empty = pull/use the tag as-is)")
                ctx = prompt(scr, "num_ctx (empty = default)") if base else None
                if base is not None:
                    w.start("add model", job_add(path, cfg, [tag], base or None, int(ctx) if ctx and ctx.isdigit() else None, pull=not base)); st["rows_at"] = 0
        elif k == ord("f") and tab == 3:
            name = prompt(scr, "family name (ollama library name, e.g. granite4.1)")
            sizes = prompt(scr, "sizes, comma separated (e.g. 3b,8b)") if name else None
            quants = prompt(scr, "quantizations, comma separated (e.g. q4_K_M,q6_K,q8_0)") if sizes else None
            ctx = prompt(scr, "num_ctx (empty = default)") if quants else None
            if quants:
                w.start("add family", job_family(path, cfg, name, [x.strip() for x in sizes.split(",") if x.strip()], [x.strip() for x in quants.split(",") if x.strip()], int(ctx) if ctx and ctx.isdigit() else None))
        elif k == ord("a") and tab == 5: w.start("advise", job_advise(cfg))
        elif k == ord("s") and tab == 4: w.start("selfcheck", job_setup(cfg, "selfcheck"))
        elif k == ord("p") and tab == 4: w.start("prepare", job_setup(cfg, "prepare"))
        elif k == ord("t") and tab == 5: w.start("tune", job_tune(cfg))
        return False

    def click(x, y, bstate):
        tab = st["tab"]
        if bstate & up: return act(curses.KEY_UP)
        if bstate & down: return act(curses.KEY_DOWN)
        if not bstate & curses.BUTTON1_PRESSED: return False
        if y == 0:
            for i, (x0, x1) in enumerate(tab_spans()):
                if x0 <= x < x1: return act(ord("1") + i)
        for (by, x0, x1, key) in hit:
            if by == y and x0 <= x < x1: return act(10 if key == "\n" else ord(key))
        if tab == 3 and st["rows"]:
            i = y - 1 + st["off"] - MODELS_HEADER
            if 0 <= i < len(st["rows"]):
                again = i == st["sel"]; st["sel"] = i
                if again: return act(10)                        # clicking the already-selected row adds it
        return False

    while True:
        h, wd = scr.getmaxyx(); scr.erase(); tab = st["tab"]
        scr.addnstr(0, 0, " " * (wd - 1), wd - 1, curses.A_REVERSE); scr.addnstr(0, 1, "llmeval", 7, curses.A_REVERSE | curses.A_BOLD)
        for i, (x0, x1) in enumerate(tab_spans()):
            scr.addnstr(0, x0, (f"[{i + 1}]{TABS[i]}" if i == tab else f" {i + 1} {TABS[i]}").ljust(x1 - x0), max(0, min(x1 - x0, wd - x0 - 1)),
                        curses.A_REVERSE | (curses.A_BOLD if i == tab else 0))
        if tab == 3 and time.time() - st["rows_at"] > 5 and not w.busy:
            try: st["rows"] = modelmgmt.installed_info()
            except Exception: st["rows"] = None
            st["rows_at"] = time.time(); st["sel"] = min(st["sel"], max(0, len(st["rows"] or []) - 1))
        if tab == 0: lines = results_lines()
        elif tab == 2: lines = fit_lines()
        elif tab == 3: lines = models_lines(st["rows"], st["sel"]) + [""] + list(w.log)[-6:]
        elif tab == 4: lines = setup_lines(cfg) + list(w.log)[-8:]
        else:
            done, total = w.progress
            lines = [("RUNNING: " + w.busy + (f"  {done}/{total}" if total else "")) if w.busy else "idle", ""] + list(w.log)
        st["off"] = max(0, min(st["off"], max(0, len(lines) - (h - 3))))
        for i, l in enumerate(lines[st["off"]:st["off"] + h - 3]): scr.addnstr(1 + i, 1, l, wd - 2)
        hit.clear(); x = 1
        for label, key in BUTTONS.get(tab, []):        # clickable buttons on the row above the footer
            txt = f" {label} "
            if x + len(txt) < wd: scr.addnstr(h - 2, x, txt, len(txt), curses.A_REVERSE); hit.append((h - 2, x, x + len(txt), key)); x += len(txt) + 1
        scr.addnstr(h - 1, 0, f' q quit  1-6/left/right tabs  up/down/PgUp/PgDn/wheel scroll  click tabs, buttons, rows  {"job: " + w.busy if w.busy else ""}', wd - 1, curses.A_DIM)
        scr.refresh()
        k = scr.getch()
        if k == curses.KEY_MOUSE:
            try: _, mx, my, _, bstate = curses.getmouse()
            except curses.error: continue
            if click(mx, my, bstate): return
        elif k != -1 and act(k): return
