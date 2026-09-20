"""Terminal UI (stdlib curses). Tabs: 1 Results  2 Run  3 Context fit  4 Models  5 Setup  6 Tune  7 Loop.
Mouse: click tabs, buttons and model rows (click a selected row again to add it); wheel scrolls.
Keys: 1-7 or left/right switch tab | up/down/PgUp/PgDn scroll
      tab 1: P cycle Pareto policy (balanced / max_quality / fastest / pareto)
      tab 2: r start run | w context quality sweep across hardware-safe contexts | x stop
      tab 3: m measure hardware fit | o optimize hardware feasibility envelope
      tab 4: up/down select, Enter add selected installed model, n new model, f new family, d discover, y apply, c clean losers, R refresh
      tab 5: s selfcheck, p prepare, i verify benchmark integrity
      tab 6: t tune lead model (staged confirmation), F tune family, a advisor hypotheses, C compare variants | x stop
      tab 7: L run full workflow (discover -> fit -> benchmark -> tune -> pareto) | D discover & fit | x stop | q quit.
Long jobs run in one worker thread; the GPU lock keeps them serial and the UI never loads a model itself.
"""
import collections, curses, json, threading, time
from . import advisor, config, discover as discovermod, families as fam, fit as fitmod, guardrails, prune as prunemod, models as modelmgmt, ollama, pareto, provenance, report, runner, store, tasks as T
from .harnesses import REGISTRY

TABS = ["Results", "Run", "Context fit", "Models", "Setup", "Tune", "Loop"]
POLICIES = ["balanced", "max_quality", "fastest", "pareto"]
POLICY_TYPES = [
    ("balanced", "Tradeoff between coding pass rate, latency, and VRAM footprint"),
    ("max_quality", "Strict prioritization of highest benchmark pass rate & accuracy"),
    ("fastest", "Lowest task completion latency & highest tok/s generation"),
    ("pareto", "All non-dominated models on the multi-objective Pareto frontier")
]

# ---- content (pure functions returning lines; unit-testable without a terminal) ----
def results_lines(policy="balanced"):
    s = report.summary()
    if not s: return ["No results yet. Press 2, then r to start a run (or `llmeval import-legacy`)."]
    out = [f'{"Harness":9} {"Model":24} {"Pass":>7} {"Rate":>5} {"95% CI":>13} {"Wall s":>7} {"Reqs":>5} {"PeakTok":>8} {"Loop":>4} {"T/O":>4}', "-" * 88]
    for x in s:
        ci_str = f'[{int(100*x["ci_low"])}%,{int(100*x["ci_high"])}%]'
        out.append(f'{x["harness"]:9} {x["model"][:24]:24} {x["passed"]:>3}/{x["n"]:<3} {100*x["rate"]:>4.0f}% {ci_str:>13} {x["wall"] or 0:>7.1f} {x["reqs"] or 0:>5.0f} {x["peak"] or 0:>8} {x["loops"]:>4} {x["timeouts"]:>4}')
    
    pt = report.per_task(); tasks = sorted({t for d in pt.values() for t in d})
    out += ["", "Pass count per task", "-" * 78, " " * 35 + " ".join(t[:2] for t in tasks)]
    for (h, m), d in sorted(pt.items()):
        out.append(f"{h:9} {m[:24]:24} " + " ".join(f"{sum(d[t])}/{len(d[t])}" if t in d else " - " for t in tasks))
    
    # Context Quality Curve (benchmarking context size explicitly)
    curve = report.context_curve()
    if curve and len({c["num_ctx"] for c in curve}) > 1:
        out += ["", "Context Quality Curve (pass rate & latency vs num_ctx)", "-" * 78]
        out.append(f'{"Model":24} {"Context":>8} {"Pass":>6} {"Rate":>5} {"95% CI":>13} {"Median Wall":>12} {"Failures":>8}')
        for c in curve:
            ci = f'[{int(100*c["ci_low"])}%,{int(100*c["ci_high"])}%]'
            out.append(f'{c["model"][:24]:24} {c["num_ctx"]//1024:>7}k {c["passed"]:>2}/{c["n"]:<3} {100*c["rate"]:>4.0f}% {ci:>13} {c["median_wall"]:>11.1f}s {c["failures"]:>8}')

    # Pareto Recommendation Policies
    out += ["", "Pareto Recommendation Policies (press P to cycle active policy):", "-" * 88]
    for pol_name, desc in POLICY_TYPES:
        mark = "[*]" if pol_name == policy else "[ ]"
        out.append(f"  {mark} {pol_name:12}: {desc}")

    # Multi-Objective Pareto Frontier & Family Recommendations
    for name, rows, pick, why in report.family_table(policy=policy):
        out += ["", f"Family {name}: recommended {pick['tag'] if pick else '-'} ({why})", "-" * 88]
        for r in sorted(rows, key=lambda r: (fam.params_b(r["size"]) or 0, r["quant"] or "")):
            is_pick = bool(pick and r["tag"] == pick["tag"])
            ci_str = f'[{int(100*r["ci_low"])}%,{int(100*r["ci_high"])}%]'
            out.append(f'{"*" if is_pick else " "} {r["tag"][:28]:28} {str(r["size"] or "-"):>5} {str(r["quant"] or "-"):>7} {100 * r["rate"]:>4.0f}% {ci_str:>13} n={r["n"]:<3} {r["tok_s"] or "-"} tok/s {r["size_gb"] or "-"} GB {r["gpu_pct"] or "-"}% GPU')
    return out

def run_lines(cfg, w):
    done, total = w.progress
    if w.busy:
        return [f"RUNNING: {w.busy}" + (f"  {done}/{total}" if total else ""), ""] + list(w.log)
    
    out = [
        "Evaluation Matrix & Context Sweep",
        "-" * 60,
        "Staged Evaluation Strategy:",
        "  Stage A - Screening: exploration on train tasks (reps 1-2)",
        "  Stage B - Shortlist: complete task suite across harnesses",
        "  Stage C - Context Sweep: benchmark quality across hardware-safe context sizes",
        "  Stage D - Confirmation: high repetitions before promotion",
        "",
        f"Configured Repetitions: {cfg.get('run', {}).get('reps', 3)} | Timeout: {cfg.get('run', {}).get('timeout', 180)}s",
        f"Harnesses: {', '.join(cfg.get('run', {}).get('harnesses', []))}",
        f"Models: {len(cfg.get('models', []))} configured",
        "",
        "Controls: Press r to run standard matrix | Press w for context quality sweep | Press x to stop",
        "",
        "Recent Execution Log:",
        "-" * 60
    ]
    out += list(w.log)[-10:] or ["(no trials executed this session)"]
    return out

def fit_lines():
    fit = store.rows(fitmod.FIT)
    out = [
        "Context Fit: Hardware Feasibility Envelope",
        "NOTE: Physical feasibility != coding quality. Measures VRAM & zero swap thrashing.",
        "-" * 90
    ]
    if not fit:
        return out + ["No context-fit data yet. Press m to probe (16k..64k) or o to optimize safe boundary."]
    
    latest = {(r["model"], r["num_ctx"]): r for r in fit}; ctxs = sorted({k[1] for k in latest})
    out.append(f'{"Model":26} ' + " ".join(f"{c // 1024:>3}k(GB/GPU%/tok/s)".rjust(19) for c in ctxs))
    out.append("-" * 90)
    for m in sorted({k[0] for k in latest}):
        cells = []
        for c in ctxs:
            r = latest.get((m, c))
            cells.append("error" if not r or "error" in r or not r.get("fit") else f'{r["fit"]["size_gb"]}/{r["fit"]["gpu_pct"]}%/{r["tok_s"]}')
        out.append(f"{m[:26]:26} " + " ".join(c.rjust(19) for c in cells))
    
    # Show optimized feasibility envelopes if any
    opt_models = [r for r in fit if r.get("hardware_safe_ctx") or r.get("recommended_ctx")]
    if opt_models:
        out += ["", "Empirical Feasibility Boundaries (from fit --optimize):", "-" * 60]
        for r in opt_models[-4:]:
            safe_c = r.get("hardware_safe_ctx") or r.get("recommended_ctx")
            max_c = r.get("hardware_max_ctx") or r.get("num_ctx")
            out.append(f'  {r["model"][:26]:26} safe: {safe_c//1024}k | max physical: {max_c//1024}k | boundary: {r.get("boundary_ctx", 0)//1024}k')

    out += ["", "Press m to probe static context list | Press o to empirically find max safe context"]
    return out

def models_lines(rows, sel):
    if rows is None: return ["ollama is not reachable, so installed models cannot be listed."]
    out = ["Installed ollama models  ([x] = in the config, tested by `run`)", f'{"":3}{"":4}{"Model":38} {"GB":>5}', "-" * 60]
    out += [f'{">" if i == sel else " "}  [{"x" if inc else " "}] {tag[:36]:36} {gb:>5}' for i, (tag, gb, inc) in enumerate(rows)]
    return out + ["", "Enter: add selected   n: new model (pull or build from a base)   f: new family (sizes x quantizations)   R: refresh"]

def setup_lines(cfg):
    out = ["Setup & Experimental Integrity", "-" * 60]
    # Benchmark Integrity status
    try:
        snap = guardrails.snapshot_benchmark()
        ok, violations = guardrails.verify_benchmark_integrity(snap)
        out.append(f"  Benchmark Integrity: {'VERIFIED OK (Ruler is frozen)' if ok else 'WARNING: TAMPERED'}")
        if not ok:
            out += [f"    ! {v}" for v in violations[:3]]
    except Exception:
        out.append("  Benchmark Integrity: check skipped")

    out.append("  Workspace Isolation: disposable per-trial workspace, protected files 0444 (read-only)")
    
    # Task splits
    split = cfg.get("tasks", {}).get("split", {})
    train_ids = set(split.get("train") or [])
    val_ids = set(split.get("val") or ["05-bug-across-files"])
    test_ids = set(split.get("test") or [])

    out += ["", "Tasks (Partitioning: Train / Validation / Final Test)", "-" * 60]
    for t in T.load():
        tid = t["id"]
        sp_label = "[test]" if tid in test_ids else "[val] " if tid in val_ids else "[train]"
        out.append(f"  {sp_label} {t['id']:22} {t['title'][:32]} (v:{t.get('version', '')[:6]})")

    out += ["", "Models (Effective Configuration)", "-" * 60]
    try: have = ollama.installed()
    except Exception: have = {}
    for m in cfg["models"]:
        tag = m["tag"] if ":" in m["tag"] else m["tag"] + ":latest"
        eff = m.get("effective_params") or config.effective_params(cfg, m)
        digest = m.get("config_digest") or provenance.compute_config_digest(eff)
        out.append(f"  {m['tag']:24} {'installed' if tag in have else 'not built  '} ctx {eff['num_ctx']:>6} digest {digest[:8]} {('base ' + m['base']) if m.get('base') else ''}")

    out += ["", "Harnesses", "-" * 60]
    out += [f"  {h.name:9} {'installed' if h.available() else 'MISSING  '} {'(experimental) ' if h.experimental else ''}{h.note[:40]}" for h in REGISTRY.values()]
    out += ["", f"ollama {'reachable' if have else 'NOT reachable'}; runs.jsonl has {len(store.rows())} trials"]
    return out

def tune_lines(w):
    done, total = w.progress
    if w.busy:
        return [f"RUNNING: {w.busy}" + (f"  {done}/{total}" if total else ""), ""] + list(w.log)
    
    out = [
        "Recursive Self-Improvement (RSI) Protocol",
        "-" * 60,
        "Measurement-Driven Workflow:",
        "  1. Hypothesis: LLM advisor proposes parameter/variant hypotheses (never auto-applied)",
        "  2. Exploration: evaluate candidates on train tasks (low repetitions)",
        "  3. Confirmation: verified with high repetitions (min 5 reps) and statistical tests",
        "  4. Held-out Validation: verified on independent validation tasks",
        "  5. Promotion: proposal written to results/tune-best.toml only if improvement supported",
        "",
        "Invariants Enforced:",
        "  - Frozen ruler: benchmark infrastructure is strictly immutable",
        "  - Statistical evidence required: noisy low-repetition wins are rejected",
        "  - Final test isolation: test tasks never participate in tuner decisions",
        "",
        "Controls: t: tune lead model | F: tune family | a: advisor hypotheses | C: compare variants",
        "",
        "Recent Tune & Advice History:",
        "-" * 60
    ]
    out += list(w.log)[-10:] or ["(no tuning runs executed this session)"]
    return out

def loop_lines(w, policy="balanced"):
    done, total = w.progress
    if w.busy:
        return [f"RUNNING WORKFLOW: {w.busy}" + (f"  {done}/{total}" if total else ""), ""] + list(w.log)
    
    out = [
        "Recursive Self-Improvement (RSI) End-to-End Workflow",
        "-" * 60,
        "Unified Pipeline: Discovery -> Hardware Fit -> Auto-Context -> Tune -> Pareto",
        "",
        "Autonomous Workflow Stages:",
        "  [1/5] Discover: Search Ollama library & HuggingFace for family variants",
        "  [2/5] Hardware Envelope: Probe VRAM feasibility (zero swap thrashing limit)",
        "  [3/5] Auto-Context Discovery: Sweep candidate context sizes & automatically",
        "        select best context size based on quality & latency curve",
        "  [4/5] Staged Tuning: Statistical confirmation (Wilson 95% CI, Fisher exact,",
        "        min 5 reps) on train/val splits; rejects noisy wins",
        "  [5/5] Pareto Decision: Multi-objective selection according to active policy",
        "",
        "Pareto Recommendation Policies (press P to cycle active policy):",
        "-" * 60
    ]
    for pol_name, desc in POLICY_TYPES:
        mark = "[*]" if pol_name == policy else "[ ]"
        out.append(f"  {mark} {pol_name:12}: {desc}")

    out += [
        "",
        "Controls:",
        "  L: Run Full Workflow (auto-discovers best context size)",
        "  D: Discover & Probe Hardware Fit only",
        "  P: Cycle Pareto Policy (balanced / max_quality / fastest / pareto)",
        "  x: Stop current running workflow",
        "",
        "Recent Workflow Log:",
        "-" * 60
    ]
    out += list(w.log)[-10:] or ["(no workflow executed yet - press L to start)"]
    return out

# ---- worker ----
class Worker:
    def __init__(self):
        self.plan = None
        self.log = collections.deque(maxlen=500)
        self.busy = ""
        self.progress = (0, 0)
        self.stop = False
        self.thread = None

    def start(self, name, fn):
        if self.thread and self.thread.is_alive():
            self.log.append("busy: wait for the current job")
            return
        self.stop = False
        self.busy = name
        self.progress = (0, 0)
        def go():
            try: fn(self)
            except SystemExit as e: self.log.append(f"stopped: {e}")
            except Exception as e: self.log.append(f"ERROR: {e}")
            self.busy = ""
        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()

def job_run(cfg, ctx_sweep=None):
    def fn(w):
        sweep_desc = f" with ctx_sweep={ctx_sweep}" if ctx_sweep else ""
        for ev in runner.run_matrix(cfg, ctx_sweep=ctx_sweep):
            if ev["type"] == "plan":
                w.progress = (0, ev["total"])
                w.log.append(f'{ev["total"]} trials to run ({ev["skipped"]} already done){sweep_desc}')
            elif ev["type"] == "model":
                w.log.append(f'== {ev["model"]}')
            elif ev["type"] == "trial":
                w.progress = (ev["n"], w.progress[1])
                ctx_info = f" ctx {ev.get('num_ctx')}" if ev.get("num_ctx") else ""
                w.log.append(f'{"PASS" if ev["done"] else "FAIL"} {ev["harness"]:9} {ev["task"]}{ctx_info} #{ev["rep"]} {ev["wall_s"]}s' + (" LOOP" if ev.get("loop") else "") + (" TIMEOUT" if ev["timeout"] else ""))
            if w.stop:
                w.log.append("stop requested: finishing after this trial")
                return
    return fn

def job_fit(cfg):
    def fn(w):
        for r in fitmod.measure([m["tag"] for m in cfg["models"]]):
            w.log.append(f'{r["model"]} {r["num_ctx"]}: ' + (r["error"] if "error" in r else f'{r["fit"]} {r["tok_s"]} tok/s'))
    return fn

def job_fit_opt(cfg, model_tag=None):
    def fn(w):
        tags = [model_tag] if model_tag else [m["tag"] for m in cfg["models"] if ollama.has(m["tag"])]
        if not tags:
            w.log.append("no installed models found to optimize")
            return
        for tag in tags:
            w.log.append(f"optimizing hardware feasibility for {tag}...")
            res = fitmod.optimize(tag, log=w.log.append)
            if "error" in res:
                w.log.append(f"feasibility check failed: {res['error']}")
            else:
                w.log.append(f"FEASIBILITY OK: {tag} safe_ctx={res['hardware_safe_ctx']} max_ctx={res['hardware_max_ctx']}")
    return fn

def job_setup(cfg, what):
    def fn(w):
        if what == "selfcheck":
            for t in T.load():
                w.log.append(("OK   " if T.selfcheck(t) else "BROKEN ") + t["id"])
        elif what == "integrity":
            snap = guardrails.snapshot_benchmark()
            ok, violations = guardrails.verify_benchmark_integrity(snap)
            if ok:
                w.log.append("INTEGRITY VERIFIED: All benchmark infrastructure and task files are immutable and untampered.")
            else:
                w.log.append("INTEGRITY VIOLATION DETECTED:")
                for v in violations: w.log.append(f"  ! {v}")
        else:
            config.prepare(cfg, w.log.append)
    return fn

def job_add(path, cfg, tags, base=None, ctx=None, pull=False):
    def fn(w):
        modelmgmt.add(path, tags, base, ctx, pull, w.log.append)
        cfg.clear(); cfg.update(config.load(path))
    return fn

def job_advise(cfg):
    def fn(w):
        out = advisor.advise(cfg, log=w.log.append)
        a = out["advice"]
        w.log.append("ADVISOR ANALYSIS: " + a["analysis"])
        w.log.extend(f'HYPOTHESIS {e["model"]} {e["changes"]}: {e["why"]}' for e in a["experiments"])
        w.log.extend("IDEA: " + i for i in a["ideas"])
        w.log.extend("DROPPED: " + d for d in out["dropped"])
    return fn

def job_compare(a, b):
    def fn(w):
        res = report.compare(a, b)
        w.log.append(f"COMPARE {a} vs {b}: Decision={res['decision']} (Verdict={res['verdict']})")
        w.log.append(f"  Reason: {res['reason']}")
        if res.get("comparability_notes"):
            for n in res["comparability_notes"]: w.log.append(f"  Note: {n}")
        w.log.append(f"  Observed: {a}={res[a]['passed']}/{res['n']} ({res[a]['median_wall']}s) vs {b}={res[b]['passed']}/{res['n']} ({res[b]['median_wall']}s)")
    return fn

def job_family(path, cfg, name, sizes, quants, ctx):
    def fn(w):
        if modelmgmt.add_family(path, name, sizes, quants, ctx, log=w.log.append):
            cfg.clear(); cfg.update(config.load(path))
    return fn

def job_discover(cfg, name, ctx):
    def fn(w):
        w.plan = discovermod.discover(cfg, name, ctx, log=w.log.append)
        w.log.extend(discovermod.format_plan(w.plan))
        w.log.append("press y (Models tab) to add this family to the config")
    return fn

def job_apply(path, cfg, plan):
    def fn(w):
        if discovermod.apply(path, plan, w.log.append):
            cfg.clear(); cfg.update(config.load(path))
            w.plan = None
    return fn

def job_prune(path, cfg, plan, drop):
    def fn(w):
        prunemod.execute(plan, path, drop, w.log.append)
        cfg.clear(); cfg.update(config.load(path))
    return fn

def job_tune_family(cfg, name):
    def fn(w):
        from . import tune
        for ev in tune.tune_family(cfg, name, cfg.get("tune", {}).get("holdout", ["05-bug-across-files"]), cfg["run"]["harnesses"][0], 2, cfg["run"]["timeout"]):
            w.log.append(str({k: v for k, v in ev.items() if k != "type"}) if ev["type"] not in ("family_done",) else f'FAMILY DONE confirmed={ev["confirmed"]} {ev.get("reason") or ""} {ev.get("proposal") or ""}')
    return fn

def job_tune(cfg):
    def fn(w):
        from . import tune
        m = next((m for m in cfg["models"] if m.get("base")), None)
        if not m: raise SystemExit("no model with a `base` in the config")
        w.log.append(f"Starting staged tuning for {m['tag']} (Exploration -> Confirmation -> Held-out Validation)...")
        for ev in tune.tune(m, cfg.get("tune", {}).get("holdout", ["05-bug-across-files"]), cfg["run"]["harnesses"][0], 2, cfg["run"]["timeout"], confirmation_reps=5, cfg_splits=cfg.get("tasks", {}).get("split", {})):
            if ev["type"] == "candidate":
                status = "CONFIRMED" if ev["accepted"] else "REJECTED"
                w.log.append(f"Candidate {ev.get('changed')}: score={ev.get('score')} -> {status} ({ev.get('note', '')})")
            elif ev["type"] == "done":
                w.log.append(f'DONE confirmed={ev["confirmed"]} baseline={ev["baseline"]} winner={ev["winner"]} proposal={ev.get("proposal")}')
            else:
                w.log.append(str({k: v for k, v in ev.items() if k != "type"}))
    return fn

def job_workflow(path, cfg, family_name=None, target_ctx=None, confirm_reps=5, policy="balanced", max_cycles=3):
    def fn(w):
        nonlocal family_name
        # Auto-detect family if omitted
        if not family_name:
            cfg_families = cfg.get("families", [])
            if cfg_families and cfg_families[0].get("name"):
                family_name = cfg_families[0]["name"]
            else:
                f_cands = [m.get("family") for m in cfg.get("models", []) if m.get("family")]
                family_name = f_cands[0] if f_cands else "granite4.1"

        w.log.append("=" * 65)
        w.log.append(f"STARTING AUTONOMOUS RSI LOOP: {family_name}")
        w.log.append(f"Auto-Context: ON | Max Cycles: {max_cycles} | Policy: {policy}")
        w.log.append("=" * 65)

        # Stage 1: Discovery & Registration
        w.log.append(f"== [1/5] AUTONOMOUS DISCOVERY: '{family_name}' ==")
        init_ctx = target_ctx or 32768
        plan = discovermod.discover(cfg, family_name, init_ctx, log=w.log.append)
        if plan and plan.get("models"):
            w.log.append(f"Discovered {len(plan['models'])} model variant(s). Applying to config...")
            discovermod.apply(path, plan, log=w.log.append)
            cfg.clear(); cfg.update(config.load(path))
            w.log.append("Config updated.")
        else:
            w.log.append(f"No new remote models added. Checking config for '{family_name}'...")
        if w.stop: w.log.append("Loop stopped."); return

        # Stage 2: Hardware Feasibility Envelope & Physical Safe Boundary
        w.log.append(f"== [2/5] HARDWARE ENVELOPE: Probing VRAM feasibility & limits ==")
        family_models = [m for m in cfg.get("models", []) if m.get("family") == family_name or family_name in m.get("tag", "")]
        installed = [m["tag"] for m in (family_models or cfg.get("models", [])) if ollama.has(m["tag"])]
        hardware_max_safe = 32768
        if installed:
            for tag in installed[:3]:
                w.log.append(f"Probing hardware limits for {tag}...")
                fit_results = fitmod.measure([tag], ctxs=[16384, 32768, 49152, 65536])
                for r in fit_results:
                    w.log.append(f"  {r['model']} {r['num_ctx'] // 1024}k: " + (r["error"] if "error" in r else f"{r['fit']} {r['tok_s']} tok/s"))
                    if not r.get("error") and (r.get("fit") or {}).get("gpu_pct", 0) >= 99.0:
                        hardware_max_safe = max(hardware_max_safe, r["num_ctx"])
        else:
            w.log.append("Family models not yet installed; using standard hardware baseline envelope (16k-32k).")
        if w.stop: w.log.append("Loop stopped."); return

        # Stage 3: Autonomous Context Quality Sweep & Discovery
        w.log.append(f"== [3/5] AUTONOMOUS CONTEXT SWEEP & QUALITY DISCOVERY ==")
        if target_ctx:
            candidates = [16384, target_ctx] if target_ctx > 16384 else [target_ctx]
        elif hardware_max_safe >= 65536:
            candidates = [16384, 32768, 65536]
        elif hardware_max_safe >= 32768:
            candidates = [16384, 32768]
        else:
            candidates = [8192, 16384]

        w.log.append(f"Sweeping context quality candidates: {[c // 1024 for c in candidates]}k...")
        has_family_in_cfg = any(m.get("family") == family_name for m in cfg.get("models", []))
        for ev in runner.run_matrix(cfg, family=family_name if has_family_in_cfg else None, ctx_sweep=candidates):
            if ev["type"] == "plan":
                w.progress = (0, ev["total"])
                w.log.append(f'Matrix: {ev["total"]} trials ({ev["skipped"]} cached)')
            elif ev["type"] == "trial":
                w.progress = (ev["n"], w.progress[1])
                ctx_info = f" ctx {ev.get('num_ctx')}" if ev.get("num_ctx") else ""
                w.log.append(f'{"PASS" if ev["done"] else "FAIL"} {ev["harness"]:9} {ev["task"]}{ctx_info} #{ev["rep"]} {ev["wall_s"]}s')
            if w.stop: w.log.append("Loop stopped."); return

        # Select & auto-apply best context size from quality curve
        curve = report.context_curve()
        fam_curve = [c for c in curve if (any(m["tag"] == c["model"] for m in family_models) or family_name in c["model"]) and c["n"] > 0]
        if not fam_curve:
            fam_curve = [c for c in curve if c["num_ctx"] in candidates and c["n"] > 0]

        if fam_curve:
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
            w.log.append(f"--> OPTIMAL CONTEXT DISCOVERED: {best_c // 1024}k (Pass Rate: {b_rate*100:.0f}%, Wall: {b_wall:.1f}s)")
            for m in family_models:
                fitmod.apply_context(m["tag"], best_c, config_path=path, log=w.log.append)
            cfg.clear(); cfg.update(config.load(path))
            w.log.append(f"Auto-applied best context {best_c} across Modelfiles and configuration.")
        else:
            best_c = candidates[-1]
            w.log.append(f"Defaulting to hardware-safe context: {best_c // 1024}k")

        if w.stop: w.log.append("Loop stopped."); return

        # Stage 4: Autonomous Recursive Self-Improvement (Multi-Cycle Optimization)
        from . import tune
        holdout = cfg.get("tune", {}).get("holdout", ["05-bug-across-files"])
        harness = cfg.get("run", {}).get("harnesses", ["pi"])[0]
        timeout = cfg.get("run", {}).get("timeout", 180)

        for cycle in range(1, max_cycles + 1):
            w.log.append(f"\n== [4/5] RECURSIVE CYCLE {cycle}/{max_cycles}: HYPOTHESIS & STAGED TUNING ==")
            improved = False

            if has_family_in_cfg:
                w.log.append(f"Autonomous tuning across {family_name} variants...")
                for ev in tune.tune_family(cfg, family_name, holdout, harness, 2, timeout, auto_apply=True, config_path=path):
                    if ev["type"] == "candidate":
                        status = "CONFIRMED" if ev.get("accepted") else "REJECTED"
                        w.log.append(f"Variant {ev.get('candidate')}: score={ev.get('score')} -> {status}")
                    elif ev["type"] == "family_done":
                        if ev.get("confirmed"):
                            improved = True
                            w.log.append(f"CYCLE {cycle} WINNER CONFIRMED & PROMOTED: {ev.get('winner')}")
                        else:
                            w.log.append(f"Cycle {cycle}: No variant improvement confirmed ({ev.get('reason')}).")
                    if w.stop: w.log.append("Loop stopped."); return
            else:
                lead = next((m for m in cfg.get("models", []) if m.get("base")), None)
                if lead:
                    w.log.append(f"Autonomous tuning lead model {lead['tag']} (confirmation_reps={confirm_reps})...")
                    for ev in tune.tune(lead, holdout, harness, 2, timeout, confirmation_reps=confirm_reps, cfg_splits=cfg.get("tasks", {}).get("split", {}), auto_apply=True, config_path=path):
                        if ev["type"] == "candidate":
                            status = "CONFIRMED" if ev.get("accepted") else "REJECTED"
                            w.log.append(f"Candidate {ev.get('changed')}: score={ev.get('score')} -> {status}")
                        elif ev["type"] == "done":
                            if ev.get("confirmed"):
                                improved = True
                                w.log.append(f"CYCLE {cycle} WINNER CONFIRMED & PROMOTED: baseline={ev.get('baseline')} -> winner={ev.get('winner')}")
                            else:
                                w.log.append(f"Cycle {cycle}: No candidate achieved statistical confirmation.")
                        if w.stop: w.log.append("Loop stopped."); return
                else:
                    w.log.append("No tunable base models in configuration; skipping parameter mutation.")
                    break

            cfg.clear(); cfg.update(config.load(path))

            if not improved:
                w.log.append("RSI CONVERGENCE: System stabilized. No further statistically supported improvements.")
                break
            else:
                w.log.append(f"CYCLE {cycle} COMPLETED: Winner promoted as new baseline for recursive optimization.")

        # Stage 5: Multi-Objective Pareto Analysis & Final Recommendation
        w.log.append(f"\n== [5/5] MULTI-OBJECTIVE PARETO DECISION ({policy.upper()} POLICY) ==")
        tables = report.family_table(policy=policy)
        matching = [t for t in tables if t[0] == family_name] or tables
        for f_name, rows, pick, why in matching:
            if pick:
                ci = f"[{int(100*pick.get('ci_low', 0))}%, {int(100*pick.get('ci_high', 0))}%]"
                w.log.append(f"WINNER ({policy}): {pick['tag']} | Rate: {100*pick.get('rate', 0):.0f}% {ci} | {pick.get('tok_s', '-')} tok/s | {pick.get('size_gb', '-')} GB")
                w.log.append(f"Rationale: {why}")
            else:
                w.log.append(f"No candidate met criteria for {f_name}.")
        w.log.append("=" * 65)
        w.log.append("AUTONOMOUS RSI LOOP FINISHED. See Results tab (1) for full metrics.")
        w.log.append("=" * 65)
    return fn

def job_workflow_discover_and_fit(path, cfg, family_name, target_ctx=32768):
    def fn(w):
        w.log.append("=" * 60)
        w.log.append(f"DISCOVERY & FIT ONLY: {family_name}")
        w.log.append("=" * 60)
        w.log.append(f"== [1/2] DISCOVERING MODELS ==")
        plan = discovermod.discover(cfg, family_name, target_ctx, log=w.log.append)
        if plan and plan.get("models"):
            w.log.append(f"Discovered {len(plan['models'])} model(s). Applying to config...")
            discovermod.apply(path, plan, log=w.log.append)
            cfg.clear(); cfg.update(config.load(path))
        if w.stop: w.log.append("Stopped."); return

        w.log.append(f"== [2/2] PROBING HARDWARE FIT ==")
        fams = [m for m in cfg.get("models", []) if m.get("family") == family_name or family_name in m.get("tag", "")]
        for m in (fams or cfg.get("models", [])):
            if ollama.has(m["tag"]):
                for r in fitmod.measure([m["tag"]]):
                    w.log.append(f"  {r['model']} {r['num_ctx']}: " + (r["error"] if "error" in r else f"{r['fit']} {r['tok_s']} tok/s"))
        w.log.append("Discovery & Fit completed.")
    return fn

# ---- curses loop ----
MODELS_HEADER = 3                      # lines above the first model row in models_lines()
BUTTONS = {                            # tab index -> [(label, key it triggers)]
    0: [("Policy (P)", "P")],
    1: [("Run (r)", "r"), ("Context sweep (w)", "w"), ("Stop (x)", "x")],
    2: [("Measure fit (m)", "m"), ("Hardware optimize (o)", "o")],
    3: [("Add selected (Enter)", "\n"), ("New model (n)", "n"), ("New family (f)", "f"), ("Discover (d)", "d"), ("Apply discovered (y)", "y"), ("Clean losers (c)", "c"), ("Refresh (R)", "R")],
    4: [("Selfcheck (s)", "s"), ("Prepare (p)", "p"), ("Integrity (i)", "i")],
    5: [("Tune (t)", "t"), ("Tune family (F)", "F"), ("Advise (a)", "a"), ("Compare (C)", "C"), ("Stop (x)", "x")],
    6: [("Run Workflow (L)", "L"), ("Discover & Fit (D)", "D"), ("Policy (P)", "P"), ("Stop (x)", "x")]
}

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
    st = {"tab": 0, "off": 0, "sel": 0, "rows": None, "rows_at": 0, "policy_idx": 0}; hit = []

    def act(k):
        """One handler for keyboard and mouse: clicks on buttons/tabs are translated into the same keys."""
        tab, h = st["tab"], scr.getmaxyx()[0]; rows = st["rows"]
        if k == ord("q"): return True
        if k in (ord(c) for c in "1234567"): st.update(tab=k - ord("1"), off=0, rows_at=0)
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
        elif (k == ord("P") or k == ord("p")) and tab in (0, 6):
            st["policy_idx"] = (st["policy_idx"] + 1) % len(POLICIES)
        elif k == ord("r") and tab == 1: w.start("run", job_run(cfg))
        elif k == ord("w") and tab == 1:
            sweep = prompt(scr, "context sweep sizes (e.g. 16384,32768,65536)", "16384,32768,65536")
            if sweep:
                ctxs = [int(x.strip()) for x in sweep.split(",") if x.strip().isdigit()]
                w.start("context sweep", job_run(cfg, ctx_sweep=ctxs))
        elif k == ord("m") and tab == 2: w.start("context fit", job_fit(cfg))
        elif k == ord("o") and tab == 2:
            model_tag = prompt(scr, "model tag to optimize feasibility (empty = all configured installed models)")
            w.start("hardware optimize", job_fit_opt(cfg, model_tag or None))
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
        elif k == ord("d") and tab == 3:
            name = prompt(scr, "family to discover on ollama/HuggingFace (e.g. granite4.1)")
            if name:
                ctx = prompt(scr, "target num_ctx (empty = 32768)")
                w.start("discover", job_discover(cfg, name, int(ctx) if ctx and ctx.isdigit() else 32768))
        elif k == ord("y") and tab == 3:
            if w.plan: w.start("apply plan", job_apply(path, cfg, w.plan))
            else: w.log.append("no discovered plan yet: press d first")
        elif k == ord("c") and tab == 3:
            p = prunemod.plan(cfg); w.log.extend(prunemod.format_plan(p))
            if p["delete"]:
                ans = prompt(scr, f"type DELETE to remove {len(p['delete'])} models, DELETE+BASES to also free unused base models (anything else cancels)")
                if ans in ("DELETE", "DELETE+BASES"):
                    if ans == "DELETE+BASES": p = prunemod.plan(cfg, include_bases=True)
                    w.start("prune", job_prune(path, cfg, p, False))
                else: w.log.append("cancelled: nothing deleted")
        elif k == ord("F") and tab == 5:
            name = prompt(scr, "family to tune (must have variants with a base in the config)")
            if name: w.start("tune family", job_tune_family(cfg, name))
        elif k == ord("a") and tab == 5: w.start("advise", job_advise(cfg))
        elif k == ord("C") and tab == 5:
            va = prompt(scr, "baseline variant label (e.g. base)")
            if va:
                vb = prompt(scr, "candidate variant label (e.g. v1)")
                if vb: w.start(f"compare {va} vs {vb}", job_compare(va, vb))
        elif k == ord("s") and tab == 4: w.start("selfcheck", job_setup(cfg, "selfcheck")); st["off"] = 999
        elif k == ord("p") and tab == 4: w.start("prepare", job_setup(cfg, "prepare")); st["off"] = 999
        elif k == ord("i") and tab == 4: w.start("integrity check", job_setup(cfg, "integrity")); st["off"] = 999
        elif k == ord("t") and tab == 5: w.start("tune", job_tune(cfg))
        elif (k == ord("L") or k == ord("l")) and tab == 6:
            default_fam = (cfg.get("families", [{}])[0].get("name") if cfg.get("families") else None) or (cfg.get("models", [{}])[0].get("tag", "").split(":")[0] if cfg.get("models") else "granite4.1")
            name = prompt(scr, f"family name for autonomous loop (Enter = '{default_fam}')", default_fam)
            if name is not None:
                chosen_fam = name.strip() or default_fam
                w.start(f"workflow {chosen_fam}", job_workflow(path, cfg, chosen_fam, target_ctx=None, confirm_reps=5, policy=POLICIES[st["policy_idx"]]))
        elif (k == ord("D") or k == ord("d")) and tab == 6:
            name = prompt(scr, "family name to discover and fit (e.g. granite4.1)")
            if name:
                w.start(f"discover & fit {name}", job_workflow_discover_and_fit(path, cfg, name))
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
                if again: return act(10)
        return False

    while True:
        h, wd = scr.getmaxyx(); scr.erase(); tab = st["tab"]
        curr_policy = POLICIES[st["policy_idx"]]

        # Title bar
        scr.addnstr(0, 0, " " * (wd - 1), wd - 1, curses.A_REVERSE)
        scr.addnstr(0, 1, "llmeval", 7, curses.A_REVERSE | curses.A_BOLD)
        for i, (x0, x1) in enumerate(tab_spans()):
            scr.addnstr(0, x0, (f"[{i + 1}]{TABS[i]}" if i == tab else f" {i + 1} {TABS[i]}").ljust(x1 - x0), max(0, min(x1 - x0, wd - x0 - 1)),
                        curses.A_REVERSE | (curses.A_BOLD if i == tab else 0))
        
        # Periodic model table refresh
        if tab == 3 and time.time() - st["rows_at"] > 5 and not w.busy:
            try: st["rows"] = modelmgmt.installed_info()
            except Exception: st["rows"] = None
            st["rows_at"] = time.time(); st["sel"] = min(st["sel"], max(0, len(st["rows"] or []) - 1))
            
        if tab == 0: lines = results_lines(policy=curr_policy)
        elif tab == 1: lines = run_lines(cfg, w)
        elif tab == 2: lines = fit_lines()
        elif tab == 3: lines = models_lines(st["rows"], st["sel"]) + [""] + list(w.log)[-6:]
        elif tab == 4: lines = setup_lines(cfg) + list(w.log)[-8:]
        elif tab == 5: lines = tune_lines(w)
        elif tab == 6: lines = loop_lines(w, policy=curr_policy)
        else: lines = list(w.log)

        if (tab == 4 or tab == 6) and w.busy: st["off"] = max(0, len(lines) - (h - 3))
        st["off"] = max(0, min(st["off"], max(0, len(lines) - (h - 3))))
        for i, l in enumerate(lines[st["off"]:st["off"] + h - 3]): scr.addnstr(1 + i, 1, l, wd - 2)
        
        # Bottom button bar
        hit.clear(); x = 1
        for label, key in BUTTONS.get(tab, []):
            txt = f" {label} "
            if x + len(txt) < wd:
                scr.addnstr(h - 2, x, txt, len(txt), curses.A_REVERSE)
                hit.append((h - 2, x, x + len(txt), key))
                x += len(txt) + 1
        
        # Footer
        footer_msg = f' q quit  1-7 tabs  arrows scroll  {"job: " + w.busy if w.busy else "idle"} | policy: {curr_policy}'
        scr.addnstr(h - 1, 0, footer_msg, wd - 1, curses.A_DIM)
        scr.refresh()
        
        k = scr.getch()
        if k == curses.KEY_MOUSE:
            try: _, mx, my, _, bstate = curses.getmouse()
            except curses.error: continue
            if click(mx, my, bstate): return
        elif k != -1 and act(k): return
