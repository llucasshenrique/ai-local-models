"""Markdown report from results/runs.jsonl (+ results/fit.jsonl). Ranking = pass rate, then median wall time."""
import collections, statistics as st
from . import store
from .fit import FIT

def _md(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"

def summary(rows=None):
    g = collections.defaultdict(list)
    for r in (rows if rows is not None else store.rows()): g[(r["harness"], r["model"])].append(r)
    out = []
    for (h, m), v in g.items():
        ok = sum(r["done"] for r in v); tam = sum(bool(r.get("tampered")) for r in v)
        med = lambda k: (st.median([r[k] for r in v if r.get(k) is not None]) if any(r.get(k) is not None for r in v) else None)
        out.append(dict(harness=h, model=m, n=len(v), passed=ok, rate=ok / len(v), wall=med("wall_s"), calls=med("tool_calls"),
                        reqs=med("llm_requests"), peak=max([r["peak_prompt_tokens"] for r in v if r.get("peak_prompt_tokens")] or [None], key=lambda x: x or 0),
                        loops=sum(bool(r.get("loop")) for r in v), timeouts=sum(bool(r.get("timeout")) for r in v), tampered=tam))
    return sorted(out, key=lambda s: (-s["rate"], s["wall"] or 1e9))

def per_task(rows=None):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in (rows if rows is not None else store.rows()): g[(r["harness"], r["model"])][r["task"]].append(int(r["done"]))
    return g

def render():
    s = summary(); out = ["# llmeval report\n"]
    if s:
        out.append("## Harness x model (ranked by pass rate, then median wall time)\n")
        out.append(_md(["Harness", "Model", "Pass", "Rate", "Median wall (s)", "Median LLM requests", "Peak prompt tokens", "Loops", "Timeouts", "Tampered"],
            [(x["harness"], x["model"], f'{x["passed"]}/{x["n"]}', f'{100*x["rate"]:.0f}%', x["wall"], x["reqs"], x["peak"], x["loops"], x["timeouts"], x["tampered"]) for x in s]))
        pt = per_task(); tasks = sorted({t for d in pt.values() for t in d})
        out.append("## Pass count per task\n")
        out.append(_md(["Harness", "Model"] + tasks, [(h, m, *[f"{sum(d[t])}/{len(d[t])}" if t in d else "-" for t in tasks]) for (h, m), d in sorted(pt.items())]))
    fit = store.rows(FIT)
    if fit:
        latest = {(r["model"], r["num_ctx"]): r for r in fit}; models = sorted({k[0] for k in latest}); ctxs = sorted({k[1] for k in latest})
        cell = lambda r: "error" if r is None or "error" in r else f'{r["fit"]["size_gb"]} GB, {r["fit"]["gpu_pct"]}% GPU, {r["tok_s"]} tok/s' if r.get("fit") else f'{r["tok_s"]} tok/s'
        out.append("## Context fit (latest measurement per model and num_ctx)\n")
        out.append(_md(["Model"] + [f"{c//1024}k" for c in ctxs], [(m, *[cell(latest.get((m, c))) for c in ctxs]) for m in models]))
    return "\n".join(out)
