"""Markdown report from results/runs.jsonl (+ results/fit.jsonl). Ranking = pass rate, then median wall time."""
import collections, statistics as st
from . import families as fam, store
from .fit import FIT

def _md(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join("-" if c is None else str(c) for c in r) + " |" for r in rows) + "\n"

def label(r): return r["model"] + (f' [{r["variant"]}]' if r.get("variant") else "")

def summary(rows=None):
    g = collections.defaultdict(list)
    for r in (rows if rows is not None else store.rows()): g[(r["harness"], label(r))].append(r)
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
    for r in (rows if rows is not None else store.rows()): g[(r["harness"], label(r))][r["task"]].append(int(r["done"]))
    return g

def family_table(tol=0.05):
    """[(family, [variant rows], pick, why)] joining pass rate (all harnesses pooled) with the latest context-fit data."""
    runs = collections.defaultdict(list)
    for r in store.rows():
        if r.get("family"): runs[(r["family"], r["model"])].append(r)
    fit = {}
    for f in store.rows(FIT): fit[f["model"], f["num_ctx"]] = f
    out = []
    for name in sorted({k[0] for k in runs}):
        rows = []
        for (fname, tag), v in sorted(runs.items()):
            if fname != name: continue
            ctx = v[0].get("num_ctx"); f = fit.get((tag, ctx)) or {}
            rows.append(dict(tag=tag, size=v[0].get("size"), quant=v[0].get("quant"), n=len(v), rate=sum(r["done"] for r in v) / len(v),
                             wall=st.median(r["wall_s"] for r in v), tok_s=f.get("tok_s"), size_gb=(f.get("fit") or {}).get("size_gb"),
                             gpu_pct=(f.get("fit") or {}).get("gpu_pct"), ctx=ctx))
        pick, why = fam.recommend(rows, tol); out.append((name, rows, pick, why))
    return out

def render():
    s = summary(); out = ["# llmeval report\n"]
    if s:
        out.append("## Harness x model (ranked by pass rate, then median wall time)\n")
        out.append(_md(["Harness", "Model", "Pass", "Rate", "Median wall (s)", "Median LLM requests", "Peak prompt tokens", "Loops", "Timeouts", "Tampered"],
            [(x["harness"], x["model"], f'{x["passed"]}/{x["n"]}', f'{100*x["rate"]:.0f}%', x["wall"], x["reqs"], x["peak"], x["loops"], x["timeouts"], x["tampered"]) for x in s]))
        pt = per_task(); tasks = sorted({t for d in pt.values() for t in d})
        out.append("## Pass count per task\n")
        out.append(_md(["Harness", "Model"] + tasks, [(h, m, *[f"{sum(d[t])}/{len(d[t])}" if t in d else "-" for t in tasks]) for (h, m), d in sorted(pt.items())]))
    for name, rows, pick, why in family_table():
        out.append(f"## Family: {name}\n")
        out.append(_md(["Variant", "Params", "Quant", "Pass", "Median wall (s)", "tok/s", "VRAM (GB)", "GPU %", "ctx"],
            [(("**" + r["tag"] + "**") if pick and r["tag"] == pick["tag"] else r["tag"], r["size"], r["quant"], f'{100 * r["rate"]:.0f}% (n={r["n"]})',
              round(r["wall"], 1), r["tok_s"], r["size_gb"], r["gpu_pct"], r["ctx"]) for r in sorted(rows, key=lambda r: (fam.params_b(r["size"]) or 0, r["quant"] or ""))]))
        out.append(f"Recommended: **{pick['tag'] if pick else '-'}** ({why})\n")
    fit = store.rows(FIT)
    if fit:
        latest = {(r["model"], r["num_ctx"]): r for r in fit}; models = sorted({k[0] for k in latest}); ctxs = sorted({k[1] for k in latest})
        cell = lambda r: "error" if r is None or "error" in r else f'{r["fit"]["size_gb"]} GB, {r["fit"]["gpu_pct"]}% GPU, {r["tok_s"]} tok/s' if r.get("fit") else f'{r["tok_s"]} tok/s'
        out.append("## Context fit (latest measurement per model and num_ctx)\n")
        out.append(_md(["Model"] + [f"{c//1024}k" for c in ctxs], [(m, *[cell(latest.get((m, c))) for c in ctxs]) for m in models]))
    return "\n".join(out)


def compare(a, b, harness=None, model=None, tasks=None, min_gain=1, max_slowdown=0.2):
    """Paired before/after comparison of two variants on the SAME tasks. Verdict 'better' needs >= min_gain more passes and
    no more than max_slowdown extra median wall time; 'worse' is the mirror image; anything else is 'same' (noise)."""
    rows = [r for r in store.rows() if (not harness or r["harness"] == harness) and (not model or r["model"] == model)]
    va = {(r["task"], r["rep"]): r for r in rows if r.get("variant") == a}
    vb = {(r["task"], r["rep"]): r for r in rows if r.get("variant") == b}
    common = [k for k in va if k in vb and (not tasks or k[0] in tasks)]
    if not common: return {"verdict": "no overlap", "n": 0}
    pa, pb = sum(va[k]["done"] for k in common), sum(vb[k]["done"] for k in common)
    wa, wb = st.median(va[k]["wall_s"] for k in common), st.median(vb[k]["wall_s"] for k in common)
    verdict = "better" if pb - pa >= min_gain and wb <= wa * (1 + max_slowdown) else "worse" if pa - pb >= min_gain else "same"
    return {"verdict": verdict, "n": len(common), a: {"passed": pa, "median_wall": wa}, b: {"passed": pb, "median_wall": wb}}
