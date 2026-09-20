"""Markdown report from results/runs.jsonl (+ results/fit.jsonl).
Produces structured evidence: Observed facts, Statistical evidence, Resource behavior, Decision.
"""
import collections, statistics as st
from . import families as fam, pareto, provenance, stats, store
from .fit import FIT

def _md(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join("-" if c is None else str(c) for c in r) + " |" for r in rows) + "\n"

def label(r):
    return r["model"] + (f' [{r["variant"]}]' if r.get("variant") else "")

def summary(rows=None):
    """Summarizes runs by (harness, model_label) with Wilson 95% confidence intervals."""
    g = collections.defaultdict(list)
    for r in (rows if rows is not None else store.rows()):
        g[(r["harness"], label(r))].append(r)
    out = []
    for (h, m), v in g.items():
        n = len(v)
        ok = sum(1 for r in v if r.get("done"))
        tam = sum(1 for r in v if bool(r.get("tampered")))
        rate_center, ci_low, ci_high = stats.wilson_score_interval(ok, n)
        
        med = lambda k: (st.median([r[k] for r in v if r.get(k) is not None]) if any(r.get(k) is not None for r in v) else None)
        out.append(dict(
            harness=h, model=m, n=n, passed=ok, rate=ok / n,
            ci_low=ci_low, ci_high=ci_high,
            wall=med("wall_s"), calls=med("tool_calls"),
            reqs=med("llm_requests"),
            peak=max([r["peak_prompt_tokens"] for r in v if r.get("peak_prompt_tokens")] or [None], key=lambda x: x or 0),
            loops=sum(1 for r in v if bool(r.get("loop"))),
            timeouts=sum(1 for r in v if bool(r.get("timeout"))),
            tampered=tam
        ))
    return sorted(out, key=lambda s: (-s["rate"], s["wall"] or 1e9))

def per_task(rows=None):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in (rows if rows is not None else store.rows()):
        g[(r["harness"], label(r))][r["task"]].append(int(bool(r.get("done"))))
    return g

def context_curve(rows=None):
    """Evaluates the quality vs context size curve across models/harnesses."""
    g = collections.defaultdict(list)
    for r in (rows if rows is not None else store.rows()):
        ctx = r.get("num_ctx")
        if ctx:
            g[(r["model"], ctx)].append(r)
            
    out = []
    for (m, ctx), v in sorted(g.items()):
        n = len(v)
        ok = sum(1 for r in v if r.get("done"))
        p, ci_low, ci_high = stats.wilson_score_interval(ok, n)
        lat_stats = stats.numeric_summary([r.get("wall_s") for r in v])
        fails = sum(1 for r in v if not r.get("done") or r.get("timeout") or r.get("loop"))
        out.append(dict(
            model=m,
            num_ctx=ctx,
            n=n,
            passed=ok,
            rate=p,
            ci_low=ci_low,
            ci_high=ci_high,
            median_wall=lat_stats["median"],
            iqr_wall=lat_stats["iqr"],
            failures=fails
        ))
    return out

def family_table(tol=0.05, policy="balanced"):
    """[(family, [variant rows], pick, why)] joining pass rate with context-fit and Pareto analysis."""
    runs = collections.defaultdict(list)
    for r in store.rows():
        if r.get("family"):
            runs[(r["family"], r["model"])].append(r)
    fit = {}
    for f in store.rows(FIT):
        fit[f["model"], f["num_ctx"]] = f
    out = []
    for name in sorted({k[0] for k in runs}):
        rows = []
        for (fname, tag), v in sorted(runs.items()):
            if fname != name: continue
            ctx = v[0].get("num_ctx"); f = fit.get((tag, ctx)) or {}
            n = len(v)
            ok = sum(1 for r in v if r.get("done"))
            p, ci_low, ci_high = stats.wilson_score_interval(ok, n)
            rows.append(dict(
                tag=tag, size=v[0].get("size"), quant=v[0].get("quant"),
                n=n, rate=p, ci_low=ci_low, ci_high=ci_high,
                wall=st.median(r["wall_s"] for r in v if r.get("wall_s") is not None) if any(r.get("wall_s") is not None for r in v) else None,
                tok_s=f.get("tok_s"), size_gb=(f.get("fit") or {}).get("size_gb"),
                gpu_pct=(f.get("fit") or {}).get("gpu_pct"), ctx=ctx
            ))
        pick, why = fam.recommend(rows, tol, policy)
        out.append((name, rows, pick, why))
    return out

def render():
    s = summary()
    out = ["# llmeval Evidence & Evaluation Report\n"]
    
    # Section 1: Observed Facts
    out.append("## 1. Observed Facts\n")
    if s:
        out.append("### Summary Matrix\n")
        out.append(_md(
            ["Harness", "Model", "Pass", "Rate", "95% CI", "Median wall (s)", "Median Reqs", "Peak Tok", "Loops", "Timeouts", "Tampered"],
            [(x["harness"], x["model"], f'{x["passed"]}/{x["n"]}', f'{100*x["rate"]:.0f}%',
              f'[{int(100*x["ci_low"])}%, {int(100*x["ci_high"])}%]',
              x["wall"], x["reqs"], x["peak"], x["loops"], x["timeouts"], x["tampered"]) for x in s]
        ))
        
        pt = per_task()
        tasks = sorted({t for d in pt.values() for t in d})
        out.append("\n### Pass Count per Task\n")
        out.append(_md(["Harness", "Model"] + tasks, [(h, m, *[f"{sum(d[t])}/{len(d[t])}" if t in d else "-" for t in tasks]) for (h, m), d in sorted(pt.items())]))

    # Context Quality Curve
    curve = context_curve()
    if curve and len({c["num_ctx"] for c in curve}) > 1:
        out.append("\n### Context Quality Curve (num_ctx benchmark)\n")
        out.append(_md(
            ["Model", "Context", "Pass Rate", "95% CI", "Median Wall (s)", "Failures"],
            [(c["model"], f'{c["num_ctx"] // 1024}k', f'{100*c["rate"]:.0f}% ({c["passed"]}/{c["n"]})',
              f'[{int(100*c["ci_low"])}%, {int(100*c["ci_high"])}%]', c["median_wall"], c["failures"]) for c in curve]
        ))

    # Section 2: Statistical Evidence
    out.append("\n## 2. Statistical Evidence & Uncertainty\n")
    out.append("> **Inference Policy:** Differences between configurations are confirmed only when supported by statistical tests and non-overlapping confidence intervals. Small sample sizes (n < 6) are flagged as preliminary/insufficient evidence.\n")

    # Section 3: Resource Behavior & Hardware Feasibility
    out.append("\n## 3. Resource Behavior & Hardware Feasibility\n")
    fit = store.rows(FIT)
    if fit:
        latest = {(r["model"], r["num_ctx"]): r for r in fit}
        models = sorted({k[0] for k in latest}); ctxs = sorted({k[1] for k in latest})
        cell = lambda r: "error" if r is None or "error" in r else f'{r["fit"]["size_gb"]} GB, {r["fit"]["gpu_pct"]}% GPU, {r["tok_s"]} tok/s' if r.get("fit") else f'{r["tok_s"]} tok/s'
        out.append("### Hardware Context Feasibility Envelope\n")
        out.append(_md(["Model"] + [f"{c//1024}k" for c in ctxs], [(m, *[cell(latest.get((m, c))) for c in ctxs]) for m in models]))

    # Section 4: Decisions & Pareto Recommendations
    out.append("\n## 4. Multi-Objective Decisions (Pareto Frontier)\n")
    for name, rows, pick, why in family_table():
        out.append(f"### Family: {name}\n")
        out.append(_md(
            ["Variant", "Params", "Quant", "Pass", "95% CI", "Median wall (s)", "tok/s", "VRAM (GB)", "GPU %", "ctx"],
            [(("**" + r["tag"] + "**") if pick and r["tag"] == pick["tag"] else r["tag"], r["size"], r["quant"],
              f'{100 * r["rate"]:.0f}% (n={r["n"]})', f'[{int(100*r["ci_low"])}%, {int(100*r["ci_high"])}%]',
              round(r["wall"], 1) if r["wall"] is not None else "-", r["tok_s"], r["size_gb"], r["gpu_pct"], r["ctx"])
             for r in sorted(rows, key=lambda r: (fam.params_b(r["size"]) or 0, r["quant"] or ""))]
        ))
        out.append(f"**Recommended:** **{pick['tag'] if pick else '-'}** ({why})\n")

    return "\n".join(out)

def compare(a, b, harness=None, model=None, tasks=None, min_gain=1, max_slowdown=0.25):
    """
    Statistically grounded before/after comparison of two variants.
    Preserves backwards compatible return format while providing full statistical evidence.
    """
    all_rows = store.rows()
    rows_a = [r for r in all_rows if r.get("variant") == a and (not harness or r["harness"] == harness) and (not model or r["model"] == model)]
    rows_b = [r for r in all_rows if r.get("variant") == b and (not harness or r["harness"] == harness) and (not model or r["model"] == model)]

    if tasks:
        rows_a = [r for r in rows_a if r["task"] in tasks]
        rows_b = [r for r in rows_b if r["task"] in tasks]

    # Check for overlapping tasks and repetitions
    map_a = {(r["task"], r["rep"]): r for r in rows_a}
    map_b = {(r["task"], r["rep"]): r for r in rows_b}
    common_keys = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    if not common_keys:
        return {"verdict": "no overlap", "decision": "incomparable", "n": 0}

    matched_a = [map_a[k] for k in common_keys]
    matched_b = [map_b[k] for k in common_keys]

    # Check comparability
    comparable, comp_reasons = True, []
    for ra, rb in zip(matched_a, matched_b):
        ok_comp, r_reasons = provenance.check_comparability(ra, rb)
        if not ok_comp:
            comparable = False
            comp_reasons.extend(r_reasons)
            break

    pa = sum(1 for r in matched_a if r.get("done"))
    pb = sum(1 for r in matched_b if r.get("done"))
    wa = st.median(r["wall_s"] for r in matched_a if r.get("wall_s") is not None) if any(r.get("wall_s") is not None for r in matched_a) else 0.0
    wb = st.median(r["wall_s"] for r in matched_b if r.get("wall_s") is not None) if any(r.get("wall_s") is not None for r in matched_b) else 0.0

    stat_result = stats.compare_runs(matched_a, matched_b, max_slowdown=max_slowdown)
    decision = stat_result["decision"]

    # Backward-compatible verdict: "better", "same", "worse"
    if pb - pa >= min_gain and wb <= wa * (1.0 + max_slowdown):
        verdict = "better"
    elif pa - pb >= min_gain:
        verdict = "worse"
    else:
        verdict = "same"

    return {
        "verdict": verdict,
        "decision": decision,
        "reason": stat_result["reason"],
        "comparable": comparable,
        "comparability_notes": list(set(comp_reasons)),
        "n": len(common_keys),
        "statistical_evidence": stat_result["statistical_evidence"],
        "resource_behavior": stat_result["resource_behavior"],
        a: {"passed": pa, "median_wall": wa},
        b: {"passed": pb, "median_wall": wb}
    }
