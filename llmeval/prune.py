"""Clean up models that lost the ranking. Destructive, so: dry-run by default, explicit confirmation to execute, and a model is
only ever a candidate if llmeval knows it (it is in the config or in the results). Protected: the top of the ranking, each
family's recommendation, models with too little data, the advisor, and the defaults set in your Pi/opencode configs."""
import collections, json, os, re, statistics as st, subprocess
from . import families as fam, ollama, report, store

HOME = os.path.expanduser("~")

def ranking():
    """[(model, n, passed, smoothed_rate, median_wall)] best first; pooled over harnesses and reps."""
    g = collections.defaultdict(list)
    for r in store.rows(): g[r["model"]].append(r)
    rows = [(m, len(v), sum(r["done"] for r in v), (sum(r["done"] for r in v) + 1) / (len(v) + 2), st.median(r["wall_s"] for r in v)) for m, v in g.items()]
    return sorted(rows, key=lambda x: (-x[3], x[4]))

def external_defaults():
    """Model tags your other tools use by default (never pruned)."""
    out = set()
    try: out.add(json.load(open(os.path.join(HOME, ".pi/agent/settings.json"))).get("defaultModel", ""))
    except Exception: pass
    try:
        c = json.load(open(os.path.join(HOME, ".config/opencode/opencode.json")))
        out |= {str(c.get(k, "")).split("/", 1)[-1] for k in ("model", "small_model")}
    except Exception: pass
    return {x for x in out if x}

def _norm(t): return t if ":" in t else t + ":latest"

def plan(cfg, keep_top=3, margin=0.15, min_n=6, include_bases=False, advisor_tag=None):
    rank = ranking(); best = rank[0][3] if rank else 0
    inst = {m["name"]: m["size"] / 1e9 for m in ollama.api("/api/tags")["models"]}
    known = {m["tag"]: m for m in cfg["models"]}
    protected = {}                                        # tag -> why
    for i, (m, n, p, sm, w) in enumerate(rank):
        if i < keep_top: protected[m] = f"top {keep_top} of the ranking"
        if n < min_n: protected.setdefault(m, f"only {n} trials (< {min_n}): unproven, not a loser")
    for name, rows, pick, why in report.family_table():
        if pick: protected[pick["tag"]] = f"recommended in family {name}"
    if advisor_tag: protected[advisor_tag] = "advisor model"
    for t in external_defaults(): protected[t] = "default in your Pi/opencode config"
    prot = {_norm(k): v for k, v in protected.items()}
    dele, why_delete = [], {}
    fam_pick = {f: p["tag"] for f, rows, p, w in report.family_table() if p}
    for m, n, p, sm, w in rank:
        if _norm(m) in prot or _norm(m) not in inst: continue
        meta = known.get(m, {}); reasons = []
        if sm < best - margin: reasons.append(f"smoothed pass rate {100 * sm:.0f}% vs best {100 * best:.0f}% (margin {int(100 * margin)} pts), n={n}")
        f = meta.get("family") or next((r["family"] for r in store.rows() if r["model"] == m and r.get("family")), None)
        if f and f in fam_pick and _norm(fam_pick[f]) != _norm(m):
            pick = next((x for x in rank if x[0] == fam_pick[f]), None)
            if pick and sm <= pick[3] and (fam.params_b(fam.parse_variant(m)[0]) or 0) >= (fam.params_b(fam.parse_variant(fam_pick[f])[0]) or 0):
                reasons.append(f"dominated by {fam_pick[f]} (same family: not better and not smaller)")
        if reasons: dele.append({"tag": m, "kind": "tuned" if meta.get("base") or re.search(r"-agent:|^sw-|^abl-|^tune-", m) else "model", "size_gb": round(inst[_norm(m)], 1), "reason": "; ".join(reasons)})
    keep_tags = {_norm(t) for t in inst} - {_norm(d["tag"]) for d in dele}
    if include_bases:                                     # bases whose every dependent is being deleted: this is what actually frees disk
        deps = collections.defaultdict(set)
        for m in cfg["models"]:
            if m.get("base"): deps[_norm(m["base"])].add(_norm(m["tag"]))
        gone = {_norm(d["tag"]) for d in dele}
        for base, users in deps.items():
            if base in inst and users <= gone and base not in prot and base not in {_norm(d["tag"]) for d in dele}:
                dele.append({"tag": base, "kind": "base", "size_gb": round(inst[base], 1), "reason": f"base of only deleted tags ({', '.join(sorted(users))})"})
    return {"delete": dele, "protected": prot, "best": rank[0][0] if rank else None,
            "note": "Tuned tags share their weights with the base: deleting only a tuned tag frees almost nothing. Bases (--include-bases) free the disk.",
            "reclaim_gb": round(sum(d["size_gb"] for d in dele if d["kind"] != "tuned"), 1)}

def format_plan(p):
    if not p["delete"]: return ["nothing to clean: no model is a proven loser (protected models are listed with --verbose)"]
    out = [f'ranking leader: {p["best"]}; {len(p["delete"])} models would be deleted (~{p["reclaim_gb"]} GB reclaimed)']
    out += [f'  - {d["tag"]:44} {d["kind"]:6} {d["size_gb"]:>5} GB  {d["reason"]}' for d in p["delete"]]
    return out + [p["note"]]

def drop_config(path, tags):
    """Remove the [[models]] blocks of deleted tags from the config (so `prepare`/`run` do not resurrect them)."""
    lines = open(path).read().split("\n"); out, skip, dropped = [], False, []
    for i, line in enumerate(lines):
        if re.match(r"\s*\[\[", line) or (re.match(r"\s*\[", line) and not re.match(r"\s*\[models\.", line)):
            skip = False
            if line.strip() == "[[models]]":
                block = "\n".join(lines[i:i + 6])
                m = re.search(r'tag\s*=\s*"([^"]+)"', block)
                if m and m.group(1) in tags: skip = True; dropped.append(m.group(1))
        if not skip: out.append(line)
    open(path, "w").write("\n".join(out)); return dropped

def execute(p, cfg_path=None, drop=False, log=print):
    done = []
    for d in p["delete"]:
        r = subprocess.run(["ollama", "rm", d["tag"]], capture_output=True, text=True)
        log(("removed " if r.returncode == 0 else "FAILED ") + d["tag"] + ("" if r.returncode == 0 else ": " + r.stderr.strip()[:80]))
        if r.returncode == 0: done.append(d["tag"])
    if drop and cfg_path and done: log("config entries removed: " + (", ".join(drop_config(cfg_path, set(done))) or "none"))
    return done
