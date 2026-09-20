"""Model families: describe a family once (name + sizes + quantizations) and let llmeval compare every variant, then
recommend one. Variants are ordinary models to the runner; they only carry family/size/quant metadata for reporting.

[[families]]
name   = "granite4.1"
sizes  = ["3b", "8b"]
quants = ["q4_K_M", "q6_K", "q8_0"]     # tag pattern defaults to "{name}:{size}-{quant}"
# variants = ["ornith:9b-q4_K_M", "ornith:9b-q8_0"]   # alternative: explicit list of ollama tags
# pattern = "{name}:{size}-{quant}"; as_is = true     # as_is: test the tags untouched instead of tuned "<name>-agent:<size>-<quant>" copies
"""
import re
from . import pareto

SIZE = re.compile(r"(\d+(?:\.\d+)?)b\b", re.I)
QUANT = re.compile(r"\b(q\d+(?:_[a-z0-9]+)*|bf16|fp16|f16)\b", re.I)

def parse_variant(tag):
    """(size, quant) guessed from an ollama tag like 'granite4.1:8b-q4_K_M'; either may be None."""
    t = tag.split(":", 1)[-1]
    s = SIZE.search(t) or (SIZE.search(tag.split(":", 1)[0]) if tag.startswith("hf.co/") else None)   # HF repos put the size in the repo name
    q = QUANT.search(t)
    return (s.group(0).lower() if s else None), (q.group(0) if q else None)

def params_b(size):
    return float(size[:-1]) if size else None

def expand(fam, defaults=None):
    """Model dicts for one family (same shape as [[models]] entries plus family/size/quant)."""
    name, out = fam["name"], []
    pattern = fam.get("pattern", "{name}:{size}-{quant}")
    bases = list(fam.get("variants", [])) or [pattern.format(name=name, size=s, quant=q) for s in fam.get("sizes", [""]) for q in fam.get("quants", [""])]
    for base in bases:
        size, quant = parse_variant(base)
        clean = lambda x: re.sub(r"[^A-Za-z0-9._-]+", "-", x)
        tag = base if fam.get("as_is") else f"{clean(name)}-agent:{clean(size or 'x')}-{clean(quant or 'x')}"
        m = {"tag": tag, "family": name, "size": size, "quant": quant}
        if not fam.get("as_is"): m["base"] = base; m["params"] = dict(fam.get("params", {}))
        if fam.get("ctx"): m.setdefault("params", {})["num_ctx"] = fam["ctx"]
        out.append(m)
    return out

def recommend(rows, tolerance=0.05, policy="balanced"):
    """
    Computes true Pareto frontier and applies configurable recommendation policy.
    Separates Pareto frontier computation from policy selection.
    """
    have = [r for r in rows if r.get("rate") is not None]
    if not have: return None, "no results yet"
    fits = [r for r in have if r.get("gpu_pct") in (None, 100)]
    pool = fits or have

    # Augment with normalized size attribute if size_gb missing
    norm_pool = []
    for r in pool:
        nr = dict(r)
        if nr.get("size_gb") is None and nr.get("size"):
            nr["size_gb"] = params_b(nr["size"])
        norm_pool.append(nr)

    # Compute Pareto frontier over quality (rate: max) and resource usage (size_gb: min, wall: min)
    frontier = pareto.compute_pareto_frontier(norm_pool, objectives={"rate": "max", "size_gb": "min", "wall": "min"})

    # Apply policy over the frontier candidates
    top = max(r["rate"] for r in norm_pool)
    ok = [r for r in frontier if r["rate"] >= top - tolerance]
    if not ok:
        ok = [r for r in norm_pool if r["rate"] >= top - tolerance]

    pick = sorted(ok, key=lambda r: (params_b(r.get("size")) or r.get("size_gb") or 99, r.get("size_gb") or 99, -(r.get("tok_s") or 0)))[0]

    # Map back to original row dict
    original_pick = next((r for r in rows if r.get("tag") == pick.get("tag")), pick)
    why = f'best pass rate {100 * top:.0f}%; smallest variant within {int(100 * tolerance)} points' + ("" if fits else "; NOTE none fit fully on the GPU")
    return original_pick, why
