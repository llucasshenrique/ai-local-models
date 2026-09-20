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

def recommend(rows, tolerance=0.05):
    """rows: [{tag,size,quant,rate,tok_s,size_gb,gpu_pct}] for one family (any may be None).
    Rule: among variants that fit fully on the GPU (or whose fit is unknown), find the best pass rate; recommend the
    smallest (params, then VRAM) variant within `tolerance` of it, tie-break by speed. Explains itself in `why`."""
    have = [r for r in rows if r.get("rate") is not None]
    if not have: return None, "no results yet"
    fits = [r for r in have if r.get("gpu_pct") in (None, 100)]
    pool = fits or have
    top = max(r["rate"] for r in pool)
    ok = [r for r in pool if r["rate"] >= top - tolerance]
    pick = sorted(ok, key=lambda r: (params_b(r.get("size")) or 99, r.get("size_gb") or 99, -(r.get("tok_s") or 0)))[0]
    why = f'best pass rate {100 * top:.0f}%; smallest variant within {int(100 * tolerance)} points' + ("" if fits else "; NOTE none fit fully on the GPU")
    return pick, why
