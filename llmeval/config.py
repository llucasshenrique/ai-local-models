"""TOML config: run settings, default Modelfile params, and canonical effective configuration resolution.

Resolution hierarchy (single source of truth):
    global defaults
        ↓
    family defaults
        ↓
    model configuration (model-level fields + [models.params])
        ↓
    experiment configuration
        ↓
    CLI/runtime overrides
        ↓
    effective configuration
"""
import os, tomllib
from . import ollama, provenance

KNOWN_INTS = {"num_ctx", "top_k", "repeat_last_n", "num_predict"}
KNOWN_FLOATS = {"temperature", "top_p", "repeat_penalty"}

def _clean_val(k, v):
    if v is None:
        return None
    if k in KNOWN_INTS:
        try: return int(v)
        except (ValueError, TypeError): return v
    if k in KNOWN_FLOATS:
        try: return float(v)
        except (ValueError, TypeError): return v
    if k == "seed":
        try: return int(v)
        except (ValueError, TypeError): return None
    return v

def effective_params(cfg, model_or_tag, overrides=None):
    """
    Canonical single source of truth for effective model parameters.
    Resolves full parameter hierarchy without ambiguity.
    """
    if isinstance(model_or_tag, str):
        matching = next((m for m in cfg.get("models", []) if m.get("tag") == model_or_tag), None)
        model = matching if matching is not None else {"tag": model_or_tag}
    else:
        model = model_or_tag or {}

    resolved = {}

    # 1. Global defaults
    global_def = cfg.get("defaults", {})
    global_params = dict(global_def.get("params", {}))
    for k, v in global_def.items():
        if k != "params" and (k in KNOWN_INTS or k in KNOWN_FLOATS or k == "seed"):
            global_params.setdefault(k, v)
    for k, v in global_params.items():
        resolved[k] = _clean_val(k, v)

    # 2. Family defaults
    fam_name = model.get("family")
    fam_dict = None
    if fam_name:
        fam_dict = next((f for f in cfg.get("families", []) if f.get("name") == fam_name), None)
    if fam_dict:
        fam_params = dict(fam_dict.get("params", {}))
        if "ctx" in fam_dict:
            fam_params.setdefault("num_ctx", fam_dict["ctx"])
        for k, v in fam_params.items():
            resolved[k] = _clean_val(k, v)

    # 3. Model configuration (top-level model fields + [models.params])
    # Top-level fields
    for k in ("num_ctx", "temperature", "top_k", "top_p", "repeat_penalty", "repeat_last_n", "num_predict", "seed"):
        if k in model and model[k] is not None:
            resolved[k] = _clean_val(k, model[k])
    # [models.params] takes precedence if both top-level and params exist
    for k, v in model.get("params", {}).items():
        resolved[k] = _clean_val(k, v)

    # 4. Experiment configuration
    exp_params = cfg.get("experiment", {}).get("params", {}) or cfg.get("run", {}).get("params", {})
    for k, v in exp_params.items():
        resolved[k] = _clean_val(k, v)

    # 5. Runtime / CLI overrides
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                resolved[k] = _clean_val(k, v)

    # Fallback for num_ctx if not resolved
    if "num_ctx" not in resolved or resolved["num_ctx"] is None:
        tag = model.get("tag", "")
        resolved["num_ctx"] = int(_num_ctx(tag)) if tag else 16384
    else:
        resolved["num_ctx"] = int(resolved["num_ctx"])

    # Ensure seed is explicitly present (None if stochastic)
    resolved.setdefault("seed", None)

    return resolved

def resolve_model(cfg, model, overrides=None):
    """Applies canonical effective configuration to a model dict."""
    eff = effective_params(cfg, model, overrides)
    model["effective_params"] = eff
    model["params"] = dict(eff)
    model["num_ctx"] = eff["num_ctx"]
    model["config_digest"] = provenance.compute_config_digest(eff)
    return model

def load(path):
    """Loads and resolves a TOML configuration."""
    cfg = tomllib.load(open(path, "rb"))
    cfg.setdefault("run", {})
    cfg["run"].setdefault("reps", 3)
    cfg["run"].setdefault("timeout", 180)
    cfg["run"].setdefault("harnesses", ["pi", "minimal"])
    cfg.setdefault("tasks", {})
    cfg["tasks"].setdefault("split", {"train": None, "val": None, "test": None})
    
    cfg.setdefault("models", [])
    for m in cfg["models"]:
        resolve_model(cfg, m)

    from . import families
    known = {m["tag"] for m in cfg["models"]}
    for fam in cfg.get("families", []):
        for m in families.expand(fam):
            if m["tag"] in known: continue
            resolve_model(cfg, m)
            cfg["models"].append(m)
            known.add(m["tag"])
            
    return cfg

def _num_ctx(tag, default=16384):
    """num_ctx baked into an existing tag (from its Modelfile parameters); falls back to `default`."""
    try:
        for line in ollama.api("/api/show", {"model": tag}).get("parameters", "").splitlines():
            k, _, v = line.partition(" ")
            if k == "num_ctx": return int(v.strip())
    except Exception:
        pass
    return default

def prepare(cfg, log=print, family=None, dry_run=False):
    """Pull base models and (re)create tuned tags using canonical effective parameters.
    family: restrict to one family; dry_run: only list what would be pulled/created."""
    for m in cfg["models"]:
        if family and m.get("family") != family: continue
        eff = m.get("effective_params") or effective_params(cfg, m)
        if dry_run:
            log(f"{m['tag']:36} " + (f"from {m['base']}: " + ("installed" if ollama.has(m["base"]) else "WOULD PULL") if m.get("base") else "as-is: " + ("installed" if ollama.has(m["tag"]) else "MISSING")))
            continue
        if m.get("base"):
            log(f"pull {m['base']}"); ollama.pull(m["base"])
            log(f"create {m['tag']} with effective params"); ollama.create(m["tag"], m["base"], eff)
        elif not ollama.has(m["tag"]):
            log(f"WARNING: {m['tag']} is not installed and has no `base` to build it from")
