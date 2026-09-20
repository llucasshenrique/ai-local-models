"""TOML config: run settings, default Modelfile params and the model list. See evals/default.toml."""
import os, tomllib
from . import ollama

def load(path):
    cfg = tomllib.load(open(path, "rb"))
    cfg.setdefault("run", {}); cfg["run"].setdefault("reps", 3); cfg["run"].setdefault("timeout", 180)
    cfg["run"].setdefault("harnesses", ["pi", "minimal"])
    base_params = cfg.get("defaults", {}).get("params", {})
    cfg.setdefault("models", [])
    for m in cfg["models"]:
        m["params"] = {**base_params, **m.get("params", {})} if m.get("base") else m.get("params", {})
        m["num_ctx"] = int(m["params"].get("num_ctx") or _num_ctx(m["tag"]))
    from . import families
    known = {m["tag"] for m in cfg["models"]}
    for fam in cfg.get("families", []):
        for m in families.expand(fam):
            if m["tag"] in known: continue
            m["params"] = {**base_params, **m.get("params", {})} if m.get("base") else m.get("params", {})
            m["num_ctx"] = int(m["params"].get("num_ctx") or _num_ctx(m["tag"])); cfg["models"].append(m); known.add(m["tag"])
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
    """Pull base models and (re)create tuned tags. Only adds/updates the tags named in the config.
    family: restrict to one family; dry_run: only list what would be pulled/created."""
    for m in cfg["models"]:
        if family and m.get("family") != family: continue
        if dry_run:
            log(f"{m['tag']:36} " + (f"from {m['base']}: " + ("installed" if ollama.has(m["base"]) else "WOULD PULL") if m.get("base") else "as-is: " + ("installed" if ollama.has(m["tag"]) else "MISSING")))
            continue
        if m.get("base"):
            log(f"pull {m['base']}"); ollama.pull(m["base"])
            log(f"create {m['tag']}"); ollama.create(m["tag"], m["base"], m["params"])
        elif not ollama.has(m["tag"]):
            log(f"WARNING: {m['tag']} is not installed and has no `base` to build it from")
