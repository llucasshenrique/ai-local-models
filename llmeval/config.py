"""TOML config: run settings, default Modelfile params and the model list. See evals/default.toml."""
import os, tomllib
from . import ollama

def load(path):
    cfg = tomllib.load(open(path, "rb"))
    cfg.setdefault("run", {}); cfg["run"].setdefault("reps", 3); cfg["run"].setdefault("timeout", 180)
    cfg["run"].setdefault("harnesses", ["pi", "minimal"])
    base_params = cfg.get("defaults", {}).get("params", {})
    for m in cfg["models"]:
        m["params"] = {**base_params, **m.get("params", {})} if m.get("base") else m.get("params", {})
        m["num_ctx"] = int(m["params"].get("num_ctx") or _num_ctx(m["tag"]))
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

def prepare(cfg, log=print):
    """Pull base models and (re)create tuned tags. Only adds/updates the tags named in the config."""
    for m in cfg["models"]:
        if m.get("base"):
            log(f"pull {m['base']}"); ollama.pull(m["base"])
            log(f"create {m['tag']}"); ollama.create(m["tag"], m["base"], m["params"])
        elif not ollama.has(m["tag"]):
            log(f"WARNING: {m['tag']} is not installed and has no `base` to build it from")
