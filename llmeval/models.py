"""Model-list management shared by the CLI and the TUI: what is installed, what is in the config, and adding entries."""
import os
from . import config, ollama

def installed_info():
    """[(tag, size_gb, in_config)] for everything ollama has, sorted; in_config is matched with the implicit :latest."""
    have = {(m["tag"] if ":" in m["tag"] else m["tag"] + ":latest") for m in config.load(_path)["models"]} if _path else set()
    fake = os.environ.get("LLMEVAL_FAKE_MODELS")          # "tag:GB,tag:GB": deterministic list for tests/demos
    if fake:
        rows = [{"name": t.rsplit("=", 1)[0], "size": float(t.rsplit("=", 1)[1]) * 1e9} for t in fake.split(",")]
    else:
        rows = ollama.api("/api/tags")["models"]
    return [(m["name"], round(m["size"] / 1e9, 1), m["name"] in have) for m in sorted(rows, key=lambda m: m["name"])]

_path = None
def bind(path):
    global _path; _path = path

def _installed(tag):
    fake = os.environ.get("LLMEVAL_FAKE_MODELS")
    return tag in {t.rsplit("=", 1)[0] for t in fake.split(",")} if fake else ollama.has(tag)

def add(path, tags, base=None, ctx=None, pull=False, log=print):
    """Append models to the config file. Everything is validated first, so a bad tag never leaves a half-written config.
    Installed tags are used as-is; `base` builds a tuned tag from another model (created later by `prepare`)."""
    cfg = config.load(path); have = {m["tag"] for m in cfg["models"]}; blocks = []
    for tag in tags:
        if tag in have: log(f"skip {tag}: already in the config"); continue
        if base:
            block = f'\n[[models]]\ntag  = "{tag}"\nbase = "{base}"\n' + (f"[models.params]\nnum_ctx = {ctx}\n" if ctx else "")
        else:
            if pull and not _installed(tag): log(f"pulling {tag} ..."); ollama.pull(tag)
            if not _installed(tag): raise SystemExit(f"{tag} is not installed: pull it, or give a base model to build it from")
            block = f'\n[[models]]\ntag = "{tag}"\n'
        blocks.append((tag, block))
    if blocks: open(path, "a").write("".join(b for _, b in blocks))
    log("added: " + (", ".join(t for t, _ in blocks) if blocks else "nothing"))
    return [t for t, _ in blocks]


def add_family(path, name, sizes, quants, ctx=None, as_is=False, log=print):
    """Append a [[families]] block; every size x quant becomes a variant that `prepare` pulls and `run` compares."""
    cfg = config.load(path)
    if any(f["name"] == name for f in cfg.get("families", [])): log(f"skip {name}: family already in the config"); return False
    q = lambda xs: "[" + ", ".join(f'"{x}"' for x in xs) + "]"
    block = f'\n[[families]]\nname   = "{name}"\nsizes  = {q(sizes)}\nquants = {q(quants)}\n' + (f"ctx    = {ctx}\n" if ctx else "") + ("as_is  = true\n" if as_is else "")
    open(path, "a").write(block); log(f"added family {name}: {len(sizes) * len(quants)} variants"); return True
