"""Context-fit measurement: load each model at increasing num_ctx and record GPU share, footprint and tok/s."""
import os
from . import ollama, store
from .lock import gpu_lock

FIT = os.path.join(store.RESULTS, "fit.jsonl")

def measure(models, ctxs=(16384, 32768, 49152, 65536)):
    with gpu_lock():
        ollama.unload_all()
        for tag in models:
            for c in ctxs:
                try: rec = {"model": tag, "num_ctx": c, **ollama.probe(tag, c)}
                except Exception as e: rec = {"model": tag, "num_ctx": c, "error": str(e)[:200]}
                rec["ollama"] = ollama.version(); rec["digest"] = ollama.digest(tag)
                yield store.append(rec, FIT)
                ollama.stop(tag)
