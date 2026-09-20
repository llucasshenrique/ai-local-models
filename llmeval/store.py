"""One append-only results file with a single schema. A trial is identified by (harness, model, task, rep);
rows can be marked invalid later (moved aside, never silently edited)."""
import json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("LLMEVAL_RESULTS") or os.path.join(ROOT, "results")   # override for demos/tests
RUNS = os.path.join(RESULTS, "runs.jsonl")
SCHEMA_VERSION = 2

def rows(path=RUNS):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip().startswith("{")]

def key(r):
    """5-tuple trial identity matching standard harness run key."""
    return (r["harness"], r["model"], r["task"], r["rep"], r.get("variant"))

def trial_identity(r):
    """Comprehensive trial identity including context size and config digest."""
    return (r.get("harness"), r.get("model"), r.get("task"), r.get("rep"), r.get("variant"), r.get("num_ctx"), r.get("config_digest"))

def done_keys(path=RUNS, include_ctx=False):
    if include_ctx:
        return {(r["harness"], r["model"], r["task"], r["rep"], r.get("variant"), r.get("num_ctx")) for r in rows(path)}
    return {key(r) for r in rows(path)}

def append(rec, path=RUNS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {
        "schema_version": rec.get("schema_version", SCHEMA_VERSION),
        "ts": rec.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        **rec
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def invalidate(pred, reason, path=RUNS):
    """Move rows matching pred(row) to results/invalid/ with a reason; returns how many."""
    rs = rows(path); bad = [r for r in rs if pred(r)]
    if bad:
        inv = os.path.join(os.path.dirname(path), "invalid"); os.makedirs(inv, exist_ok=True)   # next to the file it came from
        with open(os.path.join(inv, "runs-invalidated.jsonl"), "a", encoding="utf-8") as f:
            for r in bad: f.write(json.dumps({**r, "invalid_reason": reason}) + "\n")
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                if r not in bad: f.write(json.dumps(r) + "\n")
    return len(bad)
