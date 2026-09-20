"""Run one trial (harness x model x task) in a throw-away repo and measure it; run_matrix() drives a whole config."""
import json, os, shutil, signal, subprocess, threading, time
from . import ollama, store, tasks as T
from .harnesses import REGISTRY
from .lock import gpu_lock
from .proxy import Proxy

LOOP_REPEATS = 3          # the same tool call + identical arguments this many times = loop; the run is killed
TRACE_DIR = os.path.join(store.RESULTS, "traces")

class Ctx:
    def __init__(self, **kw): self.__dict__.update(kw)

def run_trial(harness, model, task, rep, timeout, proxy, num_ctx, keep=False):
    h = REGISTRY[harness]; d = T.materialize(task)
    ctx = Ctx(dir=d, model=model, prompt=task["prompt"], proxy_url=proxy.url if proxy else "", num_ctx=num_ctx, task=task)
    cmd, extra = h.build(ctx)
    env = {**os.environ, "PWD": d, **extra}
    if h.needs_proxy: proxy.reset()
    t0 = time.time(); st, seen = {}, {}
    loop = False; timed_out = threading.Event()
    p = subprocess.Popen(cmd, cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    def kill():
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    timer = threading.Timer(timeout, lambda: (timed_out.set(), kill())); timer.start()
    os.makedirs(os.path.join(TRACE_DIR, harness), exist_ok=True)
    trace = open(os.path.join(TRACE_DIR, harness, f"{model.replace(':', '_')}.{task['id']}.{rep}.log"), "w")
    written = 0
    for line in p.stdout:
        if written < 200_000: trace.write(line); written += len(line)
        k = h.on_line(line, st)
        if k:
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= LOOP_REPEATS: loop = True; kill(); break
    p.wait(); timer.cancel(); trace.close(); kill()          # kill() also reaps stragglers in the process group
    wall = round(time.time() - t0, 1)
    if h.needs_proxy: st.update(proxy.stats())
    ok, tampered = T.verify(task, d)
    if not keep: shutil.rmtree(d, ignore_errors=True)
    return dict(harness=harness, model=model, task=task["id"], rep=rep, done=ok, tampered=tampered, wall_s=wall,
                timeout=timed_out.is_set(), loop=loop if h.detects_loops else None,
                tool_calls=st.get("tool_calls"), llm_requests=st.get("llm_requests"), peak_prompt_tokens=st.get("peak_prompt_tokens"),
                num_ctx=num_ctx, ollama=ollama.version(), digest=ollama.digest(model))

def run_matrix(cfg, force=False, keep=False):
    """Generator of progress events; results are appended to results/runs.jsonl. Resumable: finished trials are skipped."""
    run = cfg["run"]; models = cfg["models"]; all_tasks = T.load(run.get("tasks") or None)
    harnesses = [h for h in run["harnesses"] if h in REGISTRY]
    done = set() if force else store.done_keys()
    todo = [(m, h, t, r) for m in models for h in harnesses for t in all_tasks for r in range(1, run["reps"] + 1)
            if (h, m["tag"], t["id"], r) not in done]
    yield {"type": "plan", "total": len(todo), "skipped": len(models) * len(harnesses) * len(all_tasks) * run["reps"] - len(todo)}
    proxy = Proxy(run.get("proxy_port", 11436)) if any(REGISTRY[h].needs_proxy for h in harnesses) else None
    n = 0
    try:
        with gpu_lock():
            ollama.unload_all()
            for tag in dict.fromkeys(m["tag"] for m in models):
                mine = [x for x in todo if x[0]["tag"] == tag]
                if not mine: continue
                yield {"type": "model", "model": tag}
                for m, h, t, r in mine:
                    num_ctx = m.get("num_ctx", 16384)
                    REGISTRY[h].prepare(tag, num_ctx)
                    yield {"type": "start", "harness": h, "model": tag, "task": t["id"], "rep": r, "n": n + 1}
                    rec = store.append(run_trial(h, tag, t, r, run.get("timeout", 180), proxy, num_ctx, keep))
                    n += 1; yield {"type": "trial", **rec, "n": n}
                ollama.stop(tag)
    finally:
        if proxy: proxy.close()
    yield {"type": "done", "trials": n}
