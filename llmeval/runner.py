"""Run one trial (harness x model x task) in a throw-away repo and measure it; run_matrix() drives a whole config."""
import json, os, shutil, signal, subprocess, threading, time
from . import config, guardrails, ollama, provenance, store, tasks as T
from .harnesses import REGISTRY
from .lock import gpu_lock
from .proxy import Proxy

LOOP_REPEATS = 3          # the same tool call + identical arguments this many times = loop; the run is killed
TRACE_DIR = os.path.join(store.RESULTS, "traces")

class Ctx:
    def __init__(self, **kw): self.__dict__.update(kw)

def run_trial(harness, model, task, rep, timeout, proxy, num_ctx, keep=False,
              effective_params_dict=None, experiment_id=None, variant=None, meta=None):
    h = REGISTRY[harness]; d = T.materialize(task)
    ctx = Ctx(dir=d, model=model, prompt=task["prompt"], proxy_url=proxy.url if proxy else "", num_ctx=num_ctx, task=task)
    cmd, extra = h.build(ctx)
    env = {**os.environ, "PWD": d, **extra}
    if h.needs_proxy: proxy.reset()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time(); st, seen = {}, {}
    loop = False; timed_out = threading.Event()
    p = subprocess.Popen(cmd, cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    def kill():
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    timer = threading.Timer(timeout, lambda: (timed_out.set(), kill())); timer.start()
    os.makedirs(os.path.join(TRACE_DIR, harness), exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, harness, f"{model.replace(':', '_')}.{task['id']}.{rep}.log")
    written = 0
    with open(trace_path, "w", encoding="utf-8") as trace:
        for line in p.stdout:
            if written < 200_000: trace.write(line); written += len(line)
            k = h.on_line(line, st)
            if k:
                seen[k] = seen.get(k, 0) + 1
                if seen[k] >= LOOP_REPEATS: loop = True; kill(); break
    p.wait(); timer.cancel(); kill()          # kill() also reaps stragglers in the process group
    ts_end = time.strftime("%Y-%m-%dT%H:%M:%S")
    wall = round(time.time() - t0, 1)
    if h.needs_proxy: st.update(proxy.stats())
    ok, tampered = T.verify(task, d)
    if not keep: shutil.rmtree(d, ignore_errors=True)

    eff = dict(effective_params_dict or {})
    seed = eff.get("seed")
    task_ver = task.get("version") or provenance.compute_task_version(task)

    record = {
        "schema_version": store.SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": provenance.generate_run_id(),
        "harness": harness,
        "harness_version": provenance.compute_harness_version(harness),
        "model": model,
        "digest": ollama.digest(model),
        "family": (meta or {}).get("family"),
        "size": (meta or {}).get("size"),
        "quant": (meta or {}).get("quant"),
        "task": task["id"],
        "task_version": task_ver,
        "rep": rep,
        "commit": provenance.get_repo_commit(),
        "benchmark_version": provenance.compute_benchmark_version(),
        "config_digest": provenance.compute_config_digest(eff),
        "effective_params": eff,
        "num_ctx": num_ctx,
        "temperature": eff.get("temperature"),
        "top_k": eff.get("top_k"),
        "top_p": eff.get("top_p"),
        "repeat_penalty": eff.get("repeat_penalty"),
        "repeat_last_n": eff.get("repeat_last_n"),
        "num_predict": eff.get("num_predict"),
        "seed": seed,
        "is_reproducible": seed is not None,
        "ollama": ollama.version(),
        "hardware": provenance.get_hardware_info(),
        "os": provenance.get_os_info(),
        "ts_start": ts_start,
        "ts_end": ts_end,
        "wall_s": wall,
        "model_load_s": st.get("load_duration_s"),
        "gen_s": st.get("eval_duration_s"),
        "prompt_eval_s": st.get("prompt_eval_duration_s"),
        "prompt_tokens": st.get("prompt_tokens"),
        "gen_tokens": st.get("completion_tokens"),
        "peak_prompt_tokens": st.get("peak_prompt_tokens"),
        "tool_calls": st.get("tool_calls"),
        "llm_requests": st.get("llm_requests"),
        "timeout": timed_out.is_set(),
        "loop": loop if h.detects_loops else None,
        "tampered": tampered,
        "done": ok,
        "variant": variant
    }
    return record

def run_matrix(cfg, force=False, keep=False, family=None, variant=None, task_ids=None,
               ctx_sweep=None, experiment_id=None, reps_override=None, split_name=None):
    """Generator of progress events; results are appended to results/runs.jsonl. Resumable: finished trials are skipped."""
    # Enforce benchmark infrastructure immutability check before starting
    bm_snapshot = guardrails.snapshot_benchmark()

    run = cfg["run"]
    models = [m for m in cfg["models"] if not family or m.get("family") == family]
    
    # Task filtering with splits
    split = cfg.get("tasks", {}).get("split", {})
    all_tasks = T.load(ids=task_ids or run.get("tasks") or None, split=split, split_name=split_name)
    
    harnesses = [h for h in run["harnesses"] if h in REGISTRY]
    reps = reps_override if reps_override is not None else run.get("reps", 3)
    exp_id = experiment_id or run.get("experiment_id") or variant or "default"

    # Context sweep dimension: if specified, evaluates multiple contexts within safe envelope
    sweep = ctx_sweep or run.get("ctx_sweep")
    is_sweep = bool(sweep)

    done = set() if force else (store.done_keys(include_ctx=True) if is_sweep else store.done_keys())

    todo = []
    for m in models:
        eff = m.get("effective_params") or config.effective_params(cfg, m)
        contexts = [int(c) for c in sweep] if is_sweep else [int(m.get("num_ctx", eff.get("num_ctx", 16384)))]
        for c in contexts:
            for h in harnesses:
                for t in all_tasks:
                    for r in range(1, reps + 1):
                        k = (h, m["tag"], t["id"], r, variant, c) if is_sweep else (h, m["tag"], t["id"], r, variant)
                        if k not in done:
                            todo.append((m, h, t, r, c, eff))

    yield {"type": "plan", "total": len(todo), "skipped": (len(models) * len(harnesses) * len(all_tasks) * reps * (len(sweep) if is_sweep else 1)) - len(todo)}
    proxy = Proxy(run.get("proxy_port", 11436)) if any(REGISTRY[h].needs_proxy for h in harnesses) else None
    n = 0
    try:
        with gpu_lock():
            ollama.unload_all()
            for tag in dict.fromkeys(m["tag"] for m in models):
                mine = [x for x in todo if x[0]["tag"] == tag]
                if not mine: continue
                yield {"type": "model", "model": tag}
                for m, h, t, r, c, eff in mine:
                    # Check benchmark files remain untampered during trial
                    guardrails.assert_immutable_benchmark(bm_snapshot)

                    REGISTRY[h].prepare(tag, c)
                    yield {"type": "start", "harness": h, "model": tag, "task": t["id"], "rep": r, "num_ctx": c, "n": n + 1}
                    meta = {k: m[k] for k in ("family", "size", "quant") if m.get(k)}
                    
                    trial_eff = dict(eff)
                    trial_eff["num_ctx"] = c
                    
                    raw_record = run_trial(
                        harness=h, model=tag, task=t, rep=r,
                        timeout=run.get("timeout", 180), proxy=proxy, num_ctx=c, keep=keep,
                        effective_params_dict=trial_eff, experiment_id=exp_id, variant=variant, meta=meta
                    )
                    rec = store.append(raw_record)
                    n += 1
                    yield {"type": "trial", **rec, "n": n}
                ollama.stop(tag)
    finally:
        if proxy: proxy.close()
    yield {"type": "done", "trials": n}
