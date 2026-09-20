"""Thin ollama client: model prep, GPU-fit measurement and a speed probe. Uses OLLAMA_HOST (default 127.0.0.1:11434)."""
import json, os, subprocess, tempfile, time, urllib.request

def host():
    h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return h if "://" in h else "http://" + h

def api(path, body=None, timeout=300):
    req = urllib.request.Request(host() + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def installed():
    """{tag: digest} of models known to the server."""
    return {m["name"]: m["digest"] for m in api("/api/tags")["models"]}

def has(tag):
    t = tag if ":" in tag else tag + ":latest"
    return t in installed()

def digest(tag):
    t = tag if ":" in tag else tag + ":latest"
    return installed().get(t)

def version():
    return api("/api/version").get("version")

def pull(tag):
    if not has(tag):
        subprocess.run(["ollama", "pull", tag], check=True)

def modelfile_text(base, params):
    lines = [f"FROM {base}"]
    for k, v in params.items():
        for x in (v if isinstance(v, list) else [v]):
            lines.append(f"PARAMETER {k} {x}")
    return "\n".join(lines) + "\n"

def create(tag, base, params):
    """(Re)create `tag` from `base` with PARAMETERs. Only ever adds/updates the given tag."""
    text = modelfile_text(base, params)
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(text)
    try:
        r = subprocess.run(["ollama", "create", tag, "-f", f.name], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ollama create {tag} failed: {r.stderr.strip()[-300:]}")
    finally:
        os.unlink(f.name)
    return text

def unload_all():
    for m in api("/api/ps").get("models", []):
        subprocess.run(["ollama", "stop", m["name"]], capture_output=True)
    time.sleep(2)

def stop(tag):
    subprocess.run(["ollama", "stop", tag], capture_output=True); time.sleep(2)

def fit(tag):
    """Loaded-model footprint from /api/ps: size (GB), GPU share (%), context."""
    for m in api("/api/ps").get("models", []):
        if m["name"] == (tag if ":" in tag else tag + ":latest"):
            size, vram = m["size"], m.get("size_vram", 0)
            return {"size_gb": round(size / 1e9, 1), "gpu_pct": round(100 * vram / size) if size else None,
                    "context": m.get("context_length")}
    return None

def probe(tag, num_ctx=None, timeout=300):
    """Generate a short answer; return tok/s, load time and GPU fit. Retries without `think` if the model rejects it."""
    body = {"model": tag, "stream": False, "think": False, "options": {"num_ctx": num_ctx} if num_ctx else {},
            "messages": [{"role": "user", "content": "Write a Python function is_prime(n) with a docstring and three asserts. Code only."}]}
    t0 = time.time()
    try:
        r = api("/api/chat", body, timeout)
    except Exception:
        body.pop("think"); r = api("/api/chat", body, timeout)
    return {"tok_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1), "load_s": round(r.get("load_duration", 0) / 1e9, 1),
            "wall_s": round(time.time() - t0, 1), "fit": fit(tag)}
