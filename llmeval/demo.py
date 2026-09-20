"""Synthetic sample data so the TUI/report can be tried without a GPU or ollama: `make tui-demo`."""
import random
from . import store

def seed(n_reps=3):
    rnd = random.Random(7); tasks = ["01-add-function", "02-rename-symbol", "03-fix-bug", "04-spec-in-readme", "05-bug-across-files"]
    prof = {("pi", "ornith-agent:9b"): (0.95, 15, 4500), ("minimal", "ornith-agent:9b"): (0.9, 8, 700), ("opencode", "ornith-agent:9b"): (0.9, 50, 12000),
            ("pi", "gemma4-agent:12b"): (0.9, 28, 4300), ("aider", "gemma4-agent:12b"): (0.8, 22, 900), ("minimal", "qwen3-agent:8b"): (0.3, 20, 800)}
    for (h, m), (p, wall, peak) in prof.items():
        for t in tasks:
            for r in range(1, n_reps + 1):
                store.append(dict(harness=h, model=m, task=t, rep=r, done=rnd.random() < p, tampered=False, wall_s=round(wall * rnd.uniform(.6, 1.6), 1),
                                  timeout=False, loop=(rnd.random() < .05) if h in ("opencode", "minimal") else None, tool_calls=rnd.randint(2, 9),
                                  llm_requests=rnd.randint(3, 10), peak_prompt_tokens=int(peak * rnd.uniform(.9, 1.1)), num_ctx=32768))
    from .fit import FIT
    for m, (gb, tps) in {"ornith-agent:9b": (6.3, 66), "gemma4-agent:12b": (8.1, 45), "qwen3-agent:8b": (10.0, 44)}.items():
        for c in (16384, 32768, 49152, 65536):
            store.append({"model": m, "num_ctx": c, "tok_s": tps, "fit": {"size_gb": gb + (c // 16384 - 1) * .3, "gpu_pct": 100 if gb < 10 or c == 16384 else 85, "context": c}}, FIT)
