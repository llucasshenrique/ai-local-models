"""Context-fit measurement and automated empirical context optimization."""
import os
import time
import json
import threading
import subprocess
from . import ollama, store
from .lock import gpu_lock

FIT = os.path.join(store.RESULTS, "fit.jsonl")

def measure(models, ctxs=(16384, 32768, 49152, 65536)):
    """Standard context-fit probe across a static list of context sizes."""
    with gpu_lock():
        ollama.unload_all()
        for tag in models:
            for c in ctxs:
                try: rec = {"model": tag, "num_ctx": c, **ollama.probe(tag, c)}
                except Exception as e: rec = {"model": tag, "num_ctx": c, "error": str(e)[:200]}
                rec["ollama"] = ollama.version(); rec["digest"] = ollama.digest(tag)
                yield store.append(rec, FIT)
                ollama.stop(tag)

def read_gpu_stats():
    """Query nvidia-smi for VRAM used/free (MiB) and GPU utilization (%). Returns (used, free, util)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            parts = [x.strip() for x in res.stdout.strip().split(",")]
            return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        pass
    return 0, 0, 0

def read_mem_stats():
    """Read RAM and swap used/available in MB from /proc/meminfo (pure stdlib)."""
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split(":")
                if len(p) == 2:
                    mem[p[0].strip()] = int(p[1].split()[0])
        total_ram = mem.get("MemTotal", 0) // 1024
        avail_ram = mem.get("MemAvailable", 0) // 1024
        used_ram = total_ram - avail_ram
        total_sw = mem.get("SwapTotal", 0) // 1024
        free_sw = mem.get("SwapFree", 0) // 1024
        used_sw = total_sw - free_sw
        return used_ram, avail_ram, used_sw, free_sw
    except Exception:
        return 0, 0, 0, 0

def read_cpu_raw():
    """Read raw total and idle jiffies from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
            idle = parts[3] + parts[4]
            total = sum(parts)
            return idle, total
    except Exception:
        return 0.0, 0.0

class ContextMonitor(threading.Thread):
    """Background sampler recording peak VRAM, RAM, swap and CPU load during inference."""
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.running = True
        self.samples = []
        self.prev_cpu = read_cpu_raw()

    def run(self):
        while self.running:
            gpu_used, gpu_free, gpu_util = read_gpu_stats()
            ram_used, ram_avail, sw_used, sw_free = read_mem_stats()
            curr_idle, curr_tot = read_cpu_raw()
            cpu_pct = 0.0
            dt = curr_tot - self.prev_cpu[1]
            if dt > 0:
                d_idle = curr_idle - self.prev_cpu[0]
                cpu_pct = max(0.0, min(100.0, (1.0 - (d_idle / dt)) * 100.0))
            self.prev_cpu = (curr_idle, curr_tot)

            self.samples.append({
                "ts": time.time(),
                "gpu_used": gpu_used,
                "gpu_free": gpu_free,
                "gpu_util": gpu_util,
                "ram_used": ram_used,
                "ram_avail": ram_avail,
                "sw_used": sw_used,
                "sw_free": sw_free,
                "cpu_pct": cpu_pct
            })
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.join()

    def summary(self, baseline):
        if not self.samples:
            return baseline
        peak_gpu = max(s["gpu_used"] for s in self.samples)
        peak_ram = max(s["ram_used"] for s in self.samples)
        peak_sw = max(s["sw_used"] for s in self.samples)
        avg_cpu = sum(s["cpu_pct"] for s in self.samples) / len(self.samples)
        peak_cpu = max(s["cpu_pct"] for s in self.samples)
        return {
            "peak_gpu_mib": peak_gpu,
            "gpu_delta_mib": peak_gpu - baseline["gpu_used"],
            "peak_ram_mb": peak_ram,
            "ram_delta_mb": peak_ram - baseline["ram_used"],
            "peak_swap_mb": peak_sw,
            "swap_delta_mb": peak_sw - baseline["sw_used"],
            "avg_cpu_pct": round(avg_cpu, 1),
            "peak_cpu_pct": round(peak_cpu, 1)
        }

def build_stress_prompt(target_tokens):
    """Calibrated prompt text generator (~25 tokens per sentence)."""
    sentence = "The quick brown fox jumps over the lazy dog. In computer science, an algorithm is a finite sequence of rigorous instructions. "
    reps = max(1, int(target_tokens / 25.1))
    return (sentence * reps).strip() + "\n\nTask: Output three bullet points summarizing the text. Begin immediately."

def run_ctx_trial(tag, num_ctx, target_prompt, num_predict=48, timeout=240):
    """Executes a single controlled context trial with background hardware monitoring."""
    b_gpu_used, _, _ = read_gpu_stats()
    b_ram_used, _, b_sw_used, _ = read_mem_stats()
    baseline = {"gpu_used": b_gpu_used, "ram_used": b_ram_used, "sw_used": b_sw_used}

    prompt = build_stress_prompt(target_prompt)
    monitor = ContextMonitor(interval=0.2)
    monitor.start()
    t0 = time.time()
    rec = {
        "model": tag,
        "num_ctx": num_ctx,
        "target_prompt": target_prompt,
        "status": "UNKNOWN"
    }

    try:
        body = {
            "model": tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": 0.0
            }
        }
        res = ollama.api("/api/generate", body, timeout=timeout)
        wall_s = round(time.time() - t0, 2)
        monitor.stop()

        # Check GPU offload percentage from /api/ps
        ps_info = ollama.api("/api/ps").get("models", [])
        gpu_pct = None
        size_gb = None
        vram_gb = None
        for m in ps_info:
            if m.get("name") in (tag, tag + ":latest") or m.get("model") in (tag, tag + ":latest"):
                s, v = m.get("size", 0), m.get("size_vram", 0)
                if s > 0:
                    gpu_pct = round(100.0 * v / s, 1)
                    size_gb = round(s / 1e9, 2)
                    vram_gb = round(v / 1e9, 2)
                break

        sm = monitor.summary(baseline)
        p_count = res.get("prompt_eval_count", 0)
        p_dur_ns = res.get("prompt_eval_duration", 0)
        p_dur_s = round(p_dur_ns / 1e9, 2)
        p_tok_s = round(p_count / (p_dur_ns / 1e9), 1) if p_dur_ns > 0 else 0

        g_count = res.get("eval_count", 0)
        g_dur_ns = res.get("eval_duration", 0)
        g_dur_s = round(g_dur_ns / 1e9, 2)
        g_tok_s = round(g_count / (g_dur_ns / 1e9), 1) if g_dur_ns > 0 else 0

        rec.update({
            "status": "SUCCESS",
            "wall_s": wall_s,
            "prompt_tokens": p_count,
            "prompt_eval_s": p_dur_s,
            "prompt_tok_s": p_tok_s,
            "gen_tokens": g_count,
            "gen_eval_s": g_dur_s,
            "gen_tok_s": g_tok_s,
            "gpu_pct": gpu_pct,
            "size_gb": size_gb,
            "vram_gb": vram_gb,
            **sm
        })

        # Classification
        if sm["swap_delta_mb"] > 20:
            rec["classification"] = "SWAP_EXHAUSTION"
        elif gpu_pct is not None and gpu_pct < 99.0:
            rec["classification"] = "CPU_OFFLOAD_DEGRADATION"
        elif g_tok_s < 25.0:
            rec["classification"] = "GEN_SPEED_DEGRADATION"
        elif p_tok_s < 800.0:
            rec["classification"] = "PROMPT_SPEED_DEGRADATION"
        elif p_dur_s > 90.0:
            rec["classification"] = "LATENCY_UNACCEPTABLE"
        else:
            rec["classification"] = "PRACTICAL_ACCEPTABLE"

    except Exception as e:
        monitor.stop()
        rec.update({
            "status": "ERROR",
            "error": str(e),
            "classification": "HARD_FAILURE"
        })

    return rec

def optimize(tag, min_ctx=16384, max_ctx=262144, prompt_ratio=0.75, log=print):
    """
    Autonomous empirical context optimization loop (RSI).
    Searches for the maximum practical context size that maintains:
      - 100% GPU offload (no CPU spilling)
      - Zero swap thrashing (<20 MB swap delta)
      - Acceptable interactive prompt and generation speed
    Uses exponential expansion -> binary search -> high-stress validation.
    """
    with gpu_lock():
        log(f"=== Starting Empirical Context Optimization for {tag} ===")
        trials = []
        
        # 1. Baseline
        log(f"\n[Baseline] Testing min_ctx={min_ctx} at {int(prompt_ratio*100)}% prompt stress...")
        t_base = run_ctx_trial(tag, min_ctx, int(min_ctx * prompt_ratio))
        trials.append(t_base)
        if t_base["status"] != "SUCCESS" or t_base["classification"] != "PRACTICAL_ACCEPTABLE":
            log(f"Baseline {min_ctx} failed ({t_base.get('classification')}). Aborting.")
            return {"model": tag, "error": f"Baseline failed: {t_base.get('classification')}", "trials": trials}

        log(f"  Baseline OK: {t_base['prompt_tok_s']} tok/s prefill, {t_base['gen_tok_s']} tok/s gen, {t_base['gpu_pct']}% GPU")

        # 2. Coarse expansion (doubling)
        curr = min_ctx * 2
        safe_low = min_ctx
        viol_high = None

        while curr <= max_ctx:
            log(f"\n[Expansion] Testing num_ctx={curr}...")
            t = run_ctx_trial(tag, curr, int(curr * prompt_ratio))
            trials.append(t)
            if t["status"] == "SUCCESS" and t["classification"] == "PRACTICAL_ACCEPTABLE":
                log(f"  {curr} OK: {t['prompt_tok_s']} tok/s prefill, {t['gen_tok_s']} tok/s gen, VRAM {t.get('peak_gpu_mib')} MiB")
                safe_low = curr
                curr *= 2
            else:
                log(f"  {curr} VIOLATION: {t.get('classification')} (GPU: {t.get('gpu_pct')}%, Swap delta: {t.get('swap_delta_mb')} MB)")
                viol_high = curr
                break

        if viol_high is None:
            viol_high = curr

        # 3. Binary search refinement
        log(f"\n[Refinement] Binary search between {safe_low} and {viol_high}...")
        while viol_high - safe_low > 4096:
            mid = ((safe_low + viol_high) // 2 // 2048) * 2048
            log(f"  Bisecting at {mid}...")
            t_mid = run_ctx_trial(tag, mid, int(mid * prompt_ratio))
            trials.append(t_mid)
            if t_mid["status"] == "SUCCESS" and t_mid["classification"] == "PRACTICAL_ACCEPTABLE":
                log(f"    {mid} OK: {t_mid['gen_tok_s']} tok/s, GPU {t_mid.get('gpu_pct')}%, Swap delta {t_mid.get('swap_delta_mb')} MB")
                safe_low = mid
            else:
                log(f"    {mid} VIOLATION: {t_mid.get('classification')}")
                viol_high = mid

        # 4. High-stress validation at safe_low (85% prompt capacity)
        log(f"\n[Validation] High-stress verification at candidate {safe_low} (85% prompt load)...")
        t_val = run_ctx_trial(tag, safe_low, int(safe_low * 0.85), num_predict=64)
        trials.append(t_val)
        
        practical_max = safe_low
        if t_val["classification"] != "PRACTICAL_ACCEPTABLE":
            log(f"  Warning: {safe_low} showed {t_val['classification']} under 85% load. Backing off one step.")
            # Step down to previous power-of-two or 8k chunk
            practical_max = max(min_ctx, (safe_low * 7) // 8 // 4096 * 4096)
            log(f"  Revised practical maximum: {practical_max}")

        # 5. Production Recommendation with Safety Margin
        # Recommend ~10-15% safety buffer for GUI/multi-turn activations
        recommended_ctx = max(min_ctx, (int(practical_max * 0.85) // 8192) * 8192)
        if recommended_ctx == 0:
            recommended_ctx = min_ctx

        log("\n" + "=" * 60)
        log(f"OPTIMIZATION COMPLETE FOR {tag}")
        log(f"  Maximum Practical Context: {practical_max} tokens")
        log(f"  Boundary Violation Context: {viol_high} tokens")
        log(f"  Recommended Production Context: {recommended_ctx} tokens (with ~15% headroom)")
        log("=" * 60 + "\n")

        # Record optimal result in FIT store
        best_trial = next((tr for tr in reversed(trials) if tr.get("num_ctx") == practical_max and tr.get("status") == "SUCCESS"), t_base)
        fit_rec = {
            "model": tag,
            "num_ctx": practical_max,
            "recommended_ctx": recommended_ctx,
            "boundary_ctx": viol_high,
            "tok_s": best_trial.get("gen_tok_s"),
            "prompt_tok_s": best_trial.get("prompt_tok_s"),
            "fit": {
                "size_gb": best_trial.get("size_gb"),
                "gpu_pct": best_trial.get("gpu_pct"),
                "peak_vram_mib": best_trial.get("peak_gpu_mib")
            },
            "optimized": True
        }
        store.append(fit_rec, FIT)

        return {
            "model": tag,
            "max_practical_ctx": practical_max,
            "first_violation_ctx": viol_high,
            "recommended_ctx": recommended_ctx,
            "best_trial": best_trial,
            "trials": trials
        }

def apply_context(tag, num_ctx, config_path=None, log=print):
    """
    Applies the optimized context size across all project and user configurations:
      1. modelfiles/<tag>.Modelfile
      2. evals/default.toml (or specified config)
      3. ~/.pi/agent/models.json & config/pi-models.json
      4. ~/.config/opencode/opencode.json & config/opencode.json
      5. Re-creates the Ollama model tag with the updated Modelfile.
    """
    home = os.path.expanduser("~")
    root = store.ROOT

    # 1. Update Modelfile on disk
    modelfile_name = tag.replace(":", ".") + ".Modelfile"
    modelfile_path = os.path.join(root, "modelfiles", modelfile_name)
    if os.path.exists(modelfile_path):
        lines = open(modelfile_path).readlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.strip().startswith("PARAMETER num_ctx"):
                new_lines.append(f"PARAMETER num_ctx {num_ctx}\n")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(f"PARAMETER num_ctx {num_ctx}\n")
        open(modelfile_path, "w").writelines(new_lines)
        log(f"Updated {modelfile_path} -> PARAMETER num_ctx {num_ctx}")

    # 2. Update evals config (e.g. evals/default.toml)
    cfg_file = config_path or os.path.join(root, "evals", "default.toml")
    if os.path.exists(cfg_file):
        text = open(cfg_file).read()
        import re
        pattern = rf'(\[\[models\]\]\s*tag\s*=\s*"{re.escape(tag)}"[^\[]*?num_ctx\s*=\s*)\d+'
        if re.search(pattern, text, flags=re.DOTALL):
            new_text = re.sub(pattern, rf'\g<1>{num_ctx}', text, flags=re.DOTALL)
            open(cfg_file, "w").write(new_text)
            log(f"Updated {cfg_file} -> num_ctx = {num_ctx} for {tag}")

    # 3. Update Pi agent configuration (~/.pi/agent/models.json and config/pi-models.json)
    for pi_path in [os.path.join(home, ".pi/agent/models.json"), os.path.join(root, "config/pi-models.json")]:
        if os.path.exists(pi_path):
            try:
                c = json.load(open(pi_path))
                changed = False
                for prov in c.get("providers", {}).values():
                    for m in prov.get("models", []):
                        if m.get("id") == tag:
                            m["contextWindow"] = num_ctx
                            changed = True
                if changed:
                    json.dump(c, open(pi_path, "w"), indent=2)
                    open(pi_path, "a").write("\n")
                    log(f"Updated {pi_path} -> contextWindow = {num_ctx} for {tag}")
            except Exception as e:
                log(f"Warning: failed to update {pi_path}: {e}")

    # 4. Update Opencode configuration (~/.config/opencode/opencode.json and config/opencode.json)
    for oc_path in [os.path.join(home, ".config/opencode/opencode.json"), os.path.join(root, "config/opencode.json")]:
        if os.path.exists(oc_path):
            try:
                c = json.load(open(oc_path))
                models = c.get("provider", {}).get("ollama", {}).get("models", {})
                if tag in models:
                    models[tag].setdefault("limit", {})["context"] = num_ctx
                    json.dump(c, open(oc_path, "w"), indent=2)
                    open(oc_path, "a").write("\n")
                    log(f"Updated {oc_path} -> limit.context = {num_ctx} for {tag}")
            except Exception as e:
                log(f"Warning: failed to update {oc_path}: {e}")

    # 5. Re-create the Ollama model using the Modelfile
    if os.path.exists(modelfile_path):
        log(f"Re-creating ollama model {tag} from {modelfile_path}...")
        res = subprocess.run(["ollama", "create", tag, "-f", modelfile_path], capture_output=True, text=True)
        if res.returncode == 0:
            log(f"Successfully re-created {tag} in Ollama with num_ctx={num_ctx}!")
        else:
            log(f"Warning: ollama create returned error: {res.stderr.strip()}")
