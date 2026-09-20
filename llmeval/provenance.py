"""Provenance and reproducibility metadata for llmeval experiments."""
import hashlib
import json
import os
import platform
import subprocess
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_repo_commit():
    """Git commit of the repository, or None if not a git repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None

def get_hardware_info():
    """Hardware environment: GPU, driver, CPU, total RAM. Non-fabricating (returns None for unobserved values)."""
    gpu_name, gpu_vram, driver = None, None, None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                try: gpu_vram = int(parts[1])
                except ValueError: gpu_vram = None
                driver = parts[2]
    except Exception:
        pass

    cpu_model = None
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        if not cpu_model:
            cpu_model = platform.processor() or None
    except Exception:
        pass

    ram_mb = None
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_mb = int(line.split()[1]) // 1024
                        break
    except Exception:
        pass

    return {
        "gpu": gpu_name,
        "gpu_vram_mib": gpu_vram,
        "driver": driver,
        "cpu": cpu_model,
        "ram_mb": ram_mb
    }

def get_os_info():
    """OS and kernel information."""
    return {
        "system": platform.system() or None,
        "release": platform.release() or None,
        "platform": platform.platform() or None
    }

def compute_config_digest(effective_params):
    """Deterministic hash for effective parameters."""
    normalized = json.dumps(effective_params or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def compute_task_version(task_dict):
    """Deterministic hash of task definition: prompt, files, test, protected."""
    core = {
        "id": task_dict.get("id"),
        "prompt": task_dict.get("prompt"),
        "files": task_dict.get("files", {}),
        "test": task_dict.get("test"),
        "protected": sorted(task_dict.get("protected", []))
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

def compute_benchmark_version(tasks=None):
    """Overall benchmark version hash covering all tasks and evaluator rules."""
    if tasks is None:
        from . import tasks as T
        tasks = T.load()
    task_hashes = sorted(compute_task_version(t) for t in tasks)
    raw = "|".join(task_hashes)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def compute_harness_version(harness_name):
    """Hash of the harness implementation file if local, or a fixed identifier."""
    harness_path = os.path.join(ROOT, "harness", f"{harness_name}_agent.py")
    if os.path.exists(harness_path):
        content = open(harness_path, "rb").read()
        return hashlib.sha256(content).hexdigest()[:16]
    return "external"

def generate_run_id():
    """Generate unique run identifier."""
    return str(uuid.uuid4())

def check_comparability(rec1, rec2):
    """
    Determines whether two trial runs are comparable for paired evaluation.
    Returns (bool, list_of_reasons).
    """
    reasons = []
    if rec1.get("task") != rec2.get("task"):
        reasons.append(f"Different tasks: '{rec1.get('task')}' vs '{rec2.get('task')}'")
    
    t1_ver = rec1.get("task_version")
    t2_ver = rec2.get("task_version")
    if t1_ver and t2_ver and t1_ver != t2_ver:
        reasons.append(f"Task versions differ: {t1_ver} vs {t2_ver}")

    bm1 = rec1.get("benchmark_version")
    bm2 = rec2.get("benchmark_version")
    if bm1 and bm2 and bm1 != bm2:
        reasons.append(f"Benchmark versions differ: {bm1} vs {bm2}")

    h1 = rec1.get("harness")
    h2 = rec2.get("harness")
    if h1 != h2:
        reasons.append(f"Different harnesses: '{h1}' vs '{h2}'")

    hv1 = rec1.get("harness_version")
    hv2 = rec2.get("harness_version")
    if hv1 and hv2 and hv1 != hv2:
        reasons.append(f"Harness versions differ: {hv1} vs {hv2}")

    return len(reasons) == 0, reasons

def is_stale(rec, current_benchmark_ver=None, current_task_versions=None):
    """
    Checks if a result is stale compared to the current benchmark/tasks.
    Returns (bool, reason).
    """
    if current_benchmark_ver and rec.get("benchmark_version"):
        if rec.get("benchmark_version") != current_benchmark_ver:
            return True, f"Benchmark version changed ({rec.get('benchmark_version')} -> {current_benchmark_ver})"

    if current_task_versions and rec.get("task") in current_task_versions:
        expected = current_task_versions[rec.get("task")]
        actual = rec.get("task_version")
        if actual and actual != expected:
            return True, f"Task '{rec.get('task')}' definition changed ({actual} -> {expected})"

    return False, ""
