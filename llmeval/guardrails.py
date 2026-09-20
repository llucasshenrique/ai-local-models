"""RSI Integrity Boundaries: ensures the subject under optimization cannot alter the benchmark ruler."""
import glob
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHMARK_INFRASTRUCTURE_FILES = [
    "llmeval/tasks.py",
    "llmeval/runner.py",
    "llmeval/store.py",
    "llmeval/stats.py",
    "llmeval/pareto.py",
    "llmeval/provenance.py",
    "llmeval/sandbox.py",
    "llmeval/guardrails.py",
    "llmeval/report.py",
    "evals/default.toml",
    "SPEC.md",
    "RSI.md"
]

class BenchmarkTamperedError(RuntimeError):
    """Raised when an optimization attempt mutates benchmark infrastructure."""
    pass

def _file_hash(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def snapshot_benchmark():
    """Captures cryptographic hashes of all benchmark infrastructure files and task definitions."""
    hashes = {}
    for rel_path in BENCHMARK_INFRASTRUCTURE_FILES:
        full_path = os.path.join(ROOT, rel_path)
        if os.path.exists(full_path):
            hashes[rel_path] = _file_hash(full_path)

    # All task definitions in tasks/
    for t_path in glob.glob(os.path.join(ROOT, "tasks", "*.json")):
        rel = os.path.relpath(t_path, ROOT)
        hashes[rel] = _file_hash(t_path)

    # All test files
    for tst_path in glob.glob(os.path.join(ROOT, "tests", "*.py")):
        rel = os.path.relpath(tst_path, ROOT)
        hashes[rel] = _file_hash(tst_path)

    return hashes

def verify_benchmark_integrity(snapshot):
    """
    Verifies that no benchmark infrastructure or evaluation task has been modified since snapshot.
    Returns (bool, list_of_violations).
    """
    violations = []
    for rel_path, expected_hash in snapshot.items():
        full_path = os.path.join(ROOT, rel_path)
        current_hash = _file_hash(full_path)
        if current_hash != expected_hash:
            violations.append(f"Benchmark file '{rel_path}' was modified or missing (expected {expected_hash[:8]}, got {str(current_hash)[:8]})")

    return len(violations) == 0, violations

def assert_immutable_benchmark(snapshot):
    """Raises BenchmarkTamperedError if any benchmark infrastructure file was modified."""
    ok, violations = verify_benchmark_integrity(snapshot)
    if not ok:
        raise BenchmarkTamperedError("RSI boundary violation: Benchmark ruler was modified!\n" + "\n".join(violations))
