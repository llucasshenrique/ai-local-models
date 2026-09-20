"""Tasks are data: tasks/*.json = {id, title, prompt, files, test, protected, solution}. `selfcheck` proves each is
solvable (test fails on the starting files, passes once `solution` is applied)."""
import glob, json, os, shutil
from . import provenance, sandbox

TASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")

def load(ids=None, task_dir=TASK_DIR, split=None, split_name=None):
    """Loads tasks and computes cryptographic version hashes. Optionally filters by split."""
    tasks = []
    for p in sorted(glob.glob(os.path.join(task_dir, "*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            t = json.load(f)
        t.setdefault("test", "sh test.sh")
        t.setdefault("protected", ["test.sh"])
        t["version"] = provenance.compute_task_version(t)
        tasks.append(t)

    if split and split_name:
        allowed_ids = set(split.get(split_name) or [])
        if allowed_ids:
            tasks = [t for t in tasks if t["id"] in allowed_ids]

    if ids:
        allowed = set(ids)
        tasks = [t for t in tasks if t["id"] in allowed]

    return tasks

def materialize(task, files=None):
    """Fresh throw-away isolated workspace holding starting files with read-only protected files."""
    return sandbox.create_isolated_workspace(task, files_override=files)

def run_test(cmd, d, timeout=60):
    """Executes test command within isolated process group and resource bounds."""
    code, _ = sandbox.run_bounded_process(cmd, cwd=d, timeout=timeout)
    return code

def verify(task, d, timeout=60):
    """(passed, tampered): tests must pass AND the protected files must be byte-identical to the originals."""
    tampered = False
    for f in task["protected"]:
        target_path = os.path.join(d, f)
        if not os.path.exists(target_path):
            tampered = True
            break
        with open(target_path, "r", encoding="utf-8") as file_obj:
            if file_obj.read() != task["files"].get(f):
                tampered = True
                break

    ok = (run_test(task["test"], d, timeout) == 0)
    return (ok and not tampered), tampered

def selfcheck(task):
    d = materialize(task)
    try:
        before = run_test(task["test"], d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    d = materialize(task, task.get("solution", {}))
    try:
        after, _ = verify(task, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    return before != 0 and after
