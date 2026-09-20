"""Tasks are data: tasks/*.json = {id, title, prompt, files, test, protected, solution}. `selfcheck` proves each is
solvable (test fails on the starting files, passes once `solution` is applied)."""
import glob, json, os, subprocess, tempfile

TASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")

def load(ids=None, task_dir=TASK_DIR):
    tasks = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(task_dir, "*.json")))]
    for t in tasks:
        t.setdefault("test", "sh test.sh"); t.setdefault("protected", ["test.sh"])
    return [t for t in tasks if not ids or t["id"] in ids]

def materialize(task, files=None):
    """Fresh throw-away git repo holding the task's starting files (plus `files` overrides, used by selfcheck)."""
    d = tempfile.mkdtemp(prefix="llmeval-")
    for name, body in {**task["files"], **(files or {})}.items():
        os.makedirs(os.path.dirname(os.path.join(d, name)) or d, exist_ok=True)
        open(os.path.join(d, name), "w").write(body)
    subprocess.run("git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm init", shell=True, cwd=d)
    return d

def verify(task, d):
    """(passed, tampered): tests must pass AND the protected files must be byte-identical to the originals."""
    tampered = any(open(os.path.join(d, f)).read() != task["files"][f] if os.path.exists(os.path.join(d, f)) else True
                   for f in task["protected"])
    ok = subprocess.run(task["test"], shell=True, cwd=d, capture_output=True, timeout=60).returncode == 0
    return ok and not tampered, tampered

def selfcheck(task):
    import shutil
    d = materialize(task); before = subprocess.run(task["test"], shell=True, cwd=d, capture_output=True).returncode; shutil.rmtree(d)
    d = materialize(task, task.get("solution", {})); after, _ = verify(task, d); shutil.rmtree(d)
    return before != 0 and after
