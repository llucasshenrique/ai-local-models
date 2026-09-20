#!/usr/bin/env python3
"""Loop smoke test: 3 tiny tasks run headless through a harness in throw-away git repos.

usage: bench/smoke.py opencode MODEL [task_id ...]   e.g. smoke.py opencode gemma4-agent:12b 1 2 3
       bench/smoke.py minimal MODEL [task_id ...]    (uses minimal_agent.py)
Prints one JSON line per task: done (test passes), wall seconds, tool calls, loop flag.
Loop = the same tool name + identical arguments issued 3 times (run is killed).
"""
import json, os, subprocess, sys, tempfile, time, threading

TIMEOUT = 180
TASKS = {
 1: dict(prompt="Create calc.py with a function add(a, b) returning a + b. Then run `sh test.sh` and make it pass.",
         files={"test.sh": 'python3 -c "from calc import add; assert add(2, 3) == 5"\n'}),
 2: dict(prompt="Rename the function `compute` to `total` in both a.py and b.py (definition and usage). Then run `sh test.sh` and make it pass.",
         files={"a.py": "def compute(x):\n    return x * 2\n", "b.py": "from a import compute\n\ndef run():\n    return compute(21)\n",
                "test.sh": 'python3 -c "import b; from a import total; assert b.run() == 42 and total(1) == 2"\n'}),
 3: dict(prompt="`sh test.sh` currently fails. Fix the bug in util.py (not the test) so it passes.",
         files={"util.py": "def is_even(n):\n    return n % 2 == 1\n", "test.sh": 'python3 -c "from util import is_even; assert is_even(4) and not is_even(7)"\n'}),
}

def setup(task):
    d = tempfile.mkdtemp(prefix="smoke-")
    for name, body in TASKS[task]["files"].items():
        open(os.path.join(d, name), "w").write(body)
    subprocess.run("git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm init", shell=True, cwd=d)
    return d

def run(harness, model, task):
    d = setup(task)
    if harness == "opencode":
        cmd = ["opencode", "run", "--format", "json", "--dir", d, "--agent", "micro", "-m", f"ollama/{model}", TASKS[task]["prompt"]]
    else:
        cmd = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness", "minimal_agent.py"), model, TASKS[task]["prompt"]]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=d, env={**os.environ, "PWD": d}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    calls, seen, loop = 0, {}, False
    def watchdog():
        time.sleep(TIMEOUT)
        if p.poll() is None: p.kill()
    threading.Thread(target=watchdog, daemon=True).start()
    dump = open(os.path.join(d, "..", os.path.basename(d) + ".events"), "w")
    for line in p.stdout:
        dump.write(line)
        try: ev = json.loads(line)
        except Exception: continue
        if harness == "minimal":
            if ev.get("call"):
                calls += 1; key = ev["call"]
            else: continue
        else:
            part = ev.get("part") or {}
            if ev.get("type") != "tool_use" or (part.get("state") or {}).get("status") not in ("completed", "error"): continue
            calls += 1; key = json.dumps([part.get("tool"), (part.get("state") or {}).get("input")], sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            loop = True; p.kill(); break
    p.wait()
    wall = round(time.time() - t0, 1)
    ok = subprocess.run("sh test.sh", shell=True, cwd=d, capture_output=True).returncode == 0
    return dict(harness=harness, model=model, task=task, done=ok, wall_s=wall, tool_calls=calls, loop=loop, timeout=wall >= TIMEOUT - 1)

if __name__ == "__main__":
    h, m = sys.argv[1], sys.argv[2]
    for t in ([int(x) for x in sys.argv[3:]] or [1, 2, 3]):
        print(json.dumps(run(h, m, t)), flush=True)
