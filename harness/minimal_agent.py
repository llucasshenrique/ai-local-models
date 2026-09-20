#!/usr/bin/env python3
"""Minimal tool-loop harness for small local models (~100 lines, stdlib only).

usage: minimal_agent.py MODEL "task prompt"   (run inside the target repo directory)
Talks to ollama /api/chat with tools read_file / write_file / run_shell, max 15 steps,
aborts when the same call (tool + args) is repeated 3 times. Emits one JSON line per
call ({"call": ...}) so smoke.py can count them, and a final {"final": ...} line.
Env: OLLAMA_URL (default http://127.0.0.1:11434), NUM_CTX (default 16384).
"""
import json, os, subprocess, sys, urllib.request

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MAX_STEPS, MAX_REPEAT, OUT_CAP = 15, 3, 4000
SYSTEM = ("You are a coding agent working in the current directory. Use the tools to finish the task. "
          "Run the test command to verify. Be brief. When the test passes, reply with DONE. "
          "Never skip verification. If your code changes anything, run tests and confirm before declaring done.")
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": props}}} for n, d, props in [
    ("read_file", "Read a text file", ["path"]),
    ("write_file", "Write (overwrite) a text file with full content", ["path", "content"]),
    ("run_shell", "Run a shell command and return its output", ["command"])]]

def call_tool(name, a):
    try:
        if name == "read_file":
            return open(a["path"]).read()[:OUT_CAP]
        if name == "write_file":
            open(a["path"], "w").write(a["content"]); return "ok"
        if name == "run_shell":
            r = subprocess.run(a["command"], shell=True, capture_output=True, text=True, timeout=60)
            return (f"exit={r.returncode}\n" + r.stdout + r.stderr)[:OUT_CAP]
        return f"unknown tool {name}"
    except Exception as e:
        return f"error: {e}"

def chat(model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False, "think": False,
            "options": {"num_ctx": int(os.environ.get("NUM_CTX", 16384))}}
    req = urllib.request.Request(URL + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=170))["message"]

def main(model, task):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    seen = {}
    for _ in range(MAX_STEPS):
        msg = chat(model, messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            print(json.dumps({"final": msg.get("content", "")[:200]}), flush=True); return
        for c in calls:
            name, args = c["function"]["name"], c["function"]["arguments"]
            key = json.dumps([name, args], sort_keys=True)
            print(json.dumps({"call": key}), flush=True)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= MAX_REPEAT:
                print(json.dumps({"final": "ABORT repeated call"}), flush=True); return
            messages.append({"role": "tool", "tool_name": name, "content": call_tool(name, args)})
    print(json.dumps({"final": "ABORT max steps"}), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
