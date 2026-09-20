"""Harness adapters. Each knows how to (1) register a model with its tool, (2) build the headless command, and
(3) optionally parse its event stream for tool calls (used for loop detection). Prompt/request metrics come from the
proxy for every harness except opencode, which reports token counts in its own JSON events."""
import json, os, shutil

HOME = os.path.expanduser("~")

def _backup_once(path):
    bak = path + ".bak-llmeval"
    if os.path.exists(path) and not os.path.exists(bak): shutil.copy2(path, bak)

class Harness:
    name = binary = ""; needs_proxy = True; experimental = False; detects_loops = False; note = ""
    def available(self): return shutil.which(self.binary) is not None
    def prepare(self, model, num_ctx): pass
    def build(self, c): raise NotImplementedError
    def on_line(self, line, st): return None          # -> identical-call key or None; may update st

class Opencode(Harness):
    name, binary, needs_proxy, detects_loops = "opencode", "opencode", False, True
    cfg = os.path.join(HOME, ".config/opencode/opencode.json")
    def prepare(self, model, num_ctx):
        c = json.load(open(self.cfg)); ms = c["provider"]["ollama"]["models"]
        changed = False
        if model not in ms:
            ms[model] = {"name": model, "tool_call": True, "limit": {"context": num_ctx, "output": 4096},
                         "options": {"temperature": 0.25, "top_p": 0.9, "presence_penalty": 0.3, "frequency_penalty": 0.1}}; changed = True
        elif ms[model].get("limit", {}).get("context") != num_ctx:
            ms[model].setdefault("limit", {})["context"] = num_ctx
            changed = True
        if "micro" not in c.get("agent", {}):
            c.setdefault("agent", {})["micro"] = {"mode": "primary", "maxSteps": 15, "permission": {"doom_loop": "deny", "edit": "allow", "bash": "allow"}}; changed = True
        if changed:
            _backup_once(self.cfg); json.dump(c, open(self.cfg, "w"), indent=2); open(self.cfg, "a").write("\n")
    def build(self, c):
        return ["opencode", "run", "--format", "json", "--dir", c.dir, "--agent", "micro", "-m", f"ollama/{c.model}", c.prompt], {}
    def on_line(self, line, st):
        try: ev = json.loads(line)
        except Exception: return None
        if not isinstance(ev, dict): return None
        part = ev.get("part") or {}
        if ev.get("type") == "step_finish" and part.get("tokens"):
            st["llm_requests"] = st.get("llm_requests", 0) + 1
            st["peak_prompt_tokens"] = max(st.get("peak_prompt_tokens", 0), part["tokens"].get("input", 0))
        if ev.get("type") == "tool_use" and (part.get("state") or {}).get("status") in ("completed", "error"):
            st["tool_calls"] = st.get("tool_calls", 0) + 1
            return json.dumps([part.get("tool"), part["state"].get("input")], sort_keys=True)

class Minimal(Harness):
    name, binary, detects_loops = "minimal", "python3", True
    note = "harness/minimal_agent.py: ~100-line ollama tool loop, 15 steps, repeat guard"
    def build(self, c):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness", "minimal_agent.py")
        return ["python3", path, c.model, c.prompt], {"OLLAMA_URL": c.proxy_url, "NUM_CTX": str(c.num_ctx)}
    def on_line(self, line, st):
        try: ev = json.loads(line)
        except Exception: return None
        if isinstance(ev, dict) and ev.get("call"):
            st["tool_calls"] = st.get("tool_calls", 0) + 1; return ev["call"]

class Mini(Harness):
    name, binary = "mini", "mini"
    note = "mini-swe-agent (pipx install mini-swe-agent)"
    def prepare(self, model, num_ctx):
        d = os.path.join(HOME, ".config/mini-swe-agent"); os.makedirs(d, exist_ok=True)
        e = os.path.join(d, ".env")
        if not os.path.exists(e): open(e, "w").write('MSWEA_CONFIGURED="true"\nMSWEA_COST_TRACKING="ignore_errors"\n')
    def build(self, c):
        return ["mini", "-m", f"ollama_chat/{c.model}", "-t", c.prompt, "-y", "--exit-immediately"], \
               {"OLLAMA_API_BASE": c.proxy_url, "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_SILENT_STARTUP": "1"}

class Aider(Harness):
    name, binary = "aider", "aider"
    note = "aider (pipx install aider-chat --python python3.12); gets the editable files + names of files to create"
    def build(self, c):
        files = [f for f in c.task["files"] if f not in c.task["protected"]] + [f for f in c.task.get("solution", {}) if f not in c.task["files"]]
        return ["aider", "--model", f"ollama_chat/{c.model}", "--message", c.prompt, "--yes-always", "--no-auto-commits", "--no-show-model-warnings",
                "--no-check-update", "--no-analytics", "--test-cmd", c.task["test"], "--auto-test"] + files, {"OLLAMA_API_BASE": c.proxy_url}

class Pi(Harness):
    name, binary = "pi", "pi"
    note = "Pi coding agent = native Orca agent (npm i -g @earendil-works/pi-coding-agent)"
    cfg = os.path.join(HOME, ".pi/agent/models.json")
    def prepare(self, model, num_ctx):
        c = json.load(open(self.cfg)) if os.path.exists(self.cfg) else {"providers": {}}
        changed = False
        for prov_name, base_url in [("ollama", "http://127.0.0.1:11434/v1"), ("ollama-proxy", "http://127.0.0.1:11436/v1")]:
            p = c["providers"].setdefault(prov_name, {"api": "openai-completions", "apiKey": "ollama", "baseUrl": base_url,
                "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False}, "models": []})
            for m in p["models"]:
                if m.get("id") == model:
                    if m.get("contextWindow") != num_ctx:
                        m["contextWindow"] = num_ctx
                        changed = True
                    break
            else:
                p["models"].append({"id": model, "contextWindow": num_ctx, "maxTokens": 4096, "input": ["text"]})
                changed = True
        if changed:
            os.makedirs(os.path.dirname(self.cfg), exist_ok=True); _backup_once(self.cfg); json.dump(c, open(self.cfg, "w"), indent=2)
    def build(self, c):
        return ["pi", "--provider", "ollama-proxy", "--model", c.model, "-p", c.prompt, "--no-session"], {}

class Launch(Harness):
    """Integrations started through `ollama launch` (they talk to the real server: no proxy metrics)."""
    needs_proxy, experimental, binary = False, True, "ollama"
    args = {}
    def build(self, c): return ["ollama", "launch", self.name, "--model", c.model, "--"] + [a.replace("{prompt}", c.prompt) for a in self.args], {}

class Qwen(Launch):   name, args, note = "qwen", ["-p", "{prompt}", "--yolo"], "Qwen Code via ollama launch (experimental, untested)"
class Codex(Launch):  name, args, note = "codex", ["exec", "--skip-git-repo-check", "{prompt}"], "Codex via ollama launch (experimental, untested)"
class Cline(Launch):  name, args, note = "cline", ["-y", "{prompt}"], "Cline via ollama launch (experimental, untested)"

REGISTRY = {h.name: h for h in (Opencode(), Minimal(), Mini(), Aider(), Pi(), Qwen(), Codex(), Cline())}
