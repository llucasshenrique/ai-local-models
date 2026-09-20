"""Loop checkpoint and state caching for resilient, resumable optimization loops."""
import json, os
from datetime import datetime, timezone
from . import store

def state_path(results_dir=None):
    base = results_dir or store.RESULTS
    return os.path.join(base, ".loop_state.json")

def load_state(results_dir=None):
    p = state_path(results_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(state, results_dir=None):
    p = state_path(results_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = p + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)

def clear_state(results_dir=None):
    p = state_path(results_dir)
    if os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass

def has_checkpoint(family=None, results_dir=None):
    s = load_state(results_dir)
    if not s:
        return False
    if s.get("status") == "completed":
        return False
    if family and s.get("family") != family:
        return False
    return True

def get_checkpoint_summary(results_dir=None):
    s = load_state(results_dir)
    if not s:
        return "No checkpoint found."
    fam = s.get("family", "unknown")
    stage = s.get("current_stage", "start")
    cycle = s.get("current_cycle", 1)
    max_c = s.get("max_cycles", 3)
    status = s.get("status", "in_progress")
    err = f" (Last error: {s['last_error'][:50]})" if s.get("last_error") else ""
    return f"Family '{fam}' at stage '{stage}' (Cycle {cycle}/{max_c}, status: {status}){err}"

class LoopState:
    def __init__(self, family, policy="balanced", max_cycles=3, target_ctx=0, results_dir=None):
        self.results_dir = results_dir
        self.state = {
            "family": family,
            "policy": policy,
            "max_cycles": max_cycles,
            "target_ctx": target_ctx,
            "current_stage": "discovery",
            "current_cycle": 1,
            "status": "in_progress",
            "last_error": None,
            "stages": {
                "discovery": {"completed": False, "plan": None},
                "hardware_envelope": {"completed": False, "hardware_max_safe": 32768},
                "context_discovery": {"completed": False, "best_ctx": None, "candidates": []},
                "rsi_cycles": {"completed": False, "history": []},
                "pareto_decision": {"completed": False, "winner": None}
            }
        }

    @classmethod
    def init_or_resume(cls, family, policy="balanced", max_cycles=3, target_ctx=0, action="resume", results_dir=None):
        """
        action:
          - 'resume': continue from last checkpoint if available
          - 'restart_step': keep earlier completed stages, but reset the current stage
          - 'reset_all': clear checkpoint and start from scratch
        """
        existing = load_state(results_dir)
        if action == "reset_all" or not existing:
            clear_state(results_dir)
            inst = cls(family, policy, max_cycles, target_ctx, results_dir)
            inst.save()
            return inst

        # If existing belongs to another family and a specific family is requested, start fresh for that family
        if family and existing.get("family") and existing.get("family") != family:
            clear_state(results_dir)
            inst = cls(family, policy, max_cycles, target_ctx, results_dir)
            inst.save()
            return inst

        inst = cls(existing.get("family", family),
                   existing.get("policy", policy),
                   existing.get("max_cycles", max_cycles),
                   existing.get("target_ctx", target_ctx),
                   results_dir)
        inst.state = existing
        inst.state["status"] = "in_progress"
        inst.state["last_error"] = None

        if action == "restart_step":
            curr = inst.state.get("current_stage", "discovery")
            if curr in inst.state.get("stages", {}):
                inst.state["stages"][curr]["completed"] = False
                if curr == "rsi_cycles":
                    cyc = inst.state.get("current_cycle", 1)
                    inst.state["stages"]["rsi_cycles"]["history"] = [
                        h for h in inst.state["stages"]["rsi_cycles"].get("history", []) if h.get("cycle", 0) < cyc
                    ]

        inst.save()
        return inst

    def save(self):
        save_state(self.state, self.results_dir)

    def is_stage_completed(self, stage_name):
        return bool(self.state.get("stages", {}).get(stage_name, {}).get("completed"))

    def get_stage_data(self, stage_name, key=None):
        stage = self.state.get("stages", {}).get(stage_name, {})
        if key:
            return stage.get(key)
        return stage

    def set_stage_start(self, stage_name):
        self.state["current_stage"] = stage_name
        self.state["status"] = "in_progress"
        self.save()

    def set_stage_complete(self, stage_name, **kwargs):
        if stage_name not in self.state["stages"]:
            self.state["stages"][stage_name] = {}
        self.state["stages"][stage_name]["completed"] = True
        self.state["stages"][stage_name]["completed_at"] = datetime.now(timezone.utc).isoformat()
        for k, v in kwargs.items():
            self.state["stages"][stage_name][k] = v
        self.save()

    def set_cycle_start(self, cycle_num):
        self.state["current_cycle"] = cycle_num
        self.state["current_stage"] = "rsi_cycles"
        self.save()

    def record_cycle_result(self, cycle_num, winner=None, confirmed=False, note=""):
        hist = self.state["stages"]["rsi_cycles"].setdefault("history", [])
        entry = {
            "cycle": cycle_num,
            "winner": winner,
            "confirmed": confirmed,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        existing_idx = next((i for i, h in enumerate(hist) if h.get("cycle") == cycle_num), None)
        if existing_idx is not None:
            hist[existing_idx] = entry
        else:
            hist.append(entry)
        self.save()

    def get_completed_cycles(self):
        return [h.get("cycle") for h in self.state.get("stages", {}).get("rsi_cycles", {}).get("history", []) if h.get("confirmed") or h.get("winner")]

    def set_error(self, err_msg):
        self.state["status"] = "failed"
        self.state["last_error"] = str(err_msg)
        self.save()

    def mark_all_completed(self):
        self.state["status"] = "completed"
        self.state["current_stage"] = "completed"
        self.save()
