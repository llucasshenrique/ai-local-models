"""Unit tests for the pure parts (no GPU, no ollama): python3 -m unittest discover -s tests"""
import json, os, tempfile, unittest
from llmeval import report, store, tasks as T
from llmeval.harnesses import Opencode, Minimal
from llmeval.ollama import modelfile_text

class TaskTests(unittest.TestCase):
    def test_every_task_is_solvable_and_starts_failing(self):
        for t in T.load(): self.assertTrue(T.selfcheck(t), t["id"])

    def test_tampering_with_the_test_is_not_a_pass(self):
        t = T.load(["03-fix-bug"])[0]
        d = T.materialize(t, {**t["solution"], "test.sh": "exit 0\n"})
        ok, tampered = T.verify(t, d)
        self.assertFalse(ok); self.assertTrue(tampered)

class StoreTests(unittest.TestCase):
    def test_invalidate_moves_rows_aside(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "runs.jsonl")
            for i in (1, 2): store.append({"harness": "h", "model": "m", "task": "t", "rep": i, "done": True}, p)
            self.assertEqual(store.invalidate(lambda r: r["rep"] == 2, "test", p), 1)
            self.assertEqual([r["rep"] for r in store.rows(p)], [1])

class ParserTests(unittest.TestCase):
    def test_opencode_events_give_metrics_and_call_keys(self):
        st = {}; h = Opencode()
        h.on_line(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 11000}}}), st)
        k = h.on_line(json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"status": "completed", "input": {"command": "ls"}}}}), st)
        self.assertEqual((st["llm_requests"], st["peak_prompt_tokens"], st["tool_calls"]), (1, 11000, 1)); self.assertIn("bash", k)

    def test_minimal_ignores_non_json_lines(self):
        self.assertIsNone(Minimal().on_line("warming up", {}))

class ReportTests(unittest.TestCase):
    def test_ranking_prefers_pass_rate_then_speed(self):
        rows = [dict(harness="a", model="slow", task="t", rep=1, done=True, wall_s=50), dict(harness="a", model="fast", task="t", rep=1, done=True, wall_s=5),
                dict(harness="a", model="bad", task="t", rep=1, done=False, wall_s=1)]
        self.assertEqual([s["model"] for s in report.summary(rows)], ["fast", "slow", "bad"])

class ModelfileTests(unittest.TestCase):
    def test_params_render(self):
        t = modelfile_text("x:1", {"temperature": 0.2, "stop": ["a", "b"]})
        self.assertEqual(t, "FROM x:1\nPARAMETER temperature 0.2\nPARAMETER stop a\nPARAMETER stop b\n")

if __name__ == "__main__": unittest.main()
