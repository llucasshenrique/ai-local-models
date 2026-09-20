"""Families, recommendation rule and advisor validation (pure logic, no GPU): python3 -m unittest discover -s tests"""
import unittest
from unittest import mock
from llmeval import advisor, families as fam, report, store

class FamilyTests(unittest.TestCase):
    def test_parse_variant(self):
        self.assertEqual(fam.parse_variant("granite4.1:8b-q4_K_M"), ("8b", "q4_K_M"))
        self.assertEqual(fam.parse_variant("ornith:9b-q8_0"), ("9b", "q8_0"))
        self.assertEqual(fam.parse_variant("gemma4:12b"), ("12b", None))

    def test_expand_builds_size_x_quant_variants(self):
        ms = fam.expand({"name": "granite4.1", "sizes": ["3b", "8b"], "quants": ["q4_K_M", "q8_0"], "ctx": 32768})
        self.assertEqual(len(ms), 4)
        self.assertEqual(ms[3]["base"], "granite4.1:8b-q8_0"); self.assertEqual(ms[3]["tag"], "granite4.1-agent:8b-q8_0")
        self.assertEqual(ms[0]["params"]["num_ctx"], 32768); self.assertEqual((ms[0]["size"], ms[0]["quant"]), ("3b", "q4_K_M"))

    def test_as_is_keeps_tags_untouched(self):
        ms = fam.expand({"name": "ornith", "variants": ["ornith:9b-q4_K_M"], "as_is": True})
        self.assertEqual(ms[0]["tag"], "ornith:9b-q4_K_M"); self.assertNotIn("base", ms[0])

    def test_recommend_prefers_the_smallest_variant_within_tolerance(self):
        rows = [dict(tag="a", size="3b", rate=0.8, size_gb=2, gpu_pct=100), dict(tag="b", size="8b", rate=0.9, size_gb=5, gpu_pct=100),
                dict(tag="c", size="8b", rate=0.95, size_gb=9, gpu_pct=100)]
        self.assertEqual(fam.recommend(rows, 0.05)[0]["tag"], "b")           # 3b is 15 pts behind; 8b-b within 5 of the best 8b-c, and smaller
        self.assertEqual(fam.recommend(rows, 0.2)[0]["tag"], "a")            # a wider tolerance lets the 3b win

    def test_recommend_skips_variants_that_spill_to_cpu(self):
        rows = [dict(tag="big", size="14b", rate=1.0, size_gb=12, gpu_pct=70), dict(tag="ok", size="8b", rate=0.9, size_gb=6, gpu_pct=100)]
        pick, why = fam.recommend(rows); self.assertEqual(pick["tag"], "ok")
        self.assertIn("NOTE", fam.recommend([rows[0]])[1])                   # nothing fits: still answers, but says so

    def test_report_joins_results_with_fit(self):
        runs = [dict(harness="h", model="g-agent:8b-q4", family="g", size="8b", quant="q4", task="t", rep=i, done=True, wall_s=10, num_ctx=16384) for i in (1, 2)]
        fits = [dict(model="g-agent:8b-q4", num_ctx=16384, tok_s=70.0, fit={"size_gb": 5.3, "gpu_pct": 100})]
        with mock.patch.object(store, "rows", side_effect=lambda p=store.RUNS: fits if p != store.RUNS else runs):
            (name, rows, pick, why), = report.family_table()
        self.assertEqual((name, pick["tag"], rows[0]["tok_s"], rows[0]["gpu_pct"]), ("g", "g-agent:8b-q4", 70.0, 100))

class AdvisorTests(unittest.TestCase):
    def test_sanitize_keeps_only_safe_experiments(self):
        advice = {"analysis": "x", "experiments": [
            {"model": "m1", "changes": {"temperature": 0.4, "top_k": "30", "num_predict": 99999, "top_p": 5}, "why": "w"},
            {"model": "unknown", "changes": {"temperature": 0.1}}, {"model": "m1", "changes": {"repeat_penalty": "abc"}}], "ideas": ["i"]}
        clean, dropped = advisor.sanitize(advice, {"m1"})
        self.assertEqual(clean["experiments"], [{"model": "m1", "changes": {"temperature": 0.4, "top_k": 30}, "why": "w"}])
        self.assertTrue(any("num_predict" in d for d in dropped) and any("top_p" in d for d in dropped) and any("unknown" in d for d in dropped))

    def test_parse_json_tolerates_prose_around_the_object(self):
        self.assertEqual(advisor.parse_json('Sure! {"analysis": "ok"} hope that helps'), {"analysis": "ok"})

    def test_space_from_advice_merges_values_per_parameter(self):
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(advisor, "ADVICE", os.path.join(d, "advice.json")):
            json.dump({"advice": {"experiments": [{"model": "m", "changes": {"temperature": 0.1}}, {"model": "m", "changes": {"temperature": 0.4, "top_k": 30}},
                                                  {"model": "other", "changes": {"top_p": 0.8}}]}}, open(advisor.ADVICE, "w"))
            self.assertEqual(advisor.space_from_advice("m"), {"temperature": [0.1, 0.4], "top_k": [30]})

if __name__ == "__main__": unittest.main()
