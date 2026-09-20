"""Discovery parsers, feasibility, choice validation, prune planning and config editing (offline, no GPU, nothing deleted)."""
import os, tempfile, unittest
from unittest import mock
from llmeval import discover as dc, prune

HTML = '''
<a href="/library/g:8b" class="x"><span>g:8b</span></a><span class="font-mono">aaaaaaaaaaaa</span> • 5.3GB • 128K
<a href="/library/g:8b-q4_K_M"><span>g:8b-q4_K_M</span></a><span class="font-mono">aaaaaaaaaaaa</span> • 5.3GB • 128K
<a href="/library/g:8b-q8_0"><span>g:8b-q8_0</span></a><span class="font-mono">bbbbbbbbbbbb</span> • 9.3GB • 128K
<a href="/library/g:latest"><span>g:latest</span></a><span class="font-mono">aaaaaaaaaaaa</span> • 5.3GB
<a href="/library/g:cloud"><span>g:cloud</span></a><span class="font-mono">cccccccccccc</span> • 1GB
<a href="/library/g:3b-q4_K_M"><span>g:3b</span></a><span class="font-mono">dddddddddddd</span> • 900MB'''

class ParserTests(unittest.TestCase):
    def test_ollama_tags_dedupe_aliases_and_skip_cloud(self):
        tags = {c["tag"]: c["size_gb"] for c in dc.parse_ollama_tags(HTML, "g")}
        self.assertEqual(tags, {"g:8b-q4_K_M": 5.3, "g:8b-q8_0": 9.3, "g:3b-q4_K_M": 0.88})     # alias 8b/latest folded, cloud dropped, MB converted

    def test_hf_repo_becomes_ollama_pullable_tags(self):
        info = {"siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 5_000_000_000}, {"rfilename": "m-Q8_0-00001-of-00002.gguf", "size": 1},
                             {"rfilename": "mmproj-f16.gguf", "size": 1}, {"rfilename": "README.md", "size": 1}]}
        out = dc.parse_hf("org/Granite-4.1-8B-GGUF", info, 100)
        self.assertEqual([(c["tag"], c["size_gb"]) for c in out], [("hf.co/org/Granite-4.1-8B-GGUF:Q4_K_M", 5.0)])

class FeasibilityTests(unittest.TestCase):
    def test_tiers_use_weights_plus_overhead(self):
        c = [{"tag": "g:3b-q4_K_M", "size_gb": 2.0}, {"tag": "g:8b-q4_K_M", "size_gb": 5.3}, {"tag": "g:8b-q8_0", "size_gb": 9.3}, {"tag": "g:30b-q4_K_M", "size_gb": 17.0}]
        with mock.patch.object(dc, "overhead_gb", return_value=2.6):
            ok, oh = dc.tier(c, {"vram_gb": 11.0}, 16384)
        self.assertEqual({x["tag"]: x["tier"] for x in ok}, {"g:3b-q4_K_M": "likely", "g:8b-q4_K_M": "likely", "g:8b-q8_0": "risky"})   # 30b is dropped

    def test_heuristic_takes_top_size_then_quant_neighbour_then_one_size_down(self):
        c = [{"tag": t, "size": s, "quant": q, "tier": "likely"} for t, s, q in
             (("a", "8b", "q4_K_M"), ("b", "8b", "q6_K"), ("c", "3b", "q4_K_M"), ("d", "3b", "q8_0"))]
        self.assertEqual([x["tag"] for x in dc.heuristic(c, 3)], ["a", "b", "c"])

    def test_choose_ignores_invented_ids_and_falls_back(self):
        c = [{"tag": "a", "size": "8b", "quant": "q4_K_M", "size_gb": 5, "est_total_gb": 7, "tier": "likely", "source": "ollama"}]
        cfg = {"models": []}
        with mock.patch.object(dc.advisor, "pick_advisor", return_value=("m", "x")), mock.patch.object(dc.advisor, "ask", return_value='{"choices":[{"id":99,"why":"x"}]}'), \
             mock.patch.object(dc.ollama, "unload_all"), mock.patch.object(dc.ollama, "stop"), mock.patch.object(dc, "report") as rep:
            rep.summary.return_value = []; rep.family_table.return_value = []
            chosen, notes, adv = dc.choose(cfg, "g", c, {}, 16384, 2.6, 3, log=lambda *_: None)
        self.assertEqual([x["tag"] for x in chosen], ["a"]); self.assertIsNone(adv)          # id 99 does not exist -> heuristic

class DiscoveryTop10Tests(unittest.TestCase):
    def test_top_10_candidates_formatting(self):
        cands = [
            {"tag": "qwen2.5-coder:14b-q4_K_M", "size": "14b", "quant": "q4_K_M", "size_gb": 9.0, "est_total_gb": 11.6,
             "tier": "likely", "popularity": "21.6M pulls", "age": "1 year ago", "source": "ollama"},
            {"tag": "hf.co/org/qwen:Q4_K_M", "size": "7b", "quant": "Q4_K_M", "size_gb": 4.5, "est_total_gb": 7.1,
             "tier": "likely", "popularity": "100K dl", "age": "2024-10-01", "source": "huggingface"},
        ]
        lines = dc.format_top_candidates(cands)
        joined = "\n".join(lines)
        self.assertIn("Top 10 Candidate Models for Selection", joined)
        self.assertIn("qwen2.5-coder:14b-q4_K_M", joined)
        self.assertIn("21.6M pulls", joined)
        self.assertIn("1 year ago", joined)
        self.assertIn("100K dl", joined)
        self.assertIn("huggingface", joined)

    def test_candidate_score_prefers_popular_and_fresh(self):
        base = {"size": "7b", "quant": "q4_K_M", "tier": "likely", "source": "ollama"}
        c_fresh_pop = {**base, "popularity_num": 5_000_000, "age_days": 30}
        c_old_unpop = {**base, "popularity_num": 10, "age_days": 600}
        self.assertGreater(dc.candidate_score(c_fresh_pop), dc.candidate_score(c_old_unpop))

    def test_format_plan_includes_top_candidates(self):
        plan = {
            "name": "qwen", "candidates": 15, "feasible": 8, "ctx": 32768,
            "machine": {"gpu": "RTX 4090", "vram_gb": 24.0, "ram_gb": 64.0},
            "overhead_gb": 2.6, "advisor": "heuristic", "chosen": [{"tag": "qwen:7b", "size": "7b", "quant": "q4_K_M", "size_gb": 4.5, "tier": "likely", "source": "ollama", "why": "ok"}],
            "notes": "", "download_gb": 4.5,
            "top_candidates": [{"tag": "qwen:7b", "size": "7b", "quant": "q4_K_M", "size_gb": 4.5, "est_total_gb": 7.1, "tier": "likely", "popularity": "1M pulls", "age": "2 months ago", "source": "ollama"}]
        }
        lines = dc.format_plan(plan)
        joined = "\n".join(lines)
        self.assertIn("Top 10 Candidate Models for Selection", joined)
        self.assertIn("1M pulls", joined)

class PruneTests(unittest.TestCase):
    def test_drop_config_removes_only_the_named_model_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.toml")
            open(p, "w").write('[run]\nreps = 1\n\n[[models]]\ntag = "keep"\n\n[[models]]\ntag  = "gone"\nbase = "x"\n[models.params]\nnum_ctx = 1\n\n[[families]]\nname = "f"\n')
            self.assertEqual(prune.drop_config(p, {"gone"}), ["gone"])
            t = open(p).read()
            self.assertIn('tag = "keep"', t); self.assertNotIn("gone", t); self.assertNotIn("num_ctx = 1", t); self.assertIn('name = "f"', t)

    def test_plan_protects_winners_unproven_and_defaults_and_never_deletes_in_a_dry_run(self):
        rank = [("best", 9, 9, 10 / 11, 5.0), ("second", 9, 9, 10 / 11, 6.0), ("third", 9, 8, 9 / 11, 7.0), ("young", 3, 0, 0.2, 9.0), ("loser", 9, 1, 2 / 11, 9.0), ("fourth", 9, 7, 8 / 11, 8.0)]
        inst = {"models": [{"name": f"{t}:latest", "size": 5e9} for t, *_ in rank]}
        with mock.patch.object(prune, "ranking", return_value=rank), mock.patch.object(prune.ollama, "api", return_value=inst), \
             mock.patch.object(prune.report, "family_table", return_value=[]), mock.patch.object(prune, "external_defaults", return_value={"loser"}), \
             mock.patch.object(prune.store, "rows", return_value=[]), mock.patch("subprocess.run") as run:
            p = prune.plan({"models": []}, keep_top=3, min_n=6)
        # best/second/third: top 3; young: too few trials; loser: protected because it is a configured default;
        # fourth: 7/9 is 18 points below the leader (margin 15) and outside the top 3 -> the only real candidate
        self.assertEqual([d["tag"] for d in p["delete"]], ["fourth"])
        self.assertIn("loser:latest", p["protected"]); self.assertIn("young:latest", p["protected"]); run.assert_not_called()

if __name__ == "__main__": unittest.main()
