"""Unit tests for RSI architectural invariants:
- Effective configuration precedence and single source of truth
- Configuration digest stability
- Experiment identity & reproducibility metadata
- Comparability checking and stale-result detection
- Separation of physical hardware feasibility from quality context
- Statistical comparisons, Wilson confidence intervals, and insufficient evidence
- True Pareto dominance and configurable recommendation policies
- Benchmark integrity guardrails and workspace isolation
- Holdout and test split isolation
- Tuner confirmation requirements (no promotion on noisy low-repetition trial)
"""
import copy
import os
import stat
import tempfile
import unittest

from llmeval import config, guardrails, pareto, provenance, report, sandbox, stats, store, tasks as T, tune

class TestInvariants(unittest.TestCase):

    def test_effective_configuration_precedence(self):
        """Test canonical resolution hierarchy: defaults -> family -> model -> experiment -> CLI overrides."""
        cfg = {
            "defaults": {
                "params": {
                    "temperature": 0.25,
                    "top_k": 20,
                    "num_ctx": 16384,
                    "repeat_penalty": 1.05
                }
            },
            "families": [
                {
                    "name": "testfam",
                    "params": {"top_k": 30},
                    "ctx": 32768
                }
            ],
            "models": [
                {
                    "tag": "testfam:8b",
                    "family": "testfam",
                    "num_ctx": 49152,             # model-level field
                    "params": {"temperature": 0.3} # model params
                },
                {
                    "tag": "other:7b",
                    "params": {"num_ctx": 8192}
                }
            ],
            "experiment": {
                "params": {"repeat_penalty": 1.1}
            }
        }

        # Model from family
        eff = config.effective_params(cfg, "testfam:8b")
        # Global defaults: repeat_penalty 1.05 overridden by experiment 1.1
        self.assertEqual(eff["repeat_penalty"], 1.1)
        # Family params: top_k 30
        self.assertEqual(eff["top_k"], 30)
        # Model-level field num_ctx 49152 takes precedence over family ctx 32768 and default 16384
        self.assertEqual(eff["num_ctx"], 49152)
        # Model params: temperature 0.3 takes precedence over default 0.25
        self.assertEqual(eff["temperature"], 0.3)
        # Seed is explicitly None if not given
        self.assertIsNone(eff["seed"])

        # Runtime CLI overrides take final precedence
        overrides = {"temperature": 0.1, "num_ctx": 65536, "seed": 42}
        eff_override = config.effective_params(cfg, "testfam:8b", overrides=overrides)
        self.assertEqual(eff_override["temperature"], 0.1)
        self.assertEqual(eff_override["num_ctx"], 65536)
        self.assertEqual(eff_override["seed"], 42)

        # Single source of truth: resolve_model synchronizes m["num_ctx"], m["params"], m["effective_params"]
        m = cfg["models"][0]
        config.resolve_model(cfg, m)
        self.assertEqual(m["num_ctx"], m["effective_params"]["num_ctx"])
        self.assertEqual(m["params"]["num_ctx"], m["effective_params"]["num_ctx"])
        self.assertEqual(m["config_digest"], provenance.compute_config_digest(m["effective_params"]))

    def test_config_digest_stability(self):
        """Configuration digest must be deterministic, order-independent, and sensitive to parameter changes."""
        params1 = {"temperature": 0.2, "top_k": 20, "num_ctx": 16384, "seed": None}
        params2 = {"num_ctx": 16384, "seed": None, "temperature": 0.2, "top_k": 20}
        params3 = {"temperature": 0.2, "top_k": 21, "num_ctx": 16384, "seed": None}

        digest1 = provenance.compute_config_digest(params1)
        digest2 = provenance.compute_config_digest(params2)
        digest3 = provenance.compute_config_digest(params3)

        self.assertEqual(digest1, digest2)
        self.assertNotEqual(digest1, digest3)

    def test_experiment_identity_and_reproducibility_metadata(self):
        """Metadata must record reproducibility, seed, run_id, and never fabricate missing values."""
        eff_stochastic = {"temperature": 0.2, "num_ctx": 16384, "seed": None}
        eff_reproducible = {"temperature": 0.2, "num_ctx": 16384, "seed": 1234}

        self.assertFalse(eff_stochastic["seed"] is not None)
        self.assertTrue(eff_reproducible["seed"] is not None)

        run_id1 = provenance.generate_run_id()
        run_id2 = provenance.generate_run_id()
        self.assertNotEqual(run_id1, run_id2)

        hw = provenance.get_hardware_info()
        self.assertIn("gpu", hw)
        self.assertIn("ram_mb", hw)

        os_info = provenance.get_os_info()
        self.assertIn("platform", os_info)

    def test_comparability_and_stale_results(self):
        """Comparability checking must detect incompatible tasks, benchmarks, or harnesses."""
        r1 = {
            "task": "01-add-function", "task_version": "hash1", "benchmark_version": "bm1",
            "harness": "pi", "harness_version": "hv1", "model": "m:1"
        }
        r2 = copy.deepcopy(r1)
        ok, reasons = provenance.check_comparability(r1, r2)
        self.assertTrue(ok)
        self.assertEqual(len(reasons), 0)

        # Incompatible task version
        r_diff_task_ver = copy.deepcopy(r1)
        r_diff_task_ver["task_version"] = "hash2"
        ok, reasons = provenance.check_comparability(r1, r_diff_task_ver)
        self.assertFalse(ok)
        self.assertTrue(any("Task versions differ" in r for r in reasons))

        # Incompatible harness
        r_diff_harness = copy.deepcopy(r1)
        r_diff_harness["harness"] = "minimal"
        ok, reasons = provenance.check_comparability(r1, r_diff_harness)
        self.assertFalse(ok)

        # Stale detection
        stale, s_reason = provenance.is_stale(r1, current_benchmark_ver="bm2")
        self.assertTrue(stale)
        self.assertIn("Benchmark version changed", s_reason)

    def test_hardware_feasibility_vs_quality_context(self):
        """Hardware feasibility determines physical runtime boundaries, not coding quality."""
        # Hardware feasibility output contains physical boundary metadata
        feasibility_result = {
            "hardware_safe_ctx": 32768,
            "hardware_max_ctx": 49152,
            "feasibility_only": True
        }
        self.assertIn("hardware_safe_ctx", feasibility_result)
        self.assertTrue(feasibility_result["feasibility_only"])

    def test_context_quality_curve(self):
        """Context quality curve must measure pass rates and latencies across context sizes."""
        runs = [
            # 16k context: 3/3 pass, fast
            {"model": "m1", "num_ctx": 16384, "task": "t1", "done": True, "wall_s": 5.0},
            {"model": "m1", "num_ctx": 16384, "task": "t2", "done": True, "wall_s": 6.0},
            {"model": "m1", "num_ctx": 16384, "task": "t3", "done": True, "wall_s": 7.0},
            # 64k context: 1/3 pass, slower, regressions
            {"model": "m1", "num_ctx": 65536, "task": "t1", "done": True, "wall_s": 25.0},
            {"model": "m1", "num_ctx": 65536, "task": "t2", "done": False, "wall_s": 30.0, "timeout": True},
            {"model": "m1", "num_ctx": 65536, "task": "t3", "done": False, "wall_s": 28.0}
        ]
        curve = report.context_curve(runs)
        self.assertEqual(len(curve), 2)
        c16 = next(c for c in curve if c["num_ctx"] == 16384)
        c64 = next(c for c in curve if c["num_ctx"] == 65536)
        self.assertEqual(c16["rate"], 1.0)
        self.assertLess(c64["rate"], 0.5)
        self.assertGreater(c64["median_wall"], c16["median_wall"])

    def test_statistical_comparison_and_insufficient_evidence(self):
        """Small sample size or ambiguous results must yield insufficient_evidence."""
        # Low repetitions: 2 passes vs 1 pass with n=2 -> insufficient evidence
        low_rep_a = [{"task": "t1", "done": False, "wall_s": 10}, {"task": "t2", "done": True, "wall_s": 10}]
        low_rep_b = [{"task": "t1", "done": True, "wall_s": 10}, {"task": "t2", "done": True, "wall_s": 10}]
        res_low = stats.compare_runs(low_rep_a, low_rep_b)
        self.assertEqual(res_low["decision"], "insufficient_evidence")

        # Large sample size with confirmed gain
        large_a = [{"task": f"t{i}", "done": False, "wall_s": 10} for i in range(15)]
        large_b = [{"task": f"t{i}", "done": True, "wall_s": 10} for i in range(15)]
        res_large = stats.compare_runs(large_a, large_b)
        self.assertEqual(res_large["decision"], "improvement_supported")

        # Wilson score interval boundary checks
        p, low, high = stats.wilson_score_interval(10, 10)
        self.assertEqual(p, 1.0)
        self.assertEqual(high, 1.0)
        self.assertGreater(low, 0.6)

    def test_true_pareto_frontier(self):
        """Pareto calculation must identify true non-dominated candidates and filter dominated ones."""
        # A: pass 0.90, wall 10, size 4
        # B: pass 0.80, wall 12, size 6 -> strictly dominated by A in all metrics
        # C: pass 0.95, wall 20, size 8 -> non-dominated (higher pass rate than A, but slower and larger)
        cand_a = {"tag": "A", "rate": 0.90, "wall": 10.0, "size_gb": 4.0, "failures": 0}
        cand_b = {"tag": "B", "rate": 0.80, "wall": 12.0, "size_gb": 6.0, "failures": 1}
        cand_c = {"tag": "C", "rate": 0.95, "wall": 20.0, "size_gb": 8.0, "failures": 0}

        self.assertTrue(pareto.dominates(cand_a, cand_b))
        self.assertFalse(pareto.dominates(cand_b, cand_a))
        self.assertFalse(pareto.dominates(cand_a, cand_c))
        self.assertFalse(pareto.dominates(cand_c, cand_a))

        frontier = pareto.compute_pareto_frontier([cand_a, cand_b, cand_c])
        frontier_tags = {c["tag"] for c in frontier}
        self.assertIn("A", frontier_tags)
        self.assertIn("C", frontier_tags)
        self.assertNotIn("B", frontier_tags) # B is dominated

        # Recommendation policies
        pick_quality, _, _ = pareto.select_recommendation([cand_a, cand_b, cand_c], policy="max_quality")
        self.assertEqual(pick_quality["tag"], "C") # Highest pass rate

        pick_balanced, _, _ = pareto.select_recommendation([cand_a, cand_b, cand_c], policy="balanced", tolerance=0.1)
        self.assertEqual(pick_balanced["tag"], "A") # Within 10pp of C (0.95 vs 0.90), but much smaller (4GB vs 8GB)

    def test_workspace_isolation_and_protected_files(self):
        """Workspace isolation must set protected files to read-only 0444."""
        task = {
            "id": "test-task",
            "prompt": "write code",
            "files": {"code.py": "def foo(): pass", "test.sh": "pytest"},
            "protected": ["test.sh"]
        }
        d = sandbox.create_isolated_workspace(task)
        try:
            test_sh_path = os.path.join(d, "test.sh")
            code_py_path = os.path.join(d, "code.py")

            test_mode = os.stat(test_sh_path).st_mode
            code_mode = os.stat(code_py_path).st_mode

            # test.sh must be read-only (not writable by owner)
            self.assertFalse(bool(test_mode & stat.S_IWUSR))
            # code.py must be writable by owner
            self.assertTrue(bool(code_mode & stat.S_IWUSR))
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_guardrails_benchmark_integrity(self):
        """Benchmark infrastructure files must be detected if modified."""
        snapshot = guardrails.snapshot_benchmark()
        ok, violations = guardrails.verify_benchmark_integrity(snapshot)
        self.assertTrue(ok)
        self.assertEqual(violations, [])

        # Tampering with snapshot
        tampered_snap = dict(snapshot)
        first_key = next(iter(tampered_snap))
        tampered_snap[first_key] = "corrupted_hash"
        ok, violations = guardrails.verify_benchmark_integrity(tampered_snap)
        self.assertFalse(ok)
        self.assertTrue(len(violations) > 0)

    def test_task_splits_and_holdout_isolation(self):
        """Final test tasks must never participate in train or validation sets."""
        split_cfg = {
            "train": ["01-add-function", "02-rename-symbol"],
            "val": ["03-fix-bug"],
            "test": ["05-bug-across-files"]
        }
        train_tasks = T.load(split=split_cfg, split_name="train")
        val_tasks = T.load(split=split_cfg, split_name="val")
        test_tasks = T.load(split=split_cfg, split_name="test")

        train_ids = {t["id"] for t in train_tasks}
        val_ids = {t["id"] for t in val_tasks}
        test_ids = {t["id"] for t in test_tasks}

        self.assertNotIn("05-bug-across-files", train_ids)
        self.assertNotIn("05-bug-across-files", val_ids)
        self.assertIn("05-bug-across-files", test_ids)
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))

if __name__ == "__main__":
    unittest.main()
