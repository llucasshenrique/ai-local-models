"""Unit tests for automated empirical context optimization in llmeval.fit."""
import unittest
from unittest.mock import patch
from llmeval import fit

class TestContextOptimizer(unittest.TestCase):
    def test_build_stress_prompt(self):
        p = fit.build_stress_prompt(500)
        self.assertIn("Task: Output three bullet points", p)
        self.assertIn("algorithm is a finite sequence", p)
        # Should be scaled appropriately
        self.assertGreater(len(p), 500)

    def test_hardware_readers(self):
        gpu_u, gpu_f, gpu_util = fit.read_gpu_stats()
        self.assertIsInstance(gpu_u, int)
        self.assertIsInstance(gpu_f, int)
        self.assertIsInstance(gpu_util, int)

        ram_u, ram_a, sw_u, sw_f = fit.read_mem_stats()
        self.assertGreaterEqual(ram_u, 0)
        self.assertGreaterEqual(ram_a, 0)
        self.assertGreaterEqual(sw_u, 0)
        self.assertGreaterEqual(sw_f, 0)

        idle, tot = fit.read_cpu_raw()
        self.assertGreaterEqual(idle, 0.0)
        self.assertGreaterEqual(tot, 0.0)

    def test_context_monitor(self):
        m = fit.ContextMonitor(interval=0.05)
        m.start()
        baseline = {"gpu_used": 1000, "ram_used": 2000, "sw_used": 500}
        import time
        time.sleep(0.15)
        m.stop()
        summary = m.summary(baseline)
        self.assertIn("peak_gpu_mib", summary)
        self.assertIn("peak_ram_mb", summary)
        self.assertIn("peak_swap_mb", summary)
        self.assertIn("avg_cpu_pct", summary)

    def test_optimization_search_logic(self):
        """Test the RSI loop with mocked trials."""
        # Suppose a model is safe up to 65536, but at 131072 it violates.
        # Midpoint at 98304: suppose it passes.
        # Midpoint at 114688: passes.
        # Midpoint at 122880: passes.
        # Midpoint at 126976: fails.
        def mock_trial(tag, ctx, target_prompt, num_predict=48, timeout=240):
            is_safe = ctx <= 122880
            return {
                "model": tag,
                "num_ctx": ctx,
                "target_prompt": target_prompt,
                "status": "SUCCESS",
                "classification": "PRACTICAL_ACCEPTABLE" if is_safe else "CPU_OFFLOAD_DEGRADATION",
                "prompt_tok_s": 1800.0,
                "gen_tok_s": 55.0 if is_safe else 25.0,
                "gpu_pct": 100.0 if is_safe else 88.0,
                "peak_gpu_mib": 6000 + ctx // 32,
                "swap_delta_mb": 0 if is_safe else 50,
                "size_gb": 6.0,
                "vram_gb": 6.0 if is_safe else 5.2
            }

        with patch("llmeval.fit.run_ctx_trial", side_effect=mock_trial), \
             patch("llmeval.store.append") as mock_append:
            res = fit.optimize("test-model:9b", min_ctx=16384, max_ctx=262144, log=lambda *args: None)
            
            self.assertEqual(res["model"], "test-model:9b")
            self.assertEqual(res["max_practical_ctx"], 122880)
            self.assertGreaterEqual(res["first_violation_ctx"], 122880)
            self.assertLessEqual(res["recommended_ctx"], 122880)
            self.assertTrue(mock_append.called)

if __name__ == "__main__":
    unittest.main()
