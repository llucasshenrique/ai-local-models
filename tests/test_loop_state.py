"""Unit tests for loop state checkpointing and recovery."""
import os, tempfile, unittest
from llmeval import loop_state

class LoopStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_load_state(self):
        ls = loop_state.LoopState("ornith", policy="pareto", max_cycles=3, results_dir=self.results_dir)
        ls.set_stage_complete("discovery", plan={"test": 123})
        
        loaded = loop_state.load_state(self.results_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["family"], "ornith")
        self.assertTrue(loaded["stages"]["discovery"]["completed"])
        self.assertEqual(loaded["stages"]["discovery"]["plan"], {"test": 123})

    def test_has_checkpoint_and_summary(self):
        self.assertFalse(loop_state.has_checkpoint("ornith", results_dir=self.results_dir))
        
        ls = loop_state.LoopState("ornith", policy="balanced", max_cycles=3, results_dir=self.results_dir)
        ls.set_stage_start("context_discovery")
        
        self.assertTrue(loop_state.has_checkpoint("ornith", results_dir=self.results_dir))
        self.assertFalse(loop_state.has_checkpoint("other_family", results_dir=self.results_dir))
        
        summary = loop_state.get_checkpoint_summary(results_dir=self.results_dir)
        self.assertIn("ornith", summary)
        self.assertIn("context_discovery", summary)

    def test_resume_preserves_completed_stages(self):
        # Stage 1: Discovery & Hardware complete
        ls = loop_state.LoopState("ornith", policy="balanced", max_cycles=3, results_dir=self.results_dir)
        ls.set_stage_complete("discovery", plan={"variants": ["v1"]})
        ls.set_stage_complete("hardware_envelope", hardware_max_safe=32768)
        ls.set_stage_start("context_discovery")
        ls.set_error("Connection reset by peer")
        
        # Resume
        resumed = loop_state.LoopState.init_or_resume("ornith", action="resume", results_dir=self.results_dir)
        self.assertTrue(resumed.is_stage_completed("discovery"))
        self.assertTrue(resumed.is_stage_completed("hardware_envelope"))
        self.assertFalse(resumed.is_stage_completed("context_discovery"))
        self.assertEqual(resumed.get_stage_data("hardware_envelope", "hardware_max_safe"), 32768)
        self.assertEqual(resumed.state["status"], "in_progress")
        self.assertIsNone(resumed.state["last_error"])

    def test_restart_step_resets_current_stage_only(self):
        ls = loop_state.LoopState("ornith", policy="balanced", max_cycles=3, results_dir=self.results_dir)
        ls.set_stage_complete("discovery")
        ls.set_stage_complete("hardware_envelope")
        ls.set_stage_complete("context_discovery", best_ctx=32768)
        ls.set_cycle_start(2)
        ls.record_cycle_result(1, winner="v1", confirmed=True)
        ls.record_cycle_result(2, winner="v2_broken", confirmed=False)
        ls.set_error("Crash during cycle 2")

        # Restart step on rsi_cycles
        restarted = loop_state.LoopState.init_or_resume("ornith", action="restart_step", results_dir=self.results_dir)
        # Prior stages still complete
        self.assertTrue(restarted.is_stage_completed("discovery"))
        self.assertTrue(restarted.is_stage_completed("context_discovery"))
        # Cycle 1 winner kept, cycle 2 cleared for retry
        history = restarted.get_stage_data("rsi_cycles", "history")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["cycle"], 1)

    def test_reset_all_clears_checkpoint(self):
        ls = loop_state.LoopState("ornith", results_dir=self.results_dir)
        ls.set_stage_complete("discovery")
        
        fresh = loop_state.LoopState.init_or_resume("ornith", action="reset_all", results_dir=self.results_dir)
        self.assertFalse(fresh.is_stage_completed("discovery"))

    def test_completion_marks_done(self):
        ls = loop_state.LoopState("ornith", results_dir=self.results_dir)
        ls.mark_all_completed()
        self.assertEqual(ls.state["status"], "completed")
        self.assertFalse(loop_state.has_checkpoint("ornith", results_dir=self.results_dir))

if __name__ == "__main__":
    unittest.main()
