"""Variant labels and the paired before/after comparison used by the self-improvement loop."""
import unittest
from unittest import mock
from llmeval import report, store

def row(v, task, rep, done, wall): return dict(harness="minimal", model="m", task=task, rep=rep, done=done, wall_s=wall, variant=v)

class VariantTests(unittest.TestCase):
    def test_variant_is_part_of_the_trial_key(self):
        self.assertNotEqual(store.key(row("a", "t", 1, True, 1)), store.key(row("b", "t", 1, True, 1)))
        self.assertEqual(store.key({"harness": "h", "model": "m", "task": "t", "rep": 1})[-1], None)     # old rows keep working

    def _cmp(self, rows, **kw):
        with mock.patch.object(store, "rows", return_value=rows): return report.compare("base", "v1", **kw)

    def test_better_needs_a_real_gain_and_no_big_slowdown(self):
        base = [row("base", "t", i, i < 3, 10) for i in range(6)]
        good = [row("v1", "t", i, i < 5, 11) for i in range(6)]
        slow = [row("v1", "t", i, i < 5, 30) for i in range(6)]
        self.assertEqual(self._cmp(base + good)["verdict"], "better")
        self.assertEqual(self._cmp(base + slow)["verdict"], "same")                 # more passes but 3x slower: not accepted

    def test_worse_and_no_overlap(self):
        base = [row("base", "t", i, True, 10) for i in range(3)]
        bad = [row("v1", "t", i, i == 0, 10) for i in range(3)]
        self.assertEqual(self._cmp(base + bad)["verdict"], "worse")
        self.assertEqual(self._cmp(base)["verdict"], "no overlap")

if __name__ == "__main__": unittest.main()
