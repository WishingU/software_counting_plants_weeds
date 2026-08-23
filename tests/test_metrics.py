from __future__ import annotations

import unittest

from plant_counter.metrics import summarize_counts


class CountingMetricTests(unittest.TestCase):
    def test_summarizes_total_and_per_class_errors(self) -> None:
        summary = summarize_counts(
            [{0: 2, 1: 1}, {0: 1}],
            [{0: 1, 1: 1}, {0: 3}],
            ["wheat", "wild oat"],
        )
        self.assertEqual(summary["images"], 2)
        self.assertAlmostEqual(summary["overall"]["mae"], 1.5)
        self.assertAlmostEqual(summary["overall"]["bias"], 0.5)
        self.assertAlmostEqual(summary["per_class"]["wild oat"]["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
