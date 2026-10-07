from __future__ import annotations

import unittest

from plant_counter.train import make_parser


class TrainArgumentTests(unittest.TestCase):
    def test_explicit_batch_is_parsed_as_integer(self) -> None:
        args = make_parser().parse_args(["--data", "grass.yaml", "--batch", "8"])
        self.assertEqual(args.batch, 8)
        self.assertIsInstance(args.batch, int)

    def test_automatic_batch_sentinel_is_integer(self) -> None:
        args = make_parser().parse_args(["--data", "grass.yaml"])
        self.assertEqual(args.batch, -1)
        self.assertIsInstance(args.batch, int)


if __name__ == "__main__":
    unittest.main()
