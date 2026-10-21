"""parse_sigmoid_scale / sigmoid_scale_for_sampling (modules/SigmoidScaleInput.py): no QApplication needed.

Run from the repo root:

    venv\\Scripts\\python.exe -m unittest tests.test_sigmoid_scale_input
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.SigmoidScaleInput import parse_sigmoid_scale, sigmoid_scale_for_sampling, sigmoid_scale_text


class ParseSigmoidScaleTest(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(parse_sigmoid_scale("1.3"), 1.3)
        self.assertEqual(parse_sigmoid_scale(" 2 "), 2.0)
        self.assertEqual(parse_sigmoid_scale(1.0), 1.0)
        self.assertIsInstance(parse_sigmoid_scale("1"), float)

    def test_presets_any_case(self):
        self.assertEqual(parse_sigmoid_scale("Style"), "style")
        self.assertEqual(parse_sigmoid_scale("BALANCED"), "balanced")
        self.assertEqual(parse_sigmoid_scale(" content "), "content")

    def test_invalid(self):
        for text in ("abc", "-1", "", "0", "nan", "inf", "   "):
            with self.subTest(text=text):
                self.assertIsNone(parse_sigmoid_scale(text))
        self.assertIsNone(parse_sigmoid_scale(True))
        self.assertIsNone(parse_sigmoid_scale(-0.5))

    def test_presets_need_sigmoid_sampling(self):
        self.assertEqual(sigmoid_scale_for_sampling("content", "sigmoid"), "content")
        self.assertIsNone(sigmoid_scale_for_sampling("content", "shift"))
        self.assertEqual(sigmoid_scale_for_sampling("1.5", "shift"), 1.5)

    def test_load_text(self):
        self.assertEqual(sigmoid_scale_text(1.0), "1.0")
        self.assertEqual(sigmoid_scale_text(1.25), "1.25")
        self.assertEqual(sigmoid_scale_text("Style"), "style")
        self.assertEqual(sigmoid_scale_text("weird"), "weird")


if __name__ == "__main__":
    unittest.main()
