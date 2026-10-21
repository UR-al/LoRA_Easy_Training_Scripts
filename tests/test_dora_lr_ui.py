"""DoRA learning rate / weight decay in the Network section (learning_rate_DoRA / weight_decay_DoRA).

Run from the repo root (CPU only):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_dora_lr_ui
"""

import copy
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.NetworkUI import NetworkWidget, parse_dora_lr, parse_dora_wd

KEYS = ("learning_rate_DoRA", "weight_decay_DoRA")


def lokr_dora(n: NetworkWidget) -> None:
    n.widget.algo_select.setCurrentText("Lokr")
    n.widget.dora_enable.click()
    assert n.widget.dora_enable.isChecked()


class DoraLrTest(unittest.TestCase):
    def setUp(self):
        self.n = NetworkWidget()

    def test_parse(self):
        self.assertEqual(parse_dora_lr("1e-4"), 1e-4)
        for text in ("", "abc", "0", "-1e-4", "nan"):
            self.assertIsNone(parse_dora_lr(text), text)
        self.assertEqual(parse_dora_wd(""), 0.0)
        self.assertEqual(parse_dora_wd("1e-5"), 1e-5)
        self.assertEqual(parse_dora_wd("0"), 0.0)
        for text in ("abc", "-1e-4", "nan", "inf"):
            self.assertIsNone(parse_dora_wd(text), text)

    def test_off_writes_nothing(self):
        self.n.dora_lr_input.setText("1e-4")
        self.n.dora_wd_input.setText("0.01")
        self.assertFalse(self.n.dora_lr_input.isEnabled())
        for key in KEYS:
            self.assertNotIn(key, self.n.args)
        # lokr without DoRA: still nothing
        self.n.widget.algo_select.setCurrentText("Lokr")
        for key in KEYS:
            self.assertNotIn(key, self.n.args)

    def test_lokr_dora(self):
        lokr_dora(self.n)
        self.assertTrue(self.n.dora_lr_input.isEnabled())
        for key in KEYS:
            self.assertNotIn(key, self.n.args)  # empty lr and wd 0: nothing
        self.n.dora_lr_input.setText("1e-4")
        self.assertEqual(self.n.args["learning_rate_DoRA"], 1e-4)
        self.assertNotIn("weight_decay_DoRA", self.n.args)
        self.n.dora_wd_input.setText("0.01")
        self.assertEqual(self.n.args["weight_decay_DoRA"], 0.01)
        self.n.dora_wd_input.setText("")
        self.assertNotIn("weight_decay_DoRA", self.n.args)
        self.assertNotIn("learning_rate_DoRA", self.n.args.get("network_args", {}))

        self.n.dora_lr_input.setText("abc")
        self.assertNotIn("learning_rate_DoRA", self.n.args)
        self.assertEqual(len(self.n.get_validation_errors()), 1)
        self.n.dora_lr_input.setText("2e-4")
        self.assertEqual(self.n.get_validation_errors(), [])

        # DoRA off again (checkbox, then an algorithm without DoRA): keys leave
        self.n.dora_wd_input.setText("0.02")
        self.n.widget.dora_enable.click()
        for key in KEYS:
            self.assertNotIn(key, self.n.args)
        self.n.widget.dora_enable.click()
        self.assertEqual(self.n.args["learning_rate_DoRA"], 2e-4)
        self.n.widget.algo_select.setCurrentText("LoRA")
        for key in KEYS:
            self.assertNotIn(key, self.n.args)
        self.assertFalse(self.n.dora_lr_input.isEnabled())

    def test_load_restores(self):
        lokr_dora(self.n)
        self.n.dora_lr_input.setText("1e-4")
        self.n.dora_wd_input.setText("0.01")
        saved = copy.deepcopy(self.n.args)
        loaded = NetworkWidget()
        loaded.load_args({"network_args": copy.deepcopy(saved)})
        self.assertEqual(loaded.args["learning_rate_DoRA"], 1e-4)
        self.assertEqual(loaded.args["weight_decay_DoRA"], 0.01)
        self.assertEqual(loaded.dora_lr_input.text(), "0.0001")
        self.assertEqual(loaded.dora_wd_input.text(), "0.01")
        self.assertEqual(loaded.args, saved)

    def test_default_network_args_unchanged(self):
        self.assertEqual(
            self.n.args, {"network_dim": 32, "network_alpha": 16.0, "min_timestep": 0, "max_timestep": 1000}
        )


if __name__ == "__main__":
    unittest.main()
