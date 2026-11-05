"""Multi-GPU (Accelerate) GPU ids in the GUI (external review 2026-11-05 #2, GUI side).

The GUI could only send the process count and port; the backend needs the GPU ids for accelerate --multi_gpu --gpu_ids.
An empty GPU IDs field keeps config.json and the /train request exactly as before (the backend picks the GPUs).
The HTTP layer is a fake; no backend runs.

Run from the repo root (CPU only):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_accelerate_gpu_ids_ui
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))  # _fake_backend

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.AccelerateUI import AccelerateWidget
from main_ui_files.MainUI import MainWidget
from _fake_backend import IDLE, STARTED, VALIDATED, FakeBackend


class InTempDir(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("queue_store").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    @staticmethod
    def saved_accelerate() -> dict:
        return json.loads(Path("config.json").read_text(encoding="utf-8"))["accelerate"]


def enable(widget: AccelerateWidget, processes: int = 2) -> None:
    widget.widget.accelerate_group.setChecked(True)
    widget.widget.accelerate_group.clicked[bool].emit(True)
    widget.widget.num_processes_input.setValue(processes)


class GpuIdsFieldTest(InTempDir):
    def test_ids_are_saved_and_loaded_back(self):
        widget = AccelerateWidget()
        enable(widget)
        widget.widget.gpu_ids_input.setText(" 0, 1 ")
        self.assertEqual(self.saved_accelerate(),
                         {"enabled": True, "num_processes": 2, "main_process_port": 29500, "gpu_ids": "0, 1"})
        loaded = AccelerateWidget()
        self.assertEqual(loaded.widget.gpu_ids_input.text(), "0, 1")
        self.assertEqual(self.saved_accelerate()["gpu_ids"], "0, 1")  # loading does not drop it

    def test_empty_ids_keep_the_config_as_before(self):
        widget = AccelerateWidget()
        enable(widget, processes=3)
        self.assertEqual(self.saved_accelerate(), {"enabled": True, "num_processes": 3, "main_process_port": 29500})
        widget.widget.gpu_ids_input.setText("0,1,2")
        widget.widget.gpu_ids_input.setText("")
        self.assertNotIn("gpu_ids", self.saved_accelerate())

    def test_validation(self):
        widget = AccelerateWidget()
        ids = widget.widget.gpu_ids_input
        ids.setText("x")
        self.assertEqual(widget.get_validation_errors(), [])  # multi-GPU off: the field is not used
        enable(widget)
        for text in ("", "0,1", "1, 3", "2,0"):
            ids.setText(text)
            self.assertEqual(widget.get_validation_errors(), [], text)
        for text, needle in (("0,1,2", "lists 3 GPUs but Number of GPUs is 2"), ("a,b", "comma-separated"),
                             ("0;1", "comma-separated"), ("-1,0", "comma-separated"), ("0,,1", "comma-separated"),
                             ("1,1", "more than once")):
            ids.setText(text)
            errors = widget.get_validation_errors()
            self.assertEqual(len(errors), 1, text)
            self.assertIn(needle, errors[0], text)

    def test_ids_need_two_or_more_gpus(self):
        """With Number of GPUs 1 the backend runs a plain single-process launch on its default GPU and does not pass
        --gpu_ids, so an id there would be silently ignored (review finding: '1' on a 2-GPU PC trained on GPU 0)."""
        widget = AccelerateWidget()
        enable(widget, processes=1)
        widget.widget.gpu_ids_input.setText("1")
        errors = widget.get_validation_errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("Number of GPUs 2 or more", errors[0])
        widget.widget.gpu_ids_input.setText("")
        self.assertEqual(widget.get_validation_errors(), [])  # one GPU without ids is fine, as before


class TrainParamsTest(InTempDir):
    def train_params(self, accelerate: dict) -> dict:
        Path("config.json").write_text(json.dumps({"accelerate": accelerate}), encoding="utf-8")
        fake = FakeBackend({"/validate": (200, VALIDATED), "/train": (200, STARTED), "/is_training": (200, IDLE)})
        with mock.patch.object(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None)):
            main = MainWidget()
        toml = Path("queue_store/item.toml")
        self.assertTrue(main.save_toml(toml))
        with mock.patch("main_ui_files.MainUI.requests.get", fake.get), \
                mock.patch("main_ui_files.MainUI.requests.post", fake.post), \
                mock.patch("main_ui_files.MainUI.sleep", lambda _: None):
            self.assertTrue(main.train_helper("http://127.0.0.1:8000", toml))
        return next(params for _, route, params, _ in fake.calls if route == "/train")

    def test_gpu_ids_are_sent(self):
        params = self.train_params({"enabled": True, "num_processes": 2, "main_process_port": 29501, "gpu_ids": "1, 3"})
        self.assertEqual(params["accelerate_enabled"], "True")
        self.assertEqual(params["accelerate_num_processes"], "2")
        self.assertEqual(params["accelerate_main_process_port"], "29501")
        self.assertEqual(params["accelerate_gpu_ids"], "1,3")

    def test_without_gpu_ids_the_request_is_unchanged(self):
        params = self.train_params({"enabled": True, "num_processes": 2, "main_process_port": 29500})
        self.assertEqual(
            {key for key in params if key.startswith("accelerate")},
            {"accelerate_enabled", "accelerate_num_processes", "accelerate_main_process_port"},
        )
        params = self.train_params({"enabled": False})
        self.assertFalse(any(key.startswith("accelerate") for key in params))


if __name__ == "__main__":
    unittest.main()
