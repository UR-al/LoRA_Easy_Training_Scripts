"""Anima section fork flags (main_ui_files/AnimaUI.py): sigmoid presets, scale_resume, RamTorch, the Anima 3.8B v1.1
connector box and the speedup switches. Existing Anima args must round-trip unchanged.

Run from the repo root (CPU only):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_anima_fork_flags_ui
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

from main_ui_files.ArgsListUI import ArgsWidget
from main_ui_files.AnimaUI import AnimaWidget

# What the GUI wrote for Anima before the fork flags (a "TOML unchanged" guard)
EXISTING_ANIMA = {
    "pretrained_model_name_or_path": "models/anima.safetensors",
    "qwen3": "models/qwen3.safetensors",
    "vae": "models/qwen_image_vae.safetensors",
    "qwen3_max_token_length": 512,
    "t5_max_token_length": 512,
    "timestep_sampling": "sigmoid",
    "sigmoid_scale": 1.0,
    "attn_mode": "flash",
    "unsloth_offload_checkpointing": True,
}
EXISTING_ANIMA_SHIFT = {
    "pretrained_model_name_or_path": "models/anima.safetensors",
    "qwen3": "models/qwen3.safetensors",
    "vae": "models/qwen_image_vae.safetensors",
    "qwen3_max_token_length": 256,
    "t5_max_token_length": 512,
    "timestep_sampling": "shift",
    "sigmoid_scale": 1.3,
    "discrete_flow_shift": 2.5,
    "blocks_to_swap": 4,
    "split_attn": True,
}


def load_anima(section: dict) -> ArgsWidget:
    w = ArgsWidget()
    w.load_args({"anima_args": copy.deepcopy(section)}, {})
    return w


def anima_args(w: ArgsWidget) -> dict:
    return copy.deepcopy(w.get_args()["args"].get("anima_args"))


def click_box(box, checked: bool) -> None:
    box.setChecked(checked)
    box.clicked[bool].emit(checked)


class AnimaExistingArgsTest(unittest.TestCase):
    def test_existing_args_round_trip_unchanged(self):
        for section in (EXISTING_ANIMA, EXISTING_ANIMA_SHIFT):
            with self.subTest(sampling=section["timestep_sampling"]):
                w = load_anima(section)
                self.assertEqual(anima_args(w), section)
                again = load_anima(anima_args(w))
                self.assertEqual(anima_args(again), section)

    def test_new_controls_off_by_default(self):
        w = load_anima(EXISTING_ANIMA)
        aw = w.anima_widget.widget
        self.assertFalse(aw.anima38_box.isChecked())
        self.assertFalse(aw.use_ramtorch_enable.isChecked())
        self.assertEqual(aw.scale_resume_input.value(), 0)
        for widget_name, _ in AnimaWidget.ADVANCED_SWITCHES:
            self.assertFalse(getattr(aw, widget_name).isChecked(), widget_name)

    def test_anima_off_writes_nothing(self):
        w = ArgsWidget()
        aw = w.anima_widget.widget
        aw.timestep_sampling_selector.setCurrentText("sigmoid")
        aw.sigmoid_scale_input.setEditText("style")
        self.assertNotIn("anima_args", w.get_args()["args"])
        self.assertEqual(w.get_validation_errors(), [])


class AnimaForkFlagsTest(unittest.TestCase):
    def setUp(self):
        self.w = load_anima(EXISTING_ANIMA)
        self.aw = self.w.anima_widget.widget

    def test_sigmoid_presets_and_scale_resume(self):
        self.aw.sigmoid_scale_input.setEditText("content")
        self.assertEqual(anima_args(self.w)["sigmoid_scale"], "content")
        self.aw.scale_resume_input.setValue(3)
        self.assertEqual(anima_args(self.w)["scale_resume"], 3)
        self.aw.scale_resume_input.setValue(0)
        self.assertNotIn("scale_resume", anima_args(self.w))
        self.aw.scale_resume_input.setValue(3)

        # flux_shift: presets are invalid (validation error), scale_resume leaves
        self.aw.timestep_sampling_selector.setCurrentText("flux_shift")
        self.assertNotIn("scale_resume", anima_args(self.w))
        self.assertFalse(self.aw.scale_resume_input.isEnabled())
        self.assertTrue(any("Sigmoid Scale" in e for e in self.w.get_validation_errors()))
        self.aw.sigmoid_scale_input.setEditText("0.8")
        self.assertEqual(anima_args(self.w)["sigmoid_scale"], 0.8)
        self.assertEqual(self.w.get_validation_errors(), [])

        self.aw.timestep_sampling_selector.setCurrentText("uniform")
        self.assertNotIn("sigmoid_scale", anima_args(self.w))
        self.aw.timestep_sampling_selector.setCurrentText("sigmoid")
        self.assertEqual(anima_args(self.w)["scale_resume"], 3)

    def test_ramtorch_and_speedup_switches(self):
        self.aw.use_ramtorch_enable.click()
        self.assertIs(anima_args(self.w)["use_ramtorch"], True)
        self.aw.use_ramtorch_enable.click()
        self.assertNotIn("use_ramtorch", anima_args(self.w))
        for widget_name, key in AnimaWidget.ADVANCED_SWITCHES:
            getattr(self.aw, widget_name).click()
            self.assertIs(anima_args(self.w)[key], True)
            getattr(self.aw, widget_name).click()
            self.assertNotIn(key, anima_args(self.w))

    def test_anima38_box(self):
        self.aw.qwen35_model_input.setText("models/qwen35_4b.safetensors")
        self.aw.anima38_negative_selector.setCurrentText("native")
        self.assertFalse(any(k in anima_args(self.w) for k in AnimaWidget.ANIMA38_KEYS))

        click_box(self.aw.anima38_box, True)
        self.assertEqual(
            {k: v for k, v in anima_args(self.w).items() if k in AnimaWidget.ANIMA38_KEYS},
            {"anima38_connector": True, "qwen35_path": "models/qwen35_4b.safetensors",
             "anima38_sample_negative": "native"},
        )
        self.aw.anima38_negative_selector.setCurrentText("connector")
        self.aw.anima38_batch_input.setValue(4)
        self.aw.qwen35_tokenizer_input.setText("tok")
        self.assertEqual(
            {k: v for k, v in anima_args(self.w).items() if k in AnimaWidget.ANIMA38_KEYS},
            {"anima38_connector": True, "qwen35_path": "models/qwen35_4b.safetensors",
             "qwen35_tokenizer_dir": "tok", "anima38_qwen35_batch_size": 4},
        )
        self.aw.anima38_batch_input.setValue(1)
        self.assertNotIn("anima38_qwen35_batch_size", anima_args(self.w))

        # round trip
        section = anima_args(self.w)
        loaded = load_anima(section)
        self.assertEqual(anima_args(loaded), section)
        self.assertTrue(loaded.anima_widget.widget.anima38_box.isChecked())
        self.assertEqual(loaded.anima_widget.widget.qwen35_tokenizer_input.text(), "tok")

        click_box(self.aw.anima38_box, False)
        self.assertFalse(any(k in anima_args(self.w) for k in AnimaWidget.ANIMA38_KEYS))
        self.assertEqual(anima_args(self.w), EXISTING_ANIMA)

    def test_all_flags_round_trip(self):
        self.aw.sigmoid_scale_input.setEditText("style")
        self.aw.scale_resume_input.setValue(2)
        self.aw.use_ramtorch_enable.click()
        self.aw.no_lokr_compile_enable.click()
        self.aw.qwen35_model_input.setText("q35.safetensors")
        click_box(self.aw.anima38_box, True)
        section = anima_args(self.w)
        loaded = load_anima(section)
        self.assertEqual(anima_args(loaded), section)
        lw = loaded.anima_widget.widget
        self.assertEqual(lw.sigmoid_scale_input.currentText(), "style")
        self.assertEqual(lw.scale_resume_input.value(), 2)
        self.assertTrue(lw.use_ramtorch_enable.isChecked())
        self.assertTrue(lw.no_lokr_compile_enable.isChecked())


if __name__ == "__main__":
    unittest.main()
