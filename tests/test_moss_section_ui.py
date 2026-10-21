"""MOSS-Image section (main_ui_files/MossUI.py): keys, defaults, TOML round trip and model-type exclusion.

Run from the repo root (CPU only, no backend):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_moss_section_ui
"""

import copy
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

from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.ArgsListUI import ArgsWidget
from main_ui_files.MossUI import MossWidget, parse_block_list

PATHS = {
    "pretrained_model_name_or_path": "models/moss_dit.safetensors",
    "vae": "models/flux2_vae.safetensors",
    "moss_text_encoder": "models/moss_te.safetensors",
}


def turn_on(box) -> None:
    """Check a checkable group box the way a click does (setChecked, then the clicked signal)."""
    box.setChecked(True)
    box.clicked[bool].emit(True)


def turn_off(box) -> None:
    box.setChecked(False)
    box.clicked[bool].emit(False)


def moss_on(args_widget: ArgsWidget) -> MossWidget:
    moss = args_widget.moss_widget
    moss.widget.dit_model_input.setText(PATHS["pretrained_model_name_or_path"])
    moss.widget.vae_model_input.setText(PATHS["vae"])
    moss.widget.text_encoder_input.setText(PATHS["moss_text_encoder"])
    turn_on(moss.widget.moss_training_box)
    return moss


def non_empty(section: dict) -> dict:
    return {k: v for k, v in section.items() if v not in ("", {})}


def moss_args(args_widget: ArgsWidget) -> dict:
    return copy.deepcopy(args_widget.get_args()["args"].get("moss_args"))


class MossSectionKeysTest(unittest.TestCase):
    def setUp(self):
        self.w = ArgsWidget()

    def test_off_writes_nothing(self):
        moss = self.w.moss_widget
        moss.widget.dit_model_input.setText("x.safetensors")  # typing while off adds no section
        moss.widget.use_ramtorch_enable.click()
        moss.widget.timestep_sampling_selector.setCurrentText("sigmoid")
        self.assertNotIn("moss_args", self.w.get_args()["args"])
        self.assertEqual(moss.get_validation_errors(), [])

    def test_on_with_defaults_writes_only_paths_and_sampling(self):
        moss = moss_on(self.w)
        self.assertEqual(moss_args(self.w), {**PATHS, "timestep_sampling": "moss"})
        # the tokenizer folder is required unless MOSS_TOKENIZER_DIR is set
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": ""}):
            errors = self.w.get_validation_errors()
            self.assertEqual(len(errors), 1)
            self.assertIn("Tokenizer Folder is required", errors[0])
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": "C:/tok"}):
            self.assertEqual(self.w.get_validation_errors(), [])
        moss.widget.tokenizer_dir_input.setText("tok")
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": ""}):
            self.assertEqual(self.w.get_validation_errors(), [])

    def test_every_control(self):
        moss = moss_on(self.w)
        w = moss.widget
        base = {**PATHS, "timestep_sampling": "moss"}

        def expect(extra: dict, missing=()):
            got = moss_args(self.w)
            for key, value in extra.items():
                self.assertIn(key, got)
                self.assertEqual(got[key], value, key)
            for key in missing:
                self.assertNotIn(key, got)

        w.tokenizer_dir_input.setText("tok")
        expect({"moss_tokenizer_dir": "tok"})
        w.tokenizer_dir_input.setText("")
        expect({}, missing=["moss_tokenizer_dir"])
        w.tokenizer_dir_input.setText("tok")

        w.max_token_input.setValue(2048)
        expect({"moss_max_token_length": 2048})
        w.max_token_input.setValue(8192)  # the trainer refuses more than 4096
        self.assertEqual(w.max_token_input.value(), 4096)
        w.max_token_input.setValue(4096)
        expect({}, missing=["moss_max_token_length"])

        # moss sampling: logit-normal values, omitted at their defaults
        w.logit_mean_input.setValue(1.2)
        w.logit_std_input.setValue(1.0)
        w.sigma_cutoff_input.setValue(0.9)
        w.uniform_mix_input.setValue(0.05)
        expect({"moss_logit_mean": 1.2, "moss_logit_std": 1.0, "moss_sigma_cutoff": 0.9, "moss_uniform_mix": 0.05})
        w.logit_std_input.setValue(0.95)
        expect({}, missing=["moss_logit_std"])

        # sigmoid: preset sigmoid_scale + scale_resume; the moss_* values leave
        w.timestep_sampling_selector.setCurrentText("sigmoid")
        self.assertFalse(w.logit_mean_input.isEnabled())
        expect({"timestep_sampling": "sigmoid"},
               missing=["moss_logit_mean", "moss_sigma_cutoff", "moss_uniform_mix", "sigmoid_scale", "scale_resume",
                        "discrete_flow_shift"])
        w.sigmoid_scale_input.setEditText("content")
        w.scale_resume_input.setValue(2)
        expect({"sigmoid_scale": "content", "scale_resume": 2})
        w.sigmoid_scale_input.setEditText("1.0")
        expect({}, missing=["sigmoid_scale"])
        w.sigmoid_scale_input.setEditText("Style")
        expect({"sigmoid_scale": "style"})

        # shift: a preset is invalid (kept out, validation error), a number is fine; scale_resume leaves
        w.timestep_sampling_selector.setCurrentText("shift")
        expect({"timestep_sampling": "shift"}, missing=["scale_resume", "discrete_flow_shift"])
        self.assertTrue(any("Sigmoid Scale" in e for e in moss.get_validation_errors()))
        w.sigmoid_scale_input.setEditText("1.5")
        w.discrete_flow_shift_input.setValue(2.0)
        expect({"sigmoid_scale": 1.5, "discrete_flow_shift": 2.0})
        self.assertEqual(moss.get_validation_errors(), [])
        w.discrete_flow_shift_input.setValue(1.0)
        expect({}, missing=["discrete_flow_shift"])

        # sigma: discrete_flow_shift only
        w.discrete_flow_shift_input.setValue(3.0)
        w.timestep_sampling_selector.setCurrentText("sigma")
        expect({"timestep_sampling": "sigma", "discrete_flow_shift": 3.0}, missing=["sigmoid_scale"])

        # uniform: nothing extra
        w.timestep_sampling_selector.setCurrentText("uniform")
        expect({"timestep_sampling": "uniform"}, missing=["discrete_flow_shift", "sigmoid_scale", "scale_resume"])

        # back to moss: the changed logit values come back
        w.timestep_sampling_selector.setCurrentText("moss")
        expect({"moss_logit_mean": 1.2, "moss_sigma_cutoff": 0.9, "moss_uniform_mix": 0.05})

        # attention / memory
        w.attn_mode_selector.setCurrentText("flash")
        expect({"attn_mode": "flash"})
        w.attn_mode_selector.setCurrentIndex(0)
        expect({}, missing=["attn_mode"])
        items = [w.attn_mode_selector.itemText(i) for i in range(w.attn_mode_selector.count())]
        self.assertEqual(items, ["default", "torch (SDPA)", "flash", "xformers"])  # no sageattn, no duplicate sdpa
        w.attn_mode_selector.setCurrentText("torch (SDPA)")
        expect({"attn_mode": "torch"})
        w.attn_mode_selector.setCurrentText("xformers")
        expect({"attn_mode": "xformers"})
        w.split_attn_enable.click()
        w.blocks_to_swap_input.setValue(4)
        w.use_ramtorch_enable.click()
        w.unsloth_offload_checkpointing.click()
        expect({"split_attn": True, "blocks_to_swap": 4, "use_ramtorch": True, "unsloth_offload_checkpointing": True})
        self.assertTrue(any("Unsloth Offload" in e for e in moss.get_validation_errors()))
        self.assertEqual(w.blocks_to_swap_input.maximum(), 40)
        w.blocks_to_swap_input.setValue(0)
        w.use_ramtorch_enable.click()
        expect({}, missing=["blocks_to_swap", "use_ramtorch"])

        # network / samples
        w.exclude_blocks_input.setText("22")
        expect({"moss_lora_exclude_blocks": "22"})
        w.exclude_blocks_input.setText("23, 22")
        expect({"moss_lora_exclude_blocks": "22,23"})
        w.exclude_blocks_input.setText("22,x")
        self.assertTrue(any("Exclude Blocks" in e for e in moss.get_validation_errors()))
        w.exclude_blocks_input.setText("")
        expect({}, missing=["moss_lora_exclude_blocks"])
        w.sample_shift_input.setValue(4.0)
        expect({"moss_sample_shift": 4.0})
        w.sample_shift_input.setValue(5.0)
        expect({}, missing=["moss_sample_shift"])
        w.uncond_skip_layers_input.setText("")
        expect({"moss_sample_uncond_skip_layers": ""})
        w.uncond_skip_layers_input.setText("10, 20")
        expect({"moss_sample_uncond_skip_layers": "10, 20"})
        w.uncond_skip_layers_input.setText("22")
        expect({}, missing=["moss_sample_uncond_skip_layers"])

        # advanced switches
        for widget_name, key in MossWidget.ADVANCED_SWITCHES:
            getattr(w, widget_name).click()
            expect({key: True})
            getattr(w, widget_name).click()
            expect({}, missing=[key])

        self.assertEqual(moss.get_validation_errors(), [])
        self.assertEqual(
            moss_args(self.w),
            {**base, "moss_tokenizer_dir": "tok", "moss_logit_mean": 1.2, "moss_sigma_cutoff": 0.9,
             "moss_uniform_mix": 0.05, "attn_mode": "xformers", "split_attn": True,
             "unsloth_offload_checkpointing": True},
        )

    def test_toggle_off_and_on_rebuilds_from_controls(self):
        moss = moss_on(self.w)
        moss.widget.timestep_sampling_selector.setCurrentText("sigmoid")
        moss.widget.sigmoid_scale_input.setEditText("balanced")
        turn_off(moss.widget.moss_training_box)
        self.assertNotIn("moss_args", self.w.get_args()["args"])
        turn_on(moss.widget.moss_training_box)
        self.assertEqual(moss_args(self.w), {**PATHS, "timestep_sampling": "sigmoid", "sigmoid_scale": "balanced"})

    def test_parse_block_list(self):
        self.assertEqual(parse_block_list("22"), [22])
        self.assertEqual(parse_block_list("23, 22 22"), [22, 23])
        self.assertEqual(parse_block_list([1, 2]), [1, 2])
        self.assertEqual(parse_block_list(""), [])
        self.assertIsNone(parse_block_list("42"))
        self.assertIsNone(parse_block_list("-1"))
        self.assertIsNone(parse_block_list("a"))


class MossRoundTripTest(unittest.TestCase):
    def test_toml_round_trip(self):
        from main_ui_files.MainUI import MainWidget

        main = MainWidget()
        moss = moss_on(main.args_widget)
        w = moss.widget
        w.tokenizer_dir_input.setText("tok")
        w.max_token_input.setValue(1024)
        main.args_widget.general_widget.widget.max_train_selector.setCurrentIndex(0)  # epochs: Scale Resume needs them
        w.timestep_sampling_selector.setCurrentText("sigmoid")
        w.sigmoid_scale_input.setEditText("content")
        w.scale_resume_input.setValue(2)
        w.attn_mode_selector.setCurrentText("flash")
        w.split_attn_enable.click()
        w.blocks_to_swap_input.setValue(6)
        w.use_ramtorch_enable.click()
        w.exclude_blocks_input.setText("22")
        w.sample_shift_input.setValue(4.5)
        w.uncond_skip_layers_input.setText("")
        w.no_sample_cache_enable.click()
        before = copy.deepcopy(main.args_widget.get_args())
        self.assertEqual(
            before["args"]["moss_args"],
            {**PATHS, "moss_tokenizer_dir": "tok", "moss_max_token_length": 1024, "timestep_sampling": "sigmoid",
             "sigmoid_scale": "content", "scale_resume": 2, "attn_mode": "flash", "split_attn": True,
             "blocks_to_swap": 6, "use_ramtorch": True, "moss_lora_exclude_blocks": "22", "moss_sample_shift": 4.5,
             "moss_sample_uncond_skip_layers": "", "moss_no_sample_cache": True},
        )

        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "moss.toml"
            main.save_toml(toml_path)
            text = toml_path.read_text(encoding="utf-8")
            self.assertIn("[moss_args.args]", text)
            self.assertIn('moss_sample_uncond_skip_layers = ""', text)
            args, dataset_args, _ = main.process_toml(toml_path)

        loaded = MainWidget()
        loaded.args_widget.load_args(args, dataset_args)
        after = loaded.args_widget.get_args()
        self.assertEqual(after["args"]["moss_args"], before["args"]["moss_args"])
        self.assertEqual(after["args"].get("anima_args"), before["args"].get("anima_args"))
        # Any load leaves an empty base-model path in GeneralUI and an empty network_args table in NetworkUI;
        # the backend drops empty values, so compare without them
        for section in ("general_args", "network_args"):
            self.assertEqual(non_empty(after["args"][section]), non_empty(before["args"][section]), section)

        lw = loaded.args_widget.moss_widget.widget
        self.assertTrue(lw.moss_training_box.isChecked())
        self.assertEqual(lw.timestep_sampling_selector.currentText(), "sigmoid")
        self.assertEqual(lw.sigmoid_scale_input.currentText(), "content")
        self.assertEqual(lw.scale_resume_input.value(), 2)
        self.assertEqual(lw.attn_mode_selector.currentText(), "flash")
        self.assertTrue(lw.use_ramtorch_enable.isChecked())
        self.assertEqual(lw.uncond_skip_layers_input.text(), "")
        self.assertEqual(lw.exclude_blocks_input.text(), "22")
        self.assertTrue(lw.no_sample_cache_enable.isChecked())
        self.assertFalse(lw.logit_mean_input.isEnabled())
        self.assertTrue(lw.scale_resume_input.isEnabled())

        # Loading a TOML without a MOSS section switches it off again
        loaded.args_widget.load_args({}, {})
        self.assertNotIn("moss_args", loaded.args_widget.get_args()["args"])
        self.assertFalse(lw.moss_training_box.isChecked())


class MossExclusionTest(unittest.TestCase):
    def setUp(self):
        self.w = ArgsWidget()
        self.general = self.w.args_widget_array[0]

    def test_section_order(self):
        names = [widget.name for widget in self.w.args_widget_array]
        self.assertEqual(names.index("moss_args"), names.index("anima_args") + 1)

    def test_moss_on_disables_other_model_types(self):
        g = self.general.widget
        g.FP8_enable.click()
        self.assertTrue(g.FP8_enable.isChecked())
        moss_on(self.w)
        self.assertFalse(self.w.anima_widget.widget.anima_training_box.isEnabled())
        self.assertFalse(self.w.flux_widget.widget.flux_training_box.isEnabled())
        for elem in (g.sdxl_enable, g.v2_enable, g.v_param_enable, g.FP8_enable):
            self.assertFalse(elem.isEnabled(), elem.objectName())
        self.assertFalse(g.FP8_enable.isChecked())
        self.assertNotIn("fp8_base", self.w.get_args()["args"]["general_args"])
        # MOSS needs cached TE outputs: the Network section enables the control
        self.assertTrue(self.w.network_widget.widget.cache_te_outputs_enable.isEnabled())

        turn_off(self.w.moss_widget.widget.moss_training_box)
        self.assertTrue(self.w.anima_widget.widget.anima_training_box.isEnabled())
        self.assertTrue(self.w.flux_widget.widget.flux_training_box.isEnabled())
        for elem in (g.sdxl_enable, g.v2_enable, g.v_param_enable, g.FP8_enable):
            self.assertTrue(elem.isEnabled(), elem.objectName())

    def test_anima_on_disables_moss(self):
        box = self.w.moss_widget.widget.moss_training_box
        turn_on(self.w.anima_widget.widget.anima_training_box)
        self.assertFalse(box.isEnabled())
        self.assertFalse(self.general.widget.FP8_enable.isEnabled())
        turn_off(self.w.anima_widget.widget.anima_training_box)
        self.assertTrue(box.isEnabled())
        self.assertTrue(self.general.widget.FP8_enable.isEnabled())

    def test_flux_on_disables_moss(self):
        box = self.w.moss_widget.widget.moss_training_box
        turn_on(self.w.flux_widget.widget.flux_training_box)
        self.assertFalse(box.isEnabled())
        turn_off(self.w.flux_widget.widget.flux_training_box)
        self.assertTrue(box.isEnabled())

    def test_sdxl_and_v2_disable_moss(self):
        box = self.w.moss_widget.widget.moss_training_box
        self.general.widget.sdxl_enable.click()
        self.assertFalse(box.isEnabled())
        self.general.widget.sdxl_enable.click()
        self.assertTrue(box.isEnabled())
        self.general.widget.v2_enable.click()
        self.assertFalse(box.isEnabled())
        self.general.widget.v2_enable.click()
        self.assertTrue(box.isEnabled())

    def test_disabled_while_checked_writes_nothing(self):
        moss_on(self.w)
        self.w.moss_widget.external_enable_disable(True)
        self.assertNotIn("moss_args", self.w.get_args()["args"])
        ramtorch = self.w.moss_widget.widget.use_ramtorch_enable  # disabled with its box: emit by hand
        ramtorch.setChecked(True)
        ramtorch.clicked[bool].emit(True)
        self.assertNotIn("moss_args", self.w.get_args()["args"])
        self.w.moss_widget.external_enable_disable(False)
        self.assertEqual(moss_args(self.w), {**PATHS, "timestep_sampling": "moss", "use_ramtorch": True})


if __name__ == "__main__":
    unittest.main()
