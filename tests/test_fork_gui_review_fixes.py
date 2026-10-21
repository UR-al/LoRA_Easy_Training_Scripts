"""Review fixes for the fork's GUI controls: MOSS attention / LyCORIS preset / required inputs, Cache TE and caption
variants, model-type state after a load (Flux, SDXL), flow model vs MOSS, queue and TOML loading errors, DoRA keys,
Extra Args migration.

Run from the repo root (CPU only, no backend; message boxes are captured, never shown):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_fork_gui_review_fixes
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

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.ArgsListUI import ArgsWidget
from main_ui_files.MainUI import MainWidget, TomlLoadError, backend_error_text
from main_ui_files.NetworkUI import NetworkWidget
from modules import ForkArgs, TomlFunctions

MOSS_PATHS = {
    "pretrained_model_name_or_path": "models/moss_dit.safetensors",
    "vae": "models/flux2_vae.safetensors",
    "moss_text_encoder": "models/moss_te.safetensors",
}
BOXES: list = []  # (kind, title, text) of every message box


def setUpModule():
    for kind in ("warning", "critical", "information"):
        patcher = mock.patch.object(
            QtWidgets.QMessageBox, kind, staticmethod(lambda *a, kind=kind, **k: BOXES.append((kind, a[1], a[2])))
        )
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


def click(box, value: bool = True) -> None:
    """Check or uncheck a checkable group box / check box the way a click does."""
    box.setChecked(value)
    box.clicked[bool].emit(value)


def clean(value):
    """Drop empty strings / tables (the backend drops them; any load leaves a few in General / Saving / TI)."""
    if isinstance(value, dict):
        out = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in ("", {}, None)}
    return value


def moss_on(aw: ArgsWidget, tokenizer: str = "tok"):
    w = aw.moss_widget.widget
    w.dit_model_input.setText(MOSS_PATHS["pretrained_model_name_or_path"])
    w.vae_model_input.setText(MOSS_PATHS["vae"])
    w.text_encoder_input.setText(MOSS_PATHS["moss_text_encoder"])
    w.tokenizer_dir_input.setText(tokenizer)
    click(w.moss_training_box)
    return aw.moss_widget


def anima_on(aw: ArgsWidget, anima38: bool = False):
    w = aw.anima_widget.widget
    w.dit_model_input.setText("anima_dit.safetensors")
    w.qwen3_model_input.setText("qwen3.safetensors")
    w.vae_model_input.setText("qwen_image_vae.safetensors")
    click(w.anima_training_box)
    if anima38:
        w.qwen35_model_input.setText("qwen35_4b.safetensors")
        w.qwen35_tokenizer_input.setText("qwen35_tok")
        click(w.anima38_box)
    return aw.anima_widget


def save_and_process(main: MainWidget, tmp: Path, name: str = "x.toml"):
    path = tmp / name
    assert main.save_toml(path), BOXES[-1:]
    return main.process_toml(path)


class MossAttentionTest(unittest.TestCase):
    def test_entries_and_default(self):
        aw = ArgsWidget()
        moss = moss_on(aw)
        sel = moss.widget.attn_mode_selector
        self.assertEqual([sel.itemText(i) for i in range(sel.count())], ["default", "torch (SDPA)", "flash", "xformers"])
        self.assertNotIn("attn_mode", moss.args)
        tip = sel.toolTip()
        self.assertIn("xformers", tip)
        self.assertIn("General Args", tip)
        self.assertIn("sageattn is not offered", tip)

    def test_load_aliases_and_sageattn(self):
        aw = ArgsWidget()
        base = {**MOSS_PATHS, "moss_tokenizer_dir": "tok", "timestep_sampling": "moss"}
        aw.load_args({"moss_args": {**base, "attn_mode": "sdpa"}}, {})
        self.assertEqual(aw.moss_widget.args["attn_mode"], "torch")
        self.assertEqual(aw.load_warnings, [])
        aw.load_args({"moss_args": {**base, "attn_mode": "sageattn"}}, {})
        self.assertNotIn("attn_mode", aw.moss_widget.args)
        self.assertTrue(any("sageattn" in w for w in aw.load_warnings))
        aw.load_args({"moss_args": {**base, "moss_max_token_length": 8192}}, {})
        self.assertEqual(aw.moss_widget.widget.max_token_input.value(), 4096)
        self.assertTrue(any("4096" in w for w in aw.load_warnings))


class MossRequirementsTest(unittest.TestCase):
    def setUp(self):
        self.aw = ArgsWidget()
        self.n = self.aw.network_widget

    def test_cache_te_checked_on_switch_and_restored(self):
        cache = self.n.widget.cache_te_outputs_enable
        self.assertFalse(cache.isChecked())
        moss = moss_on(self.aw)
        self.assertTrue(cache.isChecked() and cache.isEnabled())
        self.assertTrue(self.n.args["cache_text_encoder_outputs"])
        self.assertEqual(self.aw.get_validation_errors(), [])
        click(moss.widget.moss_training_box, False)
        self.assertFalse(cache.isChecked())
        self.assertNotIn("cache_text_encoder_outputs", self.n.args)
        # already checked before: stays checked after switching off
        self.aw.general_widget.widget.sdxl_enable.click()
        click(cache)
        self.aw.general_widget.widget.sdxl_enable.click()
        moss_on(self.aw)
        click(moss.widget.moss_training_box, False)
        self.assertTrue(cache.isChecked())

    def test_cache_te_unchecked_is_an_error(self):
        moss_on(self.aw)
        click(self.n.widget.cache_te_outputs_enable, False)
        self.assertTrue(any("needs Cache TE Outputs" in e for e in self.aw.get_validation_errors()))

    def test_cache_te_to_disk_in_extra_args_is_enough_for_moss(self):
        moss_on(self.aw)
        click(self.n.widget.cache_te_outputs_enable, False)
        self.aw.extra_args_widget.load({"cache_text_encoder_outputs_to_disk": True}, False)
        self.assertFalse(any("Cache TE" in e for e in self.aw.get_validation_errors()))

    def test_anima_switched_back_on_with_38_box_checked(self):
        anima = anima_on(self.aw, anima38=True)
        cache = self.n.widget.cache_te_outputs_enable
        click(anima.widget.anima_training_box, False)
        self.assertFalse(cache.isChecked())
        click(anima.widget.anima_training_box)
        self.assertTrue(cache.isChecked())
        self.assertEqual(self.aw.get_validation_errors(), [])

    def test_required_paths_and_tokenizer(self):
        moss = moss_on(self.aw, tokenizer="")
        moss.widget.dit_model_input.setText("")
        moss.widget.vae_model_input.setText("")
        moss.widget.text_encoder_input.setText("")
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": ""}):
            errors = moss.get_validation_errors()
        for name in ("DiT Model", "VAE Model", "Text Encoder", "Tokenizer Folder"):
            self.assertTrue(any(name in e for e in errors), name)
        self.assertIn("Required unless MOSS_TOKENIZER_DIR", moss.widget.tokenizer_dir_input.placeholderText())
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": "C:/tok"}):
            self.assertFalse(any("Tokenizer" in e for e in moss.get_validation_errors()))

    def test_anima38_requirements(self):
        anima = anima_on(self.aw, anima38=True)
        cache = self.n.widget.cache_te_outputs_enable
        self.assertTrue(cache.isChecked())  # checked with the 3.8B box
        self.assertEqual(self.aw.get_validation_errors(), [])
        w = anima.widget
        w.qwen35_model_input.setText("")
        w.qwen35_tokenizer_input.setText("")
        with mock.patch.dict(os.environ, {"ANIMA38_QWEN35_TOKENIZER_DIR": ""}):
            errors = self.aw.get_validation_errors()
        self.assertTrue(any("qwen35_path" in e for e in errors))
        # an empty Qwen3.5 Tokenizer is allowed: the trainer uses the variable or its own default
        self.assertFalse(any("Tokenizer" in e for e in errors))
        w.qwen35_model_input.setText("qwen35_4b.safetensors")
        with mock.patch.dict(os.environ, {"ANIMA38_QWEN35_TOKENIZER_DIR": ""}):
            self.assertEqual(self.aw.get_validation_errors(), [])
        placeholder = w.qwen35_tokenizer_input.placeholderText()
        self.assertIn("$ANIMA38_QWEN35_TOKENIZER_DIR", placeholder)
        self.assertIn("trainer's default", placeholder)
        self.assertNotIn("Required", placeholder)
        tip = w.qwen35_tokenizer_input.toolTip()
        self.assertIn("clear error if neither exists", tip)
        self.assertNotIn("sd-webui-forge", tip + placeholder)  # no machine path
        click(w.anima38_box, False)
        self.assertFalse(cache.isChecked())  # put back
        w.dit_model_input.setText("")
        self.assertTrue(any("Anima Args: DiT Model is required" in e for e in self.aw.get_validation_errors()))

    def test_blocks_to_swap_with_unsloth(self):
        moss = moss_on(self.aw)
        moss.widget.blocks_to_swap_input.setValue(4)
        moss.widget.unsloth_offload_checkpointing.click()
        self.assertTrue(any("Unsloth" in e for e in self.aw.get_validation_errors()))
        click(moss.widget.moss_training_box, False)
        anima = anima_on(self.aw)
        anima.widget.blocks_to_swap_input.setValue(4)
        anima.widget.unsloth_offload_checkpointing.click()
        self.assertTrue(any("Anima Args: Blocks to Swap" in e for e in self.aw.get_validation_errors()))

    def test_scale_resume_needs_epochs(self):
        moss = moss_on(self.aw)
        selector = self.aw.general_widget.widget.max_train_selector
        moss.widget.timestep_sampling_selector.setCurrentText("sigmoid")
        moss.widget.scale_resume_input.setValue(2)
        selector.setCurrentIndex(1)  # steps
        self.assertTrue(any("Scale Resume counts epochs" in e for e in self.aw.get_validation_errors()))
        selector.setCurrentIndex(0)  # epochs
        self.assertEqual(self.aw.get_validation_errors(), [])
        click(moss.widget.moss_training_box, False)
        anima = anima_on(self.aw)
        anima.widget.timestep_sampling_selector.setCurrentText("sigmoid")
        anima.widget.scale_resume_input.setValue(1)
        selector.setCurrentIndex(1)
        self.assertTrue(any("Anima Args: Scale Resume" in e for e in self.aw.get_validation_errors()))


class MossLycorisPresetTest(unittest.TestCase):
    def setUp(self):
        self.aw = ArgsWidget()
        self.n = self.aw.network_widget
        self.preset = self.n.widget.lycoris_preset_input

    def test_prefilled_and_restored(self):
        moss = moss_on(self.aw)
        self.assertEqual(self.preset.text(), "moss")
        self.assertNotIn("preset", self.n.args.get("network_args", {}))  # kohya LoRA: no preset key
        self.n.widget.algo_select.setCurrentText("Lokr")
        self.assertEqual(self.n.args["network_args"]["preset"], "moss")
        self.assertEqual(self.aw.get_validation_errors(), [])
        click(moss.widget.moss_training_box, False)
        self.assertEqual(self.preset.text(), "")
        self.assertNotIn("preset", self.n.args["network_args"])

    def test_typed_preset_is_kept(self):
        self.preset.setText("attn-mlp")
        moss = moss_on(self.aw)
        self.assertEqual(self.preset.text(), "attn-mlp")
        click(moss.widget.moss_training_box, False)
        self.assertEqual(self.preset.text(), "attn-mlp")

    def test_errors(self):
        moss = moss_on(self.aw)
        self.n.widget.algo_select.setCurrentText("Lokr")
        self.preset.setText("")
        self.assertTrue(any("LyCORIS Preset = moss" in e for e in self.aw.get_validation_errors()))
        self.preset.setText("full")
        self.assertTrue(any("LyCORIS Preset = moss" in e for e in self.aw.get_validation_errors()))
        self.preset.setText("presets/my_moss.toml")  # a preset file: its own module list
        self.assertEqual(self.aw.get_validation_errors(), [])
        moss.widget.exclude_blocks_input.setText("22")
        self.assertTrue(any("Exclude Blocks with LyCORIS" in e for e in self.aw.get_validation_errors()))
        self.preset.setText("moss-text")
        self.assertEqual(self.aw.get_validation_errors(), [])
        # kohya LoRA (networks.lora -> lora_moss): no preset needed
        self.n.widget.algo_select.setCurrentText("LoRA")
        self.preset.setText("")
        self.assertEqual(self.aw.get_validation_errors(), [])
        self.assertIn("LyCORIS Preset", moss.widget.moss_training_box.toolTip())
        self.assertIn("moss-text", moss.widget.exclude_blocks_input.toolTip())


class CaptionVariantsTest(unittest.TestCase):
    def test_key_and_subset_rule(self):
        main = MainWidget()
        aw = main.args_widget
        n = aw.network_widget
        spin = n.caption_variants_input
        self.assertFalse(spin.isEnabled())
        self.assertNotIn("cache_caption_variants", n.args)
        moss_on(aw)
        self.assertTrue(spin.isEnabled())
        for value, expected in ((1, None), (0, None), (4, 4)):
            spin.setValue(value)
            self.assertEqual(n.args.get("cache_caption_variants"), expected)
        self.assertIn("N &gt; 1", spin.toolTip())
        subset = next(iter(main.subset_widget.subsets)) if hasattr(main.subset_widget, "subsets") else None
        subset_args = next(iter(main.subset_widget.dataset_args.values()))
        subset_args["shuffle_caption"] = True
        self.assertFalse(any("Caption Variants" in e for e in main.get_validation_errors()))
        spin.setValue(1)
        self.assertTrue(any("Shuffle Caption" in e and "Caption Variants > 1" in e for e in main.get_validation_errors()))
        del subset_args["shuffle_caption"]
        subset_args["caption_tag_dropout_rate"] = 0.1
        self.assertTrue(any("Caption Tag Dropout" in e for e in main.get_validation_errors()))
        # without cached TE outputs nothing is required, and the key leaves
        spin.setValue(4)
        click(n.widget.cache_te_outputs_enable, False)
        self.assertNotIn("cache_caption_variants", n.args)
        self.assertFalse(spin.isEnabled())
        self.assertFalse(any("Caption" in e and "Variants" in e for e in main.get_validation_errors()))
        del subset

    def test_load(self):
        n = NetworkWidget()
        n.toggle_sdxl(True)
        n.load_args({"network_args": {"cache_text_encoder_outputs": True, "cache_caption_variants": 3}})
        self.assertEqual(n.caption_variants_input.value(), 3)
        self.assertEqual(n.args["cache_caption_variants"], 3)


class ModelStateAfterLoadTest(unittest.TestCase):
    """A load emits Toggled(False) of Anima / MOSS after Flux's Toggled(True); the final state must still be Flux's."""

    INITIAL = ("fresh", "moss", "anima", "sdxl", "flux")

    def setup_state(self, aw: ArgsWidget, initial: str) -> None:
        if initial == "moss":
            moss_on(aw)
        elif initial == "anima":
            anima_on(aw)
        elif initial == "sdxl":
            aw.general_widget.widget.sdxl_enable.click()
        elif initial == "flux":
            click(aw.flux_widget.widget.flux_training_box)

    def round_trip(self, prepare):
        main = MainWidget()
        prepare(main.args_widget)
        before = clean(copy.deepcopy(main.args_widget.get_args()["args"]))
        with tempfile.TemporaryDirectory() as tmp:
            args, dataset_args, _ = save_and_process(main, Path(tmp))
        results = {}
        for initial in self.INITIAL:
            loaded = MainWidget()
            self.setup_state(loaded.args_widget, initial)
            loaded.args_widget.load_args(args, dataset_args)
            results[initial] = (loaded, clean(loaded.args_widget.get_args()["args"]))
        return before, results

    def test_flux_toml_loads_back_identically(self):
        def prepare(aw):
            click(aw.flux_widget.widget.flux_training_box)
            click(aw.network_widget.widget.cache_te_outputs_enable)

        before, results = self.round_trip(prepare)
        self.assertTrue(before["network_args"]["cache_text_encoder_outputs"])
        self.assertNotIn("clip_skip", before["general_args"])
        for initial, (loaded, after) in results.items():
            self.assertEqual(after, before, initial)
            g = loaded.args_widget.general_widget.widget
            for elem in (g.v2_enable, g.sdxl_enable, g.v_param_enable):
                self.assertFalse(elem.isEnabled(), f"{initial}: {elem.objectName()}")
            self.assertTrue(loaded.args_widget.network_widget.widget.cache_te_outputs_enable.isEnabled(), initial)
            self.assertFalse(
                loaded.args_widget.general_widget.experimental_args_widget.widget.flow_model_settings_box.isEnabled()
            )

    def test_sdxl_toml_keeps_cache_te(self):
        def prepare(aw):
            aw.general_widget.widget.sdxl_enable.click()
            click(aw.network_widget.widget.cache_te_outputs_enable)

        before, results = self.round_trip(prepare)
        self.assertTrue(before["network_args"]["cache_text_encoder_outputs"])
        self.assertTrue(before["general_args"]["sdxl"])
        for initial, (_, after) in results.items():
            self.assertEqual(after, before, initial)

    def test_moss_and_anima_tomls(self):
        before, results = self.round_trip(lambda aw: moss_on(aw))
        for initial, (_, after) in results.items():
            self.assertEqual(after, before, initial)
        before, results = self.round_trip(lambda aw: anima_on(aw, anima38=True))
        for initial, (_, after) in results.items():
            self.assertEqual(after, before, initial)


class FlowModelExclusionTest(unittest.TestCase):
    def test_moss_and_flow_model(self):
        aw = ArgsWidget()
        flow = aw.general_widget.experimental_args_widget.widget.flow_model_settings_box
        moss_box = aw.moss_widget.widget.moss_training_box
        click(flow)
        self.assertFalse(moss_box.isEnabled())
        click(flow, False)
        self.assertTrue(moss_box.isEnabled())
        moss_on(aw)
        self.assertFalse(flow.isEnabled())
        self.assertFalse(flow.isChecked())
        self.assertNotIn("flow_model", aw.get_args()["args"]["general_args"])
        click(moss_box, False)
        self.assertTrue(flow.isEnabled())

    def test_flux_still_exclusive(self):
        aw = ArgsWidget()
        flow = aw.general_widget.experimental_args_widget.widget.flow_model_settings_box
        click(aw.flux_widget.widget.flux_training_box)  # crashed with endless recursion before the guard
        self.assertFalse(flow.isEnabled())
        click(aw.flux_widget.widget.flux_training_box, False)
        self.assertTrue(flow.isEnabled())


class QueueAndTomlErrorsTest(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("queue_store").mkdir()
        BOXES.clear()
        self.main = MainWidget()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_invalid_args_add_no_queue_item(self):
        queue = self.main.queue_widget
        self.main.args_widget.network_widget.dora_lr_input.setText("x")  # not active: no error
        moss_on(self.main.args_widget, tokenizer="")
        with mock.patch.dict(os.environ, {"MOSS_TOKENIZER_DIR": ""}):
            queue.add_to_queue()
        self.assertEqual(queue.elements, [])
        self.assertEqual(list(Path("queue_store").iterdir()), [])
        self.assertEqual(BOXES[-1][1], "Invalid Args")
        self.assertIn("The queue item was not added.", BOXES[-1][2])
        self.main.args_widget.moss_widget.widget.tokenizer_dir_input.setText("tok")
        queue.add_to_queue()
        self.assertEqual(len(queue.elements), 1)
        self.assertTrue(queue.elements[0].queue_file.exists())

    def test_missing_queue_file_is_an_error_not_a_dialog(self):
        errors = []
        self.main.training_error.disconnect()
        self.main.training_error.connect(lambda title, text: errors.append((title, text)))
        with mock.patch.object(TomlFunctions, "QFileDialog") as dialog:
            self.assertFalse(self.main.train_helper("http://127.0.0.1:1", Path("queue_store/missing.toml")))
            dialog.assert_not_called()
        self.assertEqual(errors[0][0], "Cannot train this item")

    def test_flat_trainer_tomls(self):
        for example in ("examples/moss/config_moss.toml", "examples/anima/config_anima38.toml",
                        "examples/anima/dataset_example.toml"):
            path = ROOT / example
            with self.assertRaises(TomlLoadError, msg=example) as ctx:
                self.main.process_toml(path)
            self.assertIn("trainer config for the command-line launchers", str(ctx.exception))
            BOXES.clear()
            self.main.load_toml(path)  # no TypeError, a message instead
            self.assertEqual(BOXES[-1][1], "Cannot load TOML")
        broken = Path("broken.toml")
        broken.write_text("a = [", encoding="utf-8")
        with self.assertRaises(TomlLoadError):
            self.main.process_toml(broken)
        self.assertIsNone(TomlFunctions.gui_toml_problem({"general_args": {"args": {}}, "subsets": [],
                                                          "train_mode": {"train_mode": "lora"}}))

    def test_backend_validation_errors_reach_a_message_box(self):
        errors = []
        self.main.training_error.disconnect()
        self.main.training_error.connect(lambda title, text: errors.append((title, text)))
        Path("config.json").write_text("{}", encoding="utf-8")
        assert self.main.save_toml(Path("queue_store/item.toml"))
        response = mock.Mock(status_code=400, text='["vae is not found"]')
        response.json.return_value = ["vae is not found", "output_dir is not found"]
        with mock.patch("main_ui_files.MainUI.requests.post", return_value=response):
            self.assertFalse(self.main.train_helper("http://127.0.0.1:1", Path("queue_store/item.toml")))
        self.assertEqual(errors, [("Backend validation failed", "vae is not found\noutput_dir is not found")])
        conflict = mock.Mock(status_code=409, text="")
        conflict.json.return_value = {"detail": "Training Already Running"}
        self.assertEqual(backend_error_text(conflict), "Training Already Running")


class MergedModelSectionsTest(unittest.TestCase):
    def test_moss_wins_and_no_box_is_stuck(self):
        aw = ArgsWidget()
        moss = {**MOSS_PATHS, "moss_tokenizer_dir": "tok", "timestep_sampling": "sigmoid", "sigmoid_scale": "style"}
        anima = {"pretrained_model_name_or_path": "a", "qwen3": "q", "vae": "v", "timestep_sampling": "shift"}
        aw.load_args({"anima_args": anima, "moss_args": moss, "network_args": {"cache_text_encoder_outputs": True}}, {})
        args = aw.get_args()["args"]
        self.assertNotIn("anima_args", args)
        self.assertEqual(args["moss_args"]["timestep_sampling"], "sigmoid")
        self.assertEqual(args["moss_args"]["sigmoid_scale"], "style")
        self.assertTrue(any("Only MOSS-Image Args was loaded" in w for w in aw.load_warnings))
        anima_box = aw.anima_widget.widget.anima_training_box
        moss_box = aw.moss_widget.widget.moss_training_box
        self.assertFalse(anima_box.isChecked())
        self.assertTrue(moss_box.isChecked() and moss_box.isEnabled())
        click(moss_box, False)
        self.assertTrue(anima_box.isEnabled())
        self.assertEqual(aw.get_validation_errors(), [])

    def test_flow_model_in_a_moss_toml_is_switched_off(self):
        aw = ArgsWidget()
        moss = {**MOSS_PATHS, "moss_tokenizer_dir": "tok", "timestep_sampling": "moss"}
        aw.load_args({"general_args": {"flow_model": True, "flow_use_ot": True}, "moss_args": moss,
                      "network_args": {"cache_text_encoder_outputs": True}}, {})
        args = aw.get_args()["args"]
        self.assertNotIn("flow_model", args["general_args"])
        self.assertEqual(clean(args["moss_args"]), moss)
        self.assertTrue(any("flow_model was switched off" in w for w in aw.load_warnings))
        self.assertEqual(aw.get_validation_errors(), [])

    def test_two_checked_boxes_are_an_error(self):
        aw = ArgsWidget()
        anima_on(aw)
        aw.moss_widget.widget.moss_training_box.setChecked(True)  # disabled, but checked
        self.assertTrue(any("More than one model type" in e for e in aw.get_validation_errors()))


class DoraReviewTest(unittest.TestCase):
    def setUp(self):
        self.n = NetworkWidget()

    def lokr_dora(self):
        self.n.widget.algo_select.setCurrentText("Lokr")
        self.n.widget.dora_enable.click()

    def test_keys_follow_written_dora_wd(self):
        self.lokr_dora()
        self.n.dora_lr_input.setText("1e-4")
        self.n.dora_wd_input.setText("1e-5")
        self.assertEqual(self.n.args["weight_decay_DoRA"], 1e-5)
        self.n.widget.algo_select.setCurrentText("OrthoLoRA")
        self.assertNotIn("dora_wd", self.n.args["network_args"])
        for key in ("learning_rate_DoRA", "weight_decay_DoRA"):
            self.assertNotIn(key, self.n.args)
        self.assertFalse(self.n.dora_lr_input.isEnabled())

    def test_small_weight_decay_round_trip(self):
        for text in ("1e-05", "5e-05", "0.0003"):
            n = NetworkWidget()
            n.widget.algo_select.setCurrentText("Lokr")
            n.widget.dora_enable.click()
            n.dora_wd_input.setText(text)
            saved = copy.deepcopy(n.args)
            self.assertEqual(saved["weight_decay_DoRA"], float(text))
            loaded = NetworkWidget()
            loaded.load_args({"network_args": copy.deepcopy(saved)})
            self.assertEqual(loaded.args, saved, text)
        self.n.widget.algo_select.setCurrentText("Lokr")
        self.n.widget.dora_enable.click()
        self.n.dora_wd_input.setText("-1")
        self.assertTrue(any("DoRA WD" in e for e in self.n.get_validation_errors()))

    def test_wd_on_output(self):
        box = self.n.dora_wd_on_output_enable
        self.assertTrue(box.isChecked())
        self.assertFalse(box.isEnabled())
        self.lokr_dora()
        self.assertTrue(box.isEnabled())
        self.assertNotIn("wd_on_output", self.n.args["network_args"])  # LyCORIS default: no key
        box.click()
        self.assertIs(self.n.args["network_args"]["wd_on_output"], False)
        saved = copy.deepcopy(self.n.args)
        loaded = NetworkWidget()
        loaded.load_args({"network_args": copy.deepcopy(saved)})
        self.assertEqual(loaded.args, saved)
        self.assertFalse(loaded.dora_wd_on_output_enable.isChecked())
        self.assertEqual([item.arg_name for item in loaded.network_args], [])  # not a free row
        self.n.widget.dora_enable.click()  # DoRA off: the key leaves
        self.assertNotIn("wd_on_output", self.n.args["network_args"])
        self.assertIn("adaptive", self.n.dora_lr_input.toolTip())

    def test_kv_list_with_bool_values_loads(self):
        n = NetworkWidget()
        n.load_args({"network_args": {"network_dim": 8, "network_args": [
            "algo=lokr", "dora_wd=True", "wd_on_output=False", "full_matrix=True", "factor=8"]}})
        self.assertIs(n.args["network_args"]["wd_on_output"], False)
        self.assertFalse(n.dora_wd_on_output_enable.isChecked())
        rows = {item.arg_name: item.arg_value for item in n.network_args}
        self.assertEqual(rows, {"full_matrix": "True", "factor": "8"})


class ExtraArgsMigrationTest(unittest.TestCase):
    def test_moves_fork_keys_into_controls(self):
        aw = ArgsWidget()
        moss = {**MOSS_PATHS, "moss_tokenizer_dir": "tok", "timestep_sampling": "moss"}
        aw.load_args({
            "moss_args": moss,
            "network_args": {"cache_text_encoder_outputs": True},
            "extra_args": {"use_ramtorch": True, "attn_mode": "flash", "moss_max_token_length": 8192,
                           "max_grad_norm": 1.0},
        }, {})
        args = aw.get_args()["args"]
        self.assertTrue(args["moss_args"]["use_ramtorch"])
        self.assertTrue(aw.moss_widget.widget.use_ramtorch_enable.isChecked())
        self.assertEqual(args["moss_args"]["attn_mode"], "flash")
        # the control cannot hold 8192 (max 4096): kept in Extra Args, with a warning and a validation error
        self.assertEqual(args["extra_args"], {"moss_max_token_length": 8192, "max_grad_norm": 1.0})
        self.assertTrue(any("Moved from Extra Args" in w and "use_ramtorch" in w for w in aw.load_warnings))
        self.assertTrue(any("Kept in Extra Args" in w and "moss_max_token_length" in w for w in aw.load_warnings))
        self.assertTrue(any("Extra Args 'moss_max_token_length'" in e for e in aw.get_validation_errors()))

    def test_anima38_keys(self):
        aw = ArgsWidget()
        anima = {"pretrained_model_name_or_path": "a", "qwen3": "q", "vae": "v", "timestep_sampling": "sigmoid",
                 "sigmoid_scale": 1.0}
        aw.load_args({
            "anima_args": anima,
            "network_args": {"cache_text_encoder_outputs": True},
            "extra_args": {"anima38_connector": True, "qwen35_path": "q35", "qwen35_tokenizer_dir": "tok",
                           "use_ramtorch": True},
        }, {})
        args = aw.get_args()["args"]
        self.assertNotIn("extra_args", args)
        self.assertTrue(aw.anima_widget.widget.anima38_box.isChecked())
        for key, value in (("anima38_connector", True), ("qwen35_path", "q35"), ("qwen35_tokenizer_dir", "tok"),
                           ("use_ramtorch", True)):
            self.assertEqual(args["anima_args"][key], value, key)
        self.assertEqual(aw.get_validation_errors(), [])

    def test_unrelated_and_inactive_keys_stay(self):
        aw = ArgsWidget()
        aw.load_args({"extra_args": {"use_ramtorch": True, "learning_rate_DoRA": 1e-4}}, {})
        self.assertEqual(aw.get_args()["args"]["extra_args"], {"use_ramtorch": True, "learning_rate_DoRA": 1e-4})
        self.assertEqual(aw.load_warnings, [])
        self.assertEqual(aw.get_validation_errors(), [])

    def test_typed_extra_arg_conflict(self):
        aw = ArgsWidget()
        moss_on(aw)
        aw.extra_args_widget.load({"use_ramtorch": True}, False)
        self.assertTrue(any("Extra Args 'use_ramtorch'" in e for e in aw.get_validation_errors()))

    def test_pure_helpers(self):
        args, moved = ForkArgs.migrate_extra_args(
            {"network_args": {"network_args": ["algo=lokr", "dora_wd=True"]},
             "extra_args": {"learning_rate_dora": 2e-4}})
        self.assertEqual(moved, [("learning_rate_DoRA", "network_args", 2e-4)])
        self.assertNotIn("extra_args", args)
        self.assertEqual(ForkArgs.unconfirmed(moved, {"network_args": {"learning_rate_DoRA": 2e-4}}), [])
        self.assertEqual(ForkArgs.unconfirmed(moved, {"network_args": {}}), moved)
        self.assertTrue(ForkArgs.same_value("moss_lora_exclude_blocks", [23, 22], "22,23"))
        self.assertTrue(ForkArgs.same_value("attn_mode", "sdpa", "torch"))
        self.assertTrue(ForkArgs.same_value("sigmoid_scale", "Style", "style"))


class TooltipTest(unittest.TestCase):
    def test_sample_negative(self):
        aw = ArgsWidget()
        tip = aw.anima_widget.widget.anima38_negative_selector.toolTip()
        self.assertIn("trainer's default", tip)
        self.assertIn("Forge's own default", tip)
        self.assertEqual(tip.count("default"), 2)


class DefaultsUnchangedTest(unittest.TestCase):
    def test_fresh_widget_writes_no_new_keys(self):
        args = ArgsWidget().get_args()["args"]
        self.assertEqual(args["network_args"],
                         {"network_dim": 32, "network_alpha": 16.0, "min_timestep": 0, "max_timestep": 1000})
        for section in ("moss_args", "anima_args", "flux_args", "extra_args"):
            self.assertNotIn(section, args)



class VerifyFindingsTest(unittest.TestCase):
    """Findings 1-8 of the second verification: UTF-8 TOMLs, the Anima 3.8B tokenizer, MOSS with DyLoRA / LoRA-FA,
    caption rules with cached TE outputs, moss presets without MOSS, preset whitespace, wd_on_output rows, unknown
    MOSS timestep_sampling / attn_mode."""

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        BOXES.clear()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    # 1
    def test_utf8_flat_trainer_toml_with_korean_comments(self):
        path = Path("config_moss_ko.toml")
        path.write_bytes((
            "# MOSS-Image 학습 설정 (명령줄 런처용)\n"
            'pretrained_model_name_or_path = "모델/moss_dit.safetensors"  # 한글 경로\n'
            'output_dir = "출력"\n'
            "network_dim = 32\n"
        ).encode("utf-8"))
        main = MainWidget()
        with self.assertRaises(TomlLoadError) as ctx:
            main.process_toml(path)
        self.assertIn("trainer config for the command-line launchers", str(ctx.exception))
        main.load_toml(path)
        self.assertEqual(BOXES[-1][1], "Cannot load TOML")
        self.assertIn("trainer config for the command-line launchers", BOXES[-1][2])
        # also when the system code page is cp949 (UTF-8 is tried first)
        with mock.patch.object(TomlFunctions.locale, "getpreferredencoding", return_value="cp949"):
            self.assertEqual(TomlFunctions.load_toml(path)["output_dir"], "출력")

    def test_locale_encoded_gui_toml_still_loads(self):
        path = Path("old_gui.toml")
        path.write_bytes('[general_args.args]\noutput_name = "한글이름"\n'.encode("cp949"))
        with mock.patch.object(TomlFunctions.locale, "getpreferredencoding", return_value="cp949"):
            self.assertEqual(TomlFunctions.load_toml(path)["general_args"]["args"]["output_name"], "한글이름")

    # 3
    def test_moss_with_dylora_or_lora_fa(self):
        aw = ArgsWidget()
        n = aw.network_widget
        n.widget.algo_select.setCurrentText("DyLoRA")
        self.assertFalse(any("DyLoRA / LoRA-FA" in e for e in aw.get_validation_errors()))  # not MOSS: fine
        moss_on(aw)
        self.assertTrue(any("DyLoRA / LoRA-FA find no MOSS-Image modules" in e for e in aw.get_validation_errors()))
        n.widget.algo_select.setCurrentText("LoRA")
        self.assertEqual(aw.get_validation_errors(), [])
        n.widget.lora_fa_enable.click()
        self.assertTrue(n.args.get("fa"))
        self.assertTrue(any("DyLoRA / LoRA-FA" in e for e in aw.get_validation_errors()))
        n.widget.lora_fa_enable.click()
        n.widget.algo_select.setCurrentText("Lokr")  # LyCORIS with preset moss (filled in by the MOSS switch)
        self.assertEqual(aw.get_validation_errors(), [])

    # 4
    def caption_errors(self, main: MainWidget) -> list[str]:
        return [e for e in main.get_validation_errors() if e.startswith("Subset ")]

    def test_caption_dropout_rate_needs_variants_except_plain_anima(self):
        for model in ("sdxl", "flux", "moss", "anima38"):
            main = MainWidget()
            aw = main.args_widget
            cache = aw.network_widget.widget.cache_te_outputs_enable
            if model == "sdxl":
                aw.general_widget.widget.sdxl_enable.click()
                click(cache)
            elif model == "flux":
                click(aw.flux_widget.widget.flux_training_box)
                click(cache)
            elif model == "moss":
                moss_on(aw)
            else:
                anima_on(aw, anima38=True)
            self.assertTrue(aw.network_widget.args.get("cache_text_encoder_outputs"), model)
            subset = next(iter(main.subset_widget.dataset_args.values()))
            subset["caption_dropout_rate"] = 0.1
            self.assertTrue(any("Caption Dropout Rate" in e and "Caption Variants > 1" in e
                                for e in self.caption_errors(main)), model)
            aw.network_widget.caption_variants_input.setValue(4)
            self.assertEqual(self.caption_errors(main), [], model)
            aw.network_widget.caption_variants_input.setValue(1)
            subset["caption_dropout_rate"] = 0.0
            self.assertEqual(self.caption_errors(main), [], model)
        # plain Anima applies caption dropout itself: allowed without variants
        main = MainWidget()
        aw = main.args_widget
        anima_on(aw)
        click(aw.network_widget.widget.cache_te_outputs_enable)
        self.assertTrue(aw.network_widget.args.get("cache_text_encoder_outputs"))
        subset = next(iter(main.subset_widget.dataset_args.values()))
        subset["caption_dropout_rate"] = 0.1
        self.assertEqual(self.caption_errors(main), [])
        subset["shuffle_caption"] = True  # still refused by plain Anima
        self.assertTrue(any("Shuffle Caption" in e for e in self.caption_errors(main)))
        # without cached outputs nothing applies
        main = MainWidget()
        main.args_widget.general_widget.widget.sdxl_enable.click()
        next(iter(main.subset_widget.dataset_args.values()))["caption_dropout_rate"] = 0.1
        self.assertEqual(self.caption_errors(main), [])
        self.assertIn("Caption Dropout Rate", main.args_widget.network_widget.caption_variants_input.toolTip())

    def test_token_warmup_with_cached_outputs(self):
        for model in ("sdxl", "anima", "moss"):
            main = MainWidget()
            aw = main.args_widget
            if model == "sdxl":
                aw.general_widget.widget.sdxl_enable.click()
                click(aw.network_widget.widget.cache_te_outputs_enable)
            elif model == "anima":
                anima_on(aw)
                click(aw.network_widget.widget.cache_te_outputs_enable)
            else:
                moss_on(aw)
            aw.network_widget.caption_variants_input.setValue(4)  # variants do not help token warmup
            subset = next(iter(main.subset_widget.dataset_args.values()))
            subset["token_warmup_min"] = 1
            subset["token_warmup_step"] = 1
            self.assertTrue(any("Token Warmup cannot be used with Cache TE Outputs" in e
                                for e in self.caption_errors(main)), model)
            subset["token_warmup_step"] = 0
            self.assertEqual(self.caption_errors(main), [], model)
        main = MainWidget()  # SD 1.5, no caching: token warmup is fine
        next(iter(main.subset_widget.dataset_args.values()))["token_warmup_step"] = 1
        self.assertEqual(self.caption_errors(main), [])

    # 5
    def test_moss_preset_without_moss(self):
        aw = ArgsWidget()
        n = aw.network_widget
        preset = n.widget.lycoris_preset_input
        n.widget.algo_select.setCurrentText("Lokr")
        for value in ("moss", "moss-text", " moss "):
            preset.setText(value)
            self.assertTrue(any("exists only for MOSS-Image training" in e for e in aw.get_validation_errors()),
                            value)
        preset.setText("attn-mlp")
        self.assertEqual(aw.get_validation_errors(), [])
        # a reloaded MOSS TOML keeps preset moss; switching MOSS off then needs the preset cleared
        moss_on(aw)
        preset.setText("moss")
        self.assertEqual(aw.get_validation_errors(), [])
        click(aw.moss_widget.widget.moss_training_box, False)
        self.assertTrue(any("exists only for MOSS-Image training" in e for e in aw.get_validation_errors()))
        n.widget.algo_select.setCurrentText("LoRA")  # kohya LoRA writes no preset
        self.assertFalse(any("exists only for MOSS-Image" in e for e in aw.get_validation_errors()))

    # 6
    def test_preset_is_saved_stripped(self):
        n = NetworkWidget()
        n.widget.algo_select.setCurrentText("Lokr")
        n.widget.lycoris_preset_input.setText("  moss ")
        self.assertEqual(n.args["network_args"]["preset"], "moss")
        n.widget.algo_select.setCurrentText("LoHa")  # change_algo writes it again
        self.assertEqual(n.args["network_args"]["preset"], "moss")
        loaded = NetworkWidget()
        loaded.load_args({"network_args": {"network_dim": 8, "network_args": {"algo": "lokr", "preset": " full "}}})
        self.assertEqual(loaded.widget.lycoris_preset_input.text(), "full")
        self.assertEqual(loaded.args["network_args"]["preset"], "full")
        aw = ArgsWidget()  # the MOSS switch writes it through its own path
        aw.network_widget.widget.algo_select.setCurrentText("Lokr")
        moss_on(aw)
        aw.network_widget.widget.lycoris_preset_input.setText(" moss-text ")
        self.assertEqual(aw.network_widget.args["network_args"]["preset"], "moss-text")
        self.assertEqual(aw.get_validation_errors(), [])

    # 7
    def test_wd_on_output_free_row_without_dora_stays(self):
        n = NetworkWidget()
        n.add_network_arg("wd_on_output", "False")
        n.modify_network_arg(n.network_args[-1])
        self.assertEqual(n.args["network_args"]["wd_on_output"], "False")
        for algo in ("Lokr", "LoHa", "LoRA"):
            n.widget.algo_select.setCurrentText(algo)
            self.assertEqual(n.args["network_args"].get("wd_on_output"), "False", algo)
        self.assertEqual([item.arg_name for item in n.network_args], ["wd_on_output"])
        loaded = NetworkWidget()
        loaded.load_args({"network_args": {"network_dim": 8,
                                           "network_args": {"algo": "lokr", "wd_on_output": False}}})
        # a free row (its value is kept as the row's text)
        self.assertEqual(str(loaded.args["network_args"]["wd_on_output"]), "False")
        self.assertEqual([item.arg_name for item in loaded.network_args], ["wd_on_output"])
        loaded.widget.algo_select.setCurrentText("LoHa")
        self.assertEqual(str(loaded.args["network_args"]["wd_on_output"]), "False")
        # the key the DoRA on output box wrote still leaves with DoRA
        d = NetworkWidget()
        d.widget.algo_select.setCurrentText("Lokr")
        d.widget.dora_enable.click()
        d.dora_wd_on_output_enable.click()
        self.assertIs(d.args["network_args"]["wd_on_output"], False)
        d.widget.dora_enable.click()
        self.assertNotIn("wd_on_output", d.args["network_args"])
        d.widget.algo_select.setCurrentText("LoHa")
        self.assertNotIn("wd_on_output", d.args["network_args"])

    def test_dora_toml_loaded_over_a_free_row_keeps_the_box_key(self):
        n = NetworkWidget()
        n.add_network_arg("wd_on_output", "True")
        n.modify_network_arg(n.network_args[-1])
        n.load_args({"network_args": {"network_dim": 8,
                                      "network_args": {"algo": "lokr", "dora_wd": True, "wd_on_output": False}}})
        self.assertIs(n.args["network_args"]["wd_on_output"], False)
        self.assertFalse(n.dora_wd_on_output_enable.isChecked())
        self.assertEqual(n.network_args, [])

    # 8
    def test_unknown_moss_timestep_sampling_and_attn_mode(self):
        aw = ArgsWidget()
        base = {**MOSS_PATHS, "moss_tokenizer_dir": "tok"}
        aw.load_args({"moss_args": {**base, "timestep_sampling": "sigma"}}, {})
        self.assertEqual(aw.moss_widget.widget.timestep_sampling_selector.currentText(), "sigma")
        self.assertEqual(aw.load_warnings, [])
        aw.load_args({"moss_args": {**base, "timestep_sampling": "flux_shift"}}, {})
        self.assertEqual(aw.moss_widget.widget.timestep_sampling_selector.currentText(), "moss")
        self.assertNotEqual(aw.moss_widget.args.get("timestep_sampling"), "flux_shift")
        self.assertNotEqual(aw.moss_widget.args.get("timestep_sampling"), "sigma")
        self.assertTrue(any('timestep_sampling = "flux_shift"' in w and "set to moss" in w
                            for w in aw.load_warnings))
        aw.load_args({"moss_args": {**base, "timestep_sampling": "moss", "attn_mode": "sdp_kernel"}}, {})
        warning = next(w for w in aw.load_warnings if "attn_mode" in w)
        self.assertIn('"sdp_kernel"', warning)
        self.assertIn("is not one the GUI offers", warning)
        self.assertNotIn("backward pass", warning)
        self.assertNotIn("attn_mode", aw.moss_widget.args)


if __name__ == "__main__":
    unittest.main()
