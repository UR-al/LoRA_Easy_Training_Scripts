"""LyCORIS Preset drop-down (modules/PresetComboBox.py) in Network Args: listed presets, "(default)" and "Custom...",
TOML round trips, the MOSS-Image auto-fill and the preset validation.

Run from the repo root (CPU only, no backend; message boxes are captured, never shown):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_lycoris_preset_combo
"""

import ast
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

from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.ArgsListUI import ArgsWidget
from main_ui_files.MainUI import MainWidget
from main_ui_files.NetworkUI import NetworkWidget
from modules.PresetComboBox import CUSTOM_LABEL, DEFAULT_LABEL, LYCORIS_PRESET_GROUPS, PresetComboBox

BOXES: list = []
LYCORIS_PRESETS = ("full", "full-lin", "attn-mlp", "attn-only", "unet-only", "unet-transformer-only",
                   "unet-convblock-only", "ia3")
ANIMA_PRESETS = ("anima", "anima-exp", "anima-qwen", "anima-qwen-exp", "anima-inpaint", "anima-full-no-excludes")
MOSS_PRESETS = ("moss", "moss-text")
ALL_PRESETS = LYCORIS_PRESETS + ANIMA_PRESETS + MOSS_PRESETS


def setUpModule():
    for kind in ("warning", "critical", "information"):
        patcher = mock.patch.object(
            QtWidgets.QMessageBox, kind, staticmethod(lambda *a, kind=kind, **k: BOXES.append((kind, a[1], a[2])))
        )
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


def click(box, value: bool = True) -> None:
    box.setChecked(value)
    box.clicked[bool].emit(value)


def moss_on(aw: ArgsWidget):
    w = aw.moss_widget.widget
    w.dit_model_input.setText("models/moss_dit.safetensors")
    w.vae_model_input.setText("models/flux2_vae.safetensors")
    w.text_encoder_input.setText("models/moss_te.safetensors")
    w.tokenizer_dir_input.setText("tok")
    click(w.moss_training_box)
    return aw.moss_widget


def choose(combo: PresetComboBox, label: str) -> None:
    """Pick an entry the way the drop-down does (by its shown text)."""
    index = combo.findText(label)
    assert index >= 0, label
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def dict_keys_of(path: Path, name: str) -> list[str]:
    """Top-level string keys of the dict assigned to ``name`` in a Python file (read only, never imported)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{name} not found in {path}")


class ItemsTest(unittest.TestCase):
    def test_order_and_modes(self):
        combo = PresetComboBox()
        self.assertEqual(combo.itemText(0), DEFAULT_LABEL)
        self.assertEqual(combo.itemText(combo.count() - 1), CUSTOM_LABEL)
        self.assertEqual(combo.presets(), ALL_PRESETS)
        self.assertEqual([title for title, _n, _a in LYCORIS_PRESET_GROUPS], ["LyCORIS", "Anima", "MOSS-Image"])
        self.assertEqual(combo.currentIndex(), 0)
        self.assertFalse(combo.isEditable())
        self.assertEqual(combo.text(), "")
        # group headers / the separator cannot be picked
        for index in range(combo.count()):
            if combo.itemText(index) not in ALL_PRESETS + (DEFAULT_LABEL, CUSTOM_LABEL):
                self.assertFalse(combo.model().item(index).isEnabled(), combo.itemText(index))
                combo.setCurrentIndex(index)
                self.assertEqual(combo.currentIndex(), 0)
        tip = combo.toolTip()
        for title in ("(default)", "LyCORIS", "Anima", "MOSS-Image", "Custom..."):
            self.assertIn(f"<b>{title}</b>", tip)
        self.assertIn("MossBlock", tip)
        self.assertIn("MossTextBlock", tip)
        self.assertEqual(tip.count("<p>"), 6)

    def test_names_match_the_backend(self):
        lycoris_config = ROOT / "backend/sd_scripts/venv/Lib/site-packages/lycoris/config.py"
        moss_presets = ROOT / "backend/sd_scripts/library/moss/lycoris_presets.py"
        if not lycoris_config.exists() or not moss_presets.exists():
            self.skipTest("backend not installed")
        self.assertEqual(set(dict_keys_of(lycoris_config, "PRESET")), set(LYCORIS_PRESETS + ANIMA_PRESETS))
        self.assertEqual(dict_keys_of(moss_presets, "MOSS_LYCORIS_PRESETS"), list(MOSS_PRESETS))
        # what the groups claim about the MOSS presets
        text = moss_presets.read_text(encoding="utf-8")
        self.assertIn('"moss": {"enable_conv": False, "unet_target_module": ["MossBlock"]', text)
        self.assertIn('"unet_target_module": ["MossBlock", "MossTextBlock"]', text)

    def test_line_edit_api(self):
        combo = PresetComboBox()
        sent = []
        combo.textChanged.connect(sent.append)
        combo.setText(" full ")
        self.assertEqual((combo.currentText(), combo.text(), combo.isEditable()), ("full", "full", False))
        combo.setText("full")  # unchanged: no signal, as QLineEdit
        combo.setText("  presets/a.toml ")
        self.assertTrue(combo.is_custom())
        self.assertTrue(combo.isEditable())
        self.assertEqual(combo.lineEdit().text(), "presets/a.toml")
        combo.setText(None)
        self.assertEqual(combo.currentText(), DEFAULT_LABEL)
        self.assertEqual(sent, ["full", "presets/a.toml", ""])
        combo.setPlaceholderText("type here")
        choose(combo, CUSTOM_LABEL)
        self.assertEqual(combo.lineEdit().placeholderText(), "type here")
        combo.clear()  # QLineEdit.clear(): back to the default entry, items stay
        self.assertEqual(combo.presets(), ALL_PRESETS)
        self.assertEqual(combo.currentIndex(), 0)


class NetworkArgsTest(unittest.TestCase):
    def setUp(self):
        self.n = NetworkWidget()
        self.combo = self.n.widget.lycoris_preset_input
        self.assertIsInstance(self.combo, PresetComboBox)

    def preset_arg(self):
        return self.n.args.get("network_args", {}).get("preset")

    def test_default_writes_nothing(self):
        self.assertFalse(self.combo.isEnabled())  # LoRA
        self.assertIn("LyCORIS Preset", self.n.widget.label.toolTip())
        for algo in ("Lokr", "LoHa", "IA3", "LoCon (LyCORIS)"):
            self.n.widget.algo_select.setCurrentText(algo)
            self.assertTrue(self.combo.isEnabled(), algo)
            self.assertEqual(self.combo.currentText(), DEFAULT_LABEL)
            self.assertIsNone(self.preset_arg(), algo)
        choose(self.combo, "attn-only")
        self.assertEqual(self.preset_arg(), "attn-only")
        choose(self.combo, DEFAULT_LABEL)
        self.assertIsNone(self.preset_arg())
        self.n.widget.algo_select.setCurrentText("LoRA")
        self.assertFalse(self.combo.isEnabled())

    def test_custom_listed_custom(self):
        self.n.widget.algo_select.setCurrentText("Lokr")
        choose(self.combo, CUSTOM_LABEL)
        self.assertTrue(self.combo.isEditable())
        self.assertEqual(self.combo.lineEdit().text(), "")
        self.assertIsNone(self.preset_arg())
        QTest.keyClicks(self.combo.lineEdit(), "  presets/mine.toml ")
        self.assertEqual(self.combo.text(), "presets/mine.toml")
        self.assertEqual(self.preset_arg(), "presets/mine.toml")
        choose(self.combo, "full-lin")
        self.assertFalse(self.combo.isEditable())
        self.assertEqual(self.preset_arg(), "full-lin")
        choose(self.combo, CUSTOM_LABEL)
        self.assertTrue(self.combo.isEditable())
        self.assertEqual(self.combo.lineEdit().text(), "")  # a fresh field
        self.assertIsNone(self.preset_arg())
        QTest.keyClicks(self.combo.lineEdit(), "my-preset")
        self.assertEqual(self.preset_arg(), "my-preset")
        self.n.widget.algo_select.setCurrentText("LoHa")  # change_algo writes it again
        self.assertEqual(self.preset_arg(), "my-preset")

    def test_picking_custom_again_starts_a_fresh_field(self):
        self.n.widget.algo_select.setCurrentText("Lokr")
        choose(self.combo, CUSTOM_LABEL)
        QTest.keyClicks(self.combo.lineEdit(), "mine")
        self.assertEqual(self.preset_arg(), "mine")
        # what Qt does when the current "Custom..." entry is picked again from the popup: the index stays, the field
        # gets the entry's label, then activated(index) is sent
        self.combo.lineEdit().setText(CUSTOM_LABEL)
        self.combo.activated.emit(self.combo.currentIndex())
        self.assertTrue(self.combo.is_custom())
        self.assertEqual(self.combo.lineEdit().text(), "")
        self.assertIsNone(self.preset_arg())  # never preset="Custom..."
        QTest.keyClicks(self.combo.lineEdit(), "x")
        self.assertEqual(self.preset_arg(), "x")

    def test_entries_explain_what_they_train(self):
        tip = lambda name: self.combo.itemData(self.combo.findText(name), QtCore.Qt.ItemDataRole.ToolTipRole)
        self.assertIn("Blocks 0-27 only", tip("anima-inpaint"))
        self.assertIn("cached text-encoder outputs", tip("anima-qwen"))
        self.assertIn("MossTextBlock", tip("moss-text"))

    def test_typed_listed_name_selects_the_entry(self):
        self.n.widget.algo_select.setCurrentText("Lokr")
        choose(self.combo, CUSTOM_LABEL)
        QTest.keyClicks(self.combo.lineEdit(), "attn-mlp")
        QTest.keyClick(self.combo.lineEdit(), QtCore.Qt.Key.Key_Return)
        APP.processEvents()
        self.assertFalse(self.combo.isEditable())
        self.assertEqual(self.combo.currentText(), "attn-mlp")
        self.assertEqual(self.preset_arg(), "attn-mlp")
        # a group header's text stays custom text
        choose(self.combo, CUSTOM_LABEL)
        QTest.keyClicks(self.combo.lineEdit(), "- Anima -")
        QTest.keyClick(self.combo.lineEdit(), QtCore.Qt.Key.Key_Return)
        APP.processEvents()
        self.assertTrue(self.combo.is_custom())
        self.assertEqual(self.preset_arg(), "- Anima -")

    def test_load(self):
        cases = (({"preset": "anima-qwen"}, "anima-qwen", False), ({"preset": " my-preset "}, "my-preset", True),
                 ({"preset": "presets/x.toml"}, "presets/x.toml", True), ({}, "", False))
        for network_args, text, custom in cases:
            with self.subTest(network_args=network_args):
                self.combo.setText("unet-only")
                self.n.load_args({"network_args": {"network_dim": 8,
                                                   "network_args": {"algo": "lokr", **network_args}}})
                self.assertEqual(self.combo.text(), text)
                self.assertEqual(self.combo.is_custom(), custom)
                self.assertEqual(self.combo.isEditable(), custom)
                if custom:
                    self.assertEqual(self.combo.lineEdit().text(), text)
                else:
                    self.assertEqual(self.combo.currentText(), text or DEFAULT_LABEL)
                self.assertEqual(self.preset_arg(), text or None)


class TomlRoundTripTest(unittest.TestCase):
    def round_trip(self, set_preset, moss: bool = False):
        main = MainWidget()
        aw = main.args_widget
        aw.network_widget.widget.algo_select.setCurrentText("Lokr")
        if moss:
            moss_on(aw)
        set_preset(aw.network_widget.widget.lycoris_preset_input)
        args = aw.network_widget.args["network_args"].get("preset")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.toml"
            self.assertTrue(main.save_toml(path), BOXES[-1:])
            text = path.read_text(encoding="utf-8")
            loaded = MainWidget()
            loaded.load_toml(path)
        return args, text, loaded.args_widget.network_widget

    def test_each_listed_preset(self):
        for name in ALL_PRESETS:
            with self.subTest(name=name):
                args, text, n = self.round_trip(lambda c: choose(c, name), moss=name in MOSS_PRESETS)
                self.assertEqual(args, name)
                self.assertIn(f'preset = "{name}"', text)
                combo = n.widget.lycoris_preset_input
                self.assertEqual(combo.currentText(), name)
                self.assertFalse(combo.isEditable())
                self.assertEqual(n.args["network_args"]["preset"], name)

    def test_default_and_custom(self):
        args, text, n = self.round_trip(lambda c: choose(c, DEFAULT_LABEL))
        self.assertIsNone(args)
        self.assertNotIn("preset", text)
        self.assertEqual(n.widget.lycoris_preset_input.currentText(), DEFAULT_LABEL)
        self.assertNotIn("preset", n.args["network_args"])

        def typed(combo):
            choose(combo, CUSTOM_LABEL)
            QTest.keyClicks(combo.lineEdit(), " presets/my_preset.toml ")

        args, text, n = self.round_trip(typed)
        self.assertEqual(args, "presets/my_preset.toml")
        self.assertIn('preset = "presets/my_preset.toml"', text)
        combo = n.widget.lycoris_preset_input
        self.assertTrue(combo.is_custom())
        self.assertTrue(combo.isEditable())
        self.assertEqual(combo.lineEdit().text(), "presets/my_preset.toml")
        self.assertEqual(n.args["network_args"]["preset"], "presets/my_preset.toml")


class MossTest(unittest.TestCase):
    def setUp(self):
        self.aw = ArgsWidget()
        self.n = self.aw.network_widget
        self.combo = self.n.widget.lycoris_preset_input
        self.n.widget.algo_select.setCurrentText("Lokr")

    def test_fill_and_restore(self):
        moss = moss_on(self.aw)
        self.assertEqual(self.combo.currentText(), "moss")
        self.assertFalse(self.combo.isEditable())
        self.assertEqual(self.n.args["network_args"]["preset"], "moss")
        self.assertEqual(self.aw.get_validation_errors(), [])
        click(moss.widget.moss_training_box, False)
        self.assertEqual(self.combo.currentText(), DEFAULT_LABEL)
        self.assertNotIn("preset", self.n.args["network_args"])

    def test_kept_entry_and_custom_text(self):
        for pick in (lambda: choose(self.combo, "moss-text"), lambda: self.combo.setText("presets/moss_mine.toml")):
            pick()
            before = (self.combo.currentIndex(), self.combo.text(), self.combo.isEditable())
            moss = moss_on(self.aw)
            self.assertEqual((self.combo.currentIndex(), self.combo.text(), self.combo.isEditable()), before)
            click(moss.widget.moss_training_box, False)
            self.assertEqual((self.combo.currentIndex(), self.combo.text(), self.combo.isEditable()), before)
            self.assertEqual(self.n.args["network_args"]["preset"], before[1])

    def test_validation(self):
        for name in MOSS_PRESETS:
            choose(self.combo, name)
            self.assertTrue(any("exists only for MOSS-Image training" in e for e in self.aw.get_validation_errors()),
                            name)
        choose(self.combo, CUSTOM_LABEL)
        QTest.keyClicks(self.combo.lineEdit(), " moss ")  # typed: still the MOSS preset
        self.assertTrue(any("exists only for MOSS-Image training" in e for e in self.aw.get_validation_errors()))
        choose(self.combo, "full")
        self.assertEqual(self.aw.get_validation_errors(), [])
        moss_on(self.aw)
        self.assertEqual(self.combo.currentText(), "full")  # not empty: left alone
        self.assertTrue(any("LyCORIS Preset = moss" in e for e in self.aw.get_validation_errors()))
        choose(self.combo, DEFAULT_LABEL)
        errors = self.aw.get_validation_errors()
        self.assertTrue(any("LyCORIS Preset = moss" in e and "an empty preset" in e for e in errors), errors)
        choose(self.combo, "moss")
        self.assertEqual(self.aw.get_validation_errors(), [])


if __name__ == "__main__":
    unittest.main()
