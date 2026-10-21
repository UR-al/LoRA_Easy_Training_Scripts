"""LyCORIS preset picker: a drop-down of the known presets plus a "Custom..." entry that turns the box into a text field.

It keeps the QLineEdit-style API the Network Args code used before (``text()``, ``setText()``, ``textChanged(str)``,
``setPlaceholderText()``), so callers read and write the preset as a plain string:

* ``""`` is the "(default)" entry: no ``preset`` key is written and LyCORIS uses its own default ("full").
* a listed name selects that entry (the box is not editable then).
* any other text is shown in Custom mode (editable) with that text - another preset name or a .toml preset file path.

Values are stripped. ``textChanged`` is sent only when the value changes, as a QLineEdit does.
"""

from contextlib import contextmanager

from PySide6 import QtCore, QtWidgets

from modules.ScrollOnSelect import ComboBox

DEFAULT_LABEL = "(default)"
CUSTOM_LABEL = "Custom..."
CUSTOM_PLACEHOLDER = "preset name or .toml preset file"

# (group title, preset names, what the group is for). The names are the keys of lycoris.config.PRESET and of the
# fork's library/moss/lycoris_presets.py MOSS_LYCORIS_PRESETS.
LYCORIS_PRESET_GROUPS = (
    (
        "LyCORIS",
        ("full", "full-lin", "attn-mlp", "attn-only", "unet-only", "unet-transformer-only", "unet-convblock-only",
         "ia3"),
        "LyCORIS built-in module sets (SD / SDXL / Flux and other DiT models): full also trains the text encoder, "
        "the unet-* presets do not, ia3 is the one for the IA3 algorithm",
    ),
    (
        "Anima",
        ("anima", "anima-exp", "anima-qwen", "anima-qwen-exp", "anima-inpaint", "anima-full-no-excludes"),
        "the Anima DiT blocks (none of them trains the llm_adapter): -qwen also trains the Qwen3 text encoder, -exp "
        "also the modulation layers, anima-full-no-excludes adds the patch / timestep embedders and final layer",
    ),
    (
        "MOSS-Image",
        ("moss", "moss-text"),
        "this fork's presets, for the MOSS-Image section only: moss targets the MOSS-Image DiT blocks (MossBlock), "
        "moss-text also the text refiner blocks (MossTextBlock)",
    ),
)

# What each Anima / MOSS-Image preset trains (from lycoris.config.PRESET and MOSS_LYCORIS_PRESETS); shown per entry.
PRESET_ITEM_TIPS = {
    "anima": "Every Anima DiT block: self-attention, cross-attention and MLP. Not the modulation layers, embedders, "
             "final layer or llm_adapter.",
    "anima-exp": "anima + the AdaLN modulation layers of every block.",
    "anima-qwen": "anima + the Qwen3 text encoder (attention, MLP). Trains the text encoder, so it does not work with "
                  "cached text-encoder outputs.",
    "anima-qwen-exp": "anima-exp + the Qwen3 text encoder. Not with cached text-encoder outputs.",
    "anima-inpaint": "Blocks 0-27 only (attention, MLP; no modulation): made for Anima base (28 blocks) - on 2.9B / 3.8B "
                     "the later blocks are not trained. Sets its own learning rates (2.5e-5 for blocks 0-17, 6e-5 for "
                     "18-27) and dim 16 for blocks 18-27.",
    "anima-full-no-excludes": "Everything in every block including the modulation layers, plus the patch and timestep "
                              "embedders and the final layer.",
    "moss": "MOSS-Image DiT blocks (MossBlock): q, k, v, o and the FFN.",
    "moss-text": "moss + the two text refiner blocks inside the MOSS-Image model (MossTextBlock); not the external "
                 "text encoder.",
}


def preset_tooltip(groups=LYCORIS_PRESET_GROUPS) -> str:
    lines = [
        "<b>LyCORIS Preset</b>: which modules the LyCORIS network trains.",
        f"<b>{DEFAULT_LABEL}</b>: writes no preset; LyCORIS then uses &quot;full&quot;.",
    ]
    lines += [f"<b>{title}</b>: {about}." for title, _names, about in groups]
    lines.append(f"<b>{CUSTOM_LABEL}</b>: type another preset name or the path of a .toml preset file.")
    return "<html><head/><body>" + "".join(f"<p>{line}</p>" for line in lines) + "</body></html>"


class PresetComboBox(ComboBox):
    """Drop-down of LyCORIS presets; "Custom..." makes it editable. See the module docstring."""

    textChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget = None, groups=LYCORIS_PRESET_GROUPS) -> None:
        super().__init__(parent)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._value = ""
        self._busy = 0
        self._placeholder = CUSTOM_PLACEHOLDER
        self._index_of: dict[str, int] = {}

        self.addItem(DEFAULT_LABEL, "")
        self._index_of[""] = 0
        for title, names, about in groups:
            self.add_section_header(title)
            self.setItemData(self.count() - 1, about, QtCore.Qt.ItemDataRole.ToolTipRole)
            for name in names:
                self.addItem(name, name)
                self.setItemData(self.count() - 1, PRESET_ITEM_TIPS.get(name, about), QtCore.Qt.ItemDataRole.ToolTipRole)
                self._index_of[name] = self.count() - 1
        self.insertSeparator(self.count())
        self.addItem(CUSTOM_LABEL)
        self._custom_index = self.count() - 1
        self._last_index = 0
        self.setCurrentIndex(0)
        self.setToolTip(preset_tooltip(groups))

        self.currentIndexChanged.connect(self._on_index_changed)
        self.editTextChanged.connect(self._on_edit_text_changed)
        self.activated.connect(self._on_activated)

    # --- QLineEdit-style API ---

    def presets(self) -> tuple[str, ...]:
        """The listed preset names (without the default entry), in order."""
        return tuple(name for name in self._index_of if name)

    def is_custom(self) -> bool:
        return self.currentIndex() == self._custom_index

    def text(self) -> str:
        if self.is_custom():
            line = self.lineEdit()
            return line.text().strip() if line is not None else ""
        data = self.itemData(self.currentIndex())
        return data.strip() if isinstance(data, str) else ""

    def setText(self, text) -> None:
        value = "" if text is None else str(text).strip()
        with self._quiet():
            if value in self._index_of:
                self.setCurrentIndex(self._index_of[value])
                self._set_custom_mode(False)
            else:
                self.setCurrentIndex(self._custom_index)
                self._set_custom_mode(True, value)
        self._last_index = self.currentIndex()
        self._emit_if_changed()

    def clear(self) -> None:  # QLineEdit.clear(), not QComboBox.clear() (which would drop the items)
        self.setText("")

    def setPlaceholderText(self, text: str) -> None:
        """Placeholder of the Custom text field (QComboBox's own placeholder only shows with no current item)."""
        self._placeholder = text
        if self.lineEdit() is not None:
            self.lineEdit().setPlaceholderText(text)

    def placeholderText(self) -> str:
        return self._placeholder

    # --- internals ---

    @contextmanager
    def _quiet(self):
        self._busy += 1
        try:
            yield
        finally:
            self._busy -= 1

    def _set_custom_mode(self, on: bool, text: str = "") -> None:
        with self._quiet():
            if on:
                if not self.isEditable():
                    self.setEditable(True)
                    line = self.lineEdit()
                    line.setCompleter(None)  # no inline completion to "Custom..." / a header
                    line.setPlaceholderText(self._placeholder)
                    line.editingFinished.connect(self._on_editing_finished)
                self.lineEdit().setText(text)
            elif self.isEditable():
                self.setEditable(False)

    def _emit_if_changed(self) -> None:
        value = self.text()
        if value != self._value:
            self._value = value
            self.textChanged.emit(value)

    def _on_index_changed(self, index: int) -> None:
        if self._busy:
            return
        if index != self._custom_index and index not in self._index_of.values():
            # a section header or the separator (e.g. Enter on its text in Custom mode): stay where we were
            typed = self.lineEdit().text() if self.lineEdit() is not None else ""
            with self._quiet():
                self.setCurrentIndex(self._last_index)
                if self._last_index == self._custom_index and self.lineEdit() is not None:
                    self.lineEdit().setText(typed)
            return
        self._last_index = index
        if index == self._custom_index:
            self._set_custom_mode(True, "")
            self.lineEdit().setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            # the closing popup may take focus back; focus the text field again once it is gone
            QtCore.QTimer.singleShot(0, self._focus_custom)
        else:
            self._set_custom_mode(False)
        self._emit_if_changed()

    def _focus_custom(self) -> None:
        if self.is_custom() and self.lineEdit() is not None:
            self.lineEdit().setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _on_activated(self, index: int) -> None:
        # Picking "Custom..." again while in Custom mode keeps the index (no currentIndexChanged) but Qt writes the
        # entry's label into the text field: start a fresh field instead of keeping "Custom..." as the value.
        # Only then: Enter in the field also sends activated(custom index), and the typed text must stay.
        line = self.lineEdit()
        if index == self._custom_index and self.is_custom() and line is not None and line.text() == CUSTOM_LABEL:
            line.setText("")
            self._focus_custom()

    def _on_edit_text_changed(self, _text: str) -> None:
        if self._busy or not self.is_custom() or _text == CUSTOM_LABEL:
            return  # CUSTOM_LABEL: Qt's own write when "Custom..." is picked again (see _on_activated)
        self._emit_if_changed()

    def _on_editing_finished(self) -> None:
        # a listed name typed in Custom mode becomes that entry
        if self._busy or not self.is_custom():
            return
        value = self.text()
        if value and value in self._index_of:
            QtCore.QTimer.singleShot(0, lambda: self.is_custom() and self.text() == value and self.setText(value))
