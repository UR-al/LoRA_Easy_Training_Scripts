"""Sigmoid scale input shared by the Anima and MOSS-Image sections.

``--sigmoid_scale`` of the fork's Anima / MOSS trainers takes a number or one of three presets
(library/timestep_presets.py): ``balanced`` (= 1.0), ``content`` (high noise) and ``style`` (low noise).
The presets only work with ``timestep_sampling = "sigmoid"``.
"""

from PySide6 import QtCore, QtWidgets

from modules.ScrollOnSelect import ComboBox

SIGMOID_PRESETS = ("balanced", "content", "style")
INVALID_STYLE = "QComboBox { border: 1px solid #c94040; }"


def parse_sigmoid_scale(text) -> float | str | None:
    """A positive number -> float, a preset name (any case) -> its lower-case name, anything else -> None."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        value = float(text)
        return value if value > 0 else None
    text = str(text).strip()
    if not text:
        return None
    if text.lower() in SIGMOID_PRESETS:
        return text.lower()
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        return None
    return value


def sigmoid_scale_for_sampling(text, sampling: str) -> float | str | None:
    """parse_sigmoid_scale, plus: a preset name is only valid with timestep_sampling = "sigmoid"."""
    value = parse_sigmoid_scale(text)
    if isinstance(value, str) and sampling != "sigmoid":
        return None
    return value


def sigmoid_scale_text(value) -> str:
    """Text shown for a loaded value: presets as their name, numbers as str(float)."""
    parsed = parse_sigmoid_scale(value)
    if isinstance(parsed, float):
        return str(parsed)
    if isinstance(parsed, str):
        return parsed
    return str(value)


class SigmoidScaleInput(ComboBox):
    """Editable combo box: type a number or pick a preset. ``valueChanged`` sends the current text."""

    valueChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.addItems(["1.0", *SIGMOID_PRESETS])
        self.setCurrentIndex(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setMinimumContentsLength(8)
        self.editTextChanged.connect(lambda text: self.valueChanged.emit(text))

    def text(self) -> str:
        return self.currentText()

    def value(self) -> float | str | None:
        return parse_sigmoid_scale(self.currentText())

    def set_value(self, value) -> None:
        self.setEditText(sigmoid_scale_text(value))

    def set_invalid(self, invalid: bool) -> None:
        self.setStyleSheet(INVALID_STYLE if invalid else "")
