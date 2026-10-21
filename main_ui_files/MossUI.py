import os
import re

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QPushButton
from modules.DragDropLineEdit import DragDropLineEdit
from modules.SigmoidScaleInput import sigmoid_scale_for_sampling
from ui_files.MossUI import (
    Ui_moss_ui, MOSS_ATTN_ALIASES, MOSS_ATTN_MODES, MOSS_MAX_TOKENS, MOSS_NUM_BLOCKS, MOSS_TIMESTEP_SAMPLING,
)
from modules.BaseWidget import BaseWidget, signals_blocked
from pathlib import Path

INVALID_LINE_STYLE = "QLineEdit { border: 1px solid #c94040; }"
TOKENIZER_DIR_ENV = "MOSS_TOKENIZER_DIR"  # library/moss/qwen35.py: the only fallback of an empty --moss_tokenizer_dir


def parse_block_list(value, num_blocks: int = MOSS_NUM_BLOCKS) -> list[int] | None:
    """"22", "22, 23", "22 23", 22 or [22, 23] -> sorted block indices; "" -> []; a bad entry -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [item for item in re.split(r"[,\s]+", str(value).strip()) if item]
    blocks = set()
    for item in items:
        if isinstance(item, bool):
            return None
        text = str(item).strip()
        if not text.isdigit():
            return None
        index = int(text)
        if index >= num_blocks:
            return None
        blocks.add(index)
    return sorted(blocks)


def block_list_text(value) -> str:
    """Text shown for a loaded block list (an array becomes "22,23")."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return "" if value is None else str(value)


class MossWidget(BaseWidget):
    Toggled = Signal(bool)  # send to general args

    # Trainer defaults (moss_train_network.py): a control at its default writes no key.
    TRAINER_DEFAULTS = {
        "moss_max_token_length": 4096,
        "moss_logit_mean": 1.09,
        "moss_logit_std": 0.95,
        "moss_sigma_cutoff": 0.95,
        "moss_uniform_mix": 0.01,
        "sigmoid_scale": 1.0,
        "scale_resume": 0,
        "discrete_flow_shift": 1.0,
        "blocks_to_swap": 0,
        "moss_sample_shift": 5.0,
        "moss_sample_uncond_skip_layers": "22",
    }
    ADVANCED_SWITCHES = (
        ("no_compile_enable", "moss_no_compile"),
        ("no_lokr_compile_enable", "moss_no_lokr_compile"),
        ("stock_lycoris_enable", "moss_stock_lycoris"),
        ("no_sample_cache_enable", "moss_no_sample_cache"),
    )

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._externally_disabled = False
        self.load_warnings: list[str] = []  # filled by load_args, shown by MainWidget.load_toml
        self.colap.set_title("MOSS-Image Args")
        self.widget = Ui_moss_ui()

        self.name = "moss_args"

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        def setup_file(elem: DragDropLineEdit, selector: QPushButton, folder=False):
            selector_icon = QIcon(str(Path("icons/more-horizontal.svg")))
            if folder:
                elem.setMode("folder")
            else:
                elem.setMode("file", [".ckpt", ".pt", ".safetensors", ".sft", ".pth"])
            elem.highlight = True
            selector.setIcon(selector_icon)

        setup_file(self.widget.dit_model_input, self.widget.dit_model_selector)
        setup_file(self.widget.vae_model_input, self.widget.vae_model_selector)
        setup_file(self.widget.text_encoder_input, self.widget.text_encoder_selector)
        setup_file(self.widget.tokenizer_dir_input, self.widget.tokenizer_dir_selector, folder=True)
        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())
        self.args = {}

    def setup_connections(self) -> None:
        self.widget.moss_training_box.clicked.connect(self.enable_disable)

        # File inputs — keys match sd_scripts argparse names
        self.widget.dit_model_input.textChanged.connect(lambda x: self._edit_if_on("pretrained_model_name_or_path", x))
        self.widget.dit_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.dit_model_input, "MOSS-Image DiT Model", "Model File")
        )
        self.widget.vae_model_input.textChanged.connect(lambda x: self._edit_if_on("vae", x))
        self.widget.vae_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.vae_model_input, "FLUX.2 VAE Model", "Model File")
        )
        self.widget.text_encoder_input.textChanged.connect(lambda x: self._edit_if_on("moss_text_encoder", x))
        self.widget.text_encoder_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.text_encoder_input, "Qwen3.5-9B Text Encoder", "Model File")
        )
        self.widget.tokenizer_dir_input.textChanged.connect(
            lambda x: self._edit_if_on("moss_tokenizer_dir", x, optional=True)
        )
        self.widget.tokenizer_dir_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.tokenizer_dir_input, "Qwen3.5 Tokenizer Folder")
        )

        self.widget.max_token_input.valueChanged.connect(lambda x: self._set_or_drop_if_on("moss_max_token_length", x))

        # Timesteps
        self.widget.timestep_sampling_selector.currentIndexChanged.connect(self.change_timestep_sampling_type)
        self.widget.discrete_flow_shift_input.valueChanged.connect(lambda _: self.change_discrete_flow_shift())
        self.widget.sigmoid_scale_input.valueChanged.connect(lambda _: self.change_sigmoid_scale())
        self.widget.scale_resume_input.valueChanged.connect(lambda _: self.change_scale_resume())
        self.widget.logit_mean_input.valueChanged.connect(lambda _: self.change_moss_logit_args())
        self.widget.logit_std_input.valueChanged.connect(lambda _: self.change_moss_logit_args())
        self.widget.sigma_cutoff_input.valueChanged.connect(lambda _: self.change_moss_logit_args())
        self.widget.uniform_mix_input.valueChanged.connect(lambda _: self.change_moss_logit_args())

        # Attention / memory
        self.widget.attn_mode_selector.currentIndexChanged.connect(self.change_attn_mode)
        self.widget.split_attn_enable.clicked.connect(lambda x: self._edit_if_on("split_attn", x, True))
        self.widget.blocks_to_swap_input.valueChanged.connect(lambda x: self._set_or_drop_if_on("blocks_to_swap", x))
        self.widget.use_ramtorch_enable.clicked.connect(lambda x: self._edit_if_on("use_ramtorch", x, True))
        self.widget.unsloth_offload_checkpointing.clicked.connect(
            lambda x: self._edit_if_on("unsloth_offload_checkpointing", x, True)
        )

        # Network / samples
        self.widget.exclude_blocks_input.textChanged.connect(lambda _: self.change_exclude_blocks())
        self.widget.sample_shift_input.valueChanged.connect(lambda x: self._set_or_drop_if_on("moss_sample_shift", x))
        self.widget.uncond_skip_layers_input.textChanged.connect(lambda _: self.change_uncond_skip_layers())

        # Advanced: disable speedups
        for widget_name, key in self.ADVANCED_SWITCHES:
            getattr(self.widget, widget_name).clicked.connect(lambda x, key=key: self._edit_if_on(key, x, True))

    def _set_or_drop(self, key: str, value: object, default: object = None) -> None:
        """Write the value, or remove the key when it equals the trainer default."""
        if default is None:
            default = self.TRAINER_DEFAULTS[key]
        if key in self.args:
            del self.args[key]
        if isinstance(default, float) and isinstance(value, (int, float)) and abs(value - default) < 1e-9:
            return
        if value == default:
            return
        self.args[key] = value

    def _edit_if_on(self, key: str, value: object, optional: bool = False) -> None:
        if self._section_on():
            self.edit_args(key, value, optional)

    def _set_or_drop_if_on(self, key: str, value: object) -> None:
        if self._section_on():
            self._set_or_drop(key, value)

    def _drop(self, *keys: str) -> None:
        for key in keys:
            if key in self.args:
                del self.args[key]

    def enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.Toggled.emit(checked)
        if not checked or self._externally_disabled:
            return  # a checked box that another model type disabled writes nothing

        self.edit_args("pretrained_model_name_or_path", self.widget.dit_model_input.text())
        self.edit_args("vae", self.widget.vae_model_input.text())
        self.edit_args("moss_text_encoder", self.widget.text_encoder_input.text())
        self.edit_args("moss_tokenizer_dir", self.widget.tokenizer_dir_input.text(), optional=True)
        self._set_or_drop("moss_max_token_length", self.widget.max_token_input.value())

        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())

        self.change_attn_mode(self.widget.attn_mode_selector.currentIndex())
        self.edit_args("split_attn", self.widget.split_attn_enable.isChecked(), True)
        self._set_or_drop("blocks_to_swap", self.widget.blocks_to_swap_input.value())
        self.edit_args("use_ramtorch", self.widget.use_ramtorch_enable.isChecked(), True)
        self.edit_args("unsloth_offload_checkpointing", self.widget.unsloth_offload_checkpointing.isChecked(), True)

        self.change_exclude_blocks()
        self._set_or_drop("moss_sample_shift", self.widget.sample_shift_input.value())
        self.change_uncond_skip_layers()

        for widget_name, key in self.ADVANCED_SWITCHES:
            self.edit_args(key, getattr(self.widget, widget_name).isChecked(), True)

    def external_enable_disable(self, disabled: bool) -> None:
        self.args = {}
        self._externally_disabled = disabled
        self.widget.moss_training_box.setEnabled(not disabled)
        if self.widget.moss_training_box.isEnabled() and self.widget.moss_training_box.isChecked():
            self.enable_disable(True)

    def _sampling(self) -> str:
        return self.widget.timestep_sampling_selector.currentText()

    def _section_on(self) -> bool:
        return self.widget.moss_training_box.isChecked() and not self._externally_disabled

    def change_timestep_sampling_type(self, index: int) -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        if self._section_on():
            self.edit_args("timestep_sampling", sampling_type)

        # moss: logit-normal of the released Base; sigmoid/shift: sigmoid_scale; sigmoid only: scale_resume;
        # sigma/shift: discrete_flow_shift
        self.widget.logit_mean_input.setEnabled(sampling_type == "moss")
        self.widget.logit_std_input.setEnabled(sampling_type == "moss")
        self.widget.sigma_cutoff_input.setEnabled(sampling_type == "moss")
        self.widget.uniform_mix_input.setEnabled(sampling_type == "moss")
        self.widget.sigmoid_scale_input.setEnabled(sampling_type in ("sigmoid", "shift"))
        self.widget.scale_resume_input.setEnabled(sampling_type == "sigmoid")
        self.widget.discrete_flow_shift_input.setEnabled(sampling_type in ("sigma", "shift"))

        self.change_moss_logit_args()
        self.change_sigmoid_scale()
        self.change_scale_resume()
        self.change_discrete_flow_shift()

    def change_moss_logit_args(self) -> None:
        keys = ("moss_logit_mean", "moss_logit_std", "moss_sigma_cutoff", "moss_uniform_mix")
        if not self._section_on() or self._sampling() != "moss":
            self._drop(*keys)
            return
        inputs = (self.widget.logit_mean_input, self.widget.logit_std_input, self.widget.sigma_cutoff_input,
                  self.widget.uniform_mix_input)
        for key, elem in zip(keys, inputs):
            self._set_or_drop(key, elem.value())

    def change_sigmoid_scale(self) -> None:
        elem = self.widget.sigmoid_scale_input
        sampling_type = self._sampling()
        if sampling_type not in ("sigmoid", "shift"):
            elem.set_invalid(False)
            if self._section_on():
                self._drop("sigmoid_scale")
            return
        value = sigmoid_scale_for_sampling(elem.currentText(), sampling_type)
        elem.set_invalid(value is None)
        if value is None or not self._section_on():
            return  # keep the previous key; get_validation_errors blocks saving and training
        self._set_or_drop("sigmoid_scale", value)

    def change_scale_resume(self) -> None:
        if not self._section_on():
            return
        if self._sampling() != "sigmoid":
            self._drop("scale_resume")
            return
        self._set_or_drop("scale_resume", self.widget.scale_resume_input.value())

    def change_discrete_flow_shift(self) -> None:
        if not self._section_on():
            return
        if self._sampling() not in ("sigma", "shift"):
            self._drop("discrete_flow_shift")
            return
        self._set_or_drop("discrete_flow_shift", self.widget.discrete_flow_shift_input.value())

    def change_attn_mode(self, index: int) -> None:
        if not self._section_on():
            return
        index = max(0, min(index, len(MOSS_ATTN_MODES) - 1))
        value = MOSS_ATTN_MODES[index][1]
        self._drop("attn_mode")
        if value is not None:
            self.edit_args("attn_mode", value)

    def change_exclude_blocks(self) -> None:
        elem = self.widget.exclude_blocks_input
        blocks = parse_block_list(elem.text())
        elem.setStyleSheet(INVALID_LINE_STYLE if blocks is None else "")
        if blocks is None or not self._section_on():
            return
        self.edit_args("moss_lora_exclude_blocks", ",".join(str(b) for b in blocks), optional=True)

    def change_uncond_skip_layers(self) -> None:
        elem = self.widget.uncond_skip_layers_input
        text = elem.text().strip()
        valid = parse_block_list(text) is not None
        elem.setStyleSheet("" if valid else INVALID_LINE_STYLE)
        if not valid or not self._section_on():
            return
        self._set_or_drop("moss_sample_uncond_skip_layers", text)

    def get_validation_errors(self) -> list[str]:
        if not self._section_on():
            return []
        errors = []
        w = self.widget
        # Required here: the backend fills an empty path from General Args otherwise (an SD checkpoint as the DiT).
        for elem, name in ((w.dit_model_input, "DiT Model"), (w.vae_model_input, "VAE Model"),
                           (w.text_encoder_input, "Text Encoder")):
            if not elem.text().strip():
                errors.append(f"MOSS-Image Args: {name} is required (General Args' Base Model / VAE are not used)")
        if not w.tokenizer_dir_input.text().strip() and not os.environ.get(TOKENIZER_DIR_ENV):
            errors.append(
                "MOSS-Image Args: Tokenizer Folder is required (the MOSS-Image tokenizer/ folder with tokenizer.json); "
                f"it may stay empty only when the {TOKENIZER_DIR_ENV} environment variable is set"
            )
        if w.blocks_to_swap_input.value() > 0 and w.unsloth_offload_checkpointing.isChecked():
            errors.append("MOSS-Image Args: Blocks to Swap cannot be combined with Unsloth Offload (use one of them)")
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        if sampling_type in ("sigmoid", "shift") and sigmoid_scale_for_sampling(
            self.widget.sigmoid_scale_input.currentText(), sampling_type
        ) is None:
            errors.append(
                "MOSS-Image Args: Sigmoid Scale must be a positive number, or balanced / content / style with "
                f"sigmoid sampling (got '{self.widget.sigmoid_scale_input.currentText()}')"
            )
        if parse_block_list(self.widget.exclude_blocks_input.text()) is None:
            errors.append(
                f"MOSS-Image Args: Exclude Blocks takes block indices 0-{MOSS_NUM_BLOCKS - 1} separated by commas "
                f"(got '{self.widget.exclude_blocks_input.text()}')"
            )
        if parse_block_list(self.widget.uncond_skip_layers_input.text()) is None:
            errors.append(
                f"MOSS-Image Args: Uncond Skip Layers takes block indices 0-{MOSS_NUM_BLOCKS - 1} or nothing "
                f"(got '{self.widget.uncond_skip_layers_input.text()}')"
            )
        return errors

    def load_args(self, args: dict) -> bool:
        args: dict = args.get(self.name, {})
        w = self.widget
        self.load_warnings = []
        controls = [
            w.moss_training_box, w.dit_model_input, w.vae_model_input, w.text_encoder_input, w.tokenizer_dir_input,
            w.max_token_input, w.timestep_sampling_selector, w.discrete_flow_shift_input, w.sigmoid_scale_input,
            w.scale_resume_input, w.logit_mean_input, w.logit_std_input, w.sigma_cutoff_input, w.uniform_mix_input,
            w.attn_mode_selector, w.split_attn_enable, w.blocks_to_swap_input, w.use_ramtorch_enable,
            w.unsloth_offload_checkpointing, w.exclude_blocks_input, w.sample_shift_input,
            w.uncond_skip_layers_input,
        ] + [getattr(w, widget_name) for widget_name, _ in self.ADVANCED_SWITCHES]

        defaults = self.TRAINER_DEFAULTS
        with signals_blocked(controls):
            w.moss_training_box.setChecked(bool(args))
            w.dit_model_input.setText(args.get("pretrained_model_name_or_path", ""))
            w.vae_model_input.setText(args.get("vae", ""))
            w.text_encoder_input.setText(args.get("moss_text_encoder", ""))
            w.tokenizer_dir_input.setText(args.get("moss_tokenizer_dir", ""))
            max_tokens = args.get("moss_max_token_length", defaults["moss_max_token_length"])
            if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > MOSS_MAX_TOKENS:
                self.load_warnings.append(
                    f"MOSS-Image Args: moss_max_token_length = {max_tokens} is above the trainer's limit of "
                    f"{MOSS_MAX_TOKENS}; Max Tokens is set to {MOSS_MAX_TOKENS}."
                )
            w.max_token_input.setValue(max_tokens)

            sampling = args.get("timestep_sampling", "moss")
            if sampling not in MOSS_TIMESTEP_SAMPLING:
                self.load_warnings.append(
                    f"MOSS-Image Args: timestep_sampling = \"{sampling}\" is not one the MOSS-Image trainer offers "
                    f"({' / '.join(MOSS_TIMESTEP_SAMPLING)}); Timestep Sampling is set to moss."
                )
                sampling = "moss"
            w.timestep_sampling_selector.setCurrentText(sampling)
            w.discrete_flow_shift_input.setValue(args.get("discrete_flow_shift", defaults["discrete_flow_shift"]))
            w.sigmoid_scale_input.set_value(args.get("sigmoid_scale", defaults["sigmoid_scale"]))
            w.scale_resume_input.setValue(args.get("scale_resume", defaults["scale_resume"]))
            w.logit_mean_input.setValue(args.get("moss_logit_mean", defaults["moss_logit_mean"]))
            w.logit_std_input.setValue(args.get("moss_logit_std", defaults["moss_logit_std"]))
            w.sigma_cutoff_input.setValue(args.get("moss_sigma_cutoff", defaults["moss_sigma_cutoff"]))
            w.uniform_mix_input.setValue(args.get("moss_uniform_mix", defaults["moss_uniform_mix"]))

            attn_mode = args.get("attn_mode")
            attn_mode = MOSS_ATTN_ALIASES.get(attn_mode, attn_mode)
            attn_values = [value for _, value in MOSS_ATTN_MODES]
            if attn_mode not in attn_values:
                self.load_warnings.append(
                    f"MOSS-Image Args: attn_mode = \"{attn_mode}\" is not one the GUI offers for MOSS-Image training "
                    "(default / torch / flash / xformers); Attention is set to default."
                )
            w.attn_mode_selector.setCurrentIndex(attn_values.index(attn_mode) if attn_mode in attn_values else 0)
            w.split_attn_enable.setChecked(bool(args.get("split_attn", False)))
            w.blocks_to_swap_input.setValue(args.get("blocks_to_swap", defaults["blocks_to_swap"]))
            w.use_ramtorch_enable.setChecked(bool(args.get("use_ramtorch", False)))
            w.unsloth_offload_checkpointing.setChecked(bool(args.get("unsloth_offload_checkpointing", False)))

            w.exclude_blocks_input.setText(block_list_text(args.get("moss_lora_exclude_blocks", "")))
            w.sample_shift_input.setValue(args.get("moss_sample_shift", defaults["moss_sample_shift"]))
            w.uncond_skip_layers_input.setText(
                block_list_text(args.get("moss_sample_uncond_skip_layers", defaults["moss_sample_uncond_skip_layers"]))
            )
            for widget_name, key in self.ADVANCED_SWITCHES:
                getattr(w, widget_name).setChecked(bool(args.get(key, False)))

        self.enable_disable(w.moss_training_box.isChecked())
        if not w.moss_training_box.isChecked():
            self.change_timestep_sampling_type(w.timestep_sampling_selector.currentIndex())  # enable states only
        return True
