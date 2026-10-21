from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QPushButton
from modules.DragDropLineEdit import DragDropLineEdit
from modules.SigmoidScaleInput import sigmoid_scale_for_sampling
from ui_files.AnimaUI import Ui_anima_ui
from modules.BaseWidget import BaseWidget, signals_blocked
from pathlib import Path


class AnimaWidget(BaseWidget):
    Toggled = Signal(bool)  # send to general args

    ANIMA38_KEYS = (
        "anima38_connector",
        "qwen35_path",
        "qwen35_tokenizer_dir",
        "anima38_sample_negative",
        "anima38_qwen35_batch_size",
    )
    ADVANCED_SWITCHES = (
        ("no_compile_enable", "anima_no_compile"),
        ("no_lokr_compile_enable", "anima_no_lokr_compile"),
        ("stock_lycoris_enable", "anima_stock_lycoris"),
        ("no_sample_cache_enable", "anima_no_sample_cache"),
    )

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Anima Args")
        self.widget = Ui_anima_ui()

        self.name = "anima_args"

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
        setup_file(self.widget.qwen3_model_input, self.widget.qwen3_model_selector)
        setup_file(self.widget.vae_model_input, self.widget.vae_model_selector)
        setup_file(self.widget.t5_tokenizer_input, self.widget.t5_tokenizer_selector, folder=True)
        setup_file(self.widget.qwen35_model_input, self.widget.qwen35_model_selector)
        setup_file(self.widget.qwen35_tokenizer_input, self.widget.qwen35_tokenizer_selector, folder=True)
        self.update_sampling_controls(self.widget.timestep_sampling_selector.currentText())

    def setup_connections(self) -> None:
        self.widget.anima_training_box.clicked.connect(self.enable_disable)
        
        # File inputs — keys match sd_scripts argparse names
        self.widget.dit_model_input.textChanged.connect(lambda x: self.edit_args("pretrained_model_name_or_path", x))
        self.widget.dit_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.dit_model_input, "Anima DiT Model", "Model File")
        )
        
        self.widget.qwen3_model_input.textChanged.connect(lambda x: self.edit_args("qwen3", x))
        self.widget.qwen3_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.qwen3_model_input, "Qwen3 Model", "Model File")
        )
        
        self.widget.vae_model_input.textChanged.connect(lambda x: self.edit_args("vae", x))
        self.widget.vae_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.vae_model_input, "VAE Model", "Model File")
        )
        
        self.widget.t5_tokenizer_input.textChanged.connect(lambda x: self.edit_args("t5_tokenizer_path", x, optional=True))
        self.widget.t5_tokenizer_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.t5_tokenizer_input, "T5 Tokenizer Folder")
        )

        # Token Lengths
        self.widget.qwen3_max_token_input.valueChanged.connect(
            lambda x: self.edit_args("qwen3_max_token_length", x)
        )
        self.widget.t5_max_token_input.valueChanged.connect(
            lambda x: self.edit_args("t5_max_token_length", x)
        )

        # Sampling Params
        self.widget.timestep_sampling_selector.currentIndexChanged.connect(self.change_timestep_sampling_type)
        self.widget.discrete_flow_shift_input.valueChanged.connect(
            lambda x: self.edit_args("discrete_flow_shift", x)
        )
        self.widget.sigmoid_scale_input.valueChanged.connect(lambda _: self.change_sigmoid_scale())
        self.widget.scale_resume_input.valueChanged.connect(lambda _: self.change_scale_resume())

        # VAE / Memory Options
        self.widget.vae_chunk_size_input.valueChanged.connect(self.change_vae_chunk_size)
        self.widget.vae_disable_cache_enable.clicked.connect(
            lambda x: self.edit_args("vae_disable_cache", x, True)
        )
        self.widget.blocks_to_swap_input.valueChanged.connect(self.change_blocks_to_swap)
        self.widget.use_ramtorch_enable.clicked.connect(lambda x: self._edit_if_on("use_ramtorch", x, True))

        # Misc
        self.widget.flash_attn_enable.clicked.connect(self.change_flash_attn)
        self.widget.split_attn_enable.clicked.connect(
            lambda x: self.edit_args("split_attn", x, True)
        )
        self.widget.unsloth_offload_checkpointing.clicked.connect(
            lambda x: self.edit_args("unsloth_offload_checkpointing", x, True)
        )

        # Anima 3.8B v1.1 connector
        self.widget.anima38_box.clicked.connect(lambda _: self.change_anima38())
        self.widget.qwen35_model_input.textChanged.connect(lambda _: self.change_anima38())
        self.widget.qwen35_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.qwen35_model_input, "Qwen3.5-4B Model", "Model File")
        )
        self.widget.qwen35_tokenizer_input.textChanged.connect(lambda _: self.change_anima38())
        self.widget.qwen35_tokenizer_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.qwen35_tokenizer_input, "Qwen3.5 Tokenizer Folder")
        )
        self.widget.anima38_negative_selector.currentIndexChanged.connect(lambda _: self.change_anima38())
        self.widget.anima38_batch_input.valueChanged.connect(lambda _: self.change_anima38())

        # Advanced: disable speedups
        for widget_name, key in self.ADVANCED_SWITCHES:
            getattr(self.widget, widget_name).clicked.connect(lambda x, key=key: self._edit_if_on(key, x, True))

    def _anima_on(self) -> bool:
        box = self.widget.anima_training_box
        return box.isChecked() and box.isEnabled()

    def _edit_if_on(self, key: str, value: object, optional: bool = False) -> None:
        if self._anima_on():
            self.edit_args(key, value, optional)

    def _drop(self, *keys: str) -> None:
        for key in keys:
            if key in self.args:
                del self.args[key]

    def enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.Toggled.emit(checked)
        if not checked or not self.widget.anima_training_box.isEnabled():
            return  # a checked box that another model type disabled writes nothing
            
        self.edit_args("pretrained_model_name_or_path", self.widget.dit_model_input.text())
        self.edit_args("qwen3", self.widget.qwen3_model_input.text())
        self.edit_args("vae", self.widget.vae_model_input.text())
        self.edit_args("t5_tokenizer_path", self.widget.t5_tokenizer_input.text(), optional=True)
        
        self.edit_args("qwen3_max_token_length", self.widget.qwen3_max_token_input.value())
        self.edit_args("t5_max_token_length", self.widget.t5_max_token_input.value())
        
        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())
        
        self.change_vae_chunk_size(self.widget.vae_chunk_size_input.value())
        self.edit_args("vae_disable_cache", self.widget.vae_disable_cache_enable.isChecked(), True)
        self.change_blocks_to_swap(self.widget.blocks_to_swap_input.value())

        self.change_flash_attn(self.widget.flash_attn_enable.isChecked())
        self.edit_args("split_attn", self.widget.split_attn_enable.isChecked(), True)
        self.edit_args("unsloth_offload_checkpointing", self.widget.unsloth_offload_checkpointing.isChecked(), True)

        self.edit_args("use_ramtorch", self.widget.use_ramtorch_enable.isChecked(), True)
        self.change_anima38()
        for widget_name, key in self.ADVANCED_SWITCHES:
            self.edit_args(key, getattr(self.widget, widget_name).isChecked(), True)

    def external_enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.widget.anima_training_box.setEnabled(not checked)
        if self.widget.anima_training_box.isEnabled() and self.widget.anima_training_box.isChecked():
            self.enable_disable(True)

    @staticmethod
    def _uses_sigmoid_scale(sampling_type: str) -> bool:
        # sigmoid, shift, and flux_shift all use sigmoid_scale to control concentration
        return sampling_type in ("sigmoid", "shift", "flux_shift")

    def update_sampling_controls(self, sampling_type: str) -> None:
        """Enable only the inputs the selected timestep sampling uses (no args change)."""
        self.widget.sigmoid_scale_input.setEnabled(self._uses_sigmoid_scale(sampling_type))
        self.widget.scale_resume_input.setEnabled(sampling_type == "sigmoid")
        self.widget.discrete_flow_shift_input.setEnabled(sampling_type in ("sigma", "shift"))

    def change_timestep_sampling_type(self, index: int) -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        self.update_sampling_controls(sampling_type)
        if not self._anima_on():
            self.change_sigmoid_scale()  # stylesheet only
            return
        self.edit_args("timestep_sampling", sampling_type)

        # Enable/Disable sigmoid scale based on selection
        self.change_sigmoid_scale()
        # scale_resume: sigmoid only, omitted at 0
        self.change_scale_resume()

        # Enable/Disable discrete flow shift based on selection
        # sigma uses it via the scheduler's internal shifted sigma table
        # shift applies it directly: sigma' = shift * sigma / (1 + (shift-1) * sigma)
        # sigmoid, flux_shift, and uniform do not use discrete_flow_shift
        uses_flow_shift = sampling_type in ("sigma", "shift")
        if uses_flow_shift:
            self.edit_args("discrete_flow_shift", self.widget.discrete_flow_shift_input.value())
        elif "discrete_flow_shift" in self.args:
            del self.args["discrete_flow_shift"]

    def change_sigmoid_scale(self) -> None:
        """A number or, with sigmoid sampling, a preset. Written whenever it is used (1.0 included), as before."""
        elem = self.widget.sigmoid_scale_input
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        if not self._uses_sigmoid_scale(sampling_type):
            elem.set_invalid(False)
            if self._anima_on():
                self._drop("sigmoid_scale")
            return
        value = sigmoid_scale_for_sampling(elem.currentText(), sampling_type)
        elem.set_invalid(value is None)
        if value is None or not self._anima_on():
            return  # keep the previous key; get_validation_errors blocks saving and training
        self.edit_args("sigmoid_scale", value)

    def change_scale_resume(self) -> None:
        if not self._anima_on():
            return
        value = self.widget.scale_resume_input.value()
        if self.widget.timestep_sampling_selector.currentText() == "sigmoid" and value > 0:
            self.edit_args("scale_resume", value)
        else:
            self._drop("scale_resume")

    def change_anima38(self) -> None:
        """Anima 3.8B v1.1 connector: no key unless the sub-box is checked; options omitted at their defaults."""
        if not self._anima_on():
            return
        self._drop(*self.ANIMA38_KEYS)
        if not self.widget.anima38_box.isChecked():
            return
        self.edit_args("anima38_connector", True)
        self.edit_args("qwen35_path", self.widget.qwen35_model_input.text())
        self.edit_args("qwen35_tokenizer_dir", self.widget.qwen35_tokenizer_input.text(), optional=True)
        negative = self.widget.anima38_negative_selector.currentText()
        if negative != "connector":
            self.edit_args("anima38_sample_negative", negative)
        batch_size = self.widget.anima38_batch_input.value()
        if batch_size != 1:
            self.edit_args("anima38_qwen35_batch_size", batch_size)

    def get_validation_errors(self) -> list[str]:
        if not self._anima_on():
            return []
        errors = []
        w = self.widget
        # Required here: the backend fills an empty DiT / VAE from General Args otherwise (an SD checkpoint as the DiT).
        for elem, name in ((w.dit_model_input, "DiT Model"), (w.qwen3_model_input, "Qwen3 Model"),
                           (w.vae_model_input, "VAE Model")):
            if not elem.text().strip():
                errors.append(f"Anima Args: {name} is required (General Args' Base Model / VAE are not used)")
        sampling_type = w.timestep_sampling_selector.currentText()
        text = w.sigmoid_scale_input.currentText()
        if self._uses_sigmoid_scale(sampling_type) and sigmoid_scale_for_sampling(text, sampling_type) is None:
            errors.append(
                "Anima Args: Sigmoid Scale must be a positive number, or balanced / content / style with "
                f"sigmoid sampling (got '{text}')"
            )
        if w.blocks_to_swap_input.value() > 0 and w.unsloth_offload_checkpointing.isChecked():
            errors.append("Anima Args: Blocks to Swap cannot be combined with Unsloth Offload (use one of them)")
        if w.anima38_box.isChecked():
            if not w.qwen35_model_input.text().strip():
                errors.append("Anima Args: the Anima 3.8B connector needs the Qwen3.5 Model (qwen35_path)")
            # An empty Qwen3.5 Tokenizer is allowed: the trainer then uses $ANIMA38_QWEN35_TOKENIZER_DIR or its own
            # default folder, and stops with a clear error if neither exists.
        return errors

    def change_vae_chunk_size(self, value: int) -> None:
        """VAE chunk size: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("vae_chunk_size", value)
        elif "vae_chunk_size" in self.args:
            del self.args["vae_chunk_size"]

    def change_blocks_to_swap(self, value: int) -> None:
        """Blocks to swap: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("blocks_to_swap", value)
        elif "blocks_to_swap" in self.args:
            del self.args["blocks_to_swap"]

    def change_flash_attn(self, checked: bool) -> None:
        """Flash attention: sets attn_mode to 'flash' when enabled, removes it when disabled."""
        if checked:
            self.edit_args("attn_mode", "flash")
        elif "attn_mode" in self.args:
            del self.args["attn_mode"]

    def update_split_attn_from_general(self, xformers_enabled: bool, sdpa_enabled: bool) -> None:
        """Called from ArgsListUI when attention mode changes in GeneralUI.
        
        When xFormers is enabled with Anima, split_attn should be forcibly enabled.
        When SDPA is enabled, split_attn is automatically enabled internally by sd_scripts.
        """
        if xformers_enabled:
            self.widget.split_attn_enable.setChecked(True)
            self.widget.split_attn_enable.setEnabled(False)
            self.edit_args("split_attn", True, True)
        else:
            self.widget.split_attn_enable.setEnabled(True)

    def load_args(self, args: dict) -> bool:
        args: dict = args.get(self.name, {})
        w = self.widget
        controls = [
            w.anima_training_box, w.dit_model_input, w.qwen3_model_input, w.vae_model_input, w.t5_tokenizer_input,
            w.qwen3_max_token_input, w.t5_max_token_input, w.timestep_sampling_selector,
            w.discrete_flow_shift_input, w.sigmoid_scale_input, w.scale_resume_input, w.vae_chunk_size_input,
            w.vae_disable_cache_enable, w.blocks_to_swap_input, w.use_ramtorch_enable, w.flash_attn_enable,
            w.split_attn_enable, w.unsloth_offload_checkpointing, w.anima38_box, w.qwen35_model_input,
            w.qwen35_tokenizer_input, w.anima38_negative_selector, w.anima38_batch_input,
        ] + [getattr(w, widget_name) for widget_name, _ in self.ADVANCED_SWITCHES]

        # Signals blocked: one enable_disable at the end rebuilds the args from the controls
        with signals_blocked(controls):
            self.widget.anima_training_box.setChecked(bool(args))
            self.widget.dit_model_input.setText(args.get("pretrained_model_name_or_path", ""))
            self.widget.qwen3_model_input.setText(args.get("qwen3", ""))
            self.widget.vae_model_input.setText(args.get("vae", ""))
            self.widget.t5_tokenizer_input.setText(args.get("t5_tokenizer_path", ""))

            self.widget.qwen3_max_token_input.setValue(args.get("qwen3_max_token_length", 512))
            self.widget.t5_max_token_input.setValue(args.get("t5_max_token_length", 512))

            self.widget.timestep_sampling_selector.setCurrentText(args.get("timestep_sampling", "sigmoid"))
            self.widget.discrete_flow_shift_input.setValue(args.get("discrete_flow_shift", 3.0))
            self.widget.sigmoid_scale_input.set_value(args.get("sigmoid_scale", 1.0))
            self.widget.scale_resume_input.setValue(args.get("scale_resume", 0))

            self.widget.vae_chunk_size_input.setValue(args.get("vae_chunk_size", 0))
            self.widget.vae_disable_cache_enable.setChecked(args.get("vae_disable_cache", False))
            self.widget.blocks_to_swap_input.setValue(args.get("blocks_to_swap", 0))
            self.widget.use_ramtorch_enable.setChecked(bool(args.get("use_ramtorch", False)))

            # flash_attn is stored as attn_mode="flash"
            self.widget.flash_attn_enable.setChecked(args.get("attn_mode", "") == "flash")
            self.widget.split_attn_enable.setChecked(args.get("split_attn", False))
            self.widget.unsloth_offload_checkpointing.setChecked(args.get("unsloth_offload_checkpointing", False))

            # Anima 3.8B v1.1 connector
            self.widget.anima38_box.setChecked(bool(args.get("anima38_connector", False)))
            self.widget.qwen35_model_input.setText(args.get("qwen35_path", ""))
            self.widget.qwen35_tokenizer_input.setText(args.get("qwen35_tokenizer_dir", ""))
            self.widget.anima38_negative_selector.setCurrentText(args.get("anima38_sample_negative", "connector"))
            self.widget.anima38_batch_input.setValue(args.get("anima38_qwen35_batch_size", 1))

            for widget_name, key in self.ADVANCED_SWITCHES:
                getattr(self.widget, widget_name).setChecked(bool(args.get(key, False)))

        self.update_sampling_controls(self.widget.timestep_sampling_selector.currentText())
        self.enable_disable(self.widget.anima_training_box.isChecked())
        if not self.widget.anima_training_box.isChecked():
            self.change_sigmoid_scale()  # stylesheet only
