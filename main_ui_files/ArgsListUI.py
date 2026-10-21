import copy

from PySide6.QtCore import Signal
from PySide6 import QtWidgets, QtCore
from main_ui_files.AccelerateUI import AccelerateWidget
from main_ui_files.FluxUI import FluxWidget
from main_ui_files.AnimaUI import AnimaWidget
from main_ui_files.MossUI import MossWidget, parse_block_list

from main_ui_files.BucketUI import BucketWidget
from main_ui_files.GeneralUI import GeneralWidget
from main_ui_files.LoggingUI import LoggingWidget
from main_ui_files.NetworkUI import NetworkWidget
from main_ui_files.NoiseOffsetUI import NoiseOffsetWidget
from main_ui_files.OptimizerUI import OptimizerWidget
from main_ui_files.SampleUI import SampleWidget
from main_ui_files.SavingUI import SavingWidget
from main_ui_files.TextualInversionUI import TextualInversionWidget
from main_ui_files.EDMLossUI import EDMLossWidget  # Add this import
from main_ui_files.ExtraArgsUI import ExtraArgsWidget
from modules.BaseWidget import BaseWidget
from modules import ForkArgs

MOSS_LYCORIS_PRESETS = ("moss", "moss-text")
# Model-type sections in GUI (and backend flatten) order
MODEL_SECTIONS = {"flux_args": "Flux Args", "anima_args": "Anima Args", "moss_args": "MOSS-Image Args"}


def lycoris_preset_is_file(preset: str) -> bool:
    """A LyCORIS preset given as a config file path (its own module list) instead of a built-in name."""
    return preset.lower().endswith((".toml", ".json")) or "/" in preset or "\\" in preset


def active(box: QtWidgets.QGroupBox) -> bool:
    """A model-type box counts only while it is checked and enabled (a checked box disabled by another model type
    writes nothing)."""
    return box.isChecked() and box.isEnabled()


class ArgsWidget(QtWidgets.QWidget):
    sdxlChecked = Signal(bool)
    cacheLatentsChecked = Signal(bool)
    keepTokensSepChecked = Signal(bool)
    maskedLossChecked = Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_widget = QtWidgets.QWidget()
        self.args_widget_array: list[BaseWidget] = []
        self.network_widget = NetworkWidget()
        self.optimizer_widget = OptimizerWidget()
        self.ti_widget = TextualInversionWidget()
        self.ti_widget.setVisible(False)
        self.flux_widget = FluxWidget()
        self.anima_widget = AnimaWidget()
        self.moss_widget = MossWidget()
        self.load_warnings: list[str] = []  # filled by load_args, shown by MainWidget.load_toml
        self._applying_model_state = False
        # What switching MOSS-Image / the Anima 3.8B connector on changed, so switching it off can put it back
        self._auto_cache_te_previous: dict[str, bool] = {}
        self._auto_moss_preset = False

        self.setup_widget()
        self.setup_args_widgets()

    def setup_widget(self) -> None:
        self.setMinimumSize(600, 300)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget.setLayout(QtWidgets.QVBoxLayout())
        self.scroll_widget.layout().setSpacing(0)
        self.scroll_widget.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.scroll_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout().addWidget(self.scroll_area)

    def setup_args_widgets(self) -> None:
        general_args = GeneralWidget()
        self.general_widget = general_args
        experimental = general_args.experimental_args_widget
        flux_box = self.flux_widget.widget.flux_training_box
        anima_box = self.anima_widget.widget.anima_training_box
        moss_box = self.moss_widget.widget.moss_training_box
        flow_box = experimental.widget.flow_model_settings_box

        general_args.colap.toggle_collapsed()
        general_args.colap.title_frame.setChecked(True)
        general_args.sdxlChecked.connect(lambda x: self.sdxlChecked.emit(x))
        general_args.cacheLatentsChecked.connect(lambda x: self.cacheLatentsChecked.emit(x))
        general_args.keepTokensSepChecked.connect(lambda x: self.keepTokensSepChecked.emit(x))

        # Each refresh_* reads every box (never the emitted bool) and does nothing when the state is unchanged.
        def refresh_flux_disabled_state() -> None:
            should_disable = (
                flow_box.isChecked()
                or general_args.widget.v_param_enable.isChecked()
                or general_args.widget.v2_enable.isChecked()
                or general_args.widget.sdxl_enable.isChecked()
                or active(anima_box)
                or active(moss_box)
            )
            # Unchanged state: nothing to do. Re-running it while Flux is on re-emits Flux.Toggled, which unchecks the
            # flow model, whose flowModelToggled calls this again: endless recursion (a crash when Flux is turned on).
            if flux_box.isEnabled() == (not should_disable):
                return
            self.flux_widget.external_enable_disable(should_disable)

        def refresh_anima_disabled_state() -> None:
            should_disable = (
                general_args.widget.v_param_enable.isChecked()
                or general_args.widget.v2_enable.isChecked()
                or general_args.widget.sdxl_enable.isChecked()
                or active(flux_box)
                or active(moss_box)
            )
            if anima_box.isEnabled() == (not should_disable):
                return
            self.anima_widget.external_enable_disable(should_disable)

        def refresh_moss_disabled_state() -> None:
            should_disable = (
                flow_box.isChecked()  # moss_train_network.py refuses --flow_model
                or general_args.widget.v_param_enable.isChecked()
                or general_args.widget.v2_enable.isChecked()
                or general_args.widget.sdxl_enable.isChecked()
                or active(flux_box)
                or active(anima_box)
            )
            if moss_box.isEnabled() == (not should_disable):
                return
            self.moss_widget.external_enable_disable(should_disable)

        def refresh_all_disabled_states() -> None:
            refresh_flux_disabled_state()
            refresh_anima_disabled_state()
            refresh_moss_disabled_state()

        # Disable FP8 while Anima or MOSS-Image is active (fp8_base is unsupported by their DiTs).
        # Reads both boxes, so turning one off does not re-enable FP8 while the other is on.
        def update_fp8_for_dit() -> None:
            dit_enabled = active(anima_box) or active(moss_box)
            general_args.widget.FP8_enable.setEnabled(not dit_enabled)
            if dit_enabled and general_args.widget.FP8_enable.isChecked():
                general_args.widget.FP8_enable.setChecked(False)
                general_args.edit_args("fp8_base", False, True)

        # Cache TE Outputs: offered for SDXL and the DiT model types (Flux, Anima, MOSS-Image)
        def sync_cache_te() -> None:
            dit_on = active(flux_box) or active(anima_box) or active(moss_box)
            self.network_widget.toggle_sdxl(dit_on or general_args.widget.sdxl_enable.isChecked())

        # The experimental flow model is refused by Flux and MOSS-Image
        def sync_flow_model(block: bool) -> None:
            if block:
                if flow_box.isEnabled() or flow_box.isChecked():
                    experimental.set_flux_enabled(True)
            elif not flow_box.isEnabled() and not general_args.widget.v_param_enable.isChecked():
                experimental.set_flux_enabled(False)

        def apply_model_state() -> None:
            """Model-type dependent controls from the final state of every box, whatever box emitted last
            (a load emits Toggled(False) of Anima / MOSS after Flux's Toggled(True))."""
            if self._applying_model_state:
                return
            self._applying_model_state = True
            try:
                flux_on, anima_on, moss_on = active(flux_box), active(anima_box), active(moss_box)
                sync_flow_model(flux_on or moss_on)
                general_args.enable_disable_model_type(flux_on or anima_on or moss_on)
                sync_cache_te()
                update_fp8_for_dit()
            finally:
                self._applying_model_state = False

        self._refresh_all_disabled_states = refresh_all_disabled_states
        self._apply_model_state = apply_model_state

        general_args.v2Checked.connect(lambda _: refresh_all_disabled_states())
        general_args.sdxlChecked.connect(lambda _: refresh_all_disabled_states())
        general_args.sdxlChecked.connect(lambda _: sync_cache_te())
        general_args.widget.v_param_enable.clicked.connect(lambda _: refresh_all_disabled_states())
        # Pass experimental_args_widget to flux_widget for cross-file logic
        self.flux_widget.experimental_args_widget = experimental
        # Flow model on disables Flux and MOSS-Image
        experimental.flowModelToggled.connect(lambda _: refresh_flux_disabled_state())
        experimental.flowModelToggled.connect(lambda _: refresh_moss_disabled_state())
        for dit_widget in (self.flux_widget, self.anima_widget, self.moss_widget):
            dit_widget.Toggled.connect(lambda _: refresh_all_disabled_states())
            dit_widget.Toggled.connect(lambda _: apply_model_state())
        refresh_all_disabled_states()
        self.flux_widget.SplitMode.connect(
            lambda x: self.network_widget.edit_network_args("train_blocks", "single" if x else False, True)
        )
        self.flux_widget.SplitQKV.connect(
            lambda x: self.network_widget.edit_network_args("split_qkv", x, True)
        )
        self.optimizer_widget.maskedLossChecked.connect(lambda x: self.maskedLossChecked.emit(x))
        self.args_widget_array.append(general_args)
        self.args_widget_array.append(self.network_widget)
        self.args_widget_array.append(self.ti_widget)
        self.args_widget_array.append(self.optimizer_widget)
        self.args_widget_array.append(SavingWidget())
        self.args_widget_array.append(BucketWidget())
        self.args_widget_array.append(NoiseOffsetWidget())
        self.args_widget_array.append(SampleWidget())
        self.args_widget_array.append(LoggingWidget())
        self.args_widget_array.append(self.flux_widget)
        self.args_widget_array.append(self.anima_widget)
        self.args_widget_array.append(self.moss_widget)

        # Connect attention mode changes to Anima's split_attn auto-enable
        def sync_attn_to_anima() -> None:
            if self.anima_widget.widget.anima_training_box.isChecked():
                self.anima_widget.update_split_attn_from_general(
                    general_args.widget.xformers_enable.isChecked(),
                    general_args.widget.sdpa_enable.isChecked(),
                )

        general_args.widget.xformers_enable.clicked.connect(lambda _: sync_attn_to_anima())
        general_args.widget.sdpa_enable.clicked.connect(lambda _: sync_attn_to_anima())
        self.anima_widget.Toggled.connect(lambda _: sync_attn_to_anima())

        # Switching MOSS-Image or the Anima 3.8B connector on (a click, not a load) checks Cache TE Outputs, which both
        # need; switching it off puts back what was there. MOSS also fills an empty LyCORIS Preset with "moss".
        moss_box.clicked.connect(self.moss_switched)
        self.anima_widget.widget.anima38_box.clicked.connect(lambda x: self.anima38_switched(x))
        anima_box.clicked.connect(lambda x: self.anima38_switched(x))  # the 3.8B box may already be checked

        self.args_widget_array.append(EDMLossWidget())
        self.accelerate_widget = AccelerateWidget()
        self.args_widget_array.append(self.accelerate_widget)
        self.extra_args_widget = ExtraArgsWidget()
        self.args_widget_array.append(self.extra_args_widget)

        for widget in self.args_widget_array:
            if widget.name == "textual_inversion_args":
                widget.setVisible(False)
            else:
                widget.setVisible(True)
            self.scroll_widget.layout().addWidget(widget)

    # --- cache TE / preset helpers for MOSS-Image and the Anima 3.8B connector ---

    def _check_cache_te_for(self, owner: str) -> None:
        box = self.network_widget.widget.cache_te_outputs_enable
        if owner not in self._auto_cache_te_previous:
            self._auto_cache_te_previous[owner] = box.isChecked()
        if not box.isChecked():
            box.setChecked(True)
        self.network_widget.enable_disable_cache_te(box.isEnabled())

    def _restore_cache_te_for(self, owner: str) -> None:
        if owner not in self._auto_cache_te_previous:
            return
        previous = self._auto_cache_te_previous.pop(owner)
        if self._auto_cache_te_previous:
            return  # the other one still needs it
        box = self.network_widget.widget.cache_te_outputs_enable
        box.setChecked(previous)
        self.network_widget.enable_disable_cache_te(box.isChecked() and box.isEnabled())

    def moss_switched(self, checked: bool) -> None:
        preset = self.network_widget.widget.lycoris_preset_input
        changed = False
        if checked and active(self.moss_widget.widget.moss_training_box):
            self._check_cache_te_for("moss")
            if not preset.text().strip():
                preset.setText("moss")
                self._auto_moss_preset = changed = True
        elif not checked:
            self._restore_cache_te_for("moss")
            if self._auto_moss_preset and preset.text().strip() == "moss":
                preset.setText("")
                changed = True
            self._auto_moss_preset = False
        if changed:
            # setText writes "preset" through textChanged; keep it out of the args for kohya LoRA, as change_algo does
            self.network_widget.edit_network_args(
                "preset", preset.text().strip() if self.network_widget.lycoris else False, True
            )

    def anima38_switched(self, checked: bool) -> None:
        if checked and self.anima_widget._anima_on() and self.anima_widget.widget.anima38_box.isChecked():
            self._check_cache_te_for("anima38")
        elif not checked:
            self._restore_cache_te_for("anima38")

    # --- training mode ---

    def set_ti_training(self) -> None:
        self.network_widget.setVisible(False)
        self.ti_widget.setVisible(True)

    def set_lora_training(self) -> None:
        self.ti_widget.setVisible(False)
        self.network_widget.setVisible(True)

    def get_args(self) -> dict:
        args = {}
        dataset_args = {}
        for widget in self.args_widget_array:
            if widget.args:
                args[widget.name] = widget.args
            if widget.dataset_args:
                dataset_args[widget.name] = widget.dataset_args
        
        # Force complete rebuild of general_args to prevent stale keys from lingering
        if "general_args" in args and hasattr(self.args_widget_array[0], "experimental_args_widget"):
            general_widget = self.args_widget_array[0]
            experimental_widget = general_widget.experimental_args_widget
            
            # Start completely fresh
            clean_general_args = {}
            
            # Define which keys belong to experimental args
            experimental_keys = {
                "flow_model", "flow_use_ot", "flow_timestep_distribution", 
                "flow_uniform_static_ratio", "flow_logit_mean", "flow_logit_std",
                "contrastive_flow_matching", "cfm_lambda", "vae_custom_scale", 
                "vae_custom_shift", "vae_reflection",
                "debiased_estimation_loss", "zero_cond_dropout",
            }
            
            # Only keep non-experimental args from general_widget
            for k, v in general_widget.args.items():
                if k not in experimental_keys:
                    clean_general_args[k] = v
            
            # Add fresh experimental args (this ensures only enabled/checked items are included)
            clean_general_args.update(experimental_widget.save_args())
            
            # Replace with completely rebuilt version
            args["general_args"] = clean_general_args

        return {"args": args, "dataset": dataset_args}

    # --- validation ---

    def _fork_control_keys(self) -> dict:
        """{key: section} of the fork keys the dedicated controls own in the current state."""
        anima_on = self.anima_widget._anima_on()
        return ForkArgs.control_keys(
            moss_on=self.moss_widget._section_on(),
            anima_on=anima_on,
            anima38_on=anima_on and self.anima_widget.widget.anima38_box.isChecked(),
            dora_on=self.network_widget.dora_written(),
            cache_te_on=bool(self.network_widget.args.get("cache_text_encoder_outputs")),
        )

    def get_cross_section_errors(self) -> list[str]:
        """Rules that need more than one section (the trainers refuse these at startup, after minutes of caching)."""
        errors: list[str] = []
        general = self.general_widget
        network = self.network_widget
        moss_on = self.moss_widget._section_on()
        anima_on = self.anima_widget._anima_on()
        anima38_on = anima_on and self.anima_widget.widget.anima38_box.isChecked()

        checked_types = [
            name for name, box in (
                ("Flux", self.flux_widget.widget.flux_training_box),
                ("Anima", self.anima_widget.widget.anima_training_box),
                ("MOSS-Image", self.moss_widget.widget.moss_training_box),
            ) if box.isChecked()
        ]
        if len(checked_types) > 1:
            errors.append(
                f"More than one model type is checked ({', '.join(checked_types)}); uncheck all but one"
            )
        for name, box in (("Flux", self.flux_widget.widget.flux_training_box),
                          ("Anima", self.anima_widget.widget.anima_training_box),
                          ("MOSS-Image", self.moss_widget.widget.moss_training_box)):
            if box.isChecked() and not box.isEnabled():
                errors.append(
                    f"{name} is checked but switched off by another setting (SDXL / v2 / v-param / flow model / another "
                    "model type), so it would not be trained; uncheck one of them"
                )

        extra = self.extra_args_widget.args
        cached = bool(network.args.get("cache_text_encoder_outputs") or extra.get("cache_text_encoder_outputs"))
        # moss_train_network.py also takes cache_text_encoder_outputs_to_disk alone (it turns caching on itself)
        cached_for_moss = cached or bool(extra.get("cache_text_encoder_outputs_to_disk"))
        if (moss_on and not cached_for_moss) or (anima38_on and not cached):
            model = "MOSS-Image" if moss_on else "Anima 3.8B (connector)"
            errors.append(
                f"{model} needs Cache TE Outputs in Network Args (its text encoder is loaded only while caching)"
            )

        if moss_on and network.lycoris:
            preset = network.widget.lycoris_preset_input.text().strip()
            exclude = parse_block_list(self.moss_widget.widget.exclude_blocks_input.text())
            if preset not in MOSS_LYCORIS_PRESETS and not (preset and lycoris_preset_is_file(preset)):
                errors.append(
                    "MOSS-Image with a LyCORIS algorithm needs LyCORIS Preset = moss (or moss-text): "
                    f"{'an empty preset' if not preset else repr(preset)} selects no MOSS-Image modules, so nothing "
                    "would be trained"
                )
            elif exclude and preset not in MOSS_LYCORIS_PRESETS:
                errors.append(
                    "MOSS-Image Args: Exclude Blocks with LyCORIS needs LyCORIS Preset = moss or moss-text "
                    f"(got {preset!r})"
                )
        if moss_on and not network.lycoris and (
                network.widget.algo_select.currentText().lower() == "dylora" or network.args.get("fa")):
            errors.append("MOSS-Image: DyLoRA / LoRA-FA find no MOSS-Image modules; use LoRA or a LyCORIS algorithm "
                          "with preset moss")
        if not moss_on and network.lycoris:
            preset = network.widget.lycoris_preset_input.text().strip()
            if preset in MOSS_LYCORIS_PRESETS:
                errors.append(
                    f"Network Args: LyCORIS Preset {preset!r} exists only for MOSS-Image training (LyCORIS has no such "
                    "preset for other models); clear the preset or switch the MOSS-Image section on"
                )

        scale_resume_users = []
        if moss_on and "scale_resume" in self.moss_widget.args:
            scale_resume_users.append("MOSS-Image Args")
        if anima_on and "scale_resume" in self.anima_widget.args:
            scale_resume_users.append("Anima Args")
        if scale_resume_users and "max_train_epochs" not in general.args and "max_train_epochs" not in extra:
            errors.append(
                f"{' / '.join(scale_resume_users)}: Scale Resume counts epochs; set Max Train Epochs in General Args "
                "(not Max Train Steps)"
            )

        owned = self._fork_control_keys()
        for name in self.extra_args_widget.args:
            section = owned.get(ForkArgs.canonical(name))
            if section:
                errors.append(
                    f"Extra Args '{name}' is set by a control in {ForkArgs.SECTION_TITLES[section]} and would override "
                    "it; remove the Extra Args row and use the control"
                )
        return errors

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []
        for widget in self.args_widget_array:
            validator = getattr(widget, "get_validation_errors", None)
            if callable(validator):
                errors.extend(validator())
        errors.extend(self.get_cross_section_errors())
        return errors

    # --- loading ---

    def load_args(self, args: dict, dataset_args: dict) -> None:
        self.load_warnings = []
        args = copy.deepcopy(args)
        model_sections = [name for name in MODEL_SECTIONS if args.get(name)]
        if len(model_sections) > 1:
            # Hand-merged TOML: one trainer runs one model type. The last section in GUI / flatten order wins
            # (MOSS-Image, then Anima, then Flux): the backend flattens it last, so its paths and timestep keys are the
            # ones that reached the trainer.
            kept, dropped = model_sections[-1], model_sections[:-1]
            for name in dropped:
                del args[name]
            self.load_warnings.append(
                f"This TOML has more than one model section ({', '.join(MODEL_SECTIONS[n] for n in model_sections)}). "
                f"Only {MODEL_SECTIONS[kept]} was loaded; {', '.join(MODEL_SECTIONS[n] for n in dropped)} "
                "was ignored."
            )
        kept_model = next((name for name in ("moss_args", "flux_args") if args.get(name)), None)
        general = (args.get("general_args") or {})
        if kept_model and general.get("flow_model"):
            # Flux and MOSS-Image refuse the experimental flow model (older GUI builds let MOSS keep it)
            general["flow_model"] = False
            self.load_warnings.append(
                f"flow_model was switched off: {MODEL_SECTIONS[kept_model]} cannot be trained as a flow model."
            )
        args, moved = ForkArgs.migrate_extra_args(args)

        self._auto_cache_te_previous = {}
        self._auto_moss_preset = False
        for widget in self.args_widget_array:
            widget.load_args(args)
            widget.load_dataset_args(dataset_args)
        self._refresh_all_disabled_states()
        self._apply_model_state()

        lost = ForkArgs.unconfirmed(moved, self.get_args()["args"])
        lost_keys = {(key, section) for key, section, _ in lost}
        kept = [(key, section) for key, section, _ in moved if (key, section) not in lost_keys]
        if kept:
            self.load_warnings.append(
                "Moved from Extra Args to their controls: "
                + ", ".join(f"{key} ({ForkArgs.SECTION_TITLES[section]})" for key, section in kept)
            )
        if lost:
            self.extra_args_widget.load({key: value for key, _, value in lost}, False)
            self.load_warnings.append(
                "Kept in Extra Args, because the control cannot hold the value (remove the row or fix the value "
                "before saving): " + ", ".join(f"{key} = {value!r}" for key, _, value in lost)
            )
        self.load_warnings.extend(self.moss_widget.load_warnings)
