# -*- coding: utf-8 -*-

################################################################################
## Hand-written in the pyuic style of ui_files/AnimaUI.py (there is no MossUI.ui).
## MOSS-Image section: moss_train_network.py of the moss-image sd-scripts fork.
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton)

from modules.DragDropLineEdit import DragDropLineEdit
from modules.ScrollOnSelect import (ComboBox, DoubleSpinBox, SpinBox)
from modules.SigmoidScaleInput import SigmoidScaleInput

# attn_mode entries: (shown text, TOML value or None = omit).
# default: attn_mode omitted, so moss_train_network.py uses xformers when General Args' xformers box is checked and
# torch SDPA otherwise. "torch" forces SDPA either way ("sdpa" in a TOML is the same value and loads as it).
# sageattn is not offered: it has no backward pass (inference only) and is not installed in the trainer venv.
MOSS_ATTN_MODES = (
    ("default", None),
    ("torch (SDPA)", "torch"),
    ("flash", "flash"),
    ("xformers", "xformers"),
)
MOSS_ATTN_ALIASES = {"sdpa": "torch"}  # TOML value -> entry value
MOSS_MAX_TOKENS = 4096  # moss_train_network.py refuses more (validate_moss_args)
MOSS_TIMESTEP_SAMPLING = ("moss", "sigma", "uniform", "sigmoid", "shift")
MOSS_NUM_BLOCKS = 42


class Ui_moss_ui(object):
    def setupUi(self, moss_ui):
        if not moss_ui.objectName():
            moss_ui.setObjectName(u"moss_ui")
        moss_ui.resize(523, 560)
        self.gridLayout = QGridLayout(moss_ui)
        self.gridLayout.setObjectName(u"gridLayout")
        self.moss_training_box = QGroupBox(moss_ui)
        self.moss_training_box.setObjectName(u"moss_training_box")
        self.moss_training_box.setCheckable(True)
        self.moss_training_box.setChecked(False)
        self.gridLayout_2 = QGridLayout(self.moss_training_box)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")

        def file_row(row, label_name, input_name, selector_name):
            label = QLabel(self.moss_training_box)
            label.setObjectName(label_name)
            self.formLayout.setWidget(row, QFormLayout.LabelRole, label)
            layout = QHBoxLayout()
            line = DragDropLineEdit(self.moss_training_box)
            line.setObjectName(input_name)
            layout.addWidget(line)
            selector = QPushButton(self.moss_training_box)
            selector.setObjectName(selector_name)
            layout.addWidget(selector)
            self.formLayout.setLayout(row, QFormLayout.FieldRole, layout)
            return label, line, selector

        # Model files
        self.label_dit, self.dit_model_input, self.dit_model_selector = file_row(
            0, u"label_dit", u"dit_model_input", u"dit_model_selector")
        self.label_vae, self.vae_model_input, self.vae_model_selector = file_row(
            1, u"label_vae", u"vae_model_input", u"vae_model_selector")
        self.label_te, self.text_encoder_input, self.text_encoder_selector = file_row(
            2, u"label_te", u"text_encoder_input", u"text_encoder_selector")
        self.label_tokenizer, self.tokenizer_dir_input, self.tokenizer_dir_selector = file_row(
            3, u"label_tokenizer", u"tokenizer_dir_input", u"tokenizer_dir_selector")

        # Token length
        self.horizontalLayout_tokens = QHBoxLayout()
        self.max_token_label = QLabel(self.moss_training_box)
        self.max_token_label.setObjectName(u"max_token_label")
        self.horizontalLayout_tokens.addWidget(self.max_token_label)
        self.max_token_input = SpinBox(self.moss_training_box)
        self.max_token_input.setObjectName(u"max_token_input")
        self.max_token_input.setMinimum(1)
        self.max_token_input.setMaximum(MOSS_MAX_TOKENS)
        self.max_token_input.setValue(4096)
        self.horizontalLayout_tokens.addWidget(self.max_token_input)
        self.horizontalLayout_tokens.addStretch(1)
        self.formLayout.setLayout(4, QFormLayout.SpanningRole, self.horizontalLayout_tokens)

        # Timestep sampling
        self.horizontalLayout_sampling = QHBoxLayout()
        self.label_sampling = QLabel(self.moss_training_box)
        self.label_sampling.setObjectName(u"label_sampling")
        self.horizontalLayout_sampling.addWidget(self.label_sampling)
        self.timestep_sampling_selector = ComboBox(self.moss_training_box)
        self.timestep_sampling_selector.setObjectName(u"timestep_sampling_selector")
        for item in MOSS_TIMESTEP_SAMPLING:
            self.timestep_sampling_selector.addItem(item)
        self.horizontalLayout_sampling.addWidget(self.timestep_sampling_selector)

        self.label_shift = QLabel(self.moss_training_box)
        self.label_shift.setObjectName(u"label_shift")
        self.horizontalLayout_sampling.addWidget(self.label_shift)
        self.discrete_flow_shift_input = DoubleSpinBox(self.moss_training_box)
        self.discrete_flow_shift_input.setObjectName(u"discrete_flow_shift_input")
        self.discrete_flow_shift_input.setDecimals(2)
        self.discrete_flow_shift_input.setMinimum(0.01)
        self.discrete_flow_shift_input.setMaximum(100.0)
        self.discrete_flow_shift_input.setSingleStep(0.1)
        self.discrete_flow_shift_input.setValue(1.0)
        self.horizontalLayout_sampling.addWidget(self.discrete_flow_shift_input)

        self.label_sigmoid = QLabel(self.moss_training_box)
        self.label_sigmoid.setObjectName(u"label_sigmoid")
        self.horizontalLayout_sampling.addWidget(self.label_sigmoid)
        self.sigmoid_scale_input = SigmoidScaleInput(self.moss_training_box)
        self.sigmoid_scale_input.setObjectName(u"sigmoid_scale_input")
        self.horizontalLayout_sampling.addWidget(self.sigmoid_scale_input)

        self.label_scale_resume = QLabel(self.moss_training_box)
        self.label_scale_resume.setObjectName(u"label_scale_resume")
        self.horizontalLayout_sampling.addWidget(self.label_scale_resume)
        self.scale_resume_input = SpinBox(self.moss_training_box)
        self.scale_resume_input.setObjectName(u"scale_resume_input")
        self.scale_resume_input.setMinimum(0)
        self.scale_resume_input.setMaximum(999)
        self.scale_resume_input.setValue(0)
        self.horizontalLayout_sampling.addWidget(self.scale_resume_input)
        self.formLayout.setLayout(5, QFormLayout.SpanningRole, self.horizontalLayout_sampling)

        # MOSS logit-normal sampling
        self.horizontalLayout_logit = QHBoxLayout()

        def double_input(name, label_name, value, minimum, maximum, step=0.01, decimals=4):
            label = QLabel(self.moss_training_box)
            label.setObjectName(label_name)
            self.horizontalLayout_logit.addWidget(label)
            spin = DoubleSpinBox(self.moss_training_box)
            spin.setObjectName(name)
            spin.setDecimals(decimals)
            spin.setMinimum(minimum)
            spin.setMaximum(maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            self.horizontalLayout_logit.addWidget(spin)
            return label, spin

        self.label_logit_mean, self.logit_mean_input = double_input(
            u"logit_mean_input", u"label_logit_mean", 1.09, -10.0, 10.0)
        self.label_logit_std, self.logit_std_input = double_input(
            u"logit_std_input", u"label_logit_std", 0.95, 0.01, 10.0)
        self.label_sigma_cutoff, self.sigma_cutoff_input = double_input(
            u"sigma_cutoff_input", u"label_sigma_cutoff", 0.95, 0.01, 1.0)
        self.label_uniform_mix, self.uniform_mix_input = double_input(
            u"uniform_mix_input", u"label_uniform_mix", 0.01, 0.0, 1.0)
        self.formLayout.setLayout(6, QFormLayout.SpanningRole, self.horizontalLayout_logit)

        # Attention
        self.horizontalLayout_attn = QHBoxLayout()
        self.label_attn_mode = QLabel(self.moss_training_box)
        self.label_attn_mode.setObjectName(u"label_attn_mode")
        self.horizontalLayout_attn.addWidget(self.label_attn_mode)
        self.attn_mode_selector = ComboBox(self.moss_training_box)
        self.attn_mode_selector.setObjectName(u"attn_mode_selector")
        for text, _ in MOSS_ATTN_MODES:
            self.attn_mode_selector.addItem(text)
        self.horizontalLayout_attn.addWidget(self.attn_mode_selector)
        self.split_attn_enable = QCheckBox(self.moss_training_box)
        self.split_attn_enable.setObjectName(u"split_attn_enable")
        self.horizontalLayout_attn.addWidget(self.split_attn_enable)
        self.horizontalLayout_attn.addStretch(1)
        self.formLayout.setLayout(7, QFormLayout.SpanningRole, self.horizontalLayout_attn)

        # Memory
        self.horizontalLayout_mem = QHBoxLayout()
        self.label_blocks_to_swap = QLabel(self.moss_training_box)
        self.label_blocks_to_swap.setObjectName(u"label_blocks_to_swap")
        self.horizontalLayout_mem.addWidget(self.label_blocks_to_swap)
        self.blocks_to_swap_input = SpinBox(self.moss_training_box)
        self.blocks_to_swap_input.setObjectName(u"blocks_to_swap_input")
        self.blocks_to_swap_input.setMinimum(0)
        self.blocks_to_swap_input.setMaximum(MOSS_NUM_BLOCKS - 2)  # the trainer asserts <= num_blocks - 2
        self.blocks_to_swap_input.setValue(0)
        self.horizontalLayout_mem.addWidget(self.blocks_to_swap_input)
        self.use_ramtorch_enable = QCheckBox(self.moss_training_box)
        self.use_ramtorch_enable.setObjectName(u"use_ramtorch_enable")
        self.horizontalLayout_mem.addWidget(self.use_ramtorch_enable)
        self.unsloth_offload_checkpointing = QCheckBox(self.moss_training_box)
        self.unsloth_offload_checkpointing.setObjectName(u"unsloth_offload_checkpointing")
        self.horizontalLayout_mem.addWidget(self.unsloth_offload_checkpointing)
        self.horizontalLayout_mem.addStretch(1)
        self.formLayout.setLayout(8, QFormLayout.SpanningRole, self.horizontalLayout_mem)

        # Network
        self.label_exclude_blocks = QLabel(self.moss_training_box)
        self.label_exclude_blocks.setObjectName(u"label_exclude_blocks")
        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.label_exclude_blocks)
        self.exclude_blocks_input = QLineEdit(self.moss_training_box)
        self.exclude_blocks_input.setObjectName(u"exclude_blocks_input")
        self.formLayout.setWidget(9, QFormLayout.FieldRole, self.exclude_blocks_input)

        # Samples
        self.horizontalLayout_samples = QHBoxLayout()
        self.label_sample_shift = QLabel(self.moss_training_box)
        self.label_sample_shift.setObjectName(u"label_sample_shift")
        self.horizontalLayout_samples.addWidget(self.label_sample_shift)
        self.sample_shift_input = DoubleSpinBox(self.moss_training_box)
        self.sample_shift_input.setObjectName(u"sample_shift_input")
        self.sample_shift_input.setDecimals(2)
        self.sample_shift_input.setMinimum(0.01)
        self.sample_shift_input.setMaximum(100.0)
        self.sample_shift_input.setSingleStep(0.1)
        self.sample_shift_input.setValue(5.0)
        self.horizontalLayout_samples.addWidget(self.sample_shift_input)
        self.label_uncond_skip = QLabel(self.moss_training_box)
        self.label_uncond_skip.setObjectName(u"label_uncond_skip")
        self.horizontalLayout_samples.addWidget(self.label_uncond_skip)
        self.uncond_skip_layers_input = QLineEdit(self.moss_training_box)
        self.uncond_skip_layers_input.setObjectName(u"uncond_skip_layers_input")
        self.uncond_skip_layers_input.setText(u"22")
        self.horizontalLayout_samples.addWidget(self.uncond_skip_layers_input)
        self.formLayout.setLayout(10, QFormLayout.SpanningRole, self.horizontalLayout_samples)

        # Advanced: disable speedups
        self.label_advanced = QLabel(self.moss_training_box)
        self.label_advanced.setObjectName(u"label_advanced")
        self.formLayout.setWidget(11, QFormLayout.LabelRole, self.label_advanced)
        self.horizontalLayout_advanced = QHBoxLayout()
        self.no_compile_enable = QCheckBox(self.moss_training_box)
        self.no_compile_enable.setObjectName(u"no_compile_enable")
        self.horizontalLayout_advanced.addWidget(self.no_compile_enable)
        self.no_lokr_compile_enable = QCheckBox(self.moss_training_box)
        self.no_lokr_compile_enable.setObjectName(u"no_lokr_compile_enable")
        self.horizontalLayout_advanced.addWidget(self.no_lokr_compile_enable)
        self.stock_lycoris_enable = QCheckBox(self.moss_training_box)
        self.stock_lycoris_enable.setObjectName(u"stock_lycoris_enable")
        self.horizontalLayout_advanced.addWidget(self.stock_lycoris_enable)
        self.no_sample_cache_enable = QCheckBox(self.moss_training_box)
        self.no_sample_cache_enable.setObjectName(u"no_sample_cache_enable")
        self.horizontalLayout_advanced.addWidget(self.no_sample_cache_enable)
        self.horizontalLayout_advanced.addStretch(1)
        self.formLayout.setLayout(11, QFormLayout.FieldRole, self.horizontalLayout_advanced)

        self.gridLayout_2.addLayout(self.formLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.moss_training_box, 0, 0, 1, 1)

        self.retranslateUi(moss_ui)
        QMetaObject.connectSlotsByName(moss_ui)

    def retranslateUi(self, moss_ui):
        def tr(text):
            return QCoreApplication.translate("moss_ui", text, None)

        def tip(text):
            return tr(u"<html><head/><body><p>" + text + u"</p></body></html>")

        moss_ui.setWindowTitle(tr(u"Form"))
        self.moss_training_box.setTitle(tr(u"Train MOSS-Image"))
        self.moss_training_box.setToolTip(tip(
            u"Train a LoRA / LyCORIS network on MOSS-Image with <code>moss_train_network.py</code>. "
            u"Needs <b>Cache TE Outputs</b> in Network Args (checked for you when this box is switched on): the "
            u"Qwen3.5-9B text encoder is loaded only while caching. With a LyCORIS algorithm (LoKr, ...) set "
            u"<b>LyCORIS Preset</b> to <code>moss</code> (or <code>moss-text</code>): LyCORIS's default preset finds no "
            u"MOSS modules and would train nothing. DiT, VAE and Text Encoder are required here; General Args' "
            u"Base Model and VAE are not used for MOSS."))

        self.label_dit.setText(tr(u"DiT Model"))
        self.label_vae.setText(tr(u"VAE Model"))
        self.label_te.setText(tr(u"Text Encoder"))
        self.label_tokenizer.setText(tr(u"Tokenizer Folder"))
        self.dit_model_input.setPlaceholderText(tr(u"Path to the MOSS-Image DiT file (.safetensors) [Required]"))
        self.dit_model_input.setToolTip(tip(
            u"MOSS-Image Base or SFT DiT (.safetensors). Maps to <code>--pretrained_model_name_or_path</code>."))
        self.vae_model_input.setPlaceholderText(tr(u"Path to the FLUX.2 small-decoder VAE file [Required]"))
        self.vae_model_input.setToolTip(tip(
            u"The FLUX.2 VAE with the small decoder that MOSS-Image uses. Maps to <code>--vae</code>."))
        self.text_encoder_input.setPlaceholderText(tr(u"Path to the Qwen3.5-9B MOSS text encoder file [Required]"))
        self.text_encoder_input.setToolTip(tip(
            u"Qwen3.5-9B text encoder (.safetensors; Base and SFT share it). Loaded only while caching text-encoder "
            u"outputs. Maps to <code>--moss_text_encoder</code>."))
        self.tokenizer_dir_input.setPlaceholderText(
            tr(u"MOSS-Image tokenizer/ folder [Required unless MOSS_TOKENIZER_DIR is set]"))
        self.tokenizer_dir_input.setToolTip(tip(
            u"Folder with the Qwen3.5 <code>tokenizer.json</code> (the <code>tokenizer/</code> folder of MOSS-Image). "
            u"Required: the trainer has no built-in default. It may stay empty only when the "
            u"<code>MOSS_TOKENIZER_DIR</code> environment variable points to such a folder for the backend. "
            u"Maps to <code>--moss_tokenizer_dir</code>."))

        self.max_token_label.setText(tr(u"Max Tokens"))
        self.max_token_input.setToolTip(tip(
            u"Text tokens per caption, chat template included. 1-4096 (the trainer refuses more); default 4096 "
            u"(omitted). Maps to <code>--moss_max_token_length</code>."))

        self.label_sampling.setText(tr(u"Timestep Sampling"))
        self.timestep_sampling_selector.setToolTip(tip(
            u"<b>moss</b>: the released Base's last pretraining distribution (logit-normal, see the four values below)."
            u"<br/><b>sigma</b>: scheduler sigma table. Uses <i>Discrete Flow Shift</i>."
            u"<br/><b>uniform</b>: flat random sampling."
            u"<br/><b>sigmoid</b>: logit-normal. Uses <i>Sigmoid Scale</i> (a number or a preset) and <i>Scale Resume</i>."
            u"<br/><b>shift</b>: logit-normal with a shift. Uses <i>Sigmoid Scale</i> (a number) and <i>Discrete Flow Shift</i>."))
        self.label_shift.setText(tr(u"Discrete Flow Shift"))
        self.discrete_flow_shift_input.setToolTip(tip(
            u"Shift for <code>sigma</code> and <code>shift</code> sampling. MOSS default: 1.0 (omitted at 1.0)."))
        self.label_sigmoid.setText(tr(u"Sigmoid Scale"))
        self.sigmoid_scale_input.setToolTip(tip(
            u"A number (sigmoid(scale * N(0,1)), default 1.0) or, with <code>sigmoid</code> sampling, a preset:"
            u"<br/><b>balanced</b> = 1.0, <b>content</b> = more high-noise steps (composition, shapes, colour layout),"
            u" <b>style</b> = more low-noise steps (line work, texture, rendering detail)."))
        self.label_scale_resume.setText(tr(u"Scale Resume"))
        self.scale_resume_input.setToolTip(tip(
            u"After Max Train Epochs, train this many more epochs with the low-noise <b>style</b> preset. "
            u"Needs <code>sigmoid</code> sampling and <b>Max Train Epochs</b> in General Args (not steps). 0 = off. "
            u"Maps to <code>--scale_resume</code>."))

        self.label_logit_mean.setText(tr(u"Logit Mean"))
        self.logit_mean_input.setToolTip(tip(u"<code>moss</code> sampling: mean of logit(sigma). Default: 1.09."))
        self.label_logit_std.setText(tr(u"Logit Std"))
        self.logit_std_input.setToolTip(tip(u"<code>moss</code> sampling: std of logit(sigma). Default: 0.95."))
        self.label_sigma_cutoff.setText(tr(u"Sigma Cutoff"))
        self.sigma_cutoff_input.setToolTip(tip(u"<code>moss</code> sampling: truncation before rescaling. Default: 0.95."))
        self.label_uniform_mix.setText(tr(u"Uniform Mix"))
        self.uniform_mix_input.setToolTip(tip(u"<code>moss</code> sampling: weight of the uniform mixture. Default: 0.01."))

        self.label_attn_mode.setText(tr(u"Attention"))
        self.attn_mode_selector.setToolTip(tip(
            u"Attention backend. <b>default</b>: no <code>attn_mode</code> key, so the trainer uses xformers when "
            u"<i>xformers</i> is checked in General Args and torch SDPA otherwise. <b>torch (SDPA)</b>: always torch "
            u"SDPA. <b>flash</b> (flash-attn varlen) is about 1.6x faster per step, but needs the <code>flash-attn</code> "
            u"package and an Ampere or newer GPU. <b>xformers</b> needs the xformers package. sageattn is not offered: "
            u"it has no backward pass. Maps to <code>--attn_mode</code>."))
        self.split_attn_enable.setText(tr(u"Split Attention"))
        self.split_attn_enable.setToolTip(tip(u"Per-sample attention: slower, lowest memory. Maps to <code>--split_attn</code>."))

        self.label_blocks_to_swap.setText(tr(u"Blocks to Swap"))
        self.blocks_to_swap_input.setToolTip(tip(
            u"Transformer blocks swapped between CPU and GPU. Lower VRAM, slower. 0 = off (at most 40). "
            u"Not together with Unsloth Offload. Maps to <code>--blocks_to_swap</code>."))
        self.use_ramtorch_enable.setText(tr(u"RamTorch"))
        self.use_ramtorch_enable.setToolTip(tip(
            u"Keep the base linear weights in pinned CPU RAM and stream them in each step: much lower VRAM, "
            u"slightly slower. Maps to <code>--use_ramtorch</code>."))
        self.unsloth_offload_checkpointing.setText(tr(u"Unsloth Offload"))
        self.unsloth_offload_checkpointing.setToolTip(tip(
            u"Offload checkpointed activations to CPU RAM with async transfers. Not with "
            u"<code>--cpu_offload_checkpointing</code> or Blocks to Swap."))

        self.label_exclude_blocks.setText(tr(u"Exclude Blocks"))
        self.exclude_blocks_input.setPlaceholderText(tr(u"Block indices left out of the network, e.g. 22 (Optional)"))
        self.exclude_blocks_input.setToolTip(tip(
            u"Comma-separated joint block indices (0-41) left out of the LoRA / LyCORIS network, e.g. <code>22</code>: "
            u"the block the official sampler's negative branch skips. With LyCORIS it needs <b>LyCORIS Preset</b> "
            u"<code>moss</code> or <code>moss-text</code>. <b>Changes the network</b>: do not change it when "
            u"resuming or continuing from network weights. Maps to <code>--moss_lora_exclude_blocks</code>."))

        self.label_sample_shift.setText(tr(u"Sample Shift"))
        self.sample_shift_input.setToolTip(tip(
            u"Sample images: sigma shift. Official 5.0 (omitted at 5.0); <code>--fs</code> in a prompt line overrides it."))
        self.label_uncond_skip.setText(tr(u"Uncond Skip Layers"))
        self.uncond_skip_layers_input.setPlaceholderText(tr(u"empty = off"))
        self.uncond_skip_layers_input.setToolTip(tip(
            u"Sample images: main blocks skipped in the negative CFG branch. Official <code>22</code> (omitted); "
            u"empty = off. Maps to <code>--moss_sample_uncond_skip_layers</code>."))

        self.label_advanced.setText(tr(u"Disable Speedups"))
        self.label_advanced.setToolTip(tip(u"Advanced: switch off the fork's speedups. Leave unchecked unless debugging."))
        self.no_compile_enable.setText(tr(u"No Compile"))
        self.no_compile_enable.setToolTip(tip(
            u"Do not torch.compile the block elementwise chains. Maps to <code>--moss_no_compile</code>."))
        self.no_lokr_compile_enable.setText(tr(u"No LoKr Compile"))
        self.no_lokr_compile_enable.setToolTip(tip(
            u"Do not torch.compile the LoKr+DoRA weight rebuild. Maps to <code>--moss_no_lokr_compile</code>."))
        self.stock_lycoris_enable.setText(tr(u"Stock LyCORIS"))
        self.stock_lycoris_enable.setToolTip(tip(
            u"Use LyCORIS's stock LoKr forward and per-call sample merges (slower; same results). "
            u"Maps to <code>--moss_stock_lycoris</code>."))
        self.no_sample_cache_enable.setText(tr(u"No Sample Cache"))
        self.no_sample_cache_enable.setToolTip(tip(
            u"Sample without the merged LoKr weight cache (about 9 GB less VRAM while sampling; slower). "
            u"Maps to <code>--moss_no_sample_cache</code>."))
