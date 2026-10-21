# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AnimaUI.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget)

from modules.DragDropLineEdit import DragDropLineEdit
from modules.ScrollOnSelect import (ComboBox, DoubleSpinBox, SpinBox)
from modules.SigmoidScaleInput import SigmoidScaleInput

class Ui_anima_ui(object):
    def setupUi(self, anima_ui):
        if not anima_ui.objectName():
            anima_ui.setObjectName(u"anima_ui")
        anima_ui.resize(523, 500)
        self.gridLayout = QGridLayout(anima_ui)
        self.gridLayout.setObjectName(u"gridLayout")
        self.anima_training_box = QGroupBox(anima_ui)
        self.anima_training_box.setObjectName(u"anima_training_box")
        self.anima_training_box.setCheckable(True)
        self.anima_training_box.setChecked(False)
        self.gridLayout_2 = QGridLayout(self.anima_training_box)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        
        # DiT Path (maps to pretrained_model_name_or_path)
        self.label = QLabel(self.anima_training_box)
        self.label.setObjectName(u"label")
        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.dit_model_input = DragDropLineEdit(self.anima_training_box)
        self.dit_model_input.setObjectName(u"dit_model_input")
        self.horizontalLayout.addWidget(self.dit_model_input)
        self.dit_model_selector = QPushButton(self.anima_training_box)
        self.dit_model_selector.setObjectName(u"dit_model_selector")
        self.horizontalLayout.addWidget(self.dit_model_selector)
        self.formLayout_3.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout)
        
        # Qwen3 Path (maps to --qwen3)
        self.label_2 = QLabel(self.anima_training_box)
        self.label_2.setObjectName(u"label_2")
        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_2)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.qwen3_model_input = DragDropLineEdit(self.anima_training_box)
        self.qwen3_model_input.setObjectName(u"qwen3_model_input")
        self.horizontalLayout_2.addWidget(self.qwen3_model_input)
        self.qwen3_model_selector = QPushButton(self.anima_training_box)
        self.qwen3_model_selector.setObjectName(u"qwen3_model_selector")
        self.horizontalLayout_2.addWidget(self.qwen3_model_selector)
        self.formLayout_3.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_2)
        
        # VAE Path (maps to --vae)
        self.label_3 = QLabel(self.anima_training_box)
        self.label_3.setObjectName(u"label_3")
        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_3)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.vae_model_input = DragDropLineEdit(self.anima_training_box)
        self.vae_model_input.setObjectName(u"vae_model_input")
        self.horizontalLayout_3.addWidget(self.vae_model_input)
        self.vae_model_selector = QPushButton(self.anima_training_box)
        self.vae_model_selector.setObjectName(u"vae_model_selector")
        self.horizontalLayout_3.addWidget(self.vae_model_selector)
        self.formLayout_3.setLayout(2, QFormLayout.FieldRole, self.horizontalLayout_3)

        # T5 Tokenizer Path
        self.label_t5 = QLabel(self.anima_training_box)
        self.label_t5.setObjectName(u"label_t5")
        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label_t5)
        self.horizontalLayout_t5 = QHBoxLayout()
        self.horizontalLayout_t5.setObjectName(u"horizontalLayout_t5")
        self.t5_tokenizer_input = DragDropLineEdit(self.anima_training_box)
        self.t5_tokenizer_input.setObjectName(u"t5_tokenizer_input")
        self.horizontalLayout_t5.addWidget(self.t5_tokenizer_input)
        self.t5_tokenizer_selector = QPushButton(self.anima_training_box)
        self.t5_tokenizer_selector.setObjectName(u"t5_tokenizer_selector")
        self.horizontalLayout_t5.addWidget(self.t5_tokenizer_selector)
        self.formLayout_3.setLayout(3, QFormLayout.FieldRole, self.horizontalLayout_t5)

        # Token Lengths
        self.horizontalLayout_tokens = QHBoxLayout()
        
        self.qwen3_len_label = QLabel(self.anima_training_box)
        self.qwen3_len_label.setObjectName(u"qwen3_len_label")
        self.horizontalLayout_tokens.addWidget(self.qwen3_len_label)
        self.qwen3_max_token_input = SpinBox(self.anima_training_box)
        self.qwen3_max_token_input.setMaximum(16777215)
        self.qwen3_max_token_input.setValue(512)
        self.horizontalLayout_tokens.addWidget(self.qwen3_max_token_input)
        
        self.t5_len_label = QLabel(self.anima_training_box)
        self.t5_len_label.setObjectName(u"t5_len_label")
        self.horizontalLayout_tokens.addWidget(self.t5_len_label)
        self.t5_max_token_input = SpinBox(self.anima_training_box)
        self.t5_max_token_input.setMaximum(16777215)
        self.t5_max_token_input.setValue(512)
        self.horizontalLayout_tokens.addWidget(self.t5_max_token_input)
        
        self.formLayout_3.setLayout(4, QFormLayout.SpanningRole, self.horizontalLayout_tokens)

        # Sampling Params
        self.horizontalLayout_sampling = QHBoxLayout()
        
        self.label_sampling = QLabel(self.anima_training_box)
        self.label_sampling.setObjectName(u"label_sampling")
        self.horizontalLayout_sampling.addWidget(self.label_sampling)
        self.timestep_sampling_selector = ComboBox(self.anima_training_box)
        self.timestep_sampling_selector.addItem("sigma")
        self.timestep_sampling_selector.addItem("uniform")
        self.timestep_sampling_selector.addItem("sigmoid")
        self.timestep_sampling_selector.addItem("shift")
        self.timestep_sampling_selector.addItem("flux_shift")
        self.horizontalLayout_sampling.addWidget(self.timestep_sampling_selector)
        
        self.label_shift = QLabel(self.anima_training_box)
        self.label_shift.setObjectName(u"label_shift")
        self.horizontalLayout_sampling.addWidget(self.label_shift)
        self.discrete_flow_shift_input = DoubleSpinBox(self.anima_training_box)
        self.discrete_flow_shift_input.setDecimals(2)
        self.discrete_flow_shift_input.setSingleStep(0.1)
        self.discrete_flow_shift_input.setValue(3.0)
        self.horizontalLayout_sampling.addWidget(self.discrete_flow_shift_input)

        self.label_sigmoid = QLabel(self.anima_training_box)
        self.label_sigmoid.setObjectName(u"label_sigmoid")
        self.horizontalLayout_sampling.addWidget(self.label_sigmoid)
        self.sigmoid_scale_input = SigmoidScaleInput(self.anima_training_box)
        self.sigmoid_scale_input.setObjectName(u"sigmoid_scale_input")
        self.horizontalLayout_sampling.addWidget(self.sigmoid_scale_input)

        self.label_scale_resume = QLabel(self.anima_training_box)
        self.label_scale_resume.setObjectName(u"label_scale_resume")
        self.horizontalLayout_sampling.addWidget(self.label_scale_resume)
        self.scale_resume_input = SpinBox(self.anima_training_box)
        self.scale_resume_input.setObjectName(u"scale_resume_input")
        self.scale_resume_input.setMinimum(0)
        self.scale_resume_input.setMaximum(999)
        self.scale_resume_input.setValue(0)
        self.horizontalLayout_sampling.addWidget(self.scale_resume_input)
        
        self.formLayout_3.setLayout(5, QFormLayout.SpanningRole, self.horizontalLayout_sampling)

        # VAE / Memory Options
        self.horizontalLayout_vae_mem = QHBoxLayout()

        self.label_vae_chunk = QLabel(self.anima_training_box)
        self.label_vae_chunk.setObjectName(u"label_vae_chunk")
        self.horizontalLayout_vae_mem.addWidget(self.label_vae_chunk)
        self.vae_chunk_size_input = SpinBox(self.anima_training_box)
        self.vae_chunk_size_input.setObjectName(u"vae_chunk_size_input")
        self.vae_chunk_size_input.setMinimum(0)
        self.vae_chunk_size_input.setMaximum(16777215)
        self.vae_chunk_size_input.setSingleStep(2)
        self.vae_chunk_size_input.setValue(0)
        self.horizontalLayout_vae_mem.addWidget(self.vae_chunk_size_input)

        self.vae_disable_cache_enable = QCheckBox(self.anima_training_box)
        self.vae_disable_cache_enable.setObjectName(u"vae_disable_cache_enable")
        self.horizontalLayout_vae_mem.addWidget(self.vae_disable_cache_enable)

        self.label_blocks_to_swap = QLabel(self.anima_training_box)
        self.label_blocks_to_swap.setObjectName(u"label_blocks_to_swap")
        self.horizontalLayout_vae_mem.addWidget(self.label_blocks_to_swap)
        self.blocks_to_swap_input = SpinBox(self.anima_training_box)
        self.blocks_to_swap_input.setObjectName(u"blocks_to_swap_input")
        self.blocks_to_swap_input.setMinimum(0)
        self.blocks_to_swap_input.setMaximum(100)
        self.blocks_to_swap_input.setValue(0)
        self.horizontalLayout_vae_mem.addWidget(self.blocks_to_swap_input)

        self.use_ramtorch_enable = QCheckBox(self.anima_training_box)
        self.use_ramtorch_enable.setObjectName(u"use_ramtorch_enable")
        self.horizontalLayout_vae_mem.addWidget(self.use_ramtorch_enable)

        self.formLayout_3.setLayout(6, QFormLayout.SpanningRole, self.horizontalLayout_vae_mem)

        # Misc Checkboxes
        self.horizontalLayout_misc = QHBoxLayout()

        self.flash_attn_enable = QCheckBox(self.anima_training_box)
        self.flash_attn_enable.setObjectName(u"flash_attn_enable")
        self.horizontalLayout_misc.addWidget(self.flash_attn_enable)

        self.split_attn_enable = QCheckBox(self.anima_training_box)
        self.split_attn_enable.setObjectName(u"split_attn_enable")
        self.horizontalLayout_misc.addWidget(self.split_attn_enable)

        self.unsloth_offload_checkpointing = QCheckBox(self.anima_training_box)
        self.unsloth_offload_checkpointing.setObjectName(u"unsloth_offload_checkpointing")
        self.horizontalLayout_misc.addWidget(self.unsloth_offload_checkpointing)
        
        self.formLayout_3.setLayout(7, QFormLayout.SpanningRole, self.horizontalLayout_misc)

        # Anima 3.8B v1.1 connector (--anima38_connector and its options)
        self.anima38_box = QGroupBox(self.anima_training_box)
        self.anima38_box.setObjectName(u"anima38_box")
        self.anima38_box.setCheckable(True)
        self.anima38_box.setChecked(False)
        self.formLayout_anima38 = QFormLayout(self.anima38_box)
        self.formLayout_anima38.setObjectName(u"formLayout_anima38")

        self.label_qwen35 = QLabel(self.anima38_box)
        self.label_qwen35.setObjectName(u"label_qwen35")
        self.formLayout_anima38.setWidget(0, QFormLayout.LabelRole, self.label_qwen35)
        self.horizontalLayout_qwen35 = QHBoxLayout()
        self.qwen35_model_input = DragDropLineEdit(self.anima38_box)
        self.qwen35_model_input.setObjectName(u"qwen35_model_input")
        self.horizontalLayout_qwen35.addWidget(self.qwen35_model_input)
        self.qwen35_model_selector = QPushButton(self.anima38_box)
        self.qwen35_model_selector.setObjectName(u"qwen35_model_selector")
        self.horizontalLayout_qwen35.addWidget(self.qwen35_model_selector)
        self.formLayout_anima38.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_qwen35)

        self.label_qwen35_tokenizer = QLabel(self.anima38_box)
        self.label_qwen35_tokenizer.setObjectName(u"label_qwen35_tokenizer")
        self.formLayout_anima38.setWidget(1, QFormLayout.LabelRole, self.label_qwen35_tokenizer)
        self.horizontalLayout_qwen35_tokenizer = QHBoxLayout()
        self.qwen35_tokenizer_input = DragDropLineEdit(self.anima38_box)
        self.qwen35_tokenizer_input.setObjectName(u"qwen35_tokenizer_input")
        self.horizontalLayout_qwen35_tokenizer.addWidget(self.qwen35_tokenizer_input)
        self.qwen35_tokenizer_selector = QPushButton(self.anima38_box)
        self.qwen35_tokenizer_selector.setObjectName(u"qwen35_tokenizer_selector")
        self.horizontalLayout_qwen35_tokenizer.addWidget(self.qwen35_tokenizer_selector)
        self.formLayout_anima38.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_qwen35_tokenizer)

        self.horizontalLayout_anima38_opts = QHBoxLayout()
        self.label_anima38_negative = QLabel(self.anima38_box)
        self.label_anima38_negative.setObjectName(u"label_anima38_negative")
        self.horizontalLayout_anima38_opts.addWidget(self.label_anima38_negative)
        self.anima38_negative_selector = ComboBox(self.anima38_box)
        self.anima38_negative_selector.setObjectName(u"anima38_negative_selector")
        self.anima38_negative_selector.addItem("connector")
        self.anima38_negative_selector.addItem("native")
        self.horizontalLayout_anima38_opts.addWidget(self.anima38_negative_selector)
        self.label_anima38_batch = QLabel(self.anima38_box)
        self.label_anima38_batch.setObjectName(u"label_anima38_batch")
        self.horizontalLayout_anima38_opts.addWidget(self.label_anima38_batch)
        self.anima38_batch_input = SpinBox(self.anima38_box)
        self.anima38_batch_input.setObjectName(u"anima38_batch_input")
        self.anima38_batch_input.setMinimum(1)
        self.anima38_batch_input.setMaximum(64)
        self.anima38_batch_input.setValue(1)
        self.horizontalLayout_anima38_opts.addWidget(self.anima38_batch_input)
        self.horizontalLayout_anima38_opts.addStretch(1)
        self.formLayout_anima38.setLayout(2, QFormLayout.SpanningRole, self.horizontalLayout_anima38_opts)

        self.formLayout_3.setWidget(8, QFormLayout.SpanningRole, self.anima38_box)

        # Advanced: disable speedups
        self.horizontalLayout_advanced = QHBoxLayout()
        self.label_advanced = QLabel(self.anima_training_box)
        self.label_advanced.setObjectName(u"label_advanced")
        self.horizontalLayout_advanced.addWidget(self.label_advanced)
        self.no_compile_enable = QCheckBox(self.anima_training_box)
        self.no_compile_enable.setObjectName(u"no_compile_enable")
        self.horizontalLayout_advanced.addWidget(self.no_compile_enable)
        self.no_lokr_compile_enable = QCheckBox(self.anima_training_box)
        self.no_lokr_compile_enable.setObjectName(u"no_lokr_compile_enable")
        self.horizontalLayout_advanced.addWidget(self.no_lokr_compile_enable)
        self.stock_lycoris_enable = QCheckBox(self.anima_training_box)
        self.stock_lycoris_enable.setObjectName(u"stock_lycoris_enable")
        self.horizontalLayout_advanced.addWidget(self.stock_lycoris_enable)
        self.no_sample_cache_enable = QCheckBox(self.anima_training_box)
        self.no_sample_cache_enable.setObjectName(u"no_sample_cache_enable")
        self.horizontalLayout_advanced.addWidget(self.no_sample_cache_enable)
        self.horizontalLayout_advanced.addStretch(1)
        self.formLayout_3.setLayout(9, QFormLayout.SpanningRole, self.horizontalLayout_advanced)

        self.gridLayout_2.addLayout(self.formLayout_3, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.anima_training_box, 0, 0, 1, 1)

        self.retranslateUi(anima_ui)
        QMetaObject.connectSlotsByName(anima_ui)

    def retranslateUi(self, anima_ui):
        anima_ui.setWindowTitle(QCoreApplication.translate("anima_ui", u"Form", None))
        self.anima_training_box.setTitle(QCoreApplication.translate("anima_ui", u"Train Anima", None))
        self.label.setText(QCoreApplication.translate("anima_ui", u"DiT Model", None))
        self.label_2.setText(QCoreApplication.translate("anima_ui", u"Qwen3 Model", None))
        self.label_3.setText(QCoreApplication.translate("anima_ui", u"VAE Model", None))
        self.label_t5.setText(QCoreApplication.translate("anima_ui", u"T5 Tokenizer", None))
        self.qwen3_len_label.setText(QCoreApplication.translate("anima_ui", u"Qwen3 Max Tokens", None))
        self.t5_len_label.setText(QCoreApplication.translate("anima_ui", u"T5 Max Tokens", None))
        self.label_sampling.setText(QCoreApplication.translate("anima_ui", u"Timestep Sampling", None))
        self.label_shift.setText(QCoreApplication.translate("anima_ui", u"Discrete Flow Shift", None))
        self.label_sigmoid.setText(QCoreApplication.translate("anima_ui", u"Sigmoid Scale", None))
        self.label_vae_chunk.setText(QCoreApplication.translate("anima_ui", u"VAE Chunk Size", None))
        self.vae_disable_cache_enable.setText(QCoreApplication.translate("anima_ui", u"VAE Disable Cache", None))
        self.label_blocks_to_swap.setText(QCoreApplication.translate("anima_ui", u"Blocks to Swap", None))
        self.flash_attn_enable.setText(QCoreApplication.translate("anima_ui", u"Flash Attention", None))
        self.split_attn_enable.setText(QCoreApplication.translate("anima_ui", u"Split Attention", None))
        self.unsloth_offload_checkpointing.setText(QCoreApplication.translate("anima_ui", u"Unsloth Offload", None))
        self.label_scale_resume.setText(QCoreApplication.translate("anima_ui", u"Scale Resume", None))
        self.use_ramtorch_enable.setText(QCoreApplication.translate("anima_ui", u"RamTorch", None))
        self.anima38_box.setTitle(QCoreApplication.translate("anima_ui", u"Anima 3.8B v1.1 connector", None))
        self.label_qwen35.setText(QCoreApplication.translate("anima_ui", u"Qwen3.5 Model", None))
        self.label_qwen35_tokenizer.setText(QCoreApplication.translate("anima_ui", u"Qwen3.5 Tokenizer", None))
        self.label_anima38_negative.setText(QCoreApplication.translate("anima_ui", u"Sample Negative", None))
        self.label_anima38_batch.setText(QCoreApplication.translate("anima_ui", u"Qwen3.5 Batch Size", None))
        self.label_advanced.setText(QCoreApplication.translate("anima_ui", u"Disable Speedups", None))
        self.no_compile_enable.setText(QCoreApplication.translate("anima_ui", u"No Compile", None))
        self.no_lokr_compile_enable.setText(QCoreApplication.translate("anima_ui", u"No LoKr Compile", None))
        self.stock_lycoris_enable.setText(QCoreApplication.translate("anima_ui", u"Stock LyCORIS", None))
        self.no_sample_cache_enable.setText(QCoreApplication.translate("anima_ui", u"No Sample Cache", None))

        # Tooltips and Placeholders
        self.dit_model_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"Path to DiT model file (.safetensors) [Required]", None))
        self.dit_model_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Path to the Anima DiT model file (.safetensors). "
            u"This maps to <code>--pretrained_model_name_or_path</code> in sd_scripts.</p></body></html>", None))

        self.qwen3_model_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"Path to Qwen3-0.6B model file or directory [Required]", None))
        self.qwen3_model_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Path to the Qwen3-0.6B text encoder model file (.safetensors) or "
            u"HuggingFace directory. Maps to <code>--qwen3</code> in sd_scripts.</p></body></html>", None))

        self.vae_model_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"Path to Qwen-Image VAE model file [Required]", None))
        self.vae_model_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Path to the Qwen-Image VAE model file (.safetensors). "
            u"Maps to <code>--vae</code> in sd_scripts.</p></body></html>", None))

        self.t5_tokenizer_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"T5 Tokenizer directory (Optional, uses built-in config if empty)", None))
        self.t5_tokenizer_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Path to the T5 tokenizer directory. If not provided, uses the built-in "
            u"<code>configs/t5_old/</code> configuration. Maps to <code>--t5_tokenizer_path</code>.</p></body></html>", None))

        self.qwen3_max_token_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Maximum token length for the Qwen3 tokenizer. Default: 512.</p></body></html>", None))
        self.t5_max_token_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Maximum token length for the T5 tokenizer. Default: 512.</p></body></html>", None))

        self.timestep_sampling_selector.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Method for sampling timesteps during training."
            u"<br/><br/>"
            u"<b>sigma</b>: Uses weighting scheme to index into the scheduler's shifted sigma table. Uses <i>Discrete Flow Shift</i>."
            u"<br/>"
            u"<b>uniform</b>: Flat random sampling across [0, 1]. No extra parameters."
            u"<br/>"
            u"<b>sigmoid</b>: Logit-normal distribution (recommended for Anima). Uses <i>Sigmoid Scale</i>."
            u"<br/>"
            u"<b>shift</b>: Logit-normal with explicit shift applied. Uses both <i>Sigmoid Scale</i> and <i>Discrete Flow Shift</i>."
            u"<br/>"
            u"<b>flux_shift</b>: Resolution-adaptive FLUX.1-style shifting. Uses <i>Sigmoid Scale</i>."
            u"</p></body></html>", None))
        self.discrete_flow_shift_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Shift value for the Rectified Flow timestep distribution. Default: 3.0."
            u"<br/><br/>"
            u"<b>Used by:</b> <code>sigma</code> (via scheduler lookup table), <code>shift</code> (direct formula)."
            u"<br/>"
            u"<b>Not used by:</b> <code>sigmoid</code>, <code>flux_shift</code>, <code>uniform</code>."
            u"</p></body></html>", None))
        self.sigmoid_scale_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Scale factor for logit-normal timestep sampling. Controls distribution concentration. Default: 1.0."
            u"<br/>Higher values → more uniform; lower → more concentrated near sigma=0.5."
            u"<br/>With <code>sigmoid</code> sampling it also takes a preset: <b>balanced</b> (= 1.0), "
            u"<b>content</b> (more high-noise steps: composition, shapes) or <b>style</b> "
            u"(more low-noise steps: line work, texture)."
            u"<br/><br/>"
            u"<b>Used by:</b> <code>sigmoid</code>, <code>shift</code>, <code>flux_shift</code>."
            u"<br/>"
            u"<b>Not used by:</b> <code>sigma</code>, <code>uniform</code>."
            u"</p></body></html>", None))

        self.vae_chunk_size_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Spatial chunk size for Qwen-Image VAE encoding/decoding. "
            u"Reduces VRAM usage at the cost of speed. Must be an even number. "
            u"Set to 0 to disable chunking (default behavior).</p></body></html>", None))
        self.vae_disable_cache_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Disable internal VAE caching mechanism to reduce VRAM usage. "
            u"Encoding/decoding will also be faster, but this differs from official behavior.</p></body></html>", None))
        self.blocks_to_swap_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Number of Transformer blocks to swap between CPU and GPU during "
            u"forward/backward passes. Reduces VRAM usage at the cost of training speed. "
            u"Set to 0 to disable (default). Not together with Unsloth Offload.</p></body></html>", None))

        self.flash_attn_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Use Flash Attention for faster training and lower VRAM usage. "
            u"Requires the <code>flash-attn</code> package. Maps to <code>--attn_mode flash</code>.</p></body></html>", None))
        self.split_attn_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Split attention computation to reduce memory usage. "
            u"Automatically enabled when xFormers attention is selected in General Args. "
            u"Maps to <code>--split_attn</code>.</p></body></html>", None))
        self.unsloth_offload_checkpointing.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Offload activations to CPU RAM using async non-blocking transfers. "
            u"Faster than standard CPU offload checkpointing. Cannot be used with "
            u"<code>--cpu_offload_checkpointing</code> or <code>--blocks_to_swap</code>.</p></body></html>", None))

        self.scale_resume_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>After Max Train Epochs, train this many more epochs with the low-noise <b>style</b> "
            u"preset. Needs <code>sigmoid</code> sampling and <b>Max Train Epochs</b> in General Args (not steps). 0 = off. "
            u"Maps to <code>--scale_resume</code>.</p></body></html>", None))
        self.use_ramtorch_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Keep the base linear weights in CPU RAM and stream them in each step: "
            u"lower VRAM, slightly slower. Maps to <code>--use_ramtorch</code>.</p></body></html>", None))
        self.anima38_box.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Train on an Anima-3.8B v1.1 bundle with its frozen Semantic Connector v2 "
            u"(Qwen3.5-4B features), the same conditioning as the Forge anima38 extension. "
            u"Maps to <code>--anima38_connector</code>. Needs <b>Cache TE Outputs</b> in Network Args (checked for you "
            u"when this box is switched on) and the Qwen3.5 Model below.</p></body></html>", None))
        self.qwen35_model_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"Path to the Qwen3.5-4B text encoder file [Required]", None))
        self.qwen35_model_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Qwen3.5-4B text encoder (.safetensors), loaded only while caching. "
            u"Maps to <code>--qwen35_path</code>.</p></body></html>", None))
        self.qwen35_tokenizer_input.setPlaceholderText(QCoreApplication.translate("anima_ui",
            u"Qwen3.5 tokenizer folder (Optional, empty uses $ANIMA38_QWEN35_TOKENIZER_DIR or the trainer's default)",
            None))
        self.qwen35_tokenizer_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Folder with the Qwen3.5 <code>tokenizer.json</code>. Optional: when empty, the "
            u"trainer uses the folder in the <code>$ANIMA38_QWEN35_TOKENIZER_DIR</code> environment variable (of the "
            u"backend), or else its own default folder, and stops with a clear error if neither exists. "
            u"Maps to <code>--qwen35_tokenizer_dir</code>.</p></body></html>", None))
        self.anima38_negative_selector.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>How the negative prompt of training samples is conditioned.<br/>"
            u"<b>connector</b>: through the Semantic Connector, like Forge with 'Anima38 negative' switched on. "
            u"This is the trainer's default (no key is written).<br/>"
            u"<b>native</b>: the plain llm_adapter, like Forge's own default setting ('Anima38 negative' off).<br/>"
            u"Maps to <code>--anima38_sample_negative</code>.</p></body></html>", None))
        self.anima38_batch_input.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Captions per Qwen3.5 forward while caching. 1 = one unpadded caption per call, "
            u"exactly like Forge (default); larger is faster but pads the batch. "
            u"Maps to <code>--anima38_qwen35_batch_size</code>.</p></body></html>", None))
        self.label_advanced.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Advanced: switch off the fork's speedups. Leave unchecked unless debugging.</p></body></html>", None))
        self.no_compile_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Do not torch.compile the block elementwise chains. "
            u"Maps to <code>--anima_no_compile</code>.</p></body></html>", None))
        self.no_lokr_compile_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Do not torch.compile the LoKr+DoRA weight rebuild. "
            u"Maps to <code>--anima_no_lokr_compile</code>.</p></body></html>", None))
        self.stock_lycoris_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Use the stock LyCORIS LoKr forward and per-call sample merges (slower; same results). "
            u"Maps to <code>--anima_stock_lycoris</code>.</p></body></html>", None))
        self.no_sample_cache_enable.setToolTip(QCoreApplication.translate("anima_ui",
            u"<html><head/><body><p>Sample without the merged LoKr weight cache (slower; same results). "
            u"Maps to <code>--anima_no_sample_cache</code>.</p></body></html>", None))
