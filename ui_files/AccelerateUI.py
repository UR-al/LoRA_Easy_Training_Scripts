# -*- coding: utf-8 -*-

################################################################################
## Custom UI file for Accelerate (Multi-GPU) settings
##
## Created for: LoRA Easy Training Scripts
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, Qt)
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QSizePolicy, QVBoxLayout, QWidget)

from modules.ScrollOnSelect import SpinBox


class Ui_accelerate_ui(object):
    def setupUi(self, accelerate_ui):
        if not accelerate_ui.objectName():
            accelerate_ui.setObjectName(u"accelerate_ui")
        accelerate_ui.resize(400, 120)

        self.verticalLayout = QVBoxLayout(accelerate_ui)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        # Main group box (checkable to enable/disable multi-GPU)
        self.accelerate_group = QGroupBox(accelerate_ui)
        self.accelerate_group.setObjectName(u"accelerate_group")
        self.accelerate_group.setCheckable(True)
        self.accelerate_group.setChecked(False)

        self.formLayout = QFormLayout(self.accelerate_group)
        self.formLayout.setObjectName(u"formLayout")

        # Number of GPUs
        self.num_processes_label = QLabel(self.accelerate_group)
        self.num_processes_label.setObjectName(u"num_processes_label")
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.num_processes_label)

        self.num_processes_input = SpinBox(self.accelerate_group)
        self.num_processes_input.setObjectName(u"num_processes_input")
        self.num_processes_input.setFocusPolicy(Qt.StrongFocus)
        self.num_processes_input.setMinimum(1)
        self.num_processes_input.setMaximum(16)
        self.num_processes_input.setValue(2)
        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.num_processes_input)

        # Main Process Port
        self.main_process_port_label = QLabel(self.accelerate_group)
        self.main_process_port_label.setObjectName(u"main_process_port_label")
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.main_process_port_label)

        self.main_process_port_input = SpinBox(self.accelerate_group)
        self.main_process_port_input.setObjectName(u"main_process_port_input")
        self.main_process_port_input.setFocusPolicy(Qt.StrongFocus)
        self.main_process_port_input.setMinimum(1024)
        self.main_process_port_input.setMaximum(65535)
        self.main_process_port_input.setValue(29500)
        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.main_process_port_input)

        # GPU IDs (accelerate --gpu_ids); empty lets the backend choose
        self.gpu_ids_label = QLabel(self.accelerate_group)
        self.gpu_ids_label.setObjectName(u"gpu_ids_label")
        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.gpu_ids_label)

        self.gpu_ids_input = QLineEdit(self.accelerate_group)
        self.gpu_ids_input.setObjectName(u"gpu_ids_input")
        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.gpu_ids_input)

        self.verticalLayout.addWidget(self.accelerate_group)

        self.retranslateUi(accelerate_ui)
        QMetaObject.connectSlotsByName(accelerate_ui)
    # setupUi

    def retranslateUi(self, accelerate_ui):
        accelerate_ui.setWindowTitle(QCoreApplication.translate("accelerate_ui", u"Form", None))
        self.accelerate_group.setTitle(QCoreApplication.translate("accelerate_ui", u"Enable Multi-GPU Training", None))
        self.accelerate_group.setToolTip(QCoreApplication.translate("accelerate_ui",
            u"<html><head/><body><p>Enable multi-GPU training using accelerate launch, your GPUs must be the same. "
            u"This will distribute the training across multiple GPUs. Please configure your accelerate first. Number of GPUs equals --num_processes=X</p></body></html>", None))
        self.num_processes_label.setText(QCoreApplication.translate("accelerate_ui", u"Number of GPUs", None))
        self.num_processes_label.setToolTip(QCoreApplication.translate("accelerate_ui",
            u"<html><head/><body><p>Number of GPUs to use for distributed training.</p></body></html>", None))
        self.main_process_port_label.setText(QCoreApplication.translate("accelerate_ui", u"Main Process Port", None))
        self.main_process_port_label.setToolTip(QCoreApplication.translate("accelerate_ui",
            u"<html><head/><body><p>Port for the main process to communicate with other processes. "
            u"Change this if the default port (29500) is already in use.</p></body></html>", None))
        self.gpu_ids_label.setText(QCoreApplication.translate("accelerate_ui", u"GPU IDs", None))
        self.gpu_ids_label.setToolTip(QCoreApplication.translate("accelerate_ui",
            u"<html><head/><body><p>Which GPUs to train on, as comma-separated GPU numbers (accelerate --gpu_ids), "
            u"one per process: 0,1 for two GPUs. Only for Number of GPUs 2 or more; leave empty to let the backend "
            u"choose.</p></body></html>", None))
        self.gpu_ids_input.setPlaceholderText(QCoreApplication.translate("accelerate_ui", u"e.g. 0,1 (empty: backend default)", None))
    # retranslateUi
