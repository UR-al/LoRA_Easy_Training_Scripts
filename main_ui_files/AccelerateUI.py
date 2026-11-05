import json
from pathlib import Path

from PySide6.QtWidgets import QWidget
from ui_files.AccelerateUI import Ui_accelerate_ui
from modules.BaseWidget import BaseWidget


def gpu_id_list(text: str) -> list[str] | None:
    """The GPU numbers of a GPU IDs field ("0, 1" -> ["0", "1"]); None when it is not comma-separated GPU numbers."""
    parts = [part.strip() for part in text.split(",")]
    if not all(part.isascii() and part.isdigit() for part in parts):
        return None
    return [str(int(part)) for part in parts]


class AccelerateWidget(BaseWidget):
    """Widget for configuring multi-GPU training with accelerate launch.

    Unlike other widgets, this one saves settings to config.json instead of
    using self.args, because accelerate settings are machine-specific and
    should not be saved to training TOML files.
    """

    CONFIG_DEFAULTS = {
        "enabled": False,
        "num_processes": 2,
        "main_process_port": 29500,
        "gpu_ids": "",
    }

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Multi-GPU Training (Accelerate)")
        self.widget = Ui_accelerate_ui()

        # Use a unique name but we won't use self.args for training config
        self.name = "accelerate_args"
        self.args = {}  # Keep empty - accelerate settings go to config.json
        self.dataset_args = {}

        self.setup_widget()
        self.setup_connections()
        self.load_from_config()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

    def setup_connections(self) -> None:
        # Save to config.json whenever any setting changes
        self.widget.accelerate_group.clicked.connect(self.save_to_config)
        self.widget.num_processes_input.valueChanged.connect(self.save_to_config)
        self.widget.main_process_port_input.valueChanged.connect(self.save_to_config)
        self.widget.gpu_ids_input.textChanged.connect(self.save_to_config)

    def get_accelerate_settings(self) -> dict:
        """Return accelerate settings for passing to backend."""
        if not self.widget.accelerate_group.isChecked():
            return {"enabled": False}

        settings = {
            "enabled": True,
            "num_processes": self.widget.num_processes_input.value(),
            "main_process_port": self.widget.main_process_port_input.value(),
        }
        gpu_ids = self.widget.gpu_ids_input.text().strip()
        if gpu_ids:  # empty: no key, the backend picks the GPUs (as before this field existed)
            settings["gpu_ids"] = gpu_ids
        return settings

    def get_validation_errors(self) -> list[str]:
        text = self.widget.gpu_ids_input.text().strip()
        if not self.widget.accelerate_group.isChecked() or not text:
            return []
        ids = gpu_id_list(text)
        if ids is None:
            return [f"Multi-GPU Training: GPU IDs must be comma-separated GPU numbers such as 0,1 (got '{text}')"]
        if len(set(ids)) != len(ids):
            return [f"Multi-GPU Training: GPU IDs names a GPU more than once ('{text}')"]
        processes = self.widget.num_processes_input.value()
        if processes < 2:
            # one process is a plain 'accelerate launch' on the backend's default GPU: the ids would be ignored
            return [
                f"Multi-GPU Training: GPU IDs apply only with Number of GPUs 2 or more (it is {processes}); "
                "clear GPU IDs or raise Number of GPUs"
            ]
        if len(ids) != processes:
            return [
                f"Multi-GPU Training: GPU IDs lists {len(ids)} GPUs but Number of GPUs is {processes}; make them match"
            ]
        return []

    def save_to_config(self) -> None:
        """Persist accelerate settings to config.json."""
        config_path = Path("config.json")
        config_dict = json.loads(config_path.read_text()) if config_path.exists() else {}
        config_dict["accelerate"] = self.get_accelerate_settings()
        config_path.write_text(json.dumps(config_dict, indent=2))

    def load_from_config(self) -> None:
        """Load accelerate settings from config.json."""
        config_path = Path("config.json")
        if not config_path.exists():
            return

        config_dict = json.loads(config_path.read_text())
        accel = config_dict.get("accelerate", {})

        self.widget.accelerate_group.setChecked(accel.get("enabled", self.CONFIG_DEFAULTS["enabled"]))
        self.widget.num_processes_input.setValue(accel.get("num_processes", self.CONFIG_DEFAULTS["num_processes"]))
        self.widget.main_process_port_input.setValue(accel.get("main_process_port", self.CONFIG_DEFAULTS["main_process_port"]))
        self.widget.gpu_ids_input.setText(str(accel.get("gpu_ids", self.CONFIG_DEFAULTS["gpu_ids"])))

    def load_args(self, args: dict) -> bool:
        """Load accelerate settings from config.json (not from TOML args)."""
        self.load_from_config()
        return True

    def load_dataset_args(self, dataset_args: dict) -> bool:
        """No dataset args for accelerate settings."""
        return True
