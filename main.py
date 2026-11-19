import contextlib
from pathlib import Path
import sys
import json
from threading import Thread

from PySide6 import QtWidgets
from qt_material import apply_stylesheet
import requests
from main_ui_files.MainWindow import MainWindow
from modules.BackendAuth import backend_headers
import subprocess
import time


# the training venv the backend runs in (PyTorch 2.14), set up by install.bat / update.bat
TRAINING_VENV = "venv_t214"


def venv_python(venv: str = TRAINING_VENV) -> Path:
    if sys.platform == "linux":
        return Path(f"backend/sd_scripts/{venv}/bin/python")
    return Path(f"backend/sd_scripts/{venv}/Scripts/python.exe")


def run_backend(venv: str = TRAINING_VENV):
    python = venv_python(venv)
    if not python.exists():
        print(f"training venv not found: {python} (run install.bat)")
        return
    with contextlib.suppress(Exception):
        subprocess.check_call(
            f"{python} backend/main.py backend", shell=sys.platform == "linux"
        )


def CreateConfig():
    return {
        "theme": {
            "location": Path("css/themes/dark_teal.xml").as_posix(),
            "is_light": False,
            "density_scale": "0",
        }
    }


def main() -> None:
    queue_store = Path("queue_store")
    if not queue_store.exists():
        queue_store.mkdir()
    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.exists() else CreateConfig()
    if "theme" not in config_dict:
        config_dict.update(CreateConfig())
    config.write_text(json.dumps(config_dict, indent=2))
    backend_thread = None
    if "run_local" in config_dict and config_dict["run_local"]:
        backend_thread = Thread(target=run_backend, daemon=True)
        backend_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    if config_dict["theme"]["location"]:
        density = config_dict["theme"].get("density_scale", "0")
        extra = {'density_scale': density} if density != "0" else {}
        
        # Add larger font size when using compact mode
        if density == "-2":
            extra['font_size'] = '15px'

        apply_stylesheet(
            app,
            theme=config_dict["theme"]["location"],
            invert_secondary=config_dict["theme"]["is_light"],
            extra=extra
        )

    window = MainWindow(app)
    window.setWindowTitle("LoRA Trainer - 67372a Fork - Refresh Branch")
    window.show()
    app.exec()
    config_dict = json.loads(config.read_text())
    if not config_dict.get("run_local"):
        return
    if window.main_widget.training_thread:
        while window.main_widget.training_thread.is_alive():
            time.sleep(5.0)
    url = window.main_widget.backend_url_input.text()
    requests.get(f"{url}/stop_server", headers=backend_headers(url))


if __name__ == "__main__":
    main()
