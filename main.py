import contextlib
import os
from pathlib import Path
import sys
import json
from threading import Thread

from PySide6 import QtWidgets
from qt_material import apply_stylesheet
import requests
from main_ui_files.MainWindow import MainWindow
import subprocess
import time


def venv_python(venv: str) -> Path:
    if sys.platform == "linux":
        return Path(f"backend/sd_scripts/{venv}/bin/python")
    return Path(f"backend/sd_scripts/{venv}/Scripts/python.exe")


def training_venv(config_dict: dict) -> str:
    """The training venv the backend runs in: backend/sd_scripts/venv_t214 (PyTorch 2.14) when chosen with
    run.bat --torch 2.14, LETS_TORCH=2.14 or "torch_profile": "2.14" in config.json, or when it is the only one
    installed; else venv (2.7)."""
    chosen = os.environ.get("LETS_TORCH", "") or str(config_dict.get("torch_profile", ""))
    for i, arg in enumerate(sys.argv):
        if arg == "--torch" and i + 1 < len(sys.argv):
            chosen = sys.argv[i + 1]
        elif arg.startswith("--torch="):
            chosen = arg.split("=", 1)[1]
    has_27, has_214 = venv_python("venv").exists(), venv_python("venv_t214").exists()
    if chosen.strip() in ("2.14", "2.14.0", "214"):
        if has_214:
            return "venv_t214"
        print("PyTorch 2.14 venv not installed (update.bat --torch 2.14 adds it): using backend/sd_scripts/venv")
    elif has_214 and not has_27:
        print("only the PyTorch 2.14 venv is installed: using backend/sd_scripts/venv_t214")
        return "venv_t214"
    return "venv"


def run_backend(venv: str = "venv"):
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
        backend_thread = Thread(target=run_backend, args=(training_venv(config_dict),), daemon=True)
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
    requests.get(f"{window.main_widget.backend_url_input.text()}/stop_server")


if __name__ == "__main__":
    main()
