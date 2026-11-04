import json
from pathlib import Path
import sys
import subprocess
import os
from sys import platform

from install_uv import ensure_uv, create_venv, uv_pip_install

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_git_install() -> None:
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=platform == "linux",
        )
    except FileNotFoundError:
        logger.error("ERROR: git is not installed, please install git")
        return False
    return True


def main():
    if not check_git_install():
        return

    uv = ensure_uv()
    venv_path = create_venv(uv, "venv", "3.11")
    uv_pip_install(uv, "-U", "-r", "requirements.txt", venv_path=venv_path)

    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.exists() else {}

    install_backend = None
    while install_backend not in ("y", "n"):
        install_backend = input("Are you using this locally? (y/n): ").lower()
    if install_backend == "n":
        config_dict["run_local"] = False
        config.write_text(json.dumps(config_dict, indent=2))
        return
    config_dict["run_local"] = True
    config.write_text(json.dumps(config_dict, indent=2))

    subprocess.check_call("git submodule update --init --recursive", shell=platform == "linux")
    os.chdir(Path("backend"))
    # install.bat --torch 2.7 / 2.14 / both (or LETS_TORCH) picks the PyTorch profile(s); without it
    # backend/installer.py asks. 2.7 goes to backend/sd_scripts/venv, 2.14 to backend/sd_scripts/venv_t214.
    subprocess.check_call([sys.executable, "installer.py", "local", *sys.argv[1:]])


if __name__ == "__main__":
    main()
