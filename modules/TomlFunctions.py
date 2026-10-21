from PySide6.QtWidgets import QFileDialog
from pathlib import Path
import locale
import toml
import json


def save_toml(args: dict, file_path: Path | None = None) -> bool:
    """Write the TOML to file_path, or to a file picked in a dialog. False when nothing was written."""
    if file_path:
        file_path.write_text(toml.dumps(args))
        return True
    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.exists() else {}

    file = QFileDialog().getSaveFileName(
        caption="Select config file save location",
        filter="Config File (*.toml)",
        dir=config_dict["toml_default"] if "toml_default" in config_dict else "",
    )[0]
    if not file:
        return False
    file = Path(file)
    config_dict["toml_default"] = file.parent.as_posix()
    config.write_text(json.dumps(config_dict, indent=2))
    file.write_text(toml.dumps(args))
    return True


NOT_GUI_TOML = (
    "This is not a GUI config TOML. It looks like a trainer config for the command-line launchers (for example "
    "examples/moss/config_moss.toml or examples/anima/config_anima38.toml) or a dataset config: the trainer reads "
    "those directly, and the GUI cannot import them. Enter the values in the GUI, or run the file with its launcher "
    "script."
)


def gui_toml_problem(loaded: dict) -> str | None:
    """None for a TOML the GUI saved (every top-level table holds "args" / "dataset_args", plus "subsets" and
    "train_mode"); otherwise why it cannot be loaded (a flat trainer config has top-level values)."""
    for key, value in loaded.items():
        if key == "subsets":
            ok = isinstance(value, list) and all(isinstance(item, dict) for item in value)
        elif key == "train_mode":
            ok = isinstance(value, dict) and "train_mode" in value
        else:
            ok = isinstance(value, dict) and bool(value) and set(value) <= {"args", "dataset_args"}
        if not ok:
            return NOT_GUI_TOML
    return None


def read_toml_text(file_path: Path) -> str:
    """The file as text: UTF-8 first (TOML's own encoding, and what the trainer configs use), then the locale
    encoding so TOMLs an older GUI saved in the system code page (cp949 here) still open."""
    raw = Path(file_path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False))


def load_toml(file_path: Path | None = None) -> dict | None:
    """Read file_path, or a file picked in a dialog when no path is given. A given path that does not exist gives None
    (no dialog: queue files are read from the training thread)."""
    if file_path:
        return toml.loads(read_toml_text(file_path)) if file_path.exists() else None
    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.exists() else {}

    file, _ = QFileDialog().getOpenFileName(
        caption="Select config file",
        filter="Config File (*.toml)",
        dir=config_dict["toml_default"] if "toml_default" in config_dict else "",
    )
    if not file:
        return
    file = Path(file)
    config_dict["toml_default"] = file.parent.as_posix()
    config.write_text(json.dumps(config_dict, indent=2))
    return toml.loads(read_toml_text(file))
