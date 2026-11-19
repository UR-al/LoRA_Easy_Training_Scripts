"""Tests for the training venv of backend/installer.py and backend/updater.py: PyTorch 2.14 in sd_scripts/venv_t214 is
the only environment they know. Which packages the installer asks uv for, that install and update only ever use
venv_t214 (whatever the command line or LETS_TORCH say, whatever other venv sits in sd_scripts), and that
sd_scripts/requirements.txt pins the triton that goes with torch 2.14. uv is mocked; nothing is installed and no
network is used."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent / "backend"
FLASH_WIN311 = ("https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/"
                "flash_attn-2.8.4+cu130torch2.14.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl")
CU130 = "https://download.pytorch.org/whl/cu130"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(BACKEND / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def installer(monkeypatch):
    monkeypatch.syspath_prepend(str(BACKEND))  # installer.py imports its sibling install_uv.py ...
    monkeypatch.delitem(sys.modules, "install_uv", raising=False)  # ... not the repo root's copy another test cached
    monkeypatch.delitem(sys.modules, "installer", raising=False)  # updater.py imports installer by name
    monkeypatch.delenv("LETS_TORCH", raising=False)
    return _load("backend_installer_torch", "installer.py")


@pytest.fixture
def calls(installer, monkeypatch):
    """Records every uv pip install the installer would run (and fails on any other subprocess call)."""
    seen = []
    monkeypatch.setattr(installer, "uv_pip_install", lambda uv, *args, venv_path="venv_t214": seen.append(args))
    monkeypatch.setattr(installer.subprocess, "call", lambda cmd: pytest.fail(f"unexpected subprocess call {cmd}"))
    monkeypatch.setattr(installer, "get_venv_python_version", lambda venv: "3.11.9")
    return seen


def _fake_venv(path: Path, torch_version=None):
    scripts = path / ("Scripts" if sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    (scripts / ("python.exe" if sys.platform == "win32" else "python")).write_text("")
    if torch_version:
        site = path / "Lib" / "site-packages"
        site.mkdir(parents=True)
        (site / f"torch-{torch_version}.dist-info").mkdir()


def test_the_installer_only_knows_torch_2_14(installer):
    assert installer.TORCH_VENV == "venv_t214"
    assert installer.TORCH_VERSION == "2.14"
    assert installer.TORCH_PACKAGES == ("torch==2.14.0", "torchvision==0.29.0")
    assert installer.TORCH_INDEX_URL == CU130
    assert installer.FLASH_ATTN_WHEELS == {("win32", 11): FLASH_WIN311}
    for gone in ("TORCH_PROFILES", "DEFAULT_TORCH_PROFILE", "parse_torch_profiles", "ask_torch_profiles",
                 "create_profile_venv", "existing_profile_venvs"):
        assert not hasattr(installer, gone), gone
    for name in ("installer.py", "updater.py", "run.bat", "run.sh"):
        text = (BACKEND / name).read_text(encoding="utf-8")
        for old in ("2.7", "cu128", "LETS_TORCH", "sd_scripts/venv/", "sd_scripts\\venv\\"):  # xformers: setup_venv test
            assert old not in text, (name, old)
    assert "sd_scripts\\venv_t214\\Scripts\\activate" in (BACKEND / "run.bat").read_text(encoding="utf-8")
    assert "sd_scripts/venv_t214/bin/activate" in (BACKEND / "run.sh").read_text(encoding="utf-8")


def test_requirements_pin_the_triton_of_torch_2_14():
    text = (BACKEND / "sd_scripts" / "requirements.txt").read_text(encoding="utf-8")
    pins = [line.strip() for line in text.splitlines() if line.strip().startswith("triton")]
    assert pins == ['triton~=3.8.0; platform_system == "Linux"',
                    'triton-windows==3.8.0.post29; platform_system == "Windows"']


def test_the_venv_is_venv_t214_and_never_another_one(installer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_venv(tmp_path / "venv")  # other venvs in sd_scripts are never reused
    _fake_venv(tmp_path / ".venv")
    created = []
    monkeypatch.setattr(installer.subprocess, "check_call", lambda cmd: created.append(cmd))
    assert installer.create_torch_venv("uv") == "venv_t214"
    assert created == [["uv", "venv", "--python", "3.11", "--seed", "venv_t214"]]
    _fake_venv(tmp_path / "venv_t214")
    assert installer.create_torch_venv("uv") == "venv_t214"
    assert len(created) == 1  # an existing venv_t214 is reused


def test_check_venv_torch_warns_about_another_torch(installer, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    _fake_venv(tmp_path / "venv_t214", "2.14.0+cu130")
    _fake_venv(tmp_path / "hand_upgraded", "2.15.0+cu130")
    assert installer.installed_torch_version("venv_t214") == "2.14.0+cu130"

    def warnings():
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    with caplog.at_level("WARNING"):
        installer.check_venv_torch("venv_t214")
        assert warnings() == []
        installer.check_venv_torch("hand_upgraded")
    assert any("2.15.0+cu130" in w for w in warnings())


def test_setup_venv_installs_torch_2_14(installer, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    venv = tmp_path / "venv_t214"
    venv.mkdir()
    installer.setup_venv("uv", str(venv))
    assert ("-U", "torch==2.14.0", "torchvision==0.29.0", "numpy~=2.2.6", "--index-url", CU130) in calls
    assert not any("xformers" in " ".join(args) for args in calls)
    assert ("--no-deps", FLASH_WIN311) in calls
    assert ("-U", "--no-deps", "torchao~=0.13.0", "--index-strategy", "unsafe-best-match",
            "--extra-index-url", CU130) in calls
    assert ("-r", "requirements.txt") in calls  # its triton pins are torch 2.14's: no override
    assert sorted(a for args in calls for a in args if a.startswith("git+")) == sorted([
        "git+https://github.com/67372a/RamTorch@feae2437e2f4ecabefdb410e28342433be8c05cd",
        "git+https://github.com/67372a/customized-optimizers@1d0e9426c862fabd14460905ff57e289572b6a34",
        "git+https://github.com/67372a/LyCORIS@c2cfd8ae5c5418ea7c77923eda4afa099bc69ba5",
    ])
    assert not any(p.name.startswith("lets_torch") for p in venv.iterdir())  # no profile marker or override file


@pytest.mark.parametrize("platform, python", [("linux", "3.11.9"), ("win32", "3.12.4")])
def test_setup_venv_without_a_flash_attn_wheel_skips_it(installer, calls, tmp_path, monkeypatch, platform, python):
    """No wheel (Linux, or another Windows Python) -> log and skip; calls fails the test on any other subprocess."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(installer, "get_venv_python_version", lambda venv: python)
    venv = tmp_path / "venv_t214"
    venv.mkdir()
    installer.setup_venv("uv", str(venv))
    assert not any("flash_attn" in " ".join(args) for args in calls)
    assert any("torch==2.14.0" in args for args in calls)


def _no_prompt(prompt=""):
    pytest.fail(f"the installer asked {prompt!r}")


@pytest.mark.parametrize("argv", [
    ["installer.py", "local"],
    ["installer.py", "local", "--torch", "2.7"],
    ["installer.py", "local", "--torch=both"],
    ["installer.py", "local", "--torch"],
])
def test_install_always_sets_up_venv_t214(installer, tmp_path, monkeypatch, argv):
    """An old --torch choice or LETS_TORCH changes nothing: no question, one venv, venv_t214."""
    (tmp_path / "sd_scripts").mkdir()
    _fake_venv(tmp_path / "sd_scripts" / "venv")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LETS_TORCH", "2.7")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("builtins.input", _no_prompt)
    monkeypatch.setattr(installer, "check_git_install", lambda: True)
    monkeypatch.setattr(installer, "setup_config", lambda colab, local: None)
    monkeypatch.setattr(installer, "ensure_uv", lambda: "uv")
    monkeypatch.setattr(installer, "setup_accelerate", lambda platform: None)
    commands, done = [], []
    monkeypatch.setattr(installer.subprocess, "check_call", lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(installer, "setup_venv", lambda uv, venv_path: done.append(venv_path))
    installer.main()
    assert done == ["venv_t214"]
    assert ["uv", "venv", "--python", "3.11", "--seed", "venv_t214"] in commands


def test_update_updates_only_venv_t214(installer, tmp_path, monkeypatch):
    (tmp_path / "sd_scripts").mkdir()
    _fake_venv(tmp_path / "sd_scripts" / "venv")  # another venv next to it is left alone
    _fake_venv(tmp_path / "sd_scripts" / "venv_t214", "2.14.0+cu130")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LETS_TORCH", "2.7")
    monkeypatch.setattr(sys, "argv", ["updater.py", "--torch", "2.7"])
    updater = _load("backend_updater_torch", "updater.py")
    monkeypatch.setattr(updater, "check_call", lambda *a, **kw: None)  # git submodule update
    monkeypatch.setattr(updater, "ensure_uv", lambda: "uv")
    monkeypatch.setattr(updater, "accelerate_config_path", lambda platform: tmp_path / "no_accelerate_config.yaml")
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, **kw: pytest.fail(f"unexpected venv creation {cmd}"))
    done = []
    monkeypatch.setattr(updater, "setup_venv", lambda uv, venv_path: done.append(venv_path))
    updater.main()
    assert done == ["venv_t214"]


def test_update_creates_venv_t214_when_missing(installer, tmp_path, monkeypatch):
    (tmp_path / "sd_scripts").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["updater.py"])
    updater = _load("backend_updater_torch_new", "updater.py")
    monkeypatch.setattr(updater, "check_call", lambda *a, **kw: None)
    monkeypatch.setattr(updater, "ensure_uv", lambda: "uv")
    monkeypatch.setattr(updater, "accelerate_config_path", lambda platform: tmp_path / "no_accelerate_config.yaml")
    created, done = [], []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, **kw: created.append(cmd))
    monkeypatch.setattr(updater, "setup_venv", lambda uv, venv_path: done.append(venv_path))
    updater.main()
    assert created == [["uv", "venv", "--python", "3.11", "--seed", "venv_t214"]]
    assert done == ["venv_t214"]
