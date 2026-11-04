"""Tests for the PyTorch profiles of backend/installer.py and backend/updater.py (2026-11-04): 2.7 in sd_scripts/venv
and 2.14 in sd_scripts/venv_t214, side by side; which packages each profile asks uv for, how profiles are chosen on
install and update. uv is mocked; nothing is installed and no network is used."""

import importlib.util
import io
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent / "backend"
FLASH_27_WIN311 = ("https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.0.post2/"
                   "flash_attn-2.8.0.post2+cu128torch2.7.1cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl")
FLASH_214_WIN311 = ("https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/"
                    "flash_attn-2.8.4+cu130torch2.14.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl")


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
    return _load("backend_installer_profiles", "installer.py")


@pytest.fixture
def calls(installer, monkeypatch):
    """Records every uv pip install the installer would run (and any other subprocess call)."""
    seen = []
    monkeypatch.setattr(installer, "uv_pip_install", lambda uv, *args, venv_path="venv": seen.append(args))
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


def test_profiles_live_in_their_own_venvs(installer):
    assert installer.DEFAULT_TORCH_PROFILE == "2.7"
    assert installer.profile_venv_dir("2.7") == "venv"
    assert installer.profile_venv_dir("2.14.0") == "venv_t214"
    for value, want in (("2.7.1", "2.7"), ("27", "2.7"), (" 214 ", "2.14")):
        assert installer.normalize_torch_profile(value) == want
    with pytest.raises(SystemExit):
        installer.normalize_torch_profile("2.13")


def test_parse_torch_profiles_from_argv_and_env(installer, monkeypatch):
    assert installer.parse_torch_profiles([]) is None
    assert installer.parse_torch_profiles(["local", "--torch", "2.14"]) == ["2.14"]
    assert installer.parse_torch_profiles(["--torch=2.7.1"]) == ["2.7"]
    assert installer.parse_torch_profiles(["--torch", "both"]) == ["2.7", "2.14"]
    with pytest.raises(SystemExit):
        installer.parse_torch_profiles(["--torch"])
    monkeypatch.setenv("LETS_TORCH", "2.14")
    assert installer.parse_torch_profiles(["local"]) == ["2.14"]
    assert installer.parse_torch_profiles(["--torch", "2.7"]) == ["2.7"]  # the command line wins over the env


def test_ask_torch_profiles_interactive_and_not(installer, monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # pipe / CI / scheduled job: no prompt, the default
    assert installer.ask_torch_profiles() == ["2.7"]
    monkeypatch.setattr(sys, "stdin", Tty())
    answers = iter(["9", "both"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert installer.ask_torch_profiles() == ["2.7", "2.14"]  # asks again until a known answer
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert installer.ask_torch_profiles() == ["2.7"]  # Enter = default

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert installer.ask_torch_profiles() == ["2.7"]


def test_the_2_14_venv_never_reuses_another_profiles_dot_venv(installer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_venv(tmp_path / ".venv")  # a 2.7 install that lives in .venv
    created = []
    monkeypatch.setattr(installer.subprocess, "check_call", lambda cmd: created.append(cmd))
    assert installer.create_profile_venv("uv", "2.14") == "venv_t214"
    assert created and created[0][-1] == "venv_t214"
    assert installer.existing_profile_venvs() == [("2.7", ".venv")]


def test_existing_profile_venvs_and_torch_check(installer, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    assert installer.existing_profile_venvs() == []
    _fake_venv(tmp_path / "venv", "2.7.1+cu128")
    _fake_venv(tmp_path / "venv_t214", "2.14.0+cu130")
    assert installer.existing_profile_venvs() == [("2.7", "venv"), ("2.14", "venv_t214")]
    assert installer.installed_torch_version("venv_t214") == "2.14.0+cu130"
    def warnings():
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    with caplog.at_level("WARNING"):
        installer.check_venv_torch("venv", "2.7")
        installer.check_venv_torch("venv_t214", "2.14")
        assert warnings() == []
        installer.check_venv_torch("venv_t214", "2.7")  # a venv holding the other torch
    assert any("2.14.0+cu130" in w for w in warnings())


def test_setup_venv_2_7_installs_the_original_set(installer, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    venv = tmp_path / "venv"
    venv.mkdir()
    installer.setup_venv("uv", str(venv), "2.7")
    assert ("-U", "torch~=2.7.1", "torchvision~=0.22.1", "numpy~=2.2.6",
            "--index-url", "https://download.pytorch.org/whl/cu128") in calls
    assert ("-U", "--no-deps", "xformers==0.0.31.post1", "--index-url", "https://download.pytorch.org/whl/cu128") in calls
    assert ("-U", "--no-deps", "torchao~=0.13.0", "--index-strategy", "unsafe-best-match",
            "--extra-index-url", "https://download.pytorch.org/whl/cu128") in calls
    assert ("--no-deps", FLASH_27_WIN311) in calls
    assert ("-r", "requirements.txt") in calls  # no override: requirements.txt's triton 3.3 pins apply
    assert installer.read_torch_profile(str(venv)) == "2.7"


def test_setup_venv_2_7_on_an_unsupported_python_only_skips_flash_attn(installer, calls, tmp_path, monkeypatch):
    """As before profiles: no wheel -> log and skip (calls fails the test on any uninstall / other subprocess)."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(installer, "get_venv_python_version", lambda venv: "3.12.4")
    venv = tmp_path / "venv"
    venv.mkdir()
    installer.setup_venv("uv", str(venv), "2.7")
    assert not any("flash_attn" in " ".join(args) for args in calls)


def test_setup_venv_2_14_installs_its_own_set(installer, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    venv = tmp_path / "venv_t214"
    venv.mkdir()
    installer.setup_venv("uv", str(venv), "2.14")
    assert ("-U", "torch==2.14.0", "torchvision==0.29.0", "numpy~=2.2.6",
            "--index-url", "https://download.pytorch.org/whl/cu130") in calls
    assert not any("xformers" in " ".join(args) for args in calls)
    assert ("--no-deps", FLASH_214_WIN311) in calls
    assert ("-U", "--no-deps", "torchao~=0.13.0", "--index-strategy", "unsafe-best-match",
            "--extra-index-url", "https://download.pytorch.org/whl/cu130") in calls
    req = [args for args in calls if args[:2] == ("-r", "requirements.txt")]
    assert req and req[0][2] == "--override"
    overrides = Path(req[0][3]).read_text(encoding="utf-8")
    assert 'triton-windows==3.8.0.post29; platform_system == "Windows"' in overrides
    assert 'triton~=3.8.0; platform_system == "Linux"' in overrides
    assert installer.read_torch_profile(str(venv)) == "2.14"


def test_setup_venv_2_14_on_linux_has_no_flash_attn(installer, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    venv = tmp_path / "venv_t214"
    venv.mkdir()
    installer.setup_venv("uv", str(venv), "2.14")
    assert not any("flash_attn" in " ".join(args) for args in calls)


def test_both_profiles_install_the_same_git_commits(installer, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    git = {}
    for profile in ("2.7", "2.14"):
        calls.clear()
        venv = tmp_path / profile
        venv.mkdir()
        installer.setup_venv("uv", str(venv), profile)
        git[profile] = sorted(a for args in calls for a in args if a.startswith("git+"))
    assert git["2.7"] == git["2.14"] and len(git["2.7"]) == 3


def test_update_targets(installer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    updater = _load("backend_updater_profiles", "updater.py")
    created = []
    monkeypatch.setattr(updater, "create_profile_venv", lambda uv, p: created.append(p) or installer.profile_venv_dir(p))
    assert updater.update_targets("uv", []) == [("2.7", "venv")]  # nothing installed yet: the default venv
    _fake_venv(tmp_path / "venv")
    assert updater.update_targets("uv", []) == [("2.7", "venv")]  # existing installs keep updating 2.7 only
    _fake_venv(tmp_path / "venv_t214")
    assert updater.update_targets("uv", []) == [("2.7", "venv"), ("2.14", "venv_t214")]  # every profile venv
    assert updater.update_targets("uv", ["--torch", "2.14"]) == [("2.14", "venv_t214")] and created == ["2.14"]
    monkeypatch.setenv("LETS_TORCH", "2.14")  # set for training: a plain update still updates every venv
    assert updater.update_targets("uv", []) == [("2.7", "venv"), ("2.14", "venv_t214")]
