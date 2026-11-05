"""Client side of the backend auth contract (external review 2026-11-05 #1).

With no token set nothing is sent, so the default local setup goes out exactly as before. With a token set, every
backend URL gets 'Authorization: Bearer <token>', a local one too (a backend started with LETS_BACKEND_TOKEN demands it
on every request, loopback included), the token from the LETS_BACKEND_TOKEN environment variable or else
config.json's "backend_token" (the environment wins). The HTTP layer is a fake; no backend runs.

Run from the repo root (CPU only):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_backend_auth
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))  # _fake_backend

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.MainUI import MainWidget
from modules import BackendAuth
from modules.NetworkManager import NetworkManager
from _fake_backend import IDLE, STARTED, VALIDATED, FakeBackend

LOCAL_URLS = ("http://127.0.0.1:8000", "http://localhost:8000", "http://LOCALHOST:8000/", "http://[::1]:8000",
              "http://127.0.0.2:8000", "http://0.0.0.0:8000")
REMOTE_URLS = ("http://192.168.0.7:8000", "https://abc.trycloudflare.com", "http://pc.tail1234.ts.net:8000",
               "http://127.0.0.1.example.com:8000")


class InTempDir(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(BackendAuth.TOKEN_ENV, None)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def write_config(self, **values):
        Path("config.json").write_text(json.dumps(values), encoding="utf-8")


class BackendHeadersTest(InTempDir):
    def test_local_backends_get_no_header_without_a_token(self):
        self.write_config(backend_url="http://127.0.0.1:8000")
        for url in LOCAL_URLS:
            self.assertEqual(BackendAuth.backend_headers(url), {}, url)

    def test_local_backends_get_a_configured_token(self):
        """A backend started with LETS_BACKEND_TOKEN refuses token-less requests from loopback too (review finding)."""
        self.write_config(backend_token="from-config")
        for url in LOCAL_URLS:
            self.assertEqual(BackendAuth.backend_headers(url), {"Authorization": "Bearer from-config"}, url)
        os.environ[BackendAuth.TOKEN_ENV] = "from-env"
        for url in LOCAL_URLS:
            self.assertEqual(BackendAuth.backend_headers(url), {"Authorization": "Bearer from-env"}, url)

    def test_remote_backend_token_from_env_then_config(self):
        for url in REMOTE_URLS:
            self.assertEqual(BackendAuth.backend_headers(url), {}, url)  # no token anywhere: nothing is sent
        self.write_config(backend_token=" from-config ")
        for url in REMOTE_URLS:
            self.assertEqual(BackendAuth.backend_headers(url), {"Authorization": "Bearer from-config"}, url)
        os.environ[BackendAuth.TOKEN_ENV] = "from-env"
        self.assertEqual(BackendAuth.backend_headers(REMOTE_URLS[0]), {"Authorization": "Bearer from-env"})
        os.environ[BackendAuth.TOKEN_ENV] = ""  # an empty variable does not hide the config token
        self.assertEqual(BackendAuth.backend_headers(REMOTE_URLS[0]), {"Authorization": "Bearer from-config"})

    def test_config_passed_in_is_used_and_a_broken_config_is_no_token(self):
        self.assertEqual(BackendAuth.backend_headers(REMOTE_URLS[0], {"backend_token": "given"}),
                         {"Authorization": "Bearer given"})
        Path("config.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(BackendAuth.backend_headers(REMOTE_URLS[0]), {})

    def test_url_without_a_host_gets_no_token(self):
        os.environ[BackendAuth.TOKEN_ENV] = "from-env"
        for url in ("", "localhost:8000", "127.0.0.1:8000"):
            self.assertEqual(BackendAuth.backend_headers(url), {}, url)


class TrainRequestsTest(InTempDir):
    def setUp(self):
        super().setUp()
        Path("queue_store").mkdir()
        for kind in ("warning", "critical", "information"):
            patcher = mock.patch.object(QtWidgets.QMessageBox, kind, staticmethod(lambda *a, **k: None))
            patcher.start()
            self.addCleanup(patcher.stop)
        sleep_patch = mock.patch("main_ui_files.MainUI.sleep", lambda _: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def run_job(self, url: str) -> FakeBackend:
        fake = FakeBackend({"/validate": (200, VALIDATED), "/train": (200, STARTED), "/is_training": (200, IDLE)})
        main = MainWidget()
        toml = Path("queue_store/item.toml")
        self.assertTrue(main.save_toml(toml))
        with mock.patch("main_ui_files.MainUI.requests.get", fake.get), \
                mock.patch("main_ui_files.MainUI.requests.post", fake.post):
            self.assertTrue(main.train_helper(url, toml))
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"])
        return fake

    def test_remote_backend_gets_the_token_on_every_request(self):
        self.write_config(backend_token="from-config")
        fake = self.run_job("http://192.168.0.7:8000")
        self.assertEqual([call[3] for call in fake.calls], [{"Authorization": "Bearer from-config"}] * 3)

    def test_local_backend_without_a_token_gets_no_header(self):
        self.write_config()
        fake = self.run_job("http://127.0.0.1:8000")
        self.assertEqual([call[3] for call in fake.calls], [{}] * 3)

    def test_local_backend_gets_a_set_token_on_every_request(self):
        self.write_config(backend_token="from-config")
        os.environ[BackendAuth.TOKEN_ENV] = "from-env"
        fake = self.run_job("http://127.0.0.1:8000")
        self.assertEqual([call[3] for call in fake.calls], [{"Authorization": "Bearer from-env"}] * 3)

    def test_stop_training_sends_the_token(self):
        self.write_config(backend_token="from-config")
        main = MainWidget()
        main.backend_url_input.setText("http://192.168.0.7:8000")
        main.training_thread = mock.Mock(is_alive=lambda: True)
        with mock.patch("main_ui_files.MainUI.requests.get") as get:
            main.start_training()
        get.assert_called_once_with("http://192.168.0.7:8000/stop_training",
                                    headers={"Authorization": "Bearer from-config"})


class NetworkManagerHeaderTest(InTempDir):
    def captured_request(self, url: str):
        manager = NetworkManager()
        old_url = manager._url
        self.addCleanup(manager.set_backend_url, old_url)
        manager.set_backend_url(url)
        with mock.patch.object(NetworkManager, "_network_manager") as qnam:
            manager.query("/tokenize", {"text": "a"}, lambda _: None)
        return qnam.get.call_args.args[0]

    def test_tokenize_query_header(self):
        os.environ[BackendAuth.TOKEN_ENV] = "from-env"
        self.assertEqual(self.captured_request("http://192.168.0.7:8000").rawHeader("Authorization").data(),
                         b"Bearer from-env")
        self.assertEqual(self.captured_request("http://localhost:8000").rawHeader("Authorization").data(),
                         b"Bearer from-env")
        os.environ[BackendAuth.TOKEN_ENV] = ""
        self.assertFalse(self.captured_request("http://localhost:8000").hasRawHeader("Authorization"))


if __name__ == "__main__":
    unittest.main()
