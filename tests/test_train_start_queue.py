"""Starting a queued job: the GUI must only drop a queue item (and its TOML) once the backend accepted /train.

External review 2026-11-05 #6: with /validate 200, /train 400 and /is_training idle, train_helper returned True, the
queued TOML was already deleted and the queue went on to the next item. The HTTP layer is a fake; no backend runs.

Run from the repo root (CPU only, message boxes are captured, never shown):

    set QT_QPA_PLATFORM=offscreen
    venv\\Scripts\\python.exe -m unittest tests.test_train_start_queue
"""

import itertools
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

import requests
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from main_ui_files.MainUI import MainWidget
from main_ui_files.SavingUI import SavingWidget
from _fake_backend import IDLE, RUNNING, STARTED, VALIDATED, Answers, FakeBackend


class TrainStartTest(unittest.TestCase):
    URL = "http://127.0.0.1:8000"

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("queue_store").mkdir()
        Path("config.json").write_text("{}", encoding="utf-8")
        self.boxes = []
        for kind in ("warning", "critical", "information"):
            patcher = mock.patch.object(
                QtWidgets.QMessageBox, kind,
                staticmethod(lambda *a, kind=kind, **k: self.boxes.append((kind, a[1], a[2]))),
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        self.main = MainWidget()
        self.main.backend_url_input.setText(self.URL)
        self.errors = []
        self.main.training_error.disconnect()
        self.main.training_error.connect(lambda title, text: self.errors.append((title, text)))
        sleep_patch = mock.patch("main_ui_files.MainUI.sleep", lambda _: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def backend(self, train_answer, is_training=(200, IDLE)) -> FakeBackend:
        fake = FakeBackend({"/validate": (200, VALIDATED), "/train": train_answer, "/is_training": is_training})
        for name in ("get", "post"):
            patcher = mock.patch(f"main_ui_files.MainUI.requests.{name}", getattr(fake, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        return fake

    def save_args_toml(self) -> None:
        """Saving Args > Save TOML on (written to auto_save_store, as no folder is given)."""
        saving = next(widget for widget in self.main.args_widget.args_widget_array if isinstance(widget, SavingWidget))
        saving.widget.save_toml_enable.setChecked(True)
        saving.enable_disable_toml(True)

    @staticmethod
    def auto_saved() -> list[str]:
        folder = Path("auto_save_store")
        return sorted(file.name for file in folder.iterdir()) if folder.exists() else []

    def warnings(self) -> list[str]:
        return [title for kind, title, _ in self.boxes if kind == "warning"]

    def queue_items(self, count: int) -> list[Path]:
        queue = self.main.queue_widget
        names = itertools.count(1)  # distinct queue file names (time.time_ns can repeat within a few ms on Windows)
        with mock.patch("modules.QueueItem.time.time_ns", lambda: next(names)):
            for _ in range(count):
                queue.add_to_queue()
        self.assertEqual(len(queue.elements), count, self.boxes)
        return [item.queue_file for item in queue.elements]

    def test_rejected_start_returns_false_and_keeps_the_toml(self):
        """The reviewer's sequence: /validate 200, /train 400, /is_training idle."""
        fake = self.backend((400, {"detail": "No Previously Validated Args"}))
        toml = Path("queue_store/item.toml")
        self.assertTrue(self.main.save_toml(toml), self.boxes)
        self.assertFalse(self.main.train_helper(self.URL, toml))
        self.assertTrue(toml.exists())
        self.assertEqual(fake.routes_called(), ["/validate", "/train"])
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.errors[0][0], "Training did not start")
        self.assertIn("No Previously Validated Args", self.errors[0][1])

    def test_rejected_start_keeps_the_queue_item_and_stops_the_queue(self):
        fake = self.backend((400, {"detail": "Invalid Train Parameters"}))
        files = self.queue_items(2)
        self.main.start_training_thread()
        self.assertEqual([item.queue_file for item in self.main.queue_widget.elements], files)
        self.assertTrue(all(file.exists() for file in files))
        self.assertEqual(fake.routes_called(), ["/validate", "/train"])  # the second item was not started
        self.assertEqual([title for title, _ in self.errors], ["Training did not start"])
        self.assertIn("Invalid Train Parameters", self.errors[0][1])
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_start_answered_with_training_false_is_a_failure(self):
        fake = self.backend((200, {"detail": "Training Already Running", "training": False}))
        files = self.queue_items(1)
        self.main.start_training_thread()
        self.assertEqual(len(self.main.queue_widget.elements), 1)
        self.assertTrue(files[0].exists())
        self.assertNotIn("/is_training", fake.routes_called())
        self.assertIn("Training Already Running", self.errors[0][1])

    def test_unreachable_train_keeps_the_queue_item(self):
        self.backend(requests.exceptions.ConnectionError("refused"))
        files = self.queue_items(1)
        self.main.start_training_thread()  # no exception out of the training thread
        self.assertEqual(len(self.main.queue_widget.elements), 1)
        self.assertTrue(files[0].exists())
        self.assertEqual(self.errors[0][0], "Connection Error")
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_rejected_validation_keeps_the_queue_item(self):
        fake = self.backend((200, STARTED))
        fake.routes["/validate"] = (400, ["vae is not found"])
        files = self.queue_items(1)
        self.main.start_training_thread()
        self.assertEqual(len(self.main.queue_widget.elements), 1)
        self.assertTrue(files[0].exists())
        self.assertEqual(self.errors, [("Backend validation failed", "vae is not found")])

    def test_accepted_starts_consume_the_queue_in_order(self):
        fake = self.backend((200, STARTED))
        files = self.queue_items(2)
        self.main.start_training_thread()
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertFalse(any(file.exists() for file in files))
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"] * 2)
        self.assertEqual(self.errors, [])

    def test_errored_training_after_an_accepted_start_stops_the_queue(self):
        """Unchanged behaviour: the started item is consumed, the next one waits."""
        fake = self.backend((200, STARTED), is_training=(200, {"training": False, "errored": True}))
        files = self.queue_items(2)
        self.main.start_training_thread()
        self.assertEqual([item.queue_file for item in self.main.queue_widget.elements], files[1:])
        self.assertFalse(files[0].exists())
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"])

    def remove_head_during_first_validate(self, fake: FakeBackend) -> None:
        queue = self.main.queue_widget
        original_post = fake.post
        removed = []

        def post_and_remove(url, **kwargs):
            response = original_post(url, **kwargs)
            if not removed:
                removed.append(queue.elements[0])
                queue.selected = queue.elements[0]
                queue.remove_from_queue()  # the user's Remove button: the item and its TOML go
            return response

        patcher = mock.patch("main_ui_files.MainUI.requests.post", post_and_remove)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_item_removed_during_validation_is_not_started(self):
        """Review finding: with Save TOML on, removing the item while /validate ran made the auto-save copy raise
        FileNotFoundError, killing the training thread with the button left at 'Stop Training'; with it off the
        removed item was still started."""
        fake = self.backend((200, STARTED))
        self.save_args_toml()
        files = self.queue_items(1)
        self.remove_head_during_first_validate(fake)
        self.main.start_training_thread()  # no exception out of the training thread
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertFalse(files[0].exists())
        self.assertEqual(fake.routes_called(), ["/validate"])
        self.assertEqual(self.auto_saved(), [])
        self.assertEqual(self.errors, [])
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_queue_moves_on_after_an_item_removed_during_validation(self):
        fake = self.backend((200, STARTED))
        files = self.queue_items(2)
        self.remove_head_during_first_validate(fake)
        self.main.start_training_thread()
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertFalse(any(file.exists() for file in files))
        self.assertEqual(fake.routes_called(), ["/validate", "/validate", "/train", "/is_training"])
        self.assertEqual(self.errors, [])

    def test_refused_start_writes_no_auto_save_toml_and_the_retry_writes_one(self):
        """Review finding: the auto-save TOML was written between /validate and /train, so a refused start left
        output_args.toml behind and the retry added output_args_1.toml."""
        fake = self.backend((400, {"detail": "Training Already Running"}))
        self.save_args_toml()
        files = self.queue_items(1)
        queued_bytes = files[0].read_bytes()
        self.main.start_training_thread()
        self.assertEqual(self.auto_saved(), [])
        self.assertTrue(files[0].exists())
        fake.routes["/train"] = (200, STARTED)
        self.main.start_training_thread()
        self.assertEqual(self.auto_saved(), ["output_args.toml"])
        self.assertEqual(Path("auto_save_store/output_args.toml").read_bytes(), queued_bytes)
        self.assertFalse(files[0].exists())
        self.assertEqual(self.main.queue_widget.elements, [])

    def save_tag_file(self, fake: FakeBackend) -> None:
        """Saving Args > Save Tag File on (written to auto_save_store), with tags in the /validate answer."""
        saving = next(widget for widget in self.main.args_widget.args_widget_array if isinstance(widget, SavingWidget))
        saving.widget.save_tag_enable.setChecked(True)
        saving.enable_disable_tag_file(True)
        fake.routes["/validate"] = (200, {**VALIDATED, "tags": {"1girl": 3}})

    def assert_unknown_start_reasons(self, text: str) -> None:
        """The "Training may have started" text names who drops the /train answer. A current backend answers /train
        before it closes its tunnel (kill_tunnel_on_train_start) or stops its server (Colab), so only older backends
        with those options, or one that crashed while starting, end here; the text must not blame a current backend.
        While a run is going, /validate and /train answer 409, but a finished run is not refused, so the text must not
        promise that a retry cannot train the item twice."""
        self.assertIn("An older backend with kill_tunnel_on_train_start", text)
        self.assertIn("an older Colab backend", text)
        self.assertNotIn("A backend with kill_tunnel_on_train_start", text)
        self.assertIn("crashed", text)
        self.assertIn('409 "Training Already Running"', text)
        self.assertIn("has already trained", text)
        self.assertNotIn("cannot start it twice", text)

    def assert_start_unknown(self, train_answer, probe) -> None:
        """Review finding 2026-11-05: /train failed and the backend could not be asked afterwards, and the GUI counted
        it as a start (the queued TOML deleted, the item dropped, the tag file and auto-save TOML written), so a
        backend that crashed while starting lost the job. Nothing shows that training started: the item and its TOML
        stay, nothing is written, the queue stops, and a warning says an older backend with kill_tunnel_on_train_start
        (or an older Colab backend) may be training it anyway."""
        fake = self.backend(train_answer, is_training=probe)
        self.save_args_toml()
        self.save_tag_file(fake)
        files = self.queue_items(2)
        self.main.start_training_thread()
        self.assertEqual([item.queue_file for item in self.main.queue_widget.elements], files)
        self.assertTrue(all(file.exists() for file in files))
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"])
        self.assertEqual(self.errors, [])
        self.assertEqual(self.warnings(), ["Training may have started"])
        text = next(text for kind, _, text in self.boxes if kind == "warning")
        self.assertIn(self.URL, text)
        self.assertIn("kept in the queue", text)
        self.assert_unknown_start_reasons(text)
        self.assertEqual(self.auto_saved(), [])  # neither the auto-save TOML nor the tag file
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_train_connection_dropped_and_backend_gone_keeps_the_item(self):
        self.assert_start_unknown(
            requests.exceptions.ConnectionError("tunnel closed"), requests.exceptions.ConnectionError("gone")
        )

    def test_train_server_error_and_backend_gone_keeps_the_item(self):
        """An older Colab backend answered None from /train (a Starlette 500) and stopped the server; a backend that
        crashed while starting looks the same."""
        self.assert_start_unknown((500, "Internal Server Error"), requests.exceptions.ConnectionError("gone"))

    def test_tunnel_error_pages_keep_the_item(self):
        self.assert_start_unknown((502, "<html>Bad gateway</html>"), (502, "<html>Bad gateway</html>"))

    def test_unknown_start_without_a_queue_does_not_mention_a_queue_item(self):
        """Review finding: with an empty queue Start Training trains queue_store/temp.toml with no queue item, so the
        warning must not say an item was kept in the queue or tell the user to remove it."""
        fake = self.backend(requests.exceptions.ConnectionError("tunnel closed"),
                            is_training=requests.exceptions.ConnectionError("gone"))
        self.save_args_toml()
        self.save_tag_file(fake)
        self.assertEqual(self.main.queue_widget.elements, [])
        self.main.start_training_thread()
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"])
        self.assertEqual(self.errors, [])
        self.assertEqual(self.warnings(), ["Training may have started"])
        text = next(text for kind, _, text in self.boxes if kind == "warning")
        self.assertIn(self.URL, text)
        self.assertIn("Nothing was queued", text)
        self.assert_unknown_start_reasons(text)
        self.assertNotIn("kept in the queue", text)
        self.assertNotIn("remove the item", text)
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertEqual(self.auto_saved(), [])
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_retry_after_an_unknown_start_starts_the_kept_item(self):
        fake = self.backend(requests.exceptions.ConnectionError("tunnel closed"),
                            is_training=requests.exceptions.ConnectionError("gone"))
        self.save_args_toml()
        self.save_tag_file(fake)
        files = self.queue_items(1)
        queued_bytes = files[0].read_bytes()
        self.main.start_training_thread()
        self.assertEqual(self.warnings(), ["Training may have started"])
        self.assertEqual(self.auto_saved(), [])
        fake.routes["/train"] = (200, STARTED)
        fake.routes["/is_training"] = (200, IDLE)
        self.main.start_training_thread()  # Start Training again: the backend now accepts it
        self.assertEqual(self.auto_saved(), ["output_args.toml", "output_tags.txt"])
        self.assertEqual(Path("auto_save_store/output_args.toml").read_bytes(), queued_bytes)
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertFalse(files[0].exists())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.main.begin_training_button.text(), "Start Training")

    def test_server_error_with_the_backend_idle_is_a_refused_start(self):
        fake = self.backend((500, "Internal Server Error"), is_training=(200, IDLE))
        files = self.queue_items(1)
        self.main.start_training_thread()
        self.assertEqual(len(self.main.queue_widget.elements), 1)
        self.assertTrue(files[0].exists())
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training"])
        self.assertEqual([title for title, _ in self.errors], ["Training did not start"])
        self.assertIn("Internal Server Error", self.errors[0][1])

    def test_server_error_with_the_backend_training_is_a_start(self):
        fake = self.backend((500, "Internal Server Error"), is_training=Answers((200, RUNNING), (200, IDLE)))
        files = self.queue_items(1)
        self.main.start_training_thread()
        self.assertEqual(self.main.queue_widget.elements, [])
        self.assertFalse(files[0].exists())
        self.assertEqual(fake.routes_called(), ["/validate", "/train", "/is_training", "/is_training"])
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
