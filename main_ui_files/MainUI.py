import contextlib
import json
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton
from main_ui_files.AccelerateUI import gpu_id_list
from main_ui_files.ArgsListUI import ArgsWidget
from main_ui_files.SubsetListUI import SubsetListWidget
from modules import ScrollOnSelect, TomlFunctions
from modules.BackendAuth import backend_headers
from modules.LineEditHighlight import LineEditWithHighlight
from main_ui_files.QueueUI import QueueWidget
from modules.Enums import TrainingModes
from pathlib import Path
from threading import Thread
from PySide6.QtCore import Signal
import requests
import toml
from requests.exceptions import ConnectionError
from time import sleep


class TomlLoadError(Exception):
    """A TOML that cannot be loaded into the GUI (missing, unreadable, or not a GUI TOML)."""


def backend_error_text(response) -> str:
    """The backend's /validate errors (a JSON list, or a dict with "detail") as text for a message box."""
    try:
        body = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(body, list):
        return "\n".join(str(item) for item in body) or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)


def training_started(response) -> bool:
    """Whether the backend accepted /train: HTTP 200, and not a body that says {"training": false}."""
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and body.get("training") is False)


class MainWidget(QWidget):
    training_error = Signal(str, str)
    training_warning = Signal(str, str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.training_thread = None
        self.main_layout = QGridLayout()
        self.args_widget = ArgsWidget()
        self.subset_widget = SubsetListWidget()
        self.queue_widget = QueueWidget()
        self.begin_training_button = QPushButton("Start Training")
        self.backend_url_input = LineEditWithHighlight()
        self.tab_widget = ScrollOnSelect.TabView()
        self.train_mode = TrainingModes.LORA

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        self.setLayout(self.main_layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.subset_widget.add_empty_subset("subset 1")
        config = Path("config.json")
        config_dict = json.loads(config.read_text()) if config.exists() else {}
        self.backend_url_input.setText(config_dict.get("backend_url", "http://127.0.0.1:8000"))
        self.backend_url_input.setPlaceholderText("Backend Server URL")
        self.tab_widget.addTab(self.args_widget, "Main Args")
        self.tab_widget.addTab(self.subset_widget, "Subset Args")

        self.main_layout.addWidget(self.tab_widget, 0, 0, 4, 1)
        self.main_layout.addWidget(self.queue_widget, 0, 1, 2, 1)
        self.main_layout.addWidget(self.backend_url_input, 2, 1, 1, 1)
        self.main_layout.addWidget(self.begin_training_button, 3, 1, 1, 1)
        self.queue_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.backend_url_input.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum
        )
        self.begin_training_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum
        )

    def setup_connections(self) -> None:
        self.args_widget.cacheLatentsChecked.connect(self.subset_widget.enable_disable_cache_latents)
        self.args_widget.keepTokensSepChecked.connect(self.subset_widget.enable_disable_variable_keep_tokens)
        self.args_widget.maskedLossChecked.connect(self.subset_widget.enable_disable_masked_loss)
        general_widget = self.args_widget.args_widget_array[0]
        bucket_widget = next(widget for widget in self.args_widget.args_widget_array if widget.name == "bucket_args")
        general_widget.widget.width_input.valueChanged.connect(lambda: self.sync_subset_inherited_args())
        general_widget.widget.height_input.valueChanged.connect(lambda: self.sync_subset_inherited_args())
        general_widget.widget.height_enable.toggled.connect(lambda: self.sync_subset_inherited_args())
        general_widget.widget.batch_size_input.valueChanged.connect(lambda: self.sync_subset_inherited_args())
        general_widget.resolution_jitter_enable.toggled.connect(lambda: self.sync_subset_inherited_args())
        for field in general_widget._jitter_fields:
            field.textChanged.connect(lambda: self.sync_subset_inherited_args())
        bucket_widget.widget.bucket_group.toggled.connect(lambda: self.sync_subset_inherited_args())
        bucket_widget.widget.min_input.valueChanged.connect(lambda: self.sync_subset_inherited_args())
        bucket_widget.widget.max_input.valueChanged.connect(lambda: self.sync_subset_inherited_args())
        self.queue_widget.saveQueue.connect(lambda x: self.save_toml(Path(x)))
        self.queue_widget.save_handler = lambda x: self.save_toml(
            Path(x), failure_note="The queue item was not added."
        )
        self.queue_widget.loadQueue.connect(lambda x: self.load_toml(Path(x)))
        self.begin_training_button.clicked.connect(self.start_training)
        self.backend_url_input.editingFinished.connect(self.update_url)
        self.training_error.connect(self.show_error)
        self.training_warning.connect(self.show_warning)
        self.sync_subset_inherited_args()

    def sync_subset_inherited_args(self) -> None:
        """Update values shown by subset controls that are not overridden."""
        general_widget = self.args_widget.args_widget_array[0]
        bucket_widget = next(widget for widget in self.args_widget.args_widget_array if widget.name == "bucket_args")
        inherited_args = {
            "resolution": general_widget.dataset_args.get(
                "resolution", general_widget.DATASET_DEFAULTS["resolution"]
            ),
            "batch_size": general_widget.dataset_args.get("batch_size", general_widget.DATASET_DEFAULTS["batch_size"]),
            "min_bucket_reso": bucket_widget.dataset_args.get("min_bucket_reso", bucket_widget.widget.min_input.value()),
            "max_bucket_reso": bucket_widget.dataset_args.get("max_bucket_reso", bucket_widget.widget.max_input.value()),
            "resolution_jitter_resolutions": general_widget.dataset_args.get("resolution_jitter_resolutions"),
            "resolution_jitter_batch_sizes": general_widget.dataset_args.get("resolution_jitter_batch_sizes"),
            "resolution_jitter_weights": general_widget.dataset_args.get("resolution_jitter_weights"),
        }
        self.subset_widget.set_inherited_dataset_args(inherited_args)

    def show_error(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, message)

    def update_url(self) -> None:
        url = self.backend_url_input.text()
        if url.endswith("/"):
            self.backend_url_input.setText(url.strip("/"))
            url = url.strip("/")
        config = Path("config.json")
        config_dict = json.loads(config.read_text()) if config.exists() else {}
        config_dict["backend_url"] = url
        config.write_text(json.dumps(config_dict, indent=2))

    def get_args(self) -> tuple[dict, dict]:
        base_args = self.args_widget.get_args()
        subset_args = self.subset_widget.dataset_args
        return base_args, subset_args

    def get_validation_errors(self) -> list[str]:
        """Args errors plus rules that need the subsets too."""
        errors = self.args_widget.get_validation_errors()
        errors.extend(self.caption_rule_errors())
        return errors

    def caption_rule_errors(self) -> list[str]:
        """Subset caption options the trainers refuse while the text encoder outputs are cached."""
        aw = self.args_widget
        network_args = aw.network_widget.args
        extra = aw.extra_args_widget.args
        anima_on = aw.anima_widget._anima_on()
        anima38_on = anima_on and aw.anima_widget.widget.anima38_box.isChecked()
        # the *_to_disk flag turns caching on by itself (MOSS-Image / Anima 3.8B without either flag get their own
        # "needs Cache TE Outputs" error)
        cached = any(
            source.get(key) for source in (network_args, extra)
            for key in ("cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk")
        )
        if not cached:
            return []
        variants = network_args.get("cache_caption_variants", 0)
        # plain Anima keeps caption dropout with cached outputs (it applies it at each step); SDXL, Flux, MOSS-Image
        # and Anima 3.8B refuse it without caption variants
        dropout_needs_variants = not (anima_on and not anima38_on)
        errors = []
        for name, subset in self.subset_widget.dataset_args.items():
            if variants <= 1:
                used = [label for key, label in (("shuffle_caption", "Shuffle Caption"),
                                                 ("caption_tag_dropout_rate", "Caption Tag Dropout"))
                        if subset.get(key)]
                if dropout_needs_variants and (subset.get("caption_dropout_rate") or 0) > 0:
                    used.append("Caption Dropout Rate")
                if used:
                    errors.append(
                        f"Subset '{name}': {' / '.join(used)} with Cache TE Outputs needs Caption Variants > 1 "
                        "(Network Args); the trainer refuses it otherwise"
                    )
            if (subset.get("token_warmup_step") or 0) > 0:
                errors.append(
                    f"Subset '{name}': Token Warmup cannot be used with Cache TE Outputs (the caption changes with the "
                    "step, so it can never be cached); switch one of them off"
                )
        return errors

    def save_toml(self, file_name: Path | None = None, failure_note: str = "") -> bool:
        """Save the args as a GUI TOML. False when nothing was written (invalid args or a cancelled dialog)."""
        validation_errors = self.get_validation_errors()
        if validation_errors:
            message = "\n".join(validation_errors + ([failure_note] if failure_note else []))
            self.training_warning.emit("Invalid Args", message)
            print("Cannot save TOML while args are invalid:")
            print(message)
            return False
        args, subset_args = self.get_args()
        new_args = {arg: {"args": val} for arg, val in args["args"].items()}
        for arg, val in args["dataset"].items():
            if arg in new_args:
                new_args[arg]["dataset_args"] = val
            else:
                new_args[arg] = {"dataset_args": val}
        new_args["subsets"] = list(subset_args.values())
        new_args["train_mode"] = {"train_mode": self.train_mode.value}
        return bool(TomlFunctions.save_toml(new_args, file_name))

    def load_toml(self, file_name: Path | None = None) -> None:
        try:
            args, dataset_args, train_mode = self.process_toml(file_name)
        except TomlLoadError as e:
            self.training_warning.emit("Cannot load TOML", str(e))
            return
        if not args and not dataset_args:
            return
        if train_mode == TrainingModes.LORA:
            self.set_train_lora()
        else:
            self.set_train_ti()
        self.args_widget.load_args(args, dataset_args)
        self.subset_widget.load_dataset_args(dataset_args)
        self.sync_subset_inherited_args()
        if self.args_widget.load_warnings:
            self.training_warning.emit("TOML loaded with changes", "\n\n".join(self.args_widget.load_warnings))

    def set_train_lora(self) -> None:
        if self.train_mode != TrainingModes.LORA:
            self.train_mode = TrainingModes.LORA
            self.args_widget.set_lora_training()

    def set_train_ti(self) -> None:
        if self.train_mode != TrainingModes.TI:
            self.train_mode = TrainingModes.TI
            self.args_widget.set_ti_training()

    def process_toml(self, file_name: Path | None = None) -> tuple[dict, dict]:
        """(args, dataset_args, train_mode) of a GUI TOML; raises TomlLoadError for a file that is missing, broken or
        not a GUI TOML (a flat trainer config)."""
        if file_name and not Path(file_name).exists():
            raise TomlLoadError(f"{file_name} does not exist.")
        try:
            loaded_args = TomlFunctions.load_toml(file_name)
        except (toml.TomlDecodeError, UnicodeDecodeError) as e:
            raise TomlLoadError(f"{file_name or 'The file'} is not a readable TOML file:\n{e}") from None
        if not loaded_args:
            return {}, {}, self.train_mode
        problem = TomlFunctions.gui_toml_problem(loaded_args)
        if problem:
            raise TomlLoadError(problem)
        args = {}
        dataset_args = {}
        if "subsets" in loaded_args:
            dataset_args["subsets"] = loaded_args["subsets"]
            del loaded_args["subsets"]

        train_mode = TrainingModes.LORA
        if "train_mode" in loaded_args:
            train_mode = TrainingModes(loaded_args["train_mode"]["train_mode"])

        for arg, val in loaded_args.items():
            if "args" in val:
                args[arg] = val["args"]
            if "dataset_args" in val:
                dataset_args[arg] = val["dataset_args"]
        return args, dataset_args, train_mode

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            with contextlib.suppress(Exception):
                url = self.backend_url_input.text()
                requests.get(f"{url}/stop_training", headers=backend_headers(url))
            self.begin_training_button.setText("Start Training")
            return
        validation_errors = self.get_validation_errors()
        if validation_errors:
            error_text = "\n".join(validation_errors)
            QtWidgets.QMessageBox.warning(self, "Invalid Args", error_text)
            print("Training blocked until the args are fixed:")
            print(error_text)
            return
        self.training_thread = Thread(target=self.start_training_thread)
        self.training_thread.start()

    def start_training_thread(self) -> None:
        self.begin_training_button.setText("Stop Training")
        url = self.backend_url_input.text()
        if self.queue_widget.elements:
            while self.queue_widget.elements:
                item = self.queue_widget.elements[0]
                queue_file = item.queue_file
                if item.isChecked():
                    self.save_toml(queue_file)
                # the item leaves the queue only once the backend has accepted the start; a refused start, or one
                # whose outcome is unknown (the backend stopped answering), keeps it (and its TOML) for a retry.
                # One the user removes before /train is not started (the queue moves on).
                if not self.train_helper(
                    url,
                    queue_file,
                    on_started=lambda item=item: self.queue_widget.remove_first_from_queue(item),
                    still_queued=lambda item=item: item in self.queue_widget.elements,
                ):
                    self.begin_training_button.setText("Start Training")
                    return
        else:
            self.save_toml(Path("queue_store/temp.toml"))
            self.train_helper(url, Path("queue_store/temp.toml"))
        self.begin_training_button.setText("Start Training")

    def train_helper(self, url: str, train_toml: Path, on_started=None, still_queued=None) -> bool:
        """Validate and start one job, then wait for it. False when it did not start, may not have started, or failed
        (the queue stops). Only once the backend accepted /train (or a failed /train is followed by /is_training saying
        it trains) is the TOML deleted, on_started called and the tag file / auto-save TOML written, so a refused start
        leaves nothing behind. When /train fails and the backend cannot be asked afterwards, the start is unknown: the
        item is kept as for a refused start, with a warning instead of an error. When still_queued() is false after
        /validate (the user removed the item meanwhile) nothing is started and True lets the queue move on."""
        try:
            args, dataset_args, train_mode = self.process_toml(train_toml)
            # read once: the auto-save copy must not depend on the file, which a queue removal deletes
            toml_bytes = Path(train_toml).read_bytes()
        except TomlLoadError as e:
            self.training_error.emit("Cannot train this item", f"{e}\n\nIts args were probably invalid when saved.")
            return False
        except OSError as e:
            self.training_error.emit("Cannot train this item", f"{train_toml} could not be read:\n{e}")
            return False
        config = json.loads(Path("config.json").read_text())
        headers = backend_headers(url, config)  # a token only for a backend on another machine

        # Include accelerate settings in validation request for proper warmup step calculation
        final_args = {
            "args": args,
            "dataset": dataset_args,
            "accelerate": config.get("accelerate", {}),
        }
        try:
            response = requests.post(f"{url}/validate", json=True, data=json.dumps(final_args), headers=headers)
        except ConnectionError as e:
            print(e)
            self.training_error.emit(
                "Connection Error",
                f"Failed to connect to the backend at {url}.\n\n"
                "Please ensure the backend is running and the URL is correct."
            )
            return False
        if response.status_code != 200:
            print(f"Item Failed: {response.text}")
            self.training_error.emit("Backend validation failed", backend_error_text(response))
            return False
        validation_data = response.json()
        if still_queued is not None and not still_queued():
            print(f"{train_toml} was removed from the queue while it was being validated; it was not started.")
            return True
        is_sdxl = str(args.get("general_args").get("sdxl", False))
        is_flux = str(bool(args.get("flux_args")))
        is_anima = str(bool(args.get("anima_args")))
        is_moss = str(bool(args.get("moss_args")))

        # Build train params including accelerate settings
        train_params = {
            "train_mode": train_mode.value,
            "sdxl": is_sdxl,
            "flux": is_flux,
            "anima": is_anima,
            "moss": is_moss,
        }

        # Add accelerate settings if enabled
        accel = config.get("accelerate", {})
        if accel.get("enabled", False):
            train_params["accelerate_enabled"] = "True"
            train_params["accelerate_num_processes"] = str(accel.get("num_processes", 2))
            train_params["accelerate_main_process_port"] = str(accel.get("main_process_port", 29500))
            gpu_ids = str(accel.get("gpu_ids") or "").strip()
            if gpu_ids:  # only when set: an empty field leaves the GPU choice to the backend, as before
                ids = gpu_id_list(gpu_ids)
                train_params["accelerate_gpu_ids"] = ",".join(ids) if ids is not None else gpu_ids

        try:
            response = requests.get(f"{url}/train", params=train_params, headers=headers)
        except requests.exceptions.RequestException as e:
            print(e)
            response = None
        if response is None or response.status_code >= 500:
            # No answer, or a server error page: ask the backend whether it trains. Idle is a refused start, training
            # is a start. No usable answer proves neither. A current backend answers /train first and only then
            # closes its tunnel (kill_tunnel_on_train_start) or stops its server (Colab), but an older backend with
            # kill_tunnel_on_train_start, or an older Colab backend, did that as training started and dropped the
            # answer, so it may be training anyway; a backend that crashed while starting looks the same, so the item
            # is kept and the queue stops. While a run is going, a backend that can be reached refuses /validate and
            # /train with 409 "Training Already Running" (the GUI asks /validate first), so a retry then cannot start
            # it twice; once that run has finished, a retry trains the item again.
            state = self.backend_training_state(url, headers)
            if state is False:  # the backend is up and idle: the start failed
                if response is None:
                    title, text = "Connection Error", f"Failed to reach the backend at {url} to start training."
                else:
                    print(f"Training did not start: {response.text}")
                    title, text = "Training did not start", backend_error_text(response)
                self.training_error.emit(title, f"{text}\n\nNothing was removed from the queue.")
                return False
            if state is None:  # the backend went away as it was asked to start: the start is unknown
                print(f"The backend stopped answering after /train; {train_toml} was kept.")
                if on_started is not None:  # a queue item
                    what, advice = "this item", (
                        "The item was kept in the queue and the queue stopped. Check the backend: if it is training "
                        "or has already trained this item, remove the item from the queue; otherwise press Start "
                        "Training to retry."
                    )
                else:  # Start Training with an empty queue trains the current args
                    what, advice = "these args", (
                        "Nothing was queued, so nothing was removed. Check the backend: if it is training or has "
                        "already trained these args, do not start them again; otherwise press Start Training to retry."
                    )
                self.training_warning.emit(
                    "Training may have started",
                    f"The backend at {url} stopped answering right after it was asked to start training.\n\n"
                    "An older backend with kill_tunnel_on_train_start closes its tunnel as training starts, and an "
                    "older Colab backend stops its server (a current backend answers first), so it may be training "
                    f"{what} anyway; a backend that crashed while starting looks the same.\n\n{advice} While a run "
                    "is going, a backend that can be reached refuses a new start (409 \"Training Already Running\"); "
                    f"once that run has finished, a retry trains {what} again.",
                )
                return False
        elif not training_started(response):
            print(f"Training did not start: {response.text}")
            self.training_error.emit(
                "Training did not start",
                f"{backend_error_text(response)}\n\nNothing was removed from the queue.",
            )
            return False
        self.start_accepted(train_toml, toml_bytes, args, validation_data, on_started)
        training = True
        while training:
            sleep(5.0)
            try:
                response = requests.get(f"{url}/is_training", headers=headers)
            except Exception:
                print("Connection Failed, assuming training has stopped.")
                return False
            if response.status_code != 200:
                print("Connection Failed, assuming training has stopped.")
                return False
            response = response.json()
            if not response["training"]:
                training = False
            if response["errored"]:
                return False
        return True

    @staticmethod
    def backend_training_state(url: str, headers: dict) -> bool | None:
        """/is_training's "training", or None when the backend cannot be reached or does not answer like the backend
        (a tunnel's error page)."""
        try:
            response = requests.get(f"{url}/is_training", headers=headers, timeout=30)
            body = response.json() if response.status_code == 200 else None
        except (requests.exceptions.RequestException, ValueError):
            return None
        if isinstance(body, dict) and isinstance(body.get("training"), bool):
            return body["training"]
        return None

    def start_accepted(self, train_toml: Path, toml_bytes: bytes, args: dict, validation_data: dict, on_started) -> None:
        """The backend took the job: the queue item and its TOML go, and the tag file and auto-save TOML are written.
        Training is running by now, so a failed write is a warning and the training thread goes on."""
        Path(train_toml).unlink(missing_ok=True)  # the user may have removed the queue item meanwhile
        if on_started is not None:
            on_started()
        saving_args = args.get("saving_args", {})
        try:
            if saving_args.get("tag_occurrence", None):
                folder = saving_args.get("tag_file_location", None)
                self.create_tag_file(
                    validation_data.get("tags", {}),
                    Path(folder) if folder else None,
                    saving_args.get("output_name", "output_tags"),
                )
            if saving_args.get("save_toml", None):
                folder = saving_args.get("save_toml_location", None)
                self.create_auto_save_toml(
                    toml_bytes,
                    Path(folder) if folder else None,
                    saving_args.get("output_name", "output_args"),
                )
        except OSError as e:
            print(e)
            self.training_warning.emit(
                "Training started, but a file was not saved",
                f"The tag file or the auto-save TOML could not be written:\n{e}",
            )

    def create_tag_file(
        self,
        tags: dict,
        output_location: Path | None = None,
        output_name: str = "output_tags",
    ) -> None:
        if not tags:
            return
        if not output_location:
            output_location = Path("auto_save_store")
        if not output_location.exists():
            output_location.mkdir()
        if output_location.is_file():
            output_location = output_location.parent
        output_location = output_location.joinpath(f"{output_name}.txt")
        with output_location.open("w", encoding="utf-8") as f:
            f.write("Below is a list of keywords used during the training of this model:\n")
            for k, v in tags.items():
                f.write(f"[{v}] {k}\n")

    def create_auto_save_toml(
        self,
        toml_bytes: bytes,
        output_location: Path | None = None,
        output_name: str = "output_toml",
    ):
        if not output_location:
            output_location = Path("auto_save_store")
        if not output_location.exists():
            output_location.mkdir()
        if output_location.is_file():
            output_location = output_location.parent
        output_location = output_location.joinpath(f"{output_name}.toml")
        offset = 1
        orig_name = output_location.stem
        while output_location.exists():
            output_location = output_location.with_stem(f"{orig_name}_{offset}")
            offset += 1
        output_location.write_bytes(toml_bytes)
