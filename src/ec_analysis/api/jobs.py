"""Own a single local analysis process and its bounded, recoverable artifacts."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import typing
import uuid

from pydantic import BaseModel

from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.state import AnalysisState


class AnalysisJob(BaseModel):
    identifier: str
    status: typing.Literal["running", "completed", "failed", "cancelled", "timed_out"]
    elapsed_seconds: float = 0
    message: str


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True)
class RunningJob:
    information: AnalysisJob
    process: subprocess.Popen[bytes]
    directory: pathlib.Path
    started_at: float


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True)
class JobManager:
    root_directory: pathlib.Path
    project_root: pathlib.Path
    timeout_seconds: float = 1800
    maximum_bytes: int = 10 * 1024 * 1024
    jobs: dict[str, RunningJob] = dataclasses.field(default_factory=dict)
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    def submit(self, contents: bytes, target_metric: str, thorough: bool) -> AnalysisJob:
        if len(contents) > self.maximum_bytes:
            raise ValueError("CSV exceeds 10 MiB")
        with self.lock:
            self.refresh_jobs()
            if any(job.information.status == "running" for job in self.jobs.values()):
                raise RuntimeError("Уже выполняется анализ. Дождитесь завершения или отмените его.")
            identifier = uuid.uuid4().hex
            directory = self.root_directory / identifier
            directory.mkdir(parents=True, exist_ok=False)
            input_path = directory / "input.csv"
            input_path.write_bytes(contents)
            try:
                dataset = DatasetLoader(target_metric=target_metric).load_csv(input_path)
                if len(dataset.records) > 100000 or len(dataset.feature_names) > 20:
                    raise ValueError("Maximum 100000 rows and 20 factors")
                if dataset.records["id"].nunique() < 10:
                    raise ValueError("At least ten paired identifiers are required")
                category_count = sum(dataset.records[name].nunique() + 1 for name in dataset.categorical_features)
                if category_count + len(dataset.numeric_features) > 128:
                    raise ValueError("Too many categories: maximum 128 encoded factors")
            except (ValueError, OSError, UnicodeError):
                input_path.unlink(missing_ok=True)
                directory.rmdir()
                raise
            command = [
                sys.executable,
                "-m",
                "ec_analysis.api.worker",
                str(directory),
                target_metric,
                "thorough" if thorough else "quick",
                str(self.project_root),
            ]
            with (directory / "worker.log").open("wb") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=log_stream,
                    stderr=log_stream,
                    start_new_session=True,
                )
            information = AnalysisJob(
                identifier=identifier, status="running", message="Обучение GAM и проверка устойчивости…"
            )
            self.jobs[identifier] = RunningJob(
                information=information, process=process, directory=directory, started_at=time.monotonic()
            )
            return information.model_copy()

    def refresh_jobs(self) -> None:
        with self.lock:
            for running_job in self.jobs.values():
                if running_job.information.status != "running":
                    continue
                running_job.information.elapsed_seconds = time.monotonic() - running_job.started_at
                return_code = running_job.process.poll()
                if return_code is not None:
                    output_directory = running_job.directory / "result"
                    complete = return_code == 0 and (output_directory / "manifest.json").is_file()
                    running_job.information.status = "completed" if complete else "failed"
                    running_job.information.message = "Расчёт завершён." if complete else "Расчёт завершился ошибкой."
                    failure_path = running_job.directory / "error.json"
                    if not complete and failure_path.is_file():
                        running_job.information.message = str(json.loads(failure_path.read_text())["error"])
                elif running_job.information.elapsed_seconds >= self.timeout_seconds:
                    self._stop_process(running_job, "timed_out", "Превышен предел времени 30 минут.")

    def read_job(self, identifier: str) -> AnalysisJob:
        self.refresh_jobs()
        with self.lock:
            return self.jobs[identifier].information.model_copy()

    def cancel_job(self, identifier: str) -> AnalysisJob:
        with self.lock:
            self.refresh_jobs()
            running_job = self.jobs[identifier]
            if running_job.information.status == "running":
                self._stop_process(running_job, "cancelled", "Расчёт отменён.")
            return running_job.information.model_copy()

    def result_directory(self, identifier: str) -> pathlib.Path:
        information = self.read_job(identifier)
        if information.status != "completed":
            raise RuntimeError("Результат ещё не готов.")
        return self.jobs[identifier].directory / "result"

    def read_result(self, identifier: str) -> AnalysisState:
        return AnalysisState.model_validate_json((self.result_directory(identifier) / "analysis.json").read_text())

    def export_path(self, identifier: str, filename: str) -> pathlib.Path:
        directory = self.result_directory(identifier)
        manifest = json.loads((directory / "manifest.json").read_text())
        if filename not in manifest["artifact_sha256"] or pathlib.Path(filename).suffix not in {".csv", ".json"}:
            raise KeyError(filename)
        if pathlib.Path(filename).name != filename:
            raise KeyError(filename)
        return directory / filename

    def shutdown(self) -> None:
        with self.lock:
            for running_job in self.jobs.values():
                if running_job.information.status == "running":
                    self._stop_process(running_job, "cancelled", "Сервис остановлен.")

    def _stop_process(
        self, running_job: RunningJob, status: typing.Literal["cancelled", "timed_out"], message: str
    ) -> None:
        if running_job.process.poll() is None:
            try:
                os.killpg(running_job.process.pid, signal.SIGTERM)
                running_job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(running_job.process.pid, signal.SIGKILL)
                running_job.process.wait(timeout=5)
            except ProcessLookupError:
                running_job.process.wait(timeout=5)
        running_job.information.status = status
        running_job.information.message = message
        running_job.information.elapsed_seconds = time.monotonic() - running_job.started_at
