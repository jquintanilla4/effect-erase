from __future__ import annotations

import collections
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.bootstrap import load_bootstrap_status
from app.core.config import Settings
from app.models.video import (
    VideoMetadata,
    load_video_metadata,
)
from app.services.void_mask_reasoner import VoidQuadmaskBuilder


PROGRESS_PREFIX = "PROGRESS_JSON:"
SUBPROCESS_LOG_LIMIT = 4000


ProgressCallback = Callable[[float], None]
StatusCallback = Callable[[str | None, str | None], None]


def void_assets_available(settings: Settings) -> bool:
    required = settings.void_required_paths()
    return all(path.exists() for path in required.values())


def _append_log_chunk(buffer: collections.deque[str], chunk: str) -> None:
    if not chunk:
        return
    buffer.append(chunk)
    total_chars = sum(len(part) for part in buffer)
    while total_chars > SUBPROCESS_LOG_LIMIT and buffer:
        total_chars -= len(buffer.popleft())


def _parse_progress_event(line: str) -> dict[str, object] | None:
    stripped = line.strip()
    if not stripped.startswith(PROGRESS_PREFIX):
        return None

    payload = stripped[len(PROGRESS_PREFIX) :]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    progress = data.get("progress")
    if not isinstance(progress, (int, float)):
        return None
    data["progress"] = max(0.0, min(float(progress), 1.0))
    return data


def _materialize_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    try:
        target_path.hardlink_to(source_path)
        return
    except OSError:
        shutil.copy2(source_path, target_path)


def _command_prefix_for_env(manager: str, env_name: str | None) -> list[str]:
    if not env_name or manager == "unknown":
        return [sys.executable]
    if manager == "conda":
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python"]
    if manager == "micromamba":
        return ["micromamba", "run", "-n", env_name, "python"]
    raise RuntimeError(f"Unsupported environment manager: {manager}")


class VoidAssetManager:
    def __init__(self, settings: Settings, command_prefix_factory: Callable[[], list[str]]) -> None:
        self.settings = settings
        self.command_prefix_factory = command_prefix_factory
        self._condition = threading.Condition()
        self._download_active = False
        self._download_progress = 0.0
        self._download_stage = "idle"
        self._download_message = ""
        self._download_error: str | None = None

    def is_download_active(self) -> bool:
        with self._condition:
            return self._download_active

    def ensure_assets(
        self,
        progress_callback: ProgressCallback | None = None,
        status_callback: StatusCallback | None = None,
    ) -> None:
        if void_assets_available(self.settings):
            if progress_callback is not None:
                progress_callback(1.0)
            if status_callback is not None:
                status_callback("download_assets", "VOID model assets are already available.")
            return

        with self._condition:
            if self._download_active:
                # Reuse the active download instead of starting a second fetch.
                last_snapshot: tuple[float, str, str] | None = None
                while self._download_active:
                    snapshot = (self._download_progress, self._download_stage, self._download_message)
                    if snapshot != last_snapshot:
                        if progress_callback is not None:
                            progress_callback(snapshot[0])
                        if status_callback is not None:
                            status_callback(snapshot[1], snapshot[2])
                        last_snapshot = snapshot
                    self._condition.wait(timeout=0.25)

                if self._download_error:
                    raise RuntimeError(self._download_error)
                if not void_assets_available(self.settings):
                    raise RuntimeError("VOID asset download finished without producing the required files.")
                if progress_callback is not None:
                    progress_callback(1.0)
                if status_callback is not None:
                    status_callback("download_assets", "VOID model assets are ready.")
                return

            self._download_active = True
            self._download_progress = 0.0
            self._download_stage = "download_assets"
            self._download_message = "Preparing VOID asset download."
            self._download_error = None

        try:
            self._run_download_process(progress_callback, status_callback)
            if not void_assets_available(self.settings):
                raise RuntimeError("VOID asset download completed but required files are still missing.")
        except Exception as error:
            with self._condition:
                self._download_error = str(error)
            raise
        finally:
            with self._condition:
                self._download_active = False
                self._condition.notify_all()

    def _handle_progress_event(
        self,
        event: dict[str, object],
        progress_callback: ProgressCallback | None,
        status_callback: StatusCallback | None,
    ) -> None:
        progress = float(event["progress"])
        stage = str(event.get("stage") or "download_assets")
        message = str(event.get("message") or "")
        with self._condition:
            # Keep the shared download state monotonic so waiters never render
            # the progress bar moving backward when they attach mid-download.
            self._download_progress = max(self._download_progress, progress)
            self._download_stage = stage
            self._download_message = message
            self._condition.notify_all()

        if progress_callback is not None:
            progress_callback(self._download_progress)
        if status_callback is not None:
            status_callback(stage, message)

    def _stream_stdout(self, stream, progress_callback: ProgressCallback | None, status_callback: StatusCallback | None) -> str:
        stdout_log: collections.deque[str] = collections.deque()
        for line in iter(stream.readline, ""):
            _append_log_chunk(stdout_log, line)
            event = _parse_progress_event(line)
            if event is not None:
                self._handle_progress_event(event, progress_callback, status_callback)
                continue
            print(line, end="", flush=True)
        return "".join(stdout_log)

    def _stream_stderr(self, stream) -> str:
        stderr_log: collections.deque[str] = collections.deque()
        while True:
            chunk = stream.read(1)
            if chunk == "":
                break
            _append_log_chunk(stderr_log, chunk)
            print(chunk, end="", file=sys.stderr, flush=True)
        return "".join(stderr_log)

    def _run_download_process(
        self,
        progress_callback: ProgressCallback | None,
        status_callback: StatusCallback | None,
    ) -> None:
        command = [
            *self.command_prefix_factory(),
            "-m",
            "app.runners.void_download_assets",
            "--base-model-id",
            self.settings.void_base_model_id,
            "--base-model-dir",
            self.settings.void_base_model_dir.as_posix(),
            "--pass1-repo-id",
            self.settings.void_pass1_repo_id,
            "--pass1-filename",
            self.settings.void_pass1_filename,
            "--pass1-path",
            self.settings.void_pass1_path.as_posix(),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            cwd=self.settings.root_dir.as_posix(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("VOID asset download failed to attach subprocess pipes.")

        errors: queue.Queue[BaseException] = queue.Queue()
        stdout_output = ""
        stderr_output = ""

        def run_reader(target, *args) -> None:
            nonlocal stdout_output, stderr_output
            try:
                result = target(*args)
                if target is self._stream_stdout:
                    stdout_output = result
                else:
                    stderr_output = result
            except BaseException as error:  # pragma: no cover - defensive streaming path
                errors.put(error)

        stdout_thread = threading.Thread(
            target=run_reader,
            args=(self._stream_stdout, process.stdout, progress_callback, status_callback),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=run_reader,
            args=(self._stream_stderr, process.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        process.stdout.close()
        process.stderr.close()

        if not errors.empty():
            raise RuntimeError("VOID asset download output streaming failed.") from errors.get()

        if return_code != 0:
            error_output = "\n".join(
                part.strip()
                for part in (stdout_output, stderr_output)
                if part and part.strip()
            )
            raise RuntimeError(
                f"VOID asset download failed with exit code {return_code}."
                + (f"\n{error_output}" if error_output else "")
            )


class RealVoidRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.asset_manager = VoidAssetManager(settings, self._command_prefix)
        self.quadmask_builder = VoidQuadmaskBuilder(settings)

    def _bootstrap_status(self):
        return load_bootstrap_status(self.settings.bootstrap_state_path)

    def _env_name(self) -> str | None:
        bootstrap_status = self._bootstrap_status()
        if bootstrap_status.activeStrategy == "split" and bootstrap_status.voidEnvName:
            return bootstrap_status.voidEnvName
        if bootstrap_status.workerEnvName:
            return bootstrap_status.workerEnvName
        return None

    def _command_prefix(self) -> list[str]:
        bootstrap_status = self._bootstrap_status()
        return _command_prefix_for_env(bootstrap_status.envManager, self._env_name())

    def assets_ready(self) -> bool:
        return void_assets_available(self.settings)

    def download_in_progress(self) -> bool:
        return self.asset_manager.is_download_active()

    def ensure_models(
        self,
        progress_callback: ProgressCallback | None = None,
        status_callback: StatusCallback | None = None,
    ) -> None:
        self.asset_manager.ensure_assets(progress_callback=progress_callback, status_callback=status_callback)

    def remove(
        self,
        source_video_path: Path,
        mask_video_path: Path,
        output_video_path: Path,
        progress_callback: ProgressCallback,
        status_callback: StatusCallback | None = None,
        *,
        background_prompt: str | None = None,
        job_id: str | None = None,
    ) -> VideoMetadata:
        source_metadata = load_video_metadata(source_video_path)
        if source_metadata.frame_count > self.settings.void_max_frames:
            raise RuntimeError(
                "VOID is currently wired for clips up to "
                f"{self.settings.void_max_frames} frames, but received {source_metadata.frame_count}."
            )

        mask_metadata = load_video_metadata(mask_video_path)
        if mask_metadata.frame_count != source_metadata.frame_count:
            raise RuntimeError(
                "Mask propagation output does not match the uploaded clip length. "
                f"Source frames={source_metadata.frame_count}, mask frames={mask_metadata.frame_count}."
            )

        progress_callback(0.0)
        if status_callback is not None:
            status_callback("preflight", "Validating VOID inputs.")

        # Use a per-job working directory so retries or parallel jobs do not
        # trample prompt files or upstream output folders for the same project.
        project_dir = output_video_path.parent
        sequence_name = "sequence"
        work_dir = project_dir / "void_jobs" / (job_id or "manual")
        data_rootdir = work_dir / "data"
        sequence_dir = data_rootdir / sequence_name
        output_dir = work_dir / "outputs"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_callback(0.05)
        self.quadmask_builder.build(
            source_video_path=source_video_path,
            mask_video_path=mask_video_path,
            sequence_dir=sequence_dir,
            background_prompt_override=background_prompt,
            status_callback=status_callback,
        )
        progress_callback(0.55)

        self.ensure_models(
            progress_callback=lambda value: progress_callback(round(0.55 + (value * 0.25), 4)),
            status_callback=status_callback,
        )

        if status_callback is not None:
            status_callback("inference", "Running VOID Pass 1 inference.")
        progress_callback(0.82)
        command = [
            *self._command_prefix(),
            "-m",
            "app.runners.void_remove",
            "--repo-dir",
            self.settings.void_repo_dir.as_posix(),
            "--data-rootdir",
            data_rootdir.as_posix(),
            "--sequence-name",
            sequence_name,
            "--output-dir",
            output_dir.as_posix(),
            "--output-path",
            output_video_path.as_posix(),
            "--base-model-dir",
            self.settings.void_base_model_dir.as_posix(),
            "--transformer-path",
            self.settings.void_pass1_path.as_posix(),
            "--sample-size",
            f"{self.settings.void_sample_height}x{self.settings.void_sample_width}",
            "--max-video-length",
            str(source_metadata.frame_count),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        completed = subprocess.run(
            command,
            cwd=self.settings.root_dir.as_posix(),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            error_output = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part and part.strip()
            )
            raise RuntimeError(
                f"VOID inference failed with exit code {completed.returncode}."
                + (f"\n{error_output}" if error_output else "")
            )
        progress_callback(0.98)
        progress_callback(1.0)
        if status_callback is not None:
            status_callback("finalize", "VOID output is ready.")
        return load_video_metadata(output_video_path)


class MockVoidRuntime:
    def assets_ready(self) -> bool:
        return True

    def download_in_progress(self) -> bool:
        return False

    def ensure_models(
        self,
        progress_callback: ProgressCallback | None = None,
        status_callback: StatusCallback | None = None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(1.0)
        if status_callback is not None:
            status_callback("download_assets", "Mock VOID assets are ready.")

    def remove(
        self,
        source_video_path: Path,
        mask_video_path: Path,
        output_video_path: Path,
        progress_callback: ProgressCallback,
        status_callback: StatusCallback | None = None,
        *,
        background_prompt: str | None = None,
        job_id: str | None = None,
    ) -> VideoMetadata:
        # Reuse the simple inpaint-style mock behavior so VOID can be exercised
        # in development without the large upstream model stack.
        source_capture = cv2.VideoCapture(str(source_video_path))
        mask_capture = cv2.VideoCapture(str(mask_video_path))
        if not source_capture.isOpened() or not mask_capture.isOpened():
            raise RuntimeError("Unable to open source or mask video.")

        width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source_capture.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[np.ndarray] = []
        index = 0
        while True:
            ok_frame, frame = source_capture.read()
            ok_mask, mask_frame = mask_capture.read()
            if not ok_frame or not ok_mask:
                break

            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
            inpainted = cv2.inpaint(frame, binary_mask, 3, cv2.INPAINT_TELEA)
            frames.append(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            index += 1
            progress_callback(min(index / max(frame_count, 1), 0.99))

        source_capture.release()
        mask_capture.release()

        from app.models.video import write_video

        meta = write_video(output_video_path, frames, fps, width, height)
        progress_callback(1.0)
        if status_callback is not None:
            status_callback("finalize", "Mock VOID output is ready.")
        return meta
