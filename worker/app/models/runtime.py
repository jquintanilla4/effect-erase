from __future__ import annotations

import collections
import contextlib
import gc
import importlib.util
import io
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.bootstrap import load_bootstrap_status
from app.core.config import Settings
from app.models.video import VideoMetadata, load_video_metadata, read_frame, write_mask_video, write_video
from app.services.void_pipeline import MockVoidRuntime, RealVoidRuntime, void_assets_available
from app.schemas.api import BootstrapStatus, PromptPoint


REMOVE_PROGRESS_PREFIX = "PROGRESS_JSON:"
SUBPROCESS_LOG_LIMIT = 4000


@dataclass
class SessionRuntimeState:
    project_id: str
    source_video_path: Path
    frame_count: int
    width: int
    height: int
    fps: float
    model_name: str = "sam3.1"
    selected_frame: int = 0
    prompts: list[PromptPoint] = field(default_factory=list)
    last_mask: np.ndarray | None = None
    backend_state: Any | None = None


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _append_log_chunk(buffer: collections.deque[str], chunk: str, *, limit: int = SUBPROCESS_LOG_LIMIT) -> None:
    if not chunk:
        return
    buffer.append(chunk)
    total_chars = sum(len(part) for part in buffer)
    while total_chars > limit and buffer:
        total_chars -= len(buffer.popleft())


def _parse_progress_event(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(REMOVE_PROGRESS_PREFIX):
        return None

    payload = stripped[len(REMOVE_PROGRESS_PREFIX) :]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    value = data.get("progress")
    if not isinstance(value, (int, float)):
        return None
    data["progress"] = max(0.0, min(float(value), 1.0))
    return data


def _mask_to_uint8(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"Expected a single 2D mask, got shape {array.shape}.")
    if array.shape != (height, width):
        array = cv2.resize(array.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (array > 0).astype(np.uint8) * 255
    return binary


def _pick_mask(outputs: dict[str, Any] | None, *, height: int, width: int, obj_id: int = 1) -> np.ndarray:
    if not outputs:
        return _empty_mask(height, width)

    mask_batch = outputs.get("out_binary_masks")
    if mask_batch is None:
        return _empty_mask(height, width)

    masks = np.asarray(mask_batch)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] == 0:
        return _empty_mask(height, width)

    obj_ids = outputs.get("out_obj_ids")
    if obj_ids is not None:
        ids = np.asarray(obj_ids).tolist()
        if obj_id in ids:
            return _mask_to_uint8(masks[ids.index(obj_id)], height=height, width=width)

    return _mask_to_uint8(masks[0], height=height, width=width)


def _pick_sam2_mask(mask_logits: Any, object_ids: Any, *, height: int, width: int, obj_id: int = 1) -> np.ndarray:
    if mask_logits is None:
        return _empty_mask(height, width)

    if hasattr(mask_logits, "detach"):
        masks = mask_logits.detach().float().cpu().numpy()
    else:
        masks = np.asarray(mask_logits)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[0] == 0:
        return _empty_mask(height, width)

    ids = object_ids.tolist() if hasattr(object_ids, "tolist") else list(object_ids)
    if obj_id in ids:
        return _mask_to_uint8(masks[ids.index(obj_id)], height=height, width=width)
    return _mask_to_uint8(masks[0], height=height, width=width)


def _save_preview_assets(frame: np.ndarray, mask: np.ndarray, output_frame_path: Path, output_mask_path: Path) -> None:
    output_frame_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_frame_path)
    Image.fromarray(mask).save(output_mask_path)


def _patch_sam31_partial_propagation_output() -> None:
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity

    if getattr(
        Sam3MultiplexTrackingWithInteractivity,
        "_effecterase_partial_output_patch_applied",
        False,
    ):
        return

    original_build_sam2_output = Sam3MultiplexTrackingWithInteractivity._build_sam2_output

    def patched_build_sam2_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        # SAM 3.1 partial propagation computes refined tracker masks for future
        # frames, but the upstream helper only returned them when that frame
        # already had cached VG output. Fresh click-based sessions therefore
        # dropped every propagated tracker mask after the annotated frame.
        obj_id_to_mask = {}
        cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
        if frame_idx in cached_frame_outputs:
            obj_id_to_mask.update(cached_frame_outputs[frame_idx])

        if refined_obj_id_to_mask is not None:
            obj_id_to_mask.update(refined_obj_id_to_mask)

        return obj_id_to_mask

    patched_build_sam2_output._effecterase_original = original_build_sam2_output
    Sam3MultiplexTrackingWithInteractivity._build_sam2_output = patched_build_sam2_output
    Sam3MultiplexTrackingWithInteractivity._effecterase_partial_output_patch_applied = True


def _runtime_mode(settings: Settings) -> str:
    if settings.use_mock_runtime:
        return "mock"
    return settings.runtime_mode


def _clear_cuda_runtime_memory() -> None:
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()


def sam_assets_available(settings: Settings) -> bool:
    return len(available_sam_models(settings)) > 0


def resolve_sam2_config_path(settings: Settings) -> Path | None:
    configured = settings.sam2_config_path
    if configured.exists():
        return configured

    try:
        import sam2
    except ImportError:
        return None

    # Split env bootstraps often rely on the installed package config instead
    # of a checked-out third_party tree, so keep the fallback resolution in one place.
    package_config = Path(sam2.__file__).resolve().parent / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
    if package_config.exists():
        return package_config
    return None


def resolve_sam2_config_name(settings: Settings) -> str | None:
    try:
        import sam2
    except ImportError:
        return None

    package_root = Path(sam2.__file__).resolve().parent
    package_config = package_root / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
    if not package_config.exists():
        return None

    # The packaged SAM2 builders call Hydra with compose(config_name=...), so
    # they need a package-relative config name instead of an absolute path.
    return package_config.relative_to(package_root).as_posix()


def available_sam_models(settings: Settings) -> list[str]:
    models: list[str] = []
    if settings.sam_allow_hf_download or settings.sam_checkpoint_path.exists():
        models.append("sam3.1")
    if settings.sam_allow_hf_download or settings.sam_legacy_checkpoint_path.exists():
        models.append("sam3")
    if settings.sam2_allow_hf_download or (
        settings.sam2_checkpoint_path.exists() and resolve_sam2_config_path(settings) is not None
    ):
        models.append("sam2.1")
    return models


def available_local_sam_models(settings: Settings) -> list[str]:
    models: list[str] = []
    sam31_config_path = settings.models_dir / "sam3.1" / "config.json"
    sam3_config_path = settings.models_dir / "sam3" / "config.json"

    # Local verification expects a complete on-disk asset set, not a runtime
    # fallback that still depends on a later Hugging Face download.
    if sam31_config_path.exists() and settings.sam_checkpoint_path.exists():
        models.append("sam3.1")
    if sam3_config_path.exists() and settings.sam_legacy_checkpoint_path.exists():
        models.append("sam3")
    if settings.sam2_checkpoint_path.exists() and resolve_sam2_config_path(settings) is not None:
        models.append("sam2.1")
    return models


def effecterase_assets_available(settings: Settings) -> bool:
    required = settings.effecterase_required_paths()
    return all(path.exists() for path in required.values())


def _pipeline_env_ready(bootstrap_status: BootstrapStatus | None, pipeline_id: str) -> bool:
    bootstrap = bootstrap_status or BootstrapStatus(
        status="missing",
        envManager="unknown",
        envNames=[],
        activeStrategy="unknown",
    )
    if bootstrap.activeStrategy == "shared":
        return bool(bootstrap.workerEnvName)
    if pipeline_id == "effecterase":
        return bool(bootstrap.removeEnvName)
    if pipeline_id == "void":
        return bool(bootstrap.voidEnvName)
    return False


def describe_remove_pipelines(
    settings: Settings,
    bootstrap_status: BootstrapStatus | None = None,
    *,
    download_states: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    mode = _runtime_mode(settings)
    download_states = download_states or {}
    pipelines = []
    definitions = [
        {
            "id": "effecterase",
            "label": "EffectErase",
            "assets_ready": effecterase_assets_available(settings),
            "gemini_configured": False,
            "lazy_models": False,
            "downloadable": False,
        },
        {
            "id": "void",
            "label": "VOID",
            "assets_ready": void_assets_available(settings),
            "gemini_configured": settings.gemini_configured(),
            "lazy_models": True,
            "downloadable": True,
        },
    ]

    for definition in definitions:
        pipeline_id = definition["id"]
        download_state = download_states.get(pipeline_id, {})
        env_ready = _pipeline_env_ready(bootstrap_status, pipeline_id)
        assets_ready = definition["assets_ready"] or mode == "mock"
        selectable = mode == "mock" or env_ready
        pipelines.append(
            {
                "id": pipeline_id,
                "label": definition["label"],
                "envReady": env_ready or mode == "mock",
                "assetsReady": assets_ready,
                "geminiConfigured": bool(definition["gemini_configured"]) or mode == "mock",
                "lazyModels": definition["lazy_models"],
                "downloadable": definition["downloadable"],
                "selectable": selectable,
                "downloadInProgress": bool(download_state.get("active")),
                "activeJobId": download_state.get("jobId"),
            }
        )
    return pipelines


@lru_cache(maxsize=1)
def sam3_fa3_state() -> tuple[bool, str]:
    """Resolve whether SAM 3.1 should use FlashAttention 3 on this worker."""
    try:
        import torch
    except Exception as error:
        return False, f"FA3 disabled because torch could not be imported: {_error_text(error)}"

    if not torch.cuda.is_available():
        return False, "FA3 disabled because CUDA is not available in this worker process."

    try:
        device = torch.cuda.get_device_properties(0)
    except Exception as error:
        return False, f"FA3 disabled because GPU properties could not be read: {_error_text(error)}"

    capability = f"{device.major}.{device.minor}"
    # FlashAttention 3 is the Hopper path today, so keep Ada/Ampere on the
    # PyTorch SDPA path even when SAM 3.1 is otherwise available.
    if device.major < 9:
        return (
            False,
            f"FA3 disabled on {device.name} (compute capability {capability}); Hopper-or-newer hardware is required.",
        )

    if importlib.util.find_spec("flash_attn_interface") is None:
        return (
            False,
            f"FA3 disabled on {device.name} because flash_attn_interface is not installed in the SAM environment.",
        )

    return True, f"FA3 enabled on {device.name} (compute capability {capability})."


class MockSamRuntime:
    def start(self, project_id: str, source_video_path: Path, model_name: str = "sam3.1") -> SessionRuntimeState:
        meta = load_video_metadata(source_video_path)
        return SessionRuntimeState(
            project_id=project_id,
            source_video_path=source_video_path,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            model_name=model_name,
        )

    def add_prompt(
        self,
        state: SessionRuntimeState,
        frame_index: int,
        points: list[PromptPoint],
        output_mask_path: Path,
        output_frame_path: Path,
    ) -> np.ndarray:
        frame = read_frame(state.source_video_path, frame_index)
        mask = np.zeros((state.height, state.width), dtype=np.uint8)

        for point in points:
            px = int(point.x * state.width)
            py = int(point.y * state.height)
            if point.label == "positive":
                cv2.circle(mask, (px, py), 52, 255, thickness=-1)
            else:
                cv2.circle(mask, (px, py), 40, 0, thickness=-1)

        if state.last_mask is not None:
            mask = np.maximum(mask, state.last_mask)

        state.selected_frame = frame_index
        state.prompts = list(points)
        state.last_mask = mask

        _save_preview_assets(frame, mask, output_frame_path, output_mask_path)
        return mask

    def propagate(self, state: SessionRuntimeState, output_mask_video_path: Path) -> VideoMetadata:
        if state.last_mask is None:
            raise ValueError("No mask exists for propagation.")

        masks = [state.last_mask for _ in range(state.frame_count)]
        return write_mask_video(output_mask_video_path, masks, state.fps, state.width, state.height)

    def release_resources(self, state: SessionRuntimeState | None = None) -> None:
        return None


class RealSamRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.predictors: dict[str, Any] = {}

    def _sam2_config_path(self) -> Path | None:
        return resolve_sam2_config_path(self.settings)

    def _sam2_config_name(self) -> str | None:
        return resolve_sam2_config_name(self.settings)

    def _checkpoint_path(self, model_name: str) -> str | None:
        checkpoint = self.settings.sam_checkpoint_for_model(model_name)
        if checkpoint.exists():
            return checkpoint.as_posix()
        if self.settings.sam_allow_hf_download:
            return None
        raise RuntimeError(
            f"Missing SAM checkpoint for {model_name}: {checkpoint}. "
            "Download the model weights or enable WORKER_SAM_ALLOW_HF_DOWNLOAD=true."
        )

    def _resolved_model_name(self, requested_model: str) -> str:
        models = available_sam_models(self.settings)
        if requested_model in models:
            return requested_model
        raise RuntimeError(
            f"Requested SAM model '{requested_model}' is not available. "
            f"Available models: {', '.join(models) if models else 'none'}."
        )

    def _sam3_use_fa3(self) -> tuple[bool, str]:
        return sam3_fa3_state()

    def _sam3_use_rope_real(self) -> bool:
        # The public SAM 3.1 checkpoint stores the legacy complex `freqs_cis`
        # buffers. Keep non-compiled runs on that checkpoint-native path and
        # only switch to the real-valued RoPE variant when compilation is
        # explicitly enabled for compatibility with that code path.
        return bool(self.settings.sam_compile)

    def _close_backend_session(self, state: SessionRuntimeState) -> None:
        if state.backend_state is None:
            return

        predictor = self.predictors.get(state.model_name)
        if predictor is None:
            state.backend_state = None
            return

        if state.model_name == "sam2.1":
            predictor.reset_state(state.backend_state)
        else:
            with self._sam3_request_context(state.model_name):
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": state.backend_state,
                    }
                )
        state.backend_state = None

    def _unload_predictors(self) -> None:
        for model_name, predictor in list(self.predictors.items()):
            with contextlib.suppress(Exception):
                if hasattr(predictor, "shutdown"):
                    predictor.shutdown()
            del predictor
            self.predictors.pop(model_name, None)

    def release_resources(self, state: SessionRuntimeState | None = None) -> None:
        # Split-env removal runs in a second Python process on the same GPU, so
        # we must tear down the in-process SAM runtime first or the LoRA load in
        # EffectErase can OOM against the still-live tracker session.
        if state is not None:
            with contextlib.suppress(Exception):
                self._close_backend_session(state)

        self._unload_predictors()
        gc.collect()
        _clear_cuda_runtime_memory()

    @contextlib.contextmanager
    def _sam3_request_context(self, model_name: str):
        # SAM 3.1 enters bf16 autocast during predictor construction, but
        # autocast is thread-local. Our FastAPI SAM routes are synchronous,
        # which means the actual click/propagation work can run in a different
        # worker thread from predictor initialization. Re-enter autocast around
        # each request so the prompt path sees the dtype policy SAM 3.1 expects.
        if model_name != "sam3.1":
            yield
            return

        import torch

        if not torch.cuda.is_available():
            yield
            return

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield

    def _build_sam3_predictor(self, predictor_kwargs: dict[str, Any]):
        if predictor_kwargs.get("version") == "sam3.1":
            _patch_sam31_partial_propagation_output()

        from sam3.model_builder import build_sam3_predictor

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            predictor = build_sam3_predictor(**predictor_kwargs)

        for line in capture.getvalue().splitlines():
            if line.startswith("Missing keys:") or line.startswith("Unexpected keys:"):
                continue
            if line.startswith("Missing keys (") or line.startswith("Unexpected keys ("):
                continue
            print(line)
        return predictor

    def _build_sam2_predictor(self):
        from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

        config_path = self._sam2_config_path()
        config_name = self._sam2_config_name()
        if self.settings.sam2_checkpoint_path.exists() and config_path is not None and config_name is not None:
            return build_sam2_video_predictor(
                config_name,
                self.settings.sam2_checkpoint_path.as_posix(),
            )
        if self.settings.sam2_allow_hf_download:
            return build_sam2_video_predictor_hf(self.settings.sam2_hf_model_id)
        raise RuntimeError(
            "SAM 2.1 fallback is not available. "
            f"Expected a local checkpoint at {self.settings.sam2_checkpoint_path} and an installed packaged SAM2 config, "
            f"or enable WORKER_SAM2_ALLOW_HF_DOWNLOAD to use {self.settings.sam2_hf_model_id}."
        )

    def _predictor(self, model_name: str):
        if model_name not in self.predictors:
            if model_name == "sam2.1":
                self.predictors[model_name] = self._build_sam2_predictor()
            else:
                predictor_kwargs = {
                    "checkpoint_path": self._checkpoint_path(model_name),
                    "version": model_name,
                    "compile": self.settings.sam_compile,
                    "async_loading_frames": self.settings.sam_async_loading_frames,
                    "max_num_objects": self.settings.sam_max_num_objects,
                    "multiplex_count": self.settings.sam_multiplex_count,
                }
                if model_name == "sam3.1":
                    predictor_kwargs["use_fa3"] = self._sam3_use_fa3()[0]
                    predictor_kwargs["use_rope_real"] = self._sam3_use_rope_real()

                predictor = self._build_sam3_predictor(predictor_kwargs)
                self.predictors[model_name] = predictor
        return self.predictors[model_name]

    def _start_backend_state(self, predictor: Any, model_name: str, source_video_path: Path) -> Any:
        if model_name == "sam2.1":
            return predictor.init_state(source_video_path.as_posix())
        with self._sam3_request_context(model_name):
            return predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": source_video_path.as_posix(),
                    "offload_video_to_cpu": False,
                }
            )["session_id"]

    def _predictor_start_error(self, model_name: str, error: BaseException) -> str:
        if model_name == "sam2.1":
            source = self.settings.sam2_checkpoint_path
        else:
            source = self.settings.sam_checkpoint_for_model(model_name)
        if model_name == "sam3.1":
            _use_fa3, fa3_reason = self._sam3_use_fa3()
            return f"Failed to initialize {model_name} from {source}: {error}. {fa3_reason}"
        return f"Failed to initialize {model_name} from {source}: {error}"

    def start(self, project_id: str, source_video_path: Path, model_name: str = "sam3.1") -> SessionRuntimeState:
        meta = load_video_metadata(source_video_path)
        resolved_model = self._resolved_model_name(model_name)
        try:
            predictor = self._predictor(resolved_model)
            response = self._start_backend_state(predictor, resolved_model, source_video_path)
        except Exception as error:
            raise RuntimeError(self._predictor_start_error(resolved_model, error)) from error
        return SessionRuntimeState(
            project_id=project_id,
            source_video_path=source_video_path,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            model_name=resolved_model,
            backend_state=response,
        )

    def add_prompt(
        self,
        state: SessionRuntimeState,
        frame_index: int,
        points: list[PromptPoint],
        output_mask_path: Path,
        output_frame_path: Path,
    ) -> np.ndarray:
        predictor = self._predictor(state.model_name)
        if state.backend_state is None:
            raise RuntimeError("SAM predictor state is not initialized.")

        if state.model_name == "sam2.1":
            _, object_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=state.backend_state,
                frame_idx=frame_index,
                obj_id=1,
                points=np.array([[point.x * state.width, point.y * state.height] for point in points], dtype=np.float32),
                labels=np.array([1 if point.label == "positive" else 0 for point in points], dtype=np.int32),
            )
            mask = _pick_sam2_mask(mask_logits, object_ids, height=state.height, width=state.width, obj_id=1)
        else:
            request = {
                "type": "add_prompt",
                "session_id": state.backend_state,
                "frame_index": frame_index,
                "points": [[point.x, point.y] for point in points],
                "point_labels": [1 if point.label == "positive" else 0 for point in points],
                "obj_id": 1,
                "clear_old_points": True,
            }
            with self._sam3_request_context(state.model_name):
                response = predictor.handle_request(request)
            mask = _pick_mask(response.get("outputs"), height=state.height, width=state.width, obj_id=1)
        frame = read_frame(state.source_video_path, frame_index)

        state.selected_frame = frame_index
        state.prompts = list(points)
        state.last_mask = mask

        _save_preview_assets(frame, mask, output_frame_path, output_mask_path)
        return mask

    def propagate(self, state: SessionRuntimeState, output_mask_video_path: Path) -> VideoMetadata:
        if state.backend_state is None:
            raise RuntimeError("SAM predictor state is not initialized.")
        if not state.prompts:
            raise ValueError("No prompt exists for propagation.")

        predictor = self._predictor(state.model_name)
        frame_to_mask: dict[int, np.ndarray] = {}
        if state.model_name == "sam2.1":
            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state.backend_state):
                frame_to_mask[int(frame_idx)] = _pick_sam2_mask(
                    mask_logits,
                    object_ids,
                    height=state.height,
                    width=state.width,
                    obj_id=1,
                )
        else:
            with self._sam3_request_context(state.model_name):
                responses = predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": state.backend_state,
                        "start_frame_index": state.selected_frame,
                        "propagation_direction": "both",
                    }
                )
                for response in responses:
                    frame_idx = int(response["frame_index"])
                    frame_to_mask[frame_idx] = _pick_mask(
                        response.get("outputs"),
                        height=state.height,
                        width=state.width,
                        obj_id=1,
                    )

        if state.last_mask is not None:
            frame_to_mask.setdefault(state.selected_frame, state.last_mask)
        if not frame_to_mask:
            raise RuntimeError("SAM propagation returned no masks.")

        masks = [frame_to_mask.get(index, _empty_mask(state.height, state.width)) for index in range(state.frame_count)]
        return write_mask_video(output_mask_video_path, masks, state.fps, state.width, state.height)


class MockEffectEraseRuntime:
    def remove(
        self,
        source_video_path: Path,
        mask_video_path: Path,
        output_video_path: Path,
        progress_callback,
        status_callback=None,
        *,
        background_prompt: str | None = None,
        job_id: str | None = None,
    ) -> VideoMetadata:
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

        meta = write_video(output_video_path, frames, fps, width, height)
        progress_callback(1.0)
        if status_callback is not None:
            status_callback("finalize", "Mock EffectErase output is ready.")
        return meta


class RealEffectEraseRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bootstrap_status = load_bootstrap_status(settings.bootstrap_state_path)

    def _remove_env_name(self) -> str | None:
        if self.bootstrap_status.activeStrategy == "split" and self.bootstrap_status.removeEnvName:
            return self.bootstrap_status.removeEnvName
        if self.bootstrap_status.workerEnvName:
            return self.bootstrap_status.workerEnvName
        return None

    def _command_prefix(self) -> list[str]:
        env_name = self._remove_env_name()
        if env_name is None or self.bootstrap_status.envManager == "unknown":
            return [sys.executable]
        if self.bootstrap_status.envManager == "conda":
            return ["conda", "run", "--no-capture-output", "-n", env_name, "python"]
        if self.bootstrap_status.envManager == "micromamba":
            return ["micromamba", "run", "-n", env_name, "python"]
        raise RuntimeError(f"Unsupported environment manager: {self.bootstrap_status.envManager}")

    def _require_assets(self) -> dict[str, Path]:
        required = self.settings.effecterase_required_paths()
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            details = ", ".join(f"{name}={required[name]}" for name in missing)
            raise RuntimeError(
                "Missing EffectErase model assets: "
                f"{details}. Download the Wan 2.1 and EffectErase weights before running removal."
            )
        return required

    def _stream_stdout(
        self,
        stream,
        progress_callback,
        status_callback,
        stdout_log: collections.deque[str],
    ) -> None:
        for line in iter(stream.readline, ""):
            _append_log_chunk(stdout_log, line)
            progress_event = _parse_progress_event(line)
            if progress_event is not None:
                progress_callback(progress_event["progress"])
                if status_callback is not None:
                    status_callback(progress_event.get("stage"), progress_event.get("message"))
                continue
            print(line, end="", flush=True)

    def _stream_stderr(self, stream, stderr_log: collections.deque[str]) -> None:
        while True:
            chunk = stream.read(1)
            if chunk == "":
                break
            _append_log_chunk(stderr_log, chunk)
            print(chunk, end="", file=sys.stderr, flush=True)

    def _stream_remove_process(
        self,
        command: list[str],
        env: dict[str, str],
        progress_callback,
        status_callback,
    ) -> tuple[int, str, str]:
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
            raise RuntimeError("EffectErase removal failed to attach subprocess pipes.")

        stdout_log: collections.deque[str] = collections.deque()
        stderr_log: collections.deque[str] = collections.deque()
        errors: queue.Queue[BaseException] = queue.Queue()

        def run_reader(target, *args) -> None:
            try:
                target(*args)
            except BaseException as error:  # pragma: no cover - defensive streaming path
                errors.put(error)

        stdout_thread = threading.Thread(
            target=run_reader,
            args=(self._stream_stdout, process.stdout, progress_callback, status_callback, stdout_log),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=run_reader,
            args=(self._stream_stderr, process.stderr, stderr_log),
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
            raise RuntimeError("EffectErase output streaming failed.") from errors.get()

        return return_code, "".join(stdout_log), "".join(stderr_log)

    def remove(
        self,
        source_video_path: Path,
        mask_video_path: Path,
        output_video_path: Path,
        progress_callback,
        status_callback=None,
        *,
        background_prompt: str | None = None,
        job_id: str | None = None,
    ) -> VideoMetadata:
        source_metadata = load_video_metadata(source_video_path)
        if source_metadata.frame_count > self.settings.effecterase_num_frames:
            raise RuntimeError(
                "EffectErase is currently wired for clips up to "
                f"{self.settings.effecterase_num_frames} frames, but received {source_metadata.frame_count}. "
                "Trim the clip first or add chunked removal support."
            )

        mask_metadata = load_video_metadata(mask_video_path)
        if mask_metadata.frame_count != source_metadata.frame_count:
            raise RuntimeError(
                "Mask propagation output does not match the uploaded clip length. "
                f"Source frames={source_metadata.frame_count}, mask frames={mask_metadata.frame_count}. "
                "Re-run mask propagation before removal."
            )

        if source_metadata.frame_count <= 0:
            raise RuntimeError(f"Source video {source_video_path} does not contain any readable frames.")

        # EffectErase can process shorter clips than the configured maximum, but
        # the runner must receive the real clip length so its latent window and
        # the propagated mask sequence stay in lockstep.
        clip_num_frames = min(source_metadata.frame_count, self.settings.effecterase_num_frames)
        required = self._require_assets()
        progress_callback(0.0)

        command = [
            *self._command_prefix(),
            "-m",
            "app.runners.effecterase_remove",
            "--fg_bg_path",
            source_video_path.as_posix(),
            "--mask_path",
            mask_video_path.as_posix(),
            "--output_path",
            output_video_path.as_posix(),
            "--num_frames",
            str(clip_num_frames),
            "--frame_interval",
            str(self.settings.effecterase_frame_interval),
            "--height",
            str(self.settings.default_height),
            "--width",
            str(self.settings.default_width),
            "--seed",
            str(self.settings.effecterase_seed),
            "--cfg",
            str(self.settings.effecterase_cfg),
            "--lora_alpha",
            str(self.settings.effecterase_lora_alpha),
            "--num_inference_steps",
            str(self.settings.effecterase_num_inference_steps),
            "--text_encoder_path",
            required["text_encoder"].as_posix(),
            "--vae_path",
            required["vae"].as_posix(),
            "--dit_path",
            required["dit"].as_posix(),
            "--image_encoder_path",
            required["image_encoder"].as_posix(),
            "--pretrained_lora_path",
            required["lora"].as_posix(),
        ]
        if self.settings.effecterase_tiled:
            command.append("--tiled")
        if self.settings.effecterase_use_teacache:
            command.append("--use_teacache")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return_code, stdout_output, stderr_output = self._stream_remove_process(
            command,
            env,
            progress_callback,
            status_callback,
        )
        if return_code != 0:
            error_output = "\n".join(
                part.strip()
                for part in [stdout_output, stderr_output]
                if part and part.strip()
            )
            raise RuntimeError(
                f"EffectErase inference failed with exit code {return_code}."
                + (f"\n{error_output}" if error_output else "")
            )
        if not output_video_path.exists():
            raise RuntimeError("EffectErase reported success but did not create an output video.")

        progress_callback(1.0)
        return load_video_metadata(output_video_path)


def build_sam_runtime(settings: Settings):
    mode = _runtime_mode(settings)
    if mode == "mock":
        return MockSamRuntime()
    if mode == "real":
        return RealSamRuntime(settings)
    if sam_assets_available(settings):
        return RealSamRuntime(settings)
    return MockSamRuntime()


def build_remove_runtime(settings: Settings, pipeline: str = "effecterase"):
    mode = _runtime_mode(settings)
    if pipeline == "void":
        if mode == "mock":
            return MockVoidRuntime()
        return RealVoidRuntime(settings)

    if mode == "mock":
        return MockEffectEraseRuntime()
    if mode == "real":
        return RealEffectEraseRuntime(settings)
    if effecterase_assets_available(settings):
        return RealEffectEraseRuntime(settings)
    return MockEffectEraseRuntime()


def describe_runtime_availability(
    settings: Settings,
    bootstrap_status: BootstrapStatus | None = None,
    *,
    download_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bootstrap = bootstrap_status or load_bootstrap_status(settings.bootstrap_state_path)
    mode = _runtime_mode(settings)
    sam_ready = sam_assets_available(settings)
    return {
        "runtimeMode": mode,
        "samReady": sam_ready,
        "envMode": bootstrap.activeStrategy,
        "removePipelines": describe_remove_pipelines(
            settings,
            bootstrap,
            download_states=download_states,
        ),
    }
