from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.models.video import VideoMetadata, iterate_video_frames, load_video_metadata, read_frame, write_mask_video, write_video
from app.schemas.api import PromptPoint


@dataclass
class SessionRuntimeState:
    project_id: str
    source_video_path: Path
    frame_count: int
    width: int
    height: int
    fps: float
    selected_frame: int = 0
    prompts: list[PromptPoint] = field(default_factory=list)
    last_mask: np.ndarray | None = None


class MockSamRuntime:
    def start(self, project_id: str, source_video_path: Path) -> SessionRuntimeState:
        meta = load_video_metadata(source_video_path)
        return SessionRuntimeState(
            project_id=project_id,
            source_video_path=source_video_path,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
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
        state.prompts.extend(points)
        state.last_mask = mask

        Image.fromarray(frame).save(output_frame_path)
        Image.fromarray(mask).save(output_mask_path)
        return mask

    def propagate(self, state: SessionRuntimeState, output_mask_video_path: Path) -> VideoMetadata:
        if state.last_mask is None:
            raise ValueError("No mask exists for propagation.")

        masks = [state.last_mask for _ in range(state.frame_count)]
        return write_mask_video(output_mask_video_path, masks, state.fps, state.width, state.height)


class MockEffectEraseRuntime:
    def remove(
        self,
        source_video_path: Path,
        mask_video_path: Path,
        output_video_path: Path,
        progress_callback,
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
        return meta

