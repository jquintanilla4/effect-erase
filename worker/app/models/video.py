from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


def load_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return VideoMetadata(video_path, width, height, fps, frame_count)


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iterate_video_frames(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def write_mask_video(output_path: Path, masks: list[np.ndarray], fps: float, width: int, height: int) -> VideoMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
    for mask in masks:
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    return VideoMetadata(output_path, width, height, fps, len(masks))


def write_video(output_path: Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> VideoMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
    for frame in frames:
        resized = cv2.resize(frame, (width, height))
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    writer.release()
    return VideoMetadata(output_path, width, height, fps, len(frames))

