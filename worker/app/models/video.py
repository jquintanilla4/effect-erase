from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

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


def _frame_to_rgb_uint8(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    # Keep every writer path on a browser-safe RGB uint8 frame shape before
    # handing the data to ffmpeg. This avoids subtle codec/container issues when
    # the source is grayscale or a resized float array.
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.ndim == 3 and resized.shape[2] == 3:
        pass
    else:
        raise RuntimeError(f"Expected a grayscale or RGB frame, got shape {resized.shape}.")

    if resized.dtype != np.uint8:
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    return resized


def _write_browser_safe_mp4(output_path: Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> None:
    # Chromium will often refuse to render OpenCV's default mp4v output even
    # though the file exists on disk. Encode previews as H.264 + yuv420p so the
    # inline <video> tag and the "open in new tab" link both work reliably.
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libopenh264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path.as_posix(),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg did not expose a writable stdin pipe.")

        for frame in frames:
            process.stdin.write(_frame_to_rgb_uint8(frame, width=width, height=height).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except Exception as error:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"Failed to encode browser-safe MP4: {error}\n{stderr}".strip()) from error
    finally:
        if process.stderr:
            process.stderr.close()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}: {stderr}".strip())


def write_mask_video(output_path: Path, masks: list[np.ndarray], fps: float, width: int, height: int) -> VideoMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_browser_safe_mp4(output_path, masks, fps, width, height)
    return VideoMetadata(output_path, width, height, fps, len(masks))


def write_video(output_path: Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> VideoMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_browser_safe_mp4(output_path, frames, fps, width, height)
    return VideoMetadata(output_path, width, height, fps, len(frames))
