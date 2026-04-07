from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.models.video import iterate_video_frames, load_video_metadata, read_frame, write_lossless_mask_video


StatusCallback = Callable[[str | None, str | None], None]

SAMPLE_POINTS = (0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0)

VLM_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "integral_belongings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "noun": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["noun"],
            },
        },
        "affected_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "noun": {"type": "string"},
                    "category": {"type": "string", "enum": ["physical", "visual_artifact"]},
                    "why": {"type": "string"},
                    "grid_localizations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame": {"type": "integer"},
                                "grid_regions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "row": {"type": "integer"},
                                            "col": {"type": "integer"},
                                        },
                                        "required": ["row", "col"],
                                    },
                                },
                            },
                            "required": ["frame", "grid_regions"],
                        },
                    },
                },
                "required": ["noun", "category"],
            },
        },
        "scene_description": {"type": "string"},
        "background_prompt": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["integral_belongings", "affected_objects", "background_prompt"],
}


@dataclass
class VoidReasoningResult:
    integral_belongings: list[dict[str, object]]
    affected_objects: list[dict[str, object]]
    scene_description: str
    background_prompt: str
    confidence: float

    def to_json(self) -> dict[str, object]:
        return {
            "integral_belongings": self.integral_belongings,
            "affected_objects": self.affected_objects,
            "scene_description": self.scene_description,
            "background_prompt": self.background_prompt,
            "confidence": self.confidence,
        }


def calculate_square_grid(width: int, height: int, min_grid: int) -> tuple[int, int]:
    aspect_ratio = width / max(height, 1)
    if width >= height:
        grid_rows = min_grid
        grid_cols = max(min_grid, round(min_grid * aspect_ratio))
    else:
        grid_cols = min_grid
        grid_rows = max(min_grid, round(min_grid / max(aspect_ratio, 1e-6)))
    return grid_rows, grid_cols


def sample_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]

    indices = []
    for point in SAMPLE_POINTS:
        index = int(round(point * (frame_count - 1)))
        indices.append(max(0, min(index, frame_count - 1)))
    return sorted(set(indices))


def _normalize_noun(value) -> str:
    return str(value or "").strip().lower()


def _clamp_grid_region(region: dict[str, object], grid_rows: int, grid_cols: int) -> dict[str, int] | None:
    try:
        row = int(region.get("row", 0))
        col = int(region.get("col", 0))
    except Exception:
        return None

    if row < 0 or col < 0 or row >= grid_rows or col >= grid_cols:
        return None
    return {"row": row, "col": col}


def normalize_reasoning(payload: dict[str, object]) -> VoidReasoningResult:
    integral_belongings = []
    for item in payload.get("integral_belongings", [])[:3]:
        if not isinstance(item, dict):
            continue
        noun = _normalize_noun(item.get("noun"))
        if not noun:
            continue
        integral_belongings.append(
            {
                "noun": noun,
                "why": str(item.get("why", "")).strip()[:200],
            }
        )

    affected_objects = []
    for item in payload.get("affected_objects", [])[:5]:
        if not isinstance(item, dict):
            continue
        noun = _normalize_noun(item.get("noun"))
        category = str(item.get("category", "physical")).strip().lower()
        if not noun or category not in {"physical", "visual_artifact"}:
            continue

        normalized = {
            "noun": noun,
            "category": category,
            "why": str(item.get("why", "")).strip()[:240],
        }

        grid_localizations = []
        for localization in item.get("grid_localizations", []):
            if not isinstance(localization, dict):
                continue
            try:
                frame = int(localization.get("frame", 0))
            except Exception:
                frame = 0
            regions = []
            for region in localization.get("grid_regions", []):
                if isinstance(region, dict):
                    regions.append(region)
            if regions:
                grid_localizations.append(
                    {
                        "frame": frame,
                        "grid_regions": regions,
                    }
                )

        if grid_localizations:
            normalized["grid_localizations"] = grid_localizations

        affected_objects.append(normalized)

    background_prompt = str(payload.get("background_prompt", "")).strip()
    scene_description = str(payload.get("scene_description", "")).strip()
    if not background_prompt:
        background_prompt = scene_description

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return VoidReasoningResult(
        integral_belongings=integral_belongings,
        affected_objects=affected_objects,
        scene_description=scene_description,
        background_prompt=background_prompt,
        confidence=max(0.0, min(confidence, 1.0)),
    )


def create_mask_overlay(first_frame: np.ndarray, primary_mask: np.ndarray) -> np.ndarray:
    overlay = first_frame.copy()
    overlay[primary_mask] = [255, 0, 0]
    return cv2.addWeighted(first_frame, 0.6, overlay, 0.4, 0)


def draw_grid(frame: np.ndarray, grid_rows: int, grid_cols: int, frame_index: int | None = None) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    cell_width = width / grid_cols
    cell_height = height / grid_rows

    for col in range(1, grid_cols):
        x = int(col * cell_width)
        cv2.line(output, (x, 0), (x, height), (255, 255, 0), 1)

    for row in range(1, grid_rows):
        y = int(row * cell_height)
        cv2.line(output, (0, y), (width, y), (255, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col in range(grid_cols):
        x = int((col + 0.5) * cell_width)
        cv2.putText(output, str(col), (x - 5, 15), font, 0.35, (255, 255, 0), 1)
    for row in range(grid_rows):
        y = int((row + 0.5) * cell_height)
        cv2.putText(output, str(row), (5, y + 5), font, 0.35, (255, 255, 0), 1)

    if frame_index is not None:
        cv2.putText(
            output,
            f"Frame {frame_index}",
            (10, max(height - 10, 20)),
            font,
            0.5,
            (255, 255, 0),
            1,
        )
    return output


def save_debug_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def build_black_mask_frames(mask_video_path: Path) -> list[np.ndarray]:
    frames = []
    for frame in iterate_video_frames(mask_video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        frames.append(np.where(gray > 127, 0, 255).astype(np.uint8))
    if not frames:
        raise RuntimeError("Mask propagation video did not contain any readable frames.")
    return frames


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def filter_masks_by_proximity(masks: list[np.ndarray], primary_mask: np.ndarray, dilation: int) -> list[np.ndarray]:
    proximity_region = dilate_mask(primary_mask, dilation)
    return [mask & proximity_region for mask in masks]


def gridify_masks(masks: list[np.ndarray], grid_rows: int, grid_cols: int) -> list[np.ndarray]:
    gridified_masks = []
    for mask in masks:
        height, width = mask.shape
        cell_width = width / grid_cols
        cell_height = height / grid_rows
        gridified = np.zeros((height, width), dtype=bool)

        for row in range(grid_rows):
            for col in range(grid_cols):
                y1 = int(row * cell_height)
                y2 = int((row + 1) * cell_height)
                x1 = int(col * cell_width)
                x2 = int((col + 1) * cell_width)
                if mask[y1:y2, x1:x2].any():
                    gridified[y1:y2, x1:x2] = True
        gridified_masks.append(gridified)
    return gridified_masks


def combine_black_and_grey_frames(black_frames: list[np.ndarray], grey_frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(black_frames) != len(grey_frames):
        raise RuntimeError(
            "Black and grey masks must have the same number of frames before building a quadmask."
        )

    combined_frames = []
    for black_frame, grey_frame in zip(black_frames, grey_frames):
        combined = np.full_like(black_frame, 255, dtype=np.uint8)
        combined[(black_frame == 0) & (grey_frame == 255)] = 0
        combined[(black_frame == 255) & (grey_frame == 127)] = 127
        combined[(black_frame == 0) & (grey_frame == 127)] = 63
        combined_frames.append(combined)
    return combined_frames


def masks_from_grid_localizations(
    grid_localizations: list[dict[str, object]],
    total_frames: int,
    frame_shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> list[np.ndarray]:
    height, width = frame_shape
    masks = [np.zeros((height, width), dtype=bool) for _ in range(total_frames)]
    if not grid_localizations:
        return masks

    sorted_localizations = sorted(
        (
            {
                "frame": max(0, min(int(item.get("frame", 0)), total_frames - 1)),
                "grid_regions": item.get("grid_regions", []),
            }
            for item in grid_localizations
            if isinstance(item, dict)
        ),
        key=lambda item: item["frame"],
    )

    for index, localization in enumerate(sorted_localizations):
        frame_index = localization["frame"]
        next_frame = (
            sorted_localizations[index + 1]["frame"]
            if index + 1 < len(sorted_localizations)
            else total_frames - 1
        )
        object_mask = np.zeros((height, width), dtype=bool)
        for region in localization["grid_regions"]:
            if not isinstance(region, dict):
                continue
            clamped = _clamp_grid_region(region, grid_rows, grid_cols)
            if clamped is None:
                continue
            y1 = int(clamped["row"] * height / grid_rows)
            y2 = int((clamped["row"] + 1) * height / grid_rows)
            x1 = int(clamped["col"] * width / grid_cols)
            x2 = int((clamped["col"] + 1) * width / grid_cols)
            object_mask[y1:y2, x1:x2] = True

        for fill_index in range(frame_index, min(next_frame + 1, total_frames)):
            masks[fill_index] |= object_mask

    return masks


class GeminiSceneAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ensure_configured(self) -> None:
        if self.settings.gemini_configured():
            return
        raise RuntimeError(
            "Gemini is not configured for the worker. Set GEMINI_API_KEY, GOOGLE_API_KEY, "
            "or WORKER_GEMINI_API_KEY before running the VOID pipeline."
        )

    def analyze(
        self,
        source_video_path: Path,
        primary_mask: np.ndarray,
        output_dir: Path,
        *,
        status_callback: StatusCallback | None = None,
    ) -> tuple[VoidReasoningResult, int, int]:
        self.ensure_configured()
        metadata = load_video_metadata(source_video_path)
        grid_rows, grid_cols = calculate_square_grid(
            metadata.width,
            metadata.height,
            self.settings.void_mask_min_grid,
        )

        # Keep the Gemini prompt inputs on disk for inspection because failures
        # here are often about model interpretation rather than code issues.
        analysis_dir = output_dir / "analysis_inputs"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        sample_indices = sample_frame_indices(metadata.frame_count)

        masked_first_frame = draw_grid(
            create_mask_overlay(read_frame(source_video_path, 0), primary_mask),
            grid_rows,
            grid_cols,
            0,
        )
        first_frame_path = analysis_dir / "first_frame_masked_grid.jpg"
        save_debug_image(first_frame_path, masked_first_frame)

        sample_paths = [first_frame_path]
        for frame_index in sample_indices:
            if frame_index == 0:
                continue
            sample_frame = draw_grid(read_frame(source_video_path, frame_index), grid_rows, grid_cols, frame_index)
            sample_path = analysis_dir / f"grid_sample_{frame_index:04d}.jpg"
            save_debug_image(sample_path, sample_frame)
            sample_paths.append(sample_path)

        if status_callback is not None:
            status_callback("analyze_scene", "Uploading VOID analysis inputs to Gemini.")

        from google import genai
        from google.genai import types

        uploaded_files = []
        try:
            client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=self.settings.gemini_timeout_ms),
            )
            uploaded_files.append(client.files.upload(file=source_video_path.as_posix()))
            for sample_path in sample_paths:
                uploaded_files.append(client.files.upload(file=sample_path.as_posix()))

            if status_callback is not None:
                status_callback("analyze_scene", "Asking Gemini to identify integral and affected objects.")

            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    self._analysis_prompt(grid_rows, grid_cols, sample_indices),
                    *uploaded_files,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=VLM_ANALYSIS_SCHEMA,
                ),
            )
            raw_text = response.text or "{}"
            reasoning = normalize_reasoning(json.loads(raw_text))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Gemini returned invalid JSON for VOID scene analysis: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Gemini scene analysis failed: {error}") from error
        finally:
            if "client" in locals():
                for uploaded_file in uploaded_files:
                    name = getattr(uploaded_file, "name", None)
                    if not name:
                        continue
                    try:
                        client.files.delete(name=name)
                    except Exception:
                        pass

        (output_dir / "vlm_analysis.json").write_text(
            json.dumps(reasoning.to_json(), indent=2),
            encoding="utf-8",
        )
        return reasoning, grid_rows, grid_cols

    def _analysis_prompt(self, grid_rows: int, grid_cols: int, sample_indices: list[int]) -> str:
        sample_text = ", ".join(str(index) for index in sample_indices)
        return f"""
You are preparing a quadmask for video object removal.

The uploaded video is the original clip. The first image shows the PRIMARY OBJECT TO REMOVE highlighted in red and overlaid with a yellow grid. The remaining images are grid reference frames from the clip. Use the full video plus the reference frames to decide what else should disappear or be masked.

Return strict JSON only.

Task:
1. Identify integral belongings that should be removed with the primary object.
   Examples: bike someone is riding, surfboard under a surfer, backpack clearly worn by the subject.
2. Identify affected objects or visual artifacts caused by removing the primary object.
   Examples: held object, chair a person is sitting on, shadow, reflection.
3. Write a clean-background prompt describing the scene after removal.

Rules:
- Be conservative. If unsure, leave an object out.
- Integral belongings are removed with the primary mask and should not also appear as affected objects.
- Use category "visual_artifact" for things like shadows or reflections.
- For visual artifacts, include `grid_localizations` using the displayed frame numbers and grid cells.
- Use frame numbers exactly as shown in the sample images.
- Do not include trajectories or movement fields.

Grid info:
- Rows: {grid_rows}
- Columns: {grid_cols}
- Sample frames: {sample_text}

Output schema:
{json.dumps(VLM_ANALYSIS_SCHEMA, indent=2)}
""".strip()


class SamTextMaskSegmenter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor = None
        self._lock = threading.Lock()

    def _candidate_checkpoint_paths(self) -> list[str | None]:
        candidates = []
        if self.settings.sam_checkpoint_path.exists():
            candidates.append(self.settings.sam_checkpoint_path.as_posix())
        if self.settings.sam_legacy_checkpoint_path.exists():
            candidates.append(self.settings.sam_legacy_checkpoint_path.as_posix())
        if self.settings.sam_allow_hf_download:
            candidates.append(None)
        if candidates:
            return candidates
        raise RuntimeError(
            "VOID affected-mask generation requires a SAM checkpoint for image grounding. "
            "Expected a local SAM3.1 or SAM3 checkpoint in the worker models directory."
        )

    def _ensure_processor(self):
        if self._processor is not None:
            return self._processor

        with self._lock:
            if self._processor is not None:
                return self._processor

            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except Exception as error:
                raise RuntimeError(
                    "SAM 3 image-grounding components are not available in the worker environment."
                ) from error

            model = None
            errors = []
            for checkpoint_path in self._candidate_checkpoint_paths():
                try:
                    # The image-grounding builder ships separately from the
                    # interactive multiplex tracker. Try the local SAM 3.1
                    # checkpoint first so VOID reuses the existing worker
                    # assets, then fall back to the legacy SAM3 image weights
                    # if the builder rejects the multiplex checkpoint shape.
                    model = build_sam3_image_model(
                        checkpoint_path=checkpoint_path,
                        load_from_HF=checkpoint_path is None,
                        compile=self.settings.sam_compile,
                    )
                    break
                except Exception as error:
                    label = checkpoint_path or "huggingface"
                    errors.append(f"{label}: {error}")

            if model is None:
                raise RuntimeError(
                    "Unable to initialize the SAM image-grounding model for VOID. "
                    + "; ".join(errors)
                )
            self._processor = Sam3Processor(model)
            return self._processor

    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        processor = self._ensure_processor()
        with self._lock:
            try:
                state = processor.set_image(image)
                output = processor.set_text_prompt(prompt=prompt, state=state)
            except Exception as error:
                raise RuntimeError(f"SAM text segmentation failed for prompt '{prompt}': {error}") from error

        masks = output.get("masks")
        if masks is None:
            return np.zeros((image.height, image.width), dtype=bool)

        array = np.asarray(masks)
        if hasattr(masks, "detach"):
            array = masks.detach().float().cpu().numpy()

        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 3:
            return array.any(axis=0).astype(bool)
        if array.ndim == 2:
            return array.astype(bool)
        return np.zeros((image.height, image.width), dtype=bool)


def segment_object_all_frames(
    video_path: Path,
    noun: str,
    segmenter: SamTextMaskSegmenter,
    frame_stride: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video for VOID segmentation: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    masks = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % max(frame_stride, 1) == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                masks.append(segmenter.segment(image, noun))
            elif masks:
                masks.append(masks[-1].copy())
            else:
                masks.append(np.zeros((frame_height, frame_width), dtype=bool))

            frame_index += 1
    finally:
        capture.release()

    if len(masks) != total_frames:
        while len(masks) < total_frames:
            masks.append(np.zeros((frame_height, frame_width), dtype=bool))
    return masks


class VoidQuadmaskBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = GeminiSceneAnalyzer(settings)
        self.segmenter = SamTextMaskSegmenter(settings)

    def build(
        self,
        *,
        source_video_path: Path,
        mask_video_path: Path,
        sequence_dir: Path,
        background_prompt_override: str | None = None,
        status_callback: StatusCallback | None = None,
    ) -> dict[str, object]:
        metadata = load_video_metadata(source_video_path)
        black_frames = build_black_mask_frames(mask_video_path)
        primary_mask = black_frames[0] == 0

        sequence_dir.mkdir(parents=True, exist_ok=True)

        reasoning, grid_rows, grid_cols = self.analyzer.analyze(
            source_video_path,
            primary_mask,
            sequence_dir,
            status_callback=status_callback,
        )
        _materialize_source(source_video_path, sequence_dir / "input_video.mp4")

        if status_callback is not None:
            status_callback("segment_integral", "Extending the black mask with Gemini-detected integral belongings.")

        remove_masks = [frame == 0 for frame in black_frames]
        for item in reasoning.integral_belongings:
            noun = item["noun"]
            if status_callback is not None:
                status_callback("segment_integral", f"Segmenting integral belonging: {noun}")
            object_masks = segment_object_all_frames(
                source_video_path,
                noun,
                self.segmenter,
                self.settings.void_mask_frame_stride,
            )
            for index, object_mask in enumerate(object_masks):
                if index < len(remove_masks):
                    remove_masks[index] |= object_mask

        black_output_frames = [np.where(mask, 0, 255).astype(np.uint8) for mask in remove_masks]
        write_lossless_mask_video(
            sequence_dir / "black_mask.mp4",
            black_output_frames,
            metadata.fps,
            metadata.width,
            metadata.height,
        )

        if status_callback is not None:
            status_callback("segment_affected", "Building Gemini-informed affected-region masks.")

        affected_masks = [np.zeros((metadata.height, metadata.width), dtype=bool) for _ in range(metadata.frame_count)]
        for item in reasoning.affected_objects:
            noun = item["noun"]
            category = str(item.get("category", "physical"))
            if status_callback is not None:
                status_callback("segment_affected", f"Segmenting affected object: {noun}")

            if category == "visual_artifact" and item.get("grid_localizations"):
                object_masks = masks_from_grid_localizations(
                    item["grid_localizations"],
                    metadata.frame_count,
                    (metadata.height, metadata.width),
                    grid_rows,
                    grid_cols,
                )
            else:
                object_masks = segment_object_all_frames(
                    source_video_path,
                    noun,
                    self.segmenter,
                    self.settings.void_mask_frame_stride,
                )

            object_masks = filter_masks_by_proximity(
                object_masks,
                primary_mask,
                self.settings.void_mask_proximity_dilation,
            )
            for index, object_mask in enumerate(object_masks):
                if index < len(affected_masks):
                    affected_masks[index] |= object_mask

        if status_callback is not None:
            status_callback("combine_quadmask", "Gridifying affected masks and combining them into a quadmask.")

        grey_masks = gridify_masks(affected_masks, grid_rows, grid_cols)
        grey_output_frames = [np.where(mask, 127, 255).astype(np.uint8) for mask in grey_masks]
        write_lossless_mask_video(
            sequence_dir / "grey_mask.mp4",
            grey_output_frames,
            metadata.fps,
            metadata.width,
            metadata.height,
        )

        quadmask_frames = combine_black_and_grey_frames(black_output_frames, grey_output_frames)
        write_lossless_mask_video(
            sequence_dir / "quadmask_0.mp4",
            quadmask_frames,
            metadata.fps,
            metadata.width,
            metadata.height,
        )

        background_prompt = (background_prompt_override or "").strip() or reasoning.background_prompt
        if not background_prompt:
            raise RuntimeError("Gemini did not return a usable background prompt for the VOID pipeline.")

        prompt_path = sequence_dir / "prompt.json"
        prompt_path.write_text(
            json.dumps({"bg": background_prompt}, indent=2),
            encoding="utf-8",
        )
        return {
            "backgroundPrompt": background_prompt,
            "analysis": reasoning.to_json(),
            "analysisPath": sequence_dir / "vlm_analysis.json",
            "blackMaskPath": sequence_dir / "black_mask.mp4",
            "greyMaskPath": sequence_dir / "grey_mask.mp4",
            "quadmaskPath": sequence_dir / "quadmask_0.mp4",
            "promptPath": prompt_path,
        }


def _materialize_source(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    try:
        target_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)
