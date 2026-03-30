from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

from diffsynth import ModelManager, WanRemovePipeline


REMOVE_PROMPT = "Remove the specified object and all related effects, then restore a clean background."
NEGATIVE_PROMPT = (
    "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal EffectErase removal runner.")
    parser.add_argument("--fg_bg_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--use_teacache", action="store_true")
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, required=True)
    parser.add_argument("--pretrained_lora_path", type=str, required=True)
    return parser


def resize_image(image: Image.Image, height: int, width: int) -> Image.Image:
    return torchvision.transforms.functional.resize(
        image,
        (height, width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )


def frame_norm_to_tensor(frame: Image.Image) -> torch.Tensor:
    tensor = torchvision.transforms.functional.to_tensor(frame)
    tensor = torchvision.transforms.functional.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    return tensor


def read_video_frames(
    video_path: str,
    num_frames: int,
    frame_interval: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Video {video_path} has only {total_frames} frames, less than required num_frames={num_frames}."
        )

    step = frame_interval if total_frames >= frame_interval * num_frames else 1
    frames: list[torch.Tensor] = []
    first_frame: Image.Image | None = None

    for index in range(num_frames):
        frame_idx = index * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        if first_frame is None:
            first_frame = frame_pil.copy()
        frames.append(frame_norm_to_tensor(resize_image(frame_pil, height, width)))

    cap.release()

    if first_frame is None or len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames from {video_path}, but read {len(frames)}.")

    video_tensor = torch.stack(frames, dim=0)
    video_tensor = rearrange(video_tensor, "t c h w -> c t h w")
    return video_tensor, first_frame


def crop_square_from_pil(
    mask_img: Image.Image,
    fg_bg_img: Image.Image,
    *,
    target_size: int = 224,
    video_mask_path: str | None = None,
) -> torch.Tensor:
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:
        mask_np = mask_np.max(axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8)

    img_np = np.array(fg_bg_img.convert("RGB"))
    height, width = mask_np.shape
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError(f"{video_mask_path or 'mask'} has no valid mask region in first frame.")

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0

    sx0 = int(np.floor(center_x - side / 2))
    sy0 = int(np.floor(center_y - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    pad_left = max(0, -sx0)
    pad_top = max(0, -sy0)
    pad_right = max(0, sx1 - width)
    pad_bottom = max(0, sy1 - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        mask_np = np.pad(
            mask_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        sx0 += pad_left
        sx1 += pad_left
        sy0 += pad_top
        sy1 += pad_top

    crop_img = img_np[sy0:sy1, sx0:sx1]
    crop_mask = mask_np[sy0:sy1, sx0:sx1][..., None]
    crop_img = crop_img * crop_mask

    crop_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float().unsqueeze(0)
    crop_tensor = torch.nn.functional.interpolate(
        crop_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    crop_tensor = crop_tensor / 255.0
    crop_tensor = crop_tensor * 2.0 - 1.0
    return crop_tensor


def frame_to_rgb_uint8(frame: object) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def fps_for_video(video_path: str, default_fps: float = 24.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default_fps
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    cap.release()
    return fps


def save_video(frames: list[object] | np.ndarray, output_path: str, ref_video_path: str) -> str:
    fps = fps_for_video(ref_video_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_file.as_posix(),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        quality=8,
    )
    try:
        for frame in frames:
            writer.append_data(frame_to_rgb_uint8(frame))
    finally:
        writer.close()
    return output_file.as_posix()


def run(args: argparse.Namespace) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("EffectErase removal requires CUDA.")

    device = torch.device("cuda")

    print("[INFO] Reading videos...")
    mask_video, mask_first_image = read_video_frames(
        args.mask_path,
        args.num_frames,
        args.frame_interval,
        args.height,
        args.width,
    )
    fg_bg_video, fg_bg_first_image = read_video_frames(
        args.fg_bg_path,
        args.num_frames,
        args.frame_interval,
        args.height,
        args.width,
    )
    fg_first_img = crop_square_from_pil(
        mask_first_image,
        fg_bg_first_image,
        target_size=224,
        video_mask_path=args.mask_path,
    )

    print("[INFO] Building model...")
    model_manager = ModelManager(device=device.type)
    model_manager.load_models(
        [
            args.dit_path,
            args.text_encoder_path,
            args.vae_path,
            args.image_encoder_path,
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora_v2(args.pretrained_lora_path, lora_alpha=args.lora_alpha)

    pipe = WanRemovePipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device=device.type,
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

    mask_video = mask_video.to(device)
    fg_bg_video = fg_bg_video.to(device)
    fg_first_img = fg_first_img.to(device)

    print("[INFO] Running inference...")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        remove_video, _ = pipe(
            video_mask=mask_video,
            video_fg_bg=fg_bg_video,
            video_bg=None,
            task="remove",
            fg_first_img=fg_first_img,
            prompt_remove=REMOVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            tiled=args.tiled,
            height=args.height,
            width=args.width,
            tea_cache_l1_thresh=0.3 if args.use_teacache else None,
            tea_cache_model_id="Wan2.1-T2V-1.3B" if args.use_teacache else None,
        )

    save_video(remove_video, args.output_path, args.fg_bg_path)
    return args.output_path


def main() -> None:
    start = time.time()
    args = build_parser().parse_args()
    output_path = run(args)
    print(f"[INFO] Saved video to: {output_path}")
    print(f"[INFO] Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
