from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PROGRESS_PREFIX = "PROGRESS_JSON:"


def emit(progress: float, stage: str, message: str) -> None:
    payload = {
        "progress": round(max(0.0, min(progress, 1.0)), 4),
        "stage": stage,
        "message": message,
    }
    print(f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal VOID removal runner.")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--data-rootdir", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--transformer-path", required=True)
    parser.add_argument("--sample-size", required=True)
    parser.add_argument("--max-video-length", type=int, required=True)
    return parser


def find_result_video(output_dir: Path, sequence_name: str) -> Path:
    candidates = sorted(
        path
        for path in output_dir.glob(f"{sequence_name}-fg=-1-*.mp4")
        if not path.name.endswith("_tuple.mp4")
    )
    if not candidates:
        raise RuntimeError(f"VOID did not create an output MP4 in {output_dir}.")
    return candidates[-1]


def main() -> int:
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output_dir)
    output_path = Path(args.output_path)
    predict_script = repo_dir / "inference" / "cogvideox_fun" / "predict_v2v.py"
    config_path = repo_dir / "config" / "quadmask_cogvideox.py"

    emit(0.05, "inference", "Launching VOID Pass 1 inference.")
    command = [
        "python",
        predict_script.as_posix(),
        "--config",
        config_path.as_posix(),
        f"--config.data.data_rootdir={args.data_rootdir}",
        f"--config.experiment.run_seqs={args.sequence_name}",
        f"--config.experiment.save_path={args.output_dir}",
        "--config.experiment.skip_if_exists=false",
        f"--config.video_model.model_name={args.base_model_dir}",
        f"--config.video_model.transformer_path={args.transformer_path}",
        f"--config.data.sample_size={args.sample_size}",
        f"--config.data.max_video_length={args.max_video_length}",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_dir.as_posix(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_output = "\n".join(
            chunk.strip()
            for chunk in (completed.stdout, completed.stderr)
            if chunk and chunk.strip()
        )
        raise RuntimeError(
            f"VOID inference failed with exit code {completed.returncode}."
            + (f"\n{error_output}" if error_output else "")
        )

    emit(0.95, "finalize", "Collecting VOID output artifact.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_video = find_result_video(output_dir, args.sequence_name)
    shutil.copy2(result_video, output_path)
    emit(1.0, "finalize", "VOID output is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
