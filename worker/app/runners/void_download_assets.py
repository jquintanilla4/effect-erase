from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PROGRESS_PREFIX = "PROGRESS_JSON:"


def emit(progress: float, stage: str, message: str) -> None:
    payload = {
        "progress": round(max(0.0, min(progress, 1.0)), 4),
        "stage": stage,
        "message": message,
    }
    print(f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download VOID model assets on demand.")
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--pass1-repo-id", required=True)
    parser.add_argument("--pass1-filename", required=True)
    parser.add_argument("--pass1-path", required=True)
    return parser


def repo_files(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    # The inference stack only needs the model files, not repo metadata docs.
    return [name for name in files if not name.startswith(".") and not name.lower().endswith((".md", ".txt"))]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    base_model_dir = Path(args.base_model_dir)
    pass1_path = Path(args.pass1_path)

    emit(0.02, "download_assets", "Loading VOID asset manifest.")
    files = repo_files(args.base_model_id)
    if not files:
        raise RuntimeError(f"No downloadable files were returned for {args.base_model_id}.")

    total_steps = len(files) + 1
    for index, filename in enumerate(files, start=1):
        emit(
            0.05 + (0.85 * (index - 1) / total_steps),
            "download_assets",
            f"Downloading base model file {index}/{len(files)}: {filename}",
        )
        ensure_parent(base_model_dir / filename)
        hf_hub_download(
            repo_id=args.base_model_id,
            filename=filename,
            repo_type="model",
            local_dir=base_model_dir.as_posix(),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    emit(0.92, "download_assets", f"Downloading VOID Pass 1 checkpoint: {args.pass1_filename}")
    ensure_parent(pass1_path)
    hf_hub_download(
        repo_id=args.pass1_repo_id,
        filename=args.pass1_filename,
        repo_type="model",
        local_dir=pass1_path.parent.as_posix(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    emit(1.0, "download_assets", "VOID model assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
