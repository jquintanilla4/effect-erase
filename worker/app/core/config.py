from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", extra="ignore")

    root_dir: Path = ROOT_DIR
    data_dir: Path | None = None
    models_dir: Path | None = None
    projects_dir: Path | None = None
    bootstrap_state_path: Path | None = None
    default_height: int = 480
    default_width: int = 832
    max_window_frames: int = 81
    overlap_frames: int = 16
    runtime_mode: str = "auto"
    use_mock_runtime: bool = False
    sam_default_version: str = "sam3.1"
    sam_fallback_model: str = "sam2.1"
    sam_checkpoint_path: Path | None = None
    sam_legacy_checkpoint_path: Path | None = None
    sam_allow_hf_download: bool = False
    sam_compile: bool = False
    sam_async_loading_frames: bool = False
    sam_max_num_objects: int = 1
    # Match the published SAM 3.1 multiplex checkpoint shape by default.
    # Operators can still override this with WORKER_SAM_MULTIPLEX_COUNT when
    # testing a custom checkpoint that was trained with a different bucket size.
    sam_multiplex_count: int = 16
    sam2_repo_dir: Path | None = None
    sam2_hf_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_allow_hf_download: bool = True
    sam2_checkpoint_path: Path | None = None
    sam2_config_path: Path | None = None
    effecterase_model_dir: Path | None = None
    effecterase_lora_path: Path | None = None
    effecterase_wan_model_dir: Path | None = None
    effecterase_num_frames: int = 81
    effecterase_frame_interval: int = 1
    effecterase_seed: int = 2025
    effecterase_cfg: float = 1.0
    effecterase_lora_alpha: float = 1.0
    effecterase_num_inference_steps: int = 50
    effecterase_tiled: bool = False
    effecterase_use_teacache: bool = False
    public_base_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        fields_set = set(self.model_fields_set)

        if "data_dir" not in fields_set or self.data_dir is None:
            self.data_dir = self.root_dir / "data"
        if "models_dir" not in fields_set or self.models_dir is None:
            self.models_dir = self.root_dir / "models"
        if "projects_dir" not in fields_set or self.projects_dir is None:
            self.projects_dir = self.data_dir / "projects"
        if "bootstrap_state_path" not in fields_set or self.bootstrap_state_path is None:
            self.bootstrap_state_path = self.data_dir / "bootstrap-status.json"
        if "sam_checkpoint_path" not in fields_set or self.sam_checkpoint_path is None:
            self.sam_checkpoint_path = self.models_dir / "sam3.1" / "sam3.1_multiplex.pt"
        if "sam_legacy_checkpoint_path" not in fields_set or self.sam_legacy_checkpoint_path is None:
            self.sam_legacy_checkpoint_path = self.models_dir / "sam3" / "sam3.pt"
        if "sam2_repo_dir" not in fields_set or self.sam2_repo_dir is None:
            self.sam2_repo_dir = self.root_dir / "third_party" / "sam2"
        if "sam2_checkpoint_path" not in fields_set or self.sam2_checkpoint_path is None:
            self.sam2_checkpoint_path = self.models_dir / "sam2.1" / "sam2.1_hiera_base_plus.pt"
        if "sam2_config_path" not in fields_set or self.sam2_config_path is None:
            self.sam2_config_path = self.sam2_repo_dir / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
        if "effecterase_model_dir" not in fields_set or self.effecterase_model_dir is None:
            self.effecterase_model_dir = self.models_dir / "EffectErase"
        if "effecterase_lora_path" not in fields_set or self.effecterase_lora_path is None:
            self.effecterase_lora_path = self.effecterase_model_dir / "EffectErase.ckpt"
        if "effecterase_wan_model_dir" not in fields_set or self.effecterase_wan_model_dir is None:
            self.effecterase_wan_model_dir = self.models_dir / "Wan-AI" / "Wan2.1-Fun-1.3B-InP"
        return self

    def use_real_runtime(self) -> bool:
        if self.use_mock_runtime:
            return False
        return self.runtime_mode in {"auto", "real"}

    def sam_checkpoint_for_model(self, model_name: str) -> Path:
        return self.sam_legacy_checkpoint_path if model_name == "sam3" else self.sam_checkpoint_path

    def effecterase_required_paths(self) -> dict[str, Path]:
        wan_root = self.effecterase_wan_model_dir
        tokenizer_root = wan_root / "google" / "umt5-xxl"
        return {
            "lora": self.effecterase_lora_path,
            "text_encoder": wan_root / "models_t5_umt5-xxl-enc-bf16.pth",
            "vae": wan_root / "Wan2.1_VAE.pth",
            "dit": wan_root / "diffusion_pytorch_model.safetensors",
            "image_encoder": wan_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "tokenizer_config": tokenizer_root / "tokenizer_config.json",
            "tokenizer_json": tokenizer_root / "tokenizer.json",
            "sentencepiece": tokenizer_root / "spiece.model",
            "special_tokens_map": tokenizer_root / "special_tokens_map.json",
        }


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.effecterase_model_dir.mkdir(parents=True, exist_ok=True)
    return settings
