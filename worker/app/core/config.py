from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WORKER_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

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
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WORKER_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("WORKER_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    gemini_timeout_ms: int = Field(
        default=120000,
        validation_alias=AliasChoices("WORKER_GEMINI_TIMEOUT_MS", "GEMINI_TIMEOUT_MS"),
    )
    void_repo_dir: Path | None = None
    void_model_dir: Path | None = None
    void_base_model_dir: Path | None = None
    void_pass1_path: Path | None = None
    void_base_model_id: str = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
    void_pass1_repo_id: str = "netflix/void-model"
    void_pass1_filename: str = "void_pass1.safetensors"
    void_max_frames: int = 197
    void_sample_height: int = 384
    void_sample_width: int = 672
    void_mask_frame_stride: int = 5
    void_mask_proximity_dilation: int = 50
    void_mask_min_grid: int = 8
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
        if "void_repo_dir" not in fields_set or self.void_repo_dir is None:
            self.void_repo_dir = self.root_dir / "third_party" / "void-model"
        if "void_model_dir" not in fields_set or self.void_model_dir is None:
            self.void_model_dir = self.models_dir / "VOID"
        if "void_base_model_dir" not in fields_set or self.void_base_model_dir is None:
            self.void_base_model_dir = self.void_model_dir / "CogVideoX-Fun-V1.5-5b-InP"
        if "void_pass1_path" not in fields_set or self.void_pass1_path is None:
            self.void_pass1_path = self.void_model_dir / self.void_pass1_filename
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

    def gemini_configured(self) -> bool:
        return bool((self.gemini_api_key or "").strip())

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

    def void_required_paths(self) -> dict[str, Path]:
        base_root = self.void_base_model_dir
        return {
            "repo_root": self.void_repo_dir,
            "base_model_root": base_root,
            "model_index": base_root / "model_index.json",
            "scheduler_config": base_root / "scheduler" / "scheduler_config.json",
            "tokenizer_config": base_root / "tokenizer" / "tokenizer_config.json",
            "text_encoder_config": base_root / "text_encoder" / "config.json",
            "transformer_config": base_root / "transformer" / "config.json",
            "vae_config": base_root / "vae" / "config.json",
            "pass1": self.void_pass1_path,
        }


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.void_model_dir.mkdir(parents=True, exist_ok=True)
    settings.effecterase_model_dir.mkdir(parents=True, exist_ok=True)
    return settings
