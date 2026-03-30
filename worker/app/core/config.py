from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", extra="ignore")

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    projects_dir: Path = ROOT_DIR / "data" / "projects"
    bootstrap_state_path: Path = ROOT_DIR / "data" / "bootstrap-status.json"
    default_height: int = 480
    default_width: int = 832
    max_window_frames: int = 81
    overlap_frames: int = 16
    use_mock_runtime: bool = True
    public_base_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    return settings

