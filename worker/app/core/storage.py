from pathlib import Path
from uuid import uuid4


class Storage:
    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project_dir(self) -> tuple[str, Path]:
        project_id = uuid4().hex[:12]
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_id, project_dir

    def project_dir(self, project_id: str) -> Path:
        path = self.projects_dir / project_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_relative_path(self, file_path: Path) -> str:
        return file_path.relative_to(self.projects_dir).as_posix()

    def artifact_path(self, file_path: Path) -> str:
        return f"/artifacts/{self.artifact_relative_path(file_path)}"

    def artifact_url(self, public_base_url: str, file_path: Path) -> str:
        return f"{public_base_url.rstrip('/')}{self.artifact_path(file_path)}"
