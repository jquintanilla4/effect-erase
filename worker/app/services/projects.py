from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.bootstrap import load_bootstrap_status
from app.core.config import Settings
from app.core.storage import Storage
from app.models.video import load_video_metadata
from app.schemas.api import BootstrapStatus, CreateProjectResponse, ProjectCreateRequest, UploadVideoResponse


class ProjectService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = Storage(settings.projects_dir)
        self.bootstrap_status = self.refresh_bootstrap_status()

    def refresh_bootstrap_status(self) -> BootstrapStatus:
        self.bootstrap_status = load_bootstrap_status(self.settings.bootstrap_state_path)
        return self.bootstrap_status

    def create_project(self, payload: ProjectCreateRequest, public_base_url: str) -> CreateProjectResponse:
        project_id, project_dir = self.storage.create_project_dir()
        metadata = {
            "projectId": project_id,
            "profileId": payload.profileId,
            "label": payload.label,
        }
        (project_dir / "project.json").write_text(json.dumps(metadata, indent=2))
        project_url = self.storage.artifact_url(public_base_url, project_dir / "project.json")
        return CreateProjectResponse(
            projectId=project_id,
            profileId=payload.profileId,
            label=payload.label,
            projectUrl=project_url,
        )

    async def save_upload(self, project_id: str, upload: UploadFile, public_base_url: str) -> UploadVideoResponse:
        project_dir = self.storage.project_dir(project_id)
        source_path = project_dir / "source.mp4"
        content = await upload.read()
        source_path.write_bytes(content)

        metadata = load_video_metadata(source_path)
        upload_metadata = {
            "projectId": project_id,
            "sourceUrl": self.storage.artifact_url(public_base_url, source_path),
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frameCount": metadata.frame_count,
        }
        (project_dir / "video.json").write_text(json.dumps(upload_metadata, indent=2))
        return UploadVideoResponse(**upload_metadata)

    def require_source_video(self, project_id: str) -> Path:
        source_path = self.storage.project_dir(project_id) / "source.mp4"
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source video not found for project.")
        return source_path
