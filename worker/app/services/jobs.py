from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException

from app.core.config import Settings
from app.models.runtime import build_remove_runtime
from app.schemas.api import JobResponse, RemoveRequest
from app.services.projects import ProjectService
from app.services.sessions import SessionService


@dataclass
class JobState:
    job_id: str
    project_id: str
    status: str = "queued"
    progress: float = 0.0
    result_url: str | None = None
    error: str | None = None


class JobService:
    def __init__(self, settings: Settings, project_service: ProjectService, session_service: SessionService) -> None:
        self.settings = settings
        self.project_service = project_service
        self.session_service = session_service
        self.runtime = build_remove_runtime(settings)
        self.jobs: dict[str, JobState] = {}

    def create_removal_job(self, payload: RemoveRequest, _: BackgroundTasks) -> JobResponse:
        state, mask_video_path = self.session_service.require_mask_video(payload.sessionId)
        if state.project_id != payload.projectId:
            raise HTTPException(status_code=400, detail="Session does not belong to the requested project.")

        source_video_path = self.project_service.require_source_video(payload.projectId)
        output_path = self.project_service.storage.project_dir(payload.projectId) / "removed_output.mp4"
        job_id = uuid4().hex[:12]
        job = JobState(job_id=job_id, project_id=payload.projectId)
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source_video_path, mask_video_path, output_path),
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id)

    def _run_job(self, job_id: str, source_video_path: Path, mask_video_path: Path, output_path: Path) -> None:
        job = self.jobs[job_id]
        job.status = "running"

        try:
            self.runtime.remove(
                source_video_path=source_video_path,
                mask_video_path=mask_video_path,
                output_video_path=output_path,
                progress_callback=lambda value: self._update_progress(job_id, value),
            )
            job.status = "completed"
            job.result_url = self.project_service.storage.artifact_url(self.settings.public_base_url, output_path)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            job.status = "failed"
            job.error = str(exc)

    def _update_progress(self, job_id: str, value: float) -> None:
        job = self.jobs[job_id]
        job.progress = round(max(0.0, min(value, 1.0)), 4)

    def get_job(self, job_id: str) -> JobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        return JobResponse(
            jobId=job.job_id,
            projectId=job.project_id,
            status=job.status,
            progress=job.progress,
            resultUrl=job.result_url,
            error=job.error,
        )
