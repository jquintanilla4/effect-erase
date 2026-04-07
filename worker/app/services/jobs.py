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
    kind: str
    pipeline: str
    project_id: str | None = None
    status: str = "queued"
    progress: float = 0.0
    stage: str | None = None
    message: str | None = None
    # Store the artifact path, not an absolute URL, so polling clients always
    # receive a result URL on the same public origin they used for the API call.
    result_path: Path | None = None
    error: str | None = None


class JobService:
    def __init__(self, settings: Settings, project_service: ProjectService, session_service: SessionService) -> None:
        self.settings = settings
        self.project_service = project_service
        self.session_service = session_service
        self.runtimes = {
            "effecterase": build_remove_runtime(settings, "effecterase"),
            "void": build_remove_runtime(settings, "void"),
        }
        self.jobs: dict[str, JobState] = {}
        self._active_download_jobs: dict[str, str] = {}
        self._download_lock = threading.Lock()

    def download_states(self) -> dict[str, dict[str, str | bool | None]]:
        with self._download_lock:
            return {
                pipeline_id: {
                    "active": bool(job_id),
                    "jobId": job_id,
                }
                for pipeline_id, job_id in self._active_download_jobs.items()
                if job_id
            }

    def create_model_download_job(self, pipeline_id: str, public_base_url: str) -> JobResponse:
        if pipeline_id != "void":
            raise HTTPException(status_code=400, detail=f"Pipeline '{pipeline_id}' does not support manual downloads.")

        runtime = self.runtimes[pipeline_id]
        if runtime.assets_ready():
            job_id = uuid4().hex[:12]
            self.jobs[job_id] = JobState(
                job_id=job_id,
                kind="model_download",
                pipeline=pipeline_id,
                status="completed",
                progress=1.0,
                stage="download_assets",
                message="VOID model assets are already available.",
            )
            return self.get_job(job_id, public_base_url)

        with self._download_lock:
            active_job_id = self._active_download_jobs.get(pipeline_id)
            if active_job_id:
                return self.get_job(active_job_id, public_base_url)

            job_id = uuid4().hex[:12]
            self.jobs[job_id] = JobState(
                job_id=job_id,
                kind="model_download",
                pipeline=pipeline_id,
                status="queued",
                stage="download_assets",
                message="Preparing VOID model download.",
            )
            self._active_download_jobs[pipeline_id] = job_id

        thread = threading.Thread(
            target=self._run_model_download_job,
            args=(job_id, pipeline_id),
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id, public_base_url)

    def create_removal_job(self, payload: RemoveRequest, _: BackgroundTasks, public_base_url: str) -> JobResponse:
        state, mask_video_path = self.session_service.require_mask_video(payload.sessionId)
        if state.project_id != payload.projectId:
            raise HTTPException(status_code=400, detail="Session does not belong to the requested project.")

        source_video_path = self.project_service.require_source_video(payload.projectId)
        output_filename = "removed_output.mp4" if payload.pipeline == "effecterase" else "removed_output_void.mp4"
        output_path = self.project_service.storage.project_dir(payload.projectId) / output_filename
        # Free the interactive SAM session before removal work starts so the
        # worker can reuse GPU memory for either the EffectErase path or the
        # VOID quadmask + inference pipeline.
        self.session_service.release_runtime_resources(payload.sessionId)
        job_id = uuid4().hex[:12]
        job = JobState(
            job_id=job_id,
            kind="remove",
            pipeline=payload.pipeline,
            project_id=payload.projectId,
            stage="queued",
            message=f"Queued {payload.pipeline} removal job.",
        )
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id,
                payload.pipeline,
                payload.backgroundPrompt,
                source_video_path,
                mask_video_path,
                output_path,
            ),
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id, public_base_url)

    def _run_model_download_job(self, job_id: str, pipeline_id: str) -> None:
        job = self.jobs[job_id]
        job.status = "running"
        runtime = self.runtimes[pipeline_id]

        try:
            runtime.ensure_models(
                progress_callback=lambda value: self._update_progress(job_id, value),
                status_callback=lambda stage, message: self._update_status(job_id, stage, message),
            )
            job.progress = 1.0
            job.status = "completed"
            job.stage = "download_assets"
            job.message = "VOID model assets are ready."
        except Exception as exc:  # pragma: no cover - defensive runtime path
            job.status = "failed"
            job.stage = "failed"
            job.error = str(exc)
            job.message = str(exc)
        finally:
            with self._download_lock:
                if self._active_download_jobs.get(pipeline_id) == job_id:
                    self._active_download_jobs.pop(pipeline_id, None)

    def _run_job(
        self,
        job_id: str,
        pipeline: str,
        background_prompt: str | None,
        source_video_path: Path,
        mask_video_path: Path,
        output_path: Path,
    ) -> None:
        job = self.jobs[job_id]
        job.status = "running"
        runtime = self.runtimes[pipeline]

        try:
            runtime.remove(
                source_video_path=source_video_path,
                mask_video_path=mask_video_path,
                output_video_path=output_path,
                progress_callback=lambda value: self._update_progress(job_id, value),
                status_callback=lambda stage, message: self._update_status(job_id, stage, message),
                background_prompt=background_prompt,
                job_id=job_id,
            )
            # The subprocess emits a final near-complete encode milestone, but
            # the API should only report a finished job once the artifact exists.
            job.progress = 1.0
            job.status = "completed"
            job.result_path = output_path
        except Exception as exc:  # pragma: no cover - defensive runtime path
            job.status = "failed"
            job.stage = "failed"
            job.error = str(exc)
            job.message = str(exc)

    def _update_progress(self, job_id: str, value: float) -> None:
        job = self.jobs[job_id]
        job.progress = round(max(0.0, min(value, 1.0)), 4)

    def _update_status(self, job_id: str, stage: str | None, message: str | None) -> None:
        job = self.jobs[job_id]
        if stage:
            job.stage = stage
        if message:
            job.message = message

    def get_job(self, job_id: str, public_base_url: str) -> JobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        return JobResponse(
            jobId=job.job_id,
            kind=job.kind,
            pipeline=job.pipeline,
            projectId=job.project_id,
            status=job.status,
            progress=job.progress,
            stage=job.stage,
            message=job.message,
            resultUrl=(
                self.project_service.storage.artifact_url(public_base_url, job.result_path)
                if job.result_path is not None
                else None
            ),
            error=job.error,
        )
