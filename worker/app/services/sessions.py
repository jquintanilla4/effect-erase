from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import Settings
from app.models.runtime import SessionRuntimeState, build_sam_runtime
from app.schemas.api import AddPromptRequest, AddPromptResponse, PropagateRequest, PropagateResponse, StartSessionRequest, StartSessionResponse
from app.services.projects import ProjectService


class SessionService:
    def __init__(self, settings: Settings, project_service: ProjectService) -> None:
        self.settings = settings
        self.project_service = project_service
        self.runtime = build_sam_runtime(settings)
        self.sessions: dict[str, SessionRuntimeState] = {}

    def start_session(self, payload: StartSessionRequest) -> StartSessionResponse:
        source_video_path = self.project_service.require_source_video(payload.projectId)
        runtime_state = self.runtime.start(payload.projectId, source_video_path, payload.model)
        session_id = uuid4().hex[:12]
        self.sessions[session_id] = runtime_state
        return StartSessionResponse(
            sessionId=session_id,
            projectId=payload.projectId,
            model=runtime_state.model_name,
            frameCount=runtime_state.frame_count,
            fps=runtime_state.fps,
            width=runtime_state.width,
            height=runtime_state.height,
        )

    def add_prompt(self, payload: AddPromptRequest) -> AddPromptResponse:
        state = self.sessions.get(payload.sessionId)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        project_dir = self.project_service.storage.project_dir(state.project_id)
        mask_path = project_dir / f"mask_preview_{payload.frameIndex}.png"
        frame_path = project_dir / f"frame_{payload.frameIndex}.png"
        self.runtime.add_prompt(
            state=state,
            frame_index=payload.frameIndex,
            points=payload.points,
            output_mask_path=mask_path,
            output_frame_path=frame_path,
        )
        return AddPromptResponse(
            sessionId=payload.sessionId,
            frameIndex=payload.frameIndex,
            promptCount=len(state.prompts),
            frameUrl=self.project_service.storage.artifact_url(self.settings.public_base_url, frame_path),
            maskUrl=self.project_service.storage.artifact_url(self.settings.public_base_url, mask_path),
        )

    def propagate(self, payload: PropagateRequest) -> PropagateResponse:
        state = self.sessions.get(payload.sessionId)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        project_dir = self.project_service.storage.project_dir(state.project_id)
        mask_video_path = project_dir / "mask_sequence.mp4"
        metadata = self.runtime.propagate(state, mask_video_path)
        return PropagateResponse(
            sessionId=payload.sessionId,
            frameCount=metadata.frame_count,
            maskVideoUrl=self.project_service.storage.artifact_url(self.settings.public_base_url, mask_video_path),
        )

    def require_mask_video(self, session_id: str) -> tuple[SessionRuntimeState, Path]:
        state = self.sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        mask_video_path = self.project_service.storage.project_dir(state.project_id) / "mask_sequence.mp4"
        if not mask_video_path.exists():
            raise HTTPException(status_code=400, detail="Mask video not found. Run propagation first.")
        return state, mask_video_path
