from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BootstrapStatus(BaseModel):
    status: str
    envManager: str
    envNames: list[str]
    activeStrategy: str
    workerEnvName: str | None = None
    samEnvName: str | None = None
    removeEnvName: str | None = None
    pythonVersion: str | None = None
    cudaBackend: str | None = None
    samFa3Status: str | None = None
    samFa3Note: str | None = None
    lastValidatedAt: str | None = None
    error: str | None = None


class BackendCapabilities(BaseModel):
    cudaAvailable: bool
    samModels: list[str]
    effectEraseAvailable: bool
    envMode: str
    maxWindowFrames: int
    defaultResolution: dict[str, int]


class ProjectCreateRequest(BaseModel):
    profileId: str
    label: str | None = None


class CreateProjectResponse(BaseModel):
    projectId: str
    profileId: str
    label: str | None = None
    projectUrl: str


class UploadVideoResponse(BaseModel):
    projectId: str
    sourceUrl: str
    width: int
    height: int
    fps: float
    frameCount: int


class StartSessionRequest(BaseModel):
    projectId: str
    model: str = "sam3.1"


class StartSessionResponse(BaseModel):
    sessionId: str
    projectId: str
    model: str
    frameCount: int
    fps: float
    width: int
    height: int


class PromptPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    label: Literal["positive", "negative"]


class AddPromptRequest(BaseModel):
    sessionId: str
    frameIndex: int
    points: list[PromptPoint]


class AddPromptResponse(BaseModel):
    sessionId: str
    frameIndex: int
    promptCount: int
    frameUrl: str
    maskUrl: str


class PropagateRequest(BaseModel):
    sessionId: str


class PropagateResponse(BaseModel):
    sessionId: str
    frameCount: int
    maskVideoUrl: str


class RemoveRequest(BaseModel):
    projectId: str
    sessionId: str


class JobResponse(BaseModel):
    jobId: str
    projectId: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    resultUrl: str | None = None
    error: str | None = None
