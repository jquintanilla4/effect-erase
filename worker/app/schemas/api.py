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
    voidEnvName: str | None = None
    pythonVersion: str | None = None
    cudaBackend: str | None = None
    samFa3Status: str | None = None
    samFa3Note: str | None = None
    storageRoot: str | None = None
    dataDir: str | None = None
    projectsDir: str | None = None
    modelsDir: str | None = None
    bootstrapStatePath: str | None = None
    hfHome: str | None = None
    hfHubCache: str | None = None
    pipCacheDir: str | None = None
    mambaRootPrefix: str | None = None
    condaEnvsPath: str | None = None
    condaPkgsDirs: str | None = None
    lastValidatedAt: str | None = None
    error: str | None = None


class RemovalPipelineCapability(BaseModel):
    id: Literal["effecterase", "void"]
    label: str
    envReady: bool
    assetsReady: bool
    geminiConfigured: bool = False
    lazyModels: bool
    downloadable: bool
    selectable: bool
    downloadInProgress: bool = False
    activeJobId: str | None = None


class BackendCapabilities(BaseModel):
    cudaAvailable: bool
    samModels: list[str]
    removalPipelines: list[RemovalPipelineCapability]
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
    maskOverlayUrl: str


class RemoveRequest(BaseModel):
    projectId: str
    sessionId: str
    pipeline: Literal["effecterase", "void"] = "effecterase"
    backgroundPrompt: str | None = None


class JobResponse(BaseModel):
    jobId: str
    projectId: str | None = None
    kind: Literal["remove", "model_download"]
    pipeline: Literal["effecterase", "void"]
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    stage: str | None = None
    message: str | None = None
    resultUrl: str | None = None
    error: str | None = None
