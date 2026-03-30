from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
import torch

from app.models.runtime import available_sam_models, describe_runtime_availability
from app.schemas.api import (
    AddPromptRequest,
    AddPromptResponse,
    BackendCapabilities,
    BootstrapStatus,
    CreateProjectResponse,
    JobResponse,
    ProjectCreateRequest,
    PropagateRequest,
    PropagateResponse,
    RemoveRequest,
    StartSessionRequest,
    StartSessionResponse,
    UploadVideoResponse,
)
from app.services.jobs import JobService
from app.services.projects import ProjectService
from app.services.sessions import SessionService

router = APIRouter()


def get_project_service(request: Request) -> ProjectService:
    return request.app.state.project_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/capabilities", response_model=BackendCapabilities)
def capabilities(request: Request) -> BackendCapabilities:
    settings = request.app.state.settings
    availability = describe_runtime_availability(settings, request.app.state.project_service.bootstrap_status)
    sam_models = available_sam_models(settings)
    return BackendCapabilities(
        cudaAvailable=torch.cuda.is_available(),
        samModels=sam_models if sam_models or availability["runtimeMode"] == "mock" else [],
        effectEraseAvailable=availability["effectEraseReady"] or availability["runtimeMode"] == "mock",
        envMode=availability["envMode"],
        maxWindowFrames=settings.max_window_frames,
        defaultResolution={
            "width": settings.default_width,
            "height": settings.default_height,
        },
    )


@router.get("/bootstrap/status", response_model=BootstrapStatus)
def bootstrap_status(project_service: ProjectService = Depends(get_project_service)) -> BootstrapStatus:
    return project_service.bootstrap_status


@router.post("/bootstrap/ensure", response_model=BootstrapStatus)
def bootstrap_ensure(project_service: ProjectService = Depends(get_project_service)) -> BootstrapStatus:
    return project_service.refresh_bootstrap_status()


@router.post("/projects", response_model=CreateProjectResponse)
def create_project(
    payload: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service),
) -> CreateProjectResponse:
    return project_service.create_project(payload)


@router.post("/projects/{project_id}/video", response_model=UploadVideoResponse)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    project_service: ProjectService = Depends(get_project_service),
) -> UploadVideoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")
    return await project_service.save_upload(project_id, file)


@router.post("/sam/start-session", response_model=StartSessionResponse)
def start_session(
    payload: StartSessionRequest,
    session_service: SessionService = Depends(get_session_service),
) -> StartSessionResponse:
    return session_service.start_session(payload)


@router.post("/sam/add-prompt", response_model=AddPromptResponse)
def add_prompt(
    payload: AddPromptRequest,
    session_service: SessionService = Depends(get_session_service),
) -> AddPromptResponse:
    return session_service.add_prompt(payload)


@router.post("/sam/propagate", response_model=PropagateResponse)
def propagate(
    payload: PropagateRequest,
    session_service: SessionService = Depends(get_session_service),
) -> PropagateResponse:
    return session_service.propagate(payload)


@router.post("/remove", response_model=JobResponse)
def remove(
    payload: RemoveRequest,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service),
) -> JobResponse:
    return job_service.create_removal_job(payload, background_tasks)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, job_service: JobService = Depends(get_job_service)) -> JobResponse:
    return job_service.get_job(job_id)
