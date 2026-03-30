from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.services.jobs import JobService
from app.services.projects import ProjectService
from app.services.sessions import SessionService


settings = get_settings()
project_service = ProjectService(settings)
session_service = SessionService(settings, project_service)
job_service = JobService(settings, project_service, session_service)

app = FastAPI(title="EffectErase Worker", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.settings = settings
app.state.project_service = project_service
app.state.session_service = session_service
app.state.job_service = job_service
app.include_router(router)
app.mount("/artifacts", StaticFiles(directory=settings.projects_dir), name="artifacts")

