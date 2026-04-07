import type {
  AddPromptResponse,
  BackendCapabilities,
  BootstrapStatus,
  CreateProjectResponse,
  JobResponse,
  PromptPoint,
  PropagateResponse,
  StartSessionResponse,
  UploadVideoResponse,
} from "./types";

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    try {
      const parsed = JSON.parse(detail) as { detail?: string };
      throw new Error(parsed.detail || detail || `Request failed with ${response.status}`);
    } catch {
      throw new Error(detail || `Request failed with ${response.status}`);
    }
  }
  return response.json() as Promise<T>;
}

export function fetchBootstrapStatus(baseUrl: string): Promise<BootstrapStatus> {
  return request(baseUrl, "/bootstrap/status");
}

export function fetchCapabilities(baseUrl: string): Promise<BackendCapabilities> {
  return request(baseUrl, "/capabilities");
}

export function createProject(baseUrl: string, profileId: string, label?: string): Promise<CreateProjectResponse> {
  return request(baseUrl, "/projects", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ profileId, label }),
  });
}

export async function uploadVideo(baseUrl: string, projectId: string, file: File): Promise<UploadVideoResponse> {
  const form = new FormData();
  form.append("file", file);
  return request(baseUrl, `/projects/${projectId}/video`, {
    method: "POST",
    body: form,
  });
}

export function startSession(baseUrl: string, projectId: string, model: string): Promise<StartSessionResponse> {
  return request(baseUrl, "/sam/start-session", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ projectId, model }),
  });
}

export function addPrompt(
  baseUrl: string,
  sessionId: string,
  frameIndex: number,
  points: PromptPoint[],
): Promise<AddPromptResponse> {
  return request(baseUrl, "/sam/add-prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sessionId, frameIndex, points }),
  });
}

export function propagate(baseUrl: string, sessionId: string): Promise<PropagateResponse> {
  return request(baseUrl, "/sam/propagate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sessionId }),
  });
}

export function removeObject(baseUrl: string, projectId: string, sessionId: string): Promise<JobResponse> {
  return removeObjectWithPipeline(baseUrl, projectId, sessionId, "effecterase");
}

export function removeObjectWithPipeline(
  baseUrl: string,
  projectId: string,
  sessionId: string,
  pipeline: "effecterase" | "void",
  backgroundPrompt?: string,
): Promise<JobResponse> {
  return request(baseUrl, "/remove", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ projectId, sessionId, pipeline, backgroundPrompt }),
  });
}

export function downloadPipelineModels(baseUrl: string, pipelineId: string): Promise<JobResponse> {
  return request(baseUrl, `/pipelines/${pipelineId}/download-models`, {
    method: "POST",
  });
}

export function fetchJob(baseUrl: string, jobId: string): Promise<JobResponse> {
  return request(baseUrl, `/jobs/${jobId}`);
}
