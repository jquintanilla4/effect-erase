export type BackendProfile = {
  id: string;
  label: string;
  type: "local" | "tailscale" | "runpod-pod";
  baseUrl: string;
  envManager: "conda" | "micromamba";
  defaultSamModel: string;
  cudaBackend: string;
};

export type BootstrapStatus = {
  status: string;
  envManager: string;
  envNames: string[];
  activeStrategy: string;
  workerEnvName?: string | null;
  samEnvName?: string | null;
  removeEnvName?: string | null;
  pythonVersion?: string | null;
  cudaBackend?: string | null;
  lastValidatedAt?: string | null;
  error?: string | null;
};

export type BackendCapabilities = {
  cudaAvailable: boolean;
  samModels: string[];
  effectEraseAvailable: boolean;
  envMode: string;
  maxWindowFrames: number;
  defaultResolution: {
    width: number;
    height: number;
  };
};

export type CreateProjectResponse = {
  projectId: string;
  profileId: string;
  label?: string | null;
  projectUrl: string;
};

export type UploadVideoResponse = {
  projectId: string;
  sourceUrl: string;
  width: number;
  height: number;
  fps: number;
  frameCount: number;
};

export type StartSessionResponse = {
  sessionId: string;
  projectId: string;
  model: string;
  frameCount: number;
  fps: number;
  width: number;
  height: number;
};

export type PromptPoint = {
  x: number;
  y: number;
  label: "positive" | "negative";
};

export type AddPromptResponse = {
  sessionId: string;
  frameIndex: number;
  promptCount: number;
  frameUrl: string;
  maskUrl: string;
};

export type PropagateResponse = {
  sessionId: string;
  frameCount: number;
  maskVideoUrl: string;
};

export type JobResponse = {
  jobId: string;
  projectId: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  resultUrl?: string | null;
  error?: string | null;
};

