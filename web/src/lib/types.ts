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
  voidEnvName?: string | null;
  pythonVersion?: string | null;
  cudaBackend?: string | null;
  samFa3Status?: string | null;
  samFa3Note?: string | null;
  storageRoot?: string | null;
  dataDir?: string | null;
  projectsDir?: string | null;
  modelsDir?: string | null;
  bootstrapStatePath?: string | null;
  hfHome?: string | null;
  hfHubCache?: string | null;
  pipCacheDir?: string | null;
  mambaRootPrefix?: string | null;
  condaEnvsPath?: string | null;
  condaPkgsDirs?: string | null;
  lastValidatedAt?: string | null;
  error?: string | null;
};

export type RemovalPipelineCapability = {
  id: "effecterase" | "void";
  label: string;
  envReady: boolean;
  assetsReady: boolean;
  geminiConfigured: boolean;
  lazyModels: boolean;
  downloadable: boolean;
  selectable: boolean;
  downloadInProgress: boolean;
  activeJobId?: string | null;
};

export type BackendCapabilities = {
  cudaAvailable: boolean;
  samModels: string[];
  removalPipelines: RemovalPipelineCapability[];
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
  maskOverlayUrl: string;
};

export type JobResponse = {
  jobId: string;
  projectId?: string | null;
  kind: "remove" | "model_download";
  pipeline: "effecterase" | "void";
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage?: string | null;
  message?: string | null;
  resultUrl?: string | null;
  error?: string | null;
};
