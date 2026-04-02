import { useEffect, useMemo, useRef, useState } from "react";

import {
  addPrompt,
  createProject,
  fetchBootstrapStatus,
  fetchCapabilities,
  fetchJob,
  propagate,
  removeObject,
  startSession,
  uploadVideo,
} from "./lib/api";
import type {
  AddPromptResponse,
  BackendCapabilities,
  BackendProfile,
  BootstrapStatus,
  JobResponse,
  PromptPoint,
  StartSessionResponse,
  UploadVideoResponse,
} from "./lib/types";

const DEFAULT_PROFILES: BackendProfile[] = [
  {
    id: "local",
    label: "Local CUDA",
    type: "local",
    baseUrl: import.meta.env.VITE_LOCAL_WORKER_URL ?? "http://localhost:8000",
    envManager: "conda",
    defaultSamModel: "sam3.1",
    cudaBackend: "cu128",
  },
  {
    id: "tailscale",
    label: "Tailscale 4090",
    type: "tailscale",
    baseUrl: import.meta.env.VITE_TAILSCALE_WORKER_URL ?? "http://gpu-box.tailnet-name.ts.net:8000",
    envManager: "conda",
    defaultSamModel: "sam3.1",
    cudaBackend: "cu128",
  },
  {
    id: "runpod-pod",
    label: "Runpod Pod",
    type: "runpod-pod",
    baseUrl: import.meta.env.VITE_RUNPOD_WORKER_URL ?? "https://pod-id-8000.proxy.runpod.net",
    envManager: "micromamba",
    defaultSamModel: "sam3.1",
    cudaBackend: "cu128",
  },
];

function withCacheBust(url: string | null | undefined, version: string | number | null | undefined): string | null {
  if (!url) {
    return null;
  }

  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${version ?? "0"}`;
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  const [profiles] = useState(DEFAULT_PROFILES);
  const [selectedProfileId, setSelectedProfileId] = useState(DEFAULT_PROFILES[0].id);
  const [baseUrlOverride, setBaseUrlOverride] = useState("");
  const [bootstrapStatus, setBootstrapStatus] = useState<BootstrapStatus | null>(null);
  const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
  const [selectedSamModel, setSelectedSamModel] = useState(DEFAULT_PROFILES[0].defaultSamModel);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [uploadInfo, setUploadInfo] = useState<UploadVideoResponse | null>(null);
  const [sourceObjectUrl, setSourceObjectUrl] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [session, setSession] = useState<StartSessionResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [currentPointLabel, setCurrentPointLabel] = useState<PromptPoint["label"]>("positive");
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [promptPreview, setPromptPreview] = useState<AddPromptResponse | null>(null);
  const [maskVideoUrl, setMaskVideoUrl] = useState<string | null>(null);
  const [maskOverlayUrl, setMaskOverlayUrl] = useState<string | null>(null);
  const [maskVideoVersion, setMaskVideoVersion] = useState(0);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.id === selectedProfileId) ?? profiles[0],
    [profiles, selectedProfileId],
  );
  const workerUrl = baseUrlOverride.trim() || selectedProfile.baseUrl;
  const promptMaskUrl = useMemo(
    () => withCacheBust(promptPreview?.maskUrl, `${promptPreview?.frameIndex ?? 0}-${promptPreview?.promptCount ?? 0}`),
    [promptPreview?.frameIndex, promptPreview?.maskUrl, promptPreview?.promptCount],
  );
  const propagatedMaskVideoUrl = useMemo(
    () => withCacheBust(maskVideoUrl, maskVideoVersion),
    [maskVideoUrl, maskVideoVersion],
  );
  const propagatedMaskOverlayUrl = useMemo(
    () => withCacheBust(maskOverlayUrl, maskVideoVersion),
    [maskOverlayUrl, maskVideoVersion],
  );
  const removalProgressPercent = useMemo(() => {
    if (!job) {
      return 0;
    }
    // The worker owns progress; the UI only clamps for display safety.
    return Math.max(0, Math.min(100, Math.round(job.progress * 100)));
  }, [job]);

  useEffect(() => {
    let ignore = false;
    async function loadStatus() {
      try {
        const [bootstrap, caps] = await Promise.all([
          fetchBootstrapStatus(workerUrl),
          fetchCapabilities(workerUrl),
        ]);
        if (!ignore) {
          setBootstrapStatus(bootstrap);
          setCapabilities(caps);
          setError(null);
        }
      } catch (err) {
        if (!ignore) {
          setBootstrapStatus(null);
          setCapabilities(null);
          setError(err instanceof Error ? err.message : "Failed to contact worker.");
        }
      }
    }
    loadStatus();
    return () => {
      ignore = true;
    };
  }, [workerUrl]);

  useEffect(() => {
    const availableModels = capabilities?.samModels ?? [];
    const preferredModel = availableModels.includes(selectedProfile.defaultSamModel)
      ? selectedProfile.defaultSamModel
      : availableModels[0] ?? selectedProfile.defaultSamModel;

    setSelectedSamModel((currentModel) => {
      if (availableModels.length === 0) {
        return selectedProfile.defaultSamModel;
      }
      if (availableModels.includes(currentModel)) {
        return currentModel;
      }
      return preferredModel;
    });
  }, [capabilities?.samModels, selectedProfile.defaultSamModel]);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const next = await fetchJob(workerUrl, job.jobId);
        setJob(next);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to refresh job status.");
      }
    }, 1500);

    return () => window.clearInterval(timer);
  }, [job, workerUrl]);

  useEffect(() => {
    if (!uploadInfo || !videoRef.current) {
      return;
    }
    const video = videoRef.current;
    const fps = uploadInfo.fps || 24;
    video.currentTime = frameIndex / fps;
  }, [frameIndex, uploadInfo]);

  useEffect(() => {
    return () => {
      if (sourceObjectUrl) {
        URL.revokeObjectURL(sourceObjectUrl);
      }
    };
  }, [sourceObjectUrl]);

  async function handleUpload() {
    if (!videoFile) {
      setError("Choose a video file first.");
      return;
    }
    if (!selectedSamModel) {
      setError("Choose a SAM model before starting a session.");
      return;
    }

    setBusy(true);
    setError(null);

    try {
      const project = await createProject(workerUrl, selectedProfile.id, videoFile.name);
      const uploaded = await uploadVideo(workerUrl, project.projectId, videoFile);
      const startedSession = await startSession(workerUrl, project.projectId, selectedSamModel);
      setProjectId(project.projectId);
      setUploadInfo(uploaded);
      setSession(startedSession);
      setFrameIndex(0);
      setPoints([]);
      setPromptPreview(null);
      setMaskVideoUrl(null);
      setMaskOverlayUrl(null);
      setMaskVideoVersion(0);
      setJob(null);

      if (sourceObjectUrl) {
        URL.revokeObjectURL(sourceObjectUrl);
      }
      setSourceObjectUrl(URL.createObjectURL(videoFile));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload video.");
    } finally {
      setBusy(false);
    }
  }

  async function handleOverlayClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!session || !overlayRef.current) {
      return;
    }

    const bounds = overlayRef.current.getBoundingClientRect();
    const point: PromptPoint = {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
      label: currentPointLabel,
    };

    const nextPoints = [...points, point];
    setPoints(nextPoints);
    setBusy(true);
    setError(null);

    try {
      const response = await addPrompt(workerUrl, session.sessionId, frameIndex, nextPoints);
      // A new prompt changes the tracked object state, so any previous propagated
      // sequence is stale and should not be previewed or reused for removal.
      setMaskVideoUrl(null);
      setMaskOverlayUrl(null);
      setMaskVideoVersion(0);
      setPromptPreview(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add prompt.");
    } finally {
      setBusy(false);
    }
  }

  async function handlePropagate() {
    if (!session) {
      setError("Create a session before propagating.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const response = await propagate(workerUrl, session.sessionId);
      setMaskVideoUrl(response.maskVideoUrl);
      setMaskOverlayUrl(response.maskOverlayUrl);
      // The worker overwrites the same artifact path, so the player URL needs a
      // fresh cache-busting token every time propagation succeeds.
      setMaskVideoVersion((currentVersion) => currentVersion + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to propagate mask.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    if (!session || !projectId) {
      setError("Upload a video and create a session before removal.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const response = await removeObject(workerUrl, projectId, session.sessionId);
      setJob(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start removal job.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-shell">
      <aside className="left-rail">
        <div className="brand">
          <p className="eyebrow">Internal Tooling</p>
          <h1>EffectErase Studio</h1>
          <p className="lede">
            Upload a clip, click the object, preview the mask, propagate it, and run removal on the selected GPU worker.
          </p>
        </div>

        <section className="panel">
          <h2>Backend</h2>
          <label>
            Profile
            <select value={selectedProfileId} onChange={(event) => setSelectedProfileId(event.target.value)}>
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Worker URL
            <input
              value={baseUrlOverride}
              onChange={(event) => setBaseUrlOverride(event.target.value)}
              placeholder={selectedProfile.baseUrl}
            />
          </label>

          <div className="status-grid">
            <StatusPill label="Env" value={bootstrapStatus?.envManager ?? "unknown"} />
            <StatusPill label="Strategy" value={bootstrapStatus?.activeStrategy ?? "unknown"} />
            <StatusPill label="CUDA" value={capabilities?.cudaAvailable ? "ready" : "unknown"} />
            <StatusPill label="Window" value={String(capabilities?.maxWindowFrames ?? "—")} />
          </div>

          {bootstrapStatus?.envNames?.length ? (
            <p className="muted">Envs: {bootstrapStatus.envNames.join(", ")}</p>
          ) : null}
          {bootstrapStatus?.samFa3Status ? (
            <p className="muted">SAM FA3: {bootstrapStatus.samFa3Status}</p>
          ) : null}
          {bootstrapStatus?.samFa3Note ? <p className="muted">{bootstrapStatus.samFa3Note}</p> : null}
        </section>

        <section className="panel">
          <h2>Source Clip</h2>
          <label>
            SAM Model
            <select
              value={selectedSamModel}
              onChange={(event) => setSelectedSamModel(event.target.value)}
              disabled={busy || (capabilities?.samModels?.length ?? 0) === 0}
            >
              {(capabilities?.samModels ?? []).map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          <input
            type="file"
            accept="video/*"
            onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)}
          />
          <button onClick={handleUpload} disabled={busy || !videoFile || !selectedSamModel}>
            {busy ? "Working…" : "Upload And Start Session"}
          </button>

          {uploadInfo ? (
            <div className="meta-list">
              <div>Project: {projectId}</div>
              <div>
                Video: {uploadInfo.width}×{uploadInfo.height}
              </div>
              <div>
                Frames: {uploadInfo.frameCount} @ {uploadInfo.fps.toFixed(2)} fps
              </div>
              <div>Session: {session?.sessionId ?? "—"}</div>
              <div>Model: {session?.model ?? selectedSamModel}</div>
            </div>
          ) : null}
        </section>

        <section className="panel">
          <h2>Prompting</h2>
          <div className="segmented-control">
            <button
              className={currentPointLabel === "positive" ? "active" : ""}
              onClick={() => setCurrentPointLabel("positive")}
            >
              Positive
            </button>
            <button
              className={currentPointLabel === "negative" ? "active" : ""}
              onClick={() => setCurrentPointLabel("negative")}
            >
              Negative
            </button>
          </div>

          <label>
            Frame Index
            <input
              type="range"
              min={0}
              max={Math.max((uploadInfo?.frameCount ?? 1) - 1, 0)}
              value={frameIndex}
              onChange={(event) => setFrameIndex(Number(event.target.value))}
              disabled={!uploadInfo}
            />
          </label>
          <div className="muted">Selected frame: {frameIndex}</div>
          <div className="muted">Prompts placed: {points.length}</div>

          <button onClick={handlePropagate} disabled={busy || !session || points.length === 0}>
            Propagate Mask
          </button>
          <button onClick={handleRemove} disabled={busy || !maskVideoUrl}>
            Remove Object
          </button>
        </section>

        {job ? (
          <section className="panel">
            <h2>Removal Job</h2>
            <div className="meta-list">
              <div>Status: {job.status}</div>
              <div>Progress: {removalProgressPercent}%</div>
              {job.resultUrl ? (
                <a href={job.resultUrl} target="_blank" rel="noreferrer">
                  Open result
                </a>
              ) : null}
              {job.error ? <div className="error-text">{job.error}</div> : null}
            </div>
          </section>
        ) : null}

        {error ? <div className="error-banner">{error}</div> : null}
      </aside>

      <main className="workspace">
        <section className="stage panel">
          <div className="stage-header">
            <div>
              <p className="eyebrow">Mask Preview</p>
              <h2>Interactive canvas</h2>
            </div>
          </div>

          <div className="video-stack">
            {sourceObjectUrl ? (
              <>
                <video ref={videoRef} src={sourceObjectUrl} controls className="source-video" />
                <div ref={overlayRef} className="video-overlay" onClick={handleOverlayClick}>
                  {points.map((point, index) => (
                    <span
                      key={`${point.x}-${point.y}-${index}`}
                      className={`point point-${point.label}`}
                      style={{
                        left: `${point.x * 100}%`,
                        top: `${point.y * 100}%`,
                      }}
                    />
                  ))}
                  {promptMaskUrl ? (
                    <img className="mask-overlay" src={promptMaskUrl} alt="Mask overlay" />
                  ) : null}
                </div>
              </>
            ) : (
              <div className="empty-stage">Upload a video to start placing prompts.</div>
            )}
          </div>
        </section>

        <section className="gallery panel">
          <div className="gallery-card">
            <p className="eyebrow">Frame</p>
            {promptPreview?.frameUrl ? <img src={promptPreview.frameUrl} alt="Selected frame" /> : <div className="placeholder" />}
          </div>
          <div className="gallery-card">
            <p className="eyebrow">Mask</p>
            {promptMaskUrl ? <img src={promptMaskUrl} alt="Prompt mask" /> : <div className="placeholder" />}
          </div>
          <div className="gallery-card">
            <p className="eyebrow">Output</p>
            {job?.resultUrl ? (
              <video src={job.resultUrl} controls className="result-video" />
            ) : (
              <div className="placeholder">Removal output appears here.</div>
            )}
          </div>
        </section>

        {propagatedMaskVideoUrl ? (
          <section className="panel propagated-preview">
            <div className="stage-header">
              <div>
                <p className="eyebrow">Propagated Preview</p>
                <h2>Mask sequence playback</h2>
              </div>
              <a href={propagatedMaskVideoUrl} target="_blank" rel="noreferrer">
                Open mask video
              </a>
            </div>

            <p className="muted">
              Review the propagated mask sequence here before moving on to removal.
            </p>

            <div className="propagated-preview-frame">
              <video
                key={propagatedMaskOverlayUrl}
                src={propagatedMaskOverlayUrl!}
                controls
                playsInline
                preload="metadata"
                className="propagated-preview-video"
              />
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
