# EffectErase App

Internal browser UI for interactive video object removal with a TypeScript frontend and a Python GPU worker.

The intended product flow is:

1. Upload a video in the browser.
2. Pick a backend target: local CUDA machine, Tailscale-connected GPU machine, or Runpod Pod.
3. Click an object on a frame to create a mask preview.
4. Propagate the mask through the clip.
5. Run object removal with EffectErase.

The repo is already structured for that workflow, but the current runtime is still partly mocked so the system can run before gated checkpoints and large model weights are installed.

## Current status

What works today:

- a React + TypeScript UI for upload, backend selection, prompt placement, mask preview, propagation, and removal job polling
- a FastAPI worker with project/session/job endpoints
- local artifact storage under `data/projects/`
- Conda/micromamba bootstrap scripts for local, Tailscale, and Runpod setups
- a mock SAM flow that generates a synthetic preview mask from prompt clicks
- a mock removal flow that uses OpenCV inpainting to create a result video

What is not wired to real model inference yet:

- real SAM 3 / SAM 3.1 predictor calls
- real EffectErase inference
- checkpoint download and weight-path management
- authenticated model access for gated upstream assets

So this repo is currently a working application scaffold and operator workflow, not yet a finished model-integrated product.

## Repo layout

- `web/`
  React + TypeScript frontend built with Vite.
- `worker/`
  FastAPI worker package, including routes, services, schemas, and mock model runtimes.
- `scripts/`
  Setup and startup scripts for Conda and micromamba-based worker environments.
- `config/`
  Example backend profile definitions.
- `data/`
  Local runtime state, project artifacts, and bootstrap metadata.
- `third_party/`
  Created by setup scripts when cloning upstream repos. Ignored by Git.

Important files:

- [`web/src/App.tsx`](./web/src/App.tsx)
  Main browser workflow.
- [`worker/app/api/routes.py`](./worker/app/api/routes.py)
  Worker HTTP API.
- [`worker/app/models/runtime.py`](./worker/app/models/runtime.py)
  Current mock SAM and mock removal implementations.
- [`scripts/setup-worker.sh`](./scripts/setup-worker.sh)
  Env creation and dependency bootstrap.
- [`scripts/start-worker.sh`](./scripts/start-worker.sh)
  Worker startup entrypoint.
- [`config/backend-profiles.example.json`](./config/backend-profiles.example.json)
  Example worker profile values.

## Architecture

The system is intentionally split into two runtimes:

- TypeScript UI
  Runs in the browser, drives upload, prompting, preview, propagation, and job status.
- Python worker
  Runs on the GPU host, owns project/session/job APIs, and will eventually own the real SAM and EffectErase model execution.

This split is deliberate because both upstream model stacks are Python-native.

## Supported deployment modes

### Local CUDA machine

- The web UI runs locally.
- The Python worker runs on the same machine.
- Env manager: `conda`

### Tailscale-connected GPU machine

- The web UI runs on your local machine.
- The Python worker runs on the remote CUDA host.
- The worker is reached over the tailnet hostname and private port.
- Env manager: `conda`

### Runpod Pod

- You SSH into the Pod.
- You clone this repo manually.
- You run the setup/start scripts manually on the Pod.
- Env manager: `micromamba`

This repo does not currently build or rely on a custom Docker image.

## Environment management

The worker bootstrap is script-driven and tries to minimize repeated manual setup.

### Default strategy

The setup script attempts a shared environment first:

- env name: `effecterase-worker`
- Python version: `3.12`
- installs PyTorch first
- installs the local worker package
- clones and installs upstream `sam3`
- clones and installs upstream `EffectErase`

If that fails and the strategy is `shared-first`, the script falls back to split envs:

- `effecterase-sam`
- `effecterase-remove`

Bootstrap status is written to:

- `data/bootstrap-status.json`

### Local and Tailscale behavior

The scripts expect `conda` to already be installed and available on `PATH`.

Commands are executed through:

```bash
conda run --no-capture-output -n <env> ...
```

### Runpod behavior

The scripts can install `micromamba` if it is not already available.

Commands are executed through:

```bash
micromamba run -n <env> ...
```

This is just repo bootstrap behavior, not a special Runpod provisioning hook.

## Setup requirements

### General

You should assume these are required on the machine hosting the worker:

- Linux
- Git
- curl
- ffmpeg support through Conda or micromamba bootstrap
- a CUDA-capable GPU for real model execution
- internet access during first bootstrap

### For local and Tailscale workers

- Conda already installed
- a working CUDA/PyTorch-compatible environment on the host

### For Runpod

- a Pod you can SSH into
- enough disk space for:
  - this repo
  - cloned upstream repos
  - model weights
  - uploaded project media

## Quick start

### Start the worker locally or on a Tailscale host

```bash
./scripts/start-worker.sh --env-manager conda
```

### Start the worker on a Runpod Pod

```bash
git clone https://github.com/jquintanilla4/effect-erase.git
cd effect-erase
./scripts/start-worker.sh --env-manager micromamba
```

### Start the web app

```bash
cd web
npm install
npm run dev
```

By default, the browser app expects the worker at:

```text
http://localhost:8000
```

## Bootstrap script behavior

The setup script:

- creates `data/projects/` and `third_party/`
- detects the env manager or uses the one you explicitly pass
- installs `micromamba` in user space when required
- creates envs if missing
- upgrades `pip`, `setuptools`, and `wheel`
- installs `torch` and `torchvision` from the chosen PyTorch index
- installs the local worker package
- clones:
  - `facebookresearch/sam3`
  - `FudanCVL/EffectErase`
- attempts a shared env first unless configured otherwise
- writes bootstrap metadata to `data/bootstrap-status.json`

### Script options

`setup-worker.sh`:

```bash
./scripts/setup-worker.sh --env-manager conda|micromamba|auto --strategy shared-first|shared|split --cuda-backend cu128
```

`start-worker.sh`:

```bash
./scripts/start-worker.sh --env-manager conda|micromamba --host 0.0.0.0 --port 8000
```

### Relevant environment variables

- `PYTHON_VERSION`
- `CUDA_BACKEND`
- `TORCH_INDEX_URL`
- `ENV_MANAGER`
- `ENV_STRATEGY`
- `SHARED_ENV_NAME`
- `SAM_ENV_NAME`
- `REMOVE_ENV_NAME`
- `WORKER_HOST`
- `WORKER_PORT`
- `SAM3_REPO_URL`
- `EFFECTERASE_REPO_URL`

Defaults are currently defined directly in [`scripts/setup-worker.sh`](./scripts/setup-worker.sh).

## Backend profiles

The current UI includes three built-in profiles:

- `local`
- `tailscale`
- `runpod-pod`

Example values live in [`config/backend-profiles.example.json`](./config/backend-profiles.example.json).

The frontend currently ships with hardcoded defaults in [`web/src/App.tsx`](./web/src/App.tsx), backed by these optional Vite env vars:

- `VITE_LOCAL_WORKER_URL`
- `VITE_TAILSCALE_WORKER_URL`
- `VITE_RUNPOD_WORKER_URL`

If you do nothing, the defaults are:

- local: `http://localhost:8000`
- tailscale: `http://gpu-box.tailnet-name.ts.net:8000`
- runpod: `https://pod-id-8000.proxy.runpod.net`

You can also override the worker URL directly in the UI.

## Current worker API

Health and bootstrap:

- `GET /health`
- `GET /capabilities`
- `GET /bootstrap/status`
- `POST /bootstrap/ensure`

Project lifecycle:

- `POST /projects`
- `POST /projects/{project_id}/video`

SAM session lifecycle:

- `POST /sam/start-session`
- `POST /sam/add-prompt`
- `POST /sam/propagate`

Removal jobs:

- `POST /remove`
- `GET /jobs/{job_id}`

Artifacts are served under:

- `/artifacts/...`

## Browser workflow

The browser app currently does this:

1. Select a backend profile.
2. Upload a video file.
3. Create a project and start a SAM session automatically.
4. Scrub to a frame index.
5. Place positive or negative click prompts on the video canvas.
6. Preview the generated mask overlay.
7. Propagate the mask sequence.
8. Start a removal job.
9. Poll job progress until a result video is available.

## Current mock behavior

The current implementation is intentionally usable before real model installs are complete.

### Mock SAM runtime

The mock SAM runtime:

- reads the selected source frame
- converts normalized click points into pixel coordinates
- draws positive circles into a binary mask
- uses negative clicks to erase areas
- saves:
  - the selected frame image
  - the preview mask image
- propagates the same mask across every frame of the clip
- writes a binary mask video as `mask_sequence.mp4`

### Mock removal runtime

The mock removal runtime:

- opens the source video and mask video
- thresholds the mask frames
- runs OpenCV `cv2.inpaint(...)` frame-by-frame
- writes the output video as `removed_output.mp4`
- reports progress through the job service

This behavior is a placeholder for real EffectErase inference.

## Real model integration plan

The worker structure is already arranged so mock runtimes can be replaced by real ones.

What is already prepared:

- upstream source checkout under `third_party/`
- env strategy metadata
- worker/session/job boundaries
- artifact storage and URL generation
- backend profile separation

What still needs to be implemented:

- real SAM session startup using the upstream predictor APIs
- real prompt-to-mask preview for the selected frame
- real propagation and mask sequence generation
- real EffectErase inference on source clip plus mask video
- model weight discovery and configuration
- optional chunking/stitching logic for longer videos if the upstream inference path requires it

## Model and checkpoint requirements

For real model execution, you should expect to provide:

- access to SAM 3 / SAM 3.1 checkpoints
- any Hugging Face auth required by the SAM stack
- EffectErase weights
- Wan model assets required by EffectErase
- any CUDA-specific PyTorch index changes for the target machine

The current repo does not yet define a final checkpoint directory layout. That should be added when the real runtimes are wired in.

## Limitations

Current limitations of the repo:

- real SAM is not connected yet
- real EffectErase is not connected yet
- worker state is file-based and in-memory, not database-backed
- UI backend profiles are not yet loaded dynamically from disk
- auth is not implemented; this is an internal tool scaffold
- multi-object workflows are not implemented
- trimming, chunked inference, and resolution-preserving output are not implemented yet

## Development notes

### Python validation

Basic syntax validation:

```bash
python3 -m compileall worker/app
```

### Shell validation

```bash
zsh -n scripts/setup-worker.sh scripts/start-worker.sh scripts/start-web.sh
```

### Frontend

Standard Vite workflow:

```bash
cd web
npm install
npm run dev
```

## Git ignore policy

The repo ignores:

- generated project artifacts under `data/projects/`
- bootstrap state
- node modules and frontend build output
- Python cache and virtualenv directories
- cloned upstream repos under `third_party/`
- common editor and local env files

See [`.gitignore`](./.gitignore).

## Troubleshooting

### `gh` or GitHub access is unrelated to worker startup

GitHub auth is only needed to clone this repo or upstream repos. It is not needed for the running app.

### `conda` is not found

Make sure Conda is installed and available on `PATH` before running:

```bash
./scripts/start-worker.sh --env-manager conda
```

### `micromamba` is not found on Runpod

That is expected on some base Pods. The setup script installs it automatically into user space.

### Shared env bootstrap fails

The script should fall back to split envs automatically when using the default `shared-first` strategy.

### The UI cannot reach the worker

Check:

- worker host and port
- Tailscale hostname and access
- Runpod proxy URL
- browser CORS/network reachability

### The app runs but the model output looks fake

That is expected right now. The repo currently uses mock runtimes until real SAM and EffectErase integration is added.

## Next implementation priorities

The highest-value next steps are:

1. Replace the mock SAM runtime with real SAM 3.1 session handling.
2. Replace the OpenCV inpainting runtime with real EffectErase inference.
3. Add model path configuration and checkpoint validation.
4. Add a dedicated operator setup section once the real weight layout is finalized.
