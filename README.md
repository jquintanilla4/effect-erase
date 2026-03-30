# EffectErase App

Internal browser UI for interactive video object removal using a TypeScript frontend and a Python GPU worker.

## Architecture

- `web/`: React + TypeScript app for upload, prompt placement, mask preview, propagation, and removal job status.
- `worker/`: FastAPI service that stores projects, exposes worker APIs, and runs mock or real model backends.
- `scripts/`: setup and start scripts for local CUDA, Tailscale-connected hosts, and Runpod Pods.
- `config/`: example backend profile definitions.

## Runtime model

- Local and Tailscale machines use `conda`.
- Runpod Pods use `micromamba`.
- The setup script attempts a shared env first and falls back to split envs if upstream installs conflict.
- The worker can run immediately with mock SAM and removal runtimes.
- Real SAM 3 / 3.1 and EffectErase integration is prepared through the setup scripts and runtime interfaces.

## Quick start

### 1. Start the worker

Local or Tailscale host:

```bash
./scripts/start-worker.sh --env-manager conda
```

Runpod Pod:

```bash
git clone <this repo>
cd effect-erase
./scripts/start-worker.sh --env-manager micromamba
```

The script:

- creates the env if it does not exist
- reuses the env if validation passes
- clones `facebookresearch/sam3` and `FudanCVL/EffectErase` into `third_party/`
- installs worker dependencies
- writes `data/bootstrap-status.json`
- starts `uvicorn`

### 2. Start the web app

```bash
cd web
npm install
npm run dev
```

By default the UI expects the worker at `http://localhost:8000`.

## Worker behavior

The current implementation includes a functional mock workflow:

- clicking on a frame generates a synthetic mask preview
- propagation writes a binary mask video aligned with the source clip
- removal runs frame-by-frame OpenCV inpainting to produce a result video

This keeps the app testable before SAM checkpoints and EffectErase weights are installed.

## Real model integration

The repo is structured so that real model runtimes can replace the mock adapters. Setup already prepares:

- upstream source checkout under `third_party/`
- separate env metadata when a shared env install fails
- bootstrap state surfaced at `GET /bootstrap/status`

You still need to provide:

- Hugging Face access for SAM checkpoints
- EffectErase model weights and Wan dependencies
- any environment-specific CUDA/PyTorch index configuration

## Useful endpoints

- `GET /health`
- `GET /capabilities`
- `GET /bootstrap/status`
- `POST /projects`
- `POST /projects/{project_id}/video`
- `POST /sam/start-session`
- `POST /sam/add-prompt`
- `POST /sam/propagate`
- `POST /remove`
- `GET /jobs/{job_id}`

