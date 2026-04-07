# EffectErase App

Internal browser UI for interactive video object removal with a TypeScript frontend and a Python GPU worker.

The intended product flow is:

1. Upload a video in the browser.
2. Pick a backend target: local CUDA machine, Tailscale-connected GPU machine, or Runpod Pod.
3. Click an object on a frame to create a mask preview.
4. Propagate the mask through the clip.
5. Run object removal with EffectErase.

## Current status

What works today:

- a React + TypeScript UI for upload, backend selection, prompt placement, mask preview, propagation, and removal job polling
- a FastAPI worker with project/session/job endpoints
- local artifact storage under `data/projects/`
- split-env bootstrap for the SAM stack and the EffectErase stack
- real SAM inference when model assets are present
- real EffectErase removal when model assets are present
- automatic fallback to mock runtimes when assets are missing and runtime mode is `auto`
- model downloads for SAM, EffectErase, and required Wan assets

What is still rough:

- EffectErase removal is currently wired for clips up to 81 frames
- worker state is file-based and in-memory, not database-backed
- multi-object workflows are not implemented yet

## Repo layout

- `web/`
  React + TypeScript frontend built with Vite.
- `worker/`
  FastAPI worker package, including routes, services, schemas, runtimes, and the internal EffectErase runner.
- `scripts/`
  Setup, asset download, and startup scripts for Conda and micromamba-based worker environments.
- `config/`
  Example backend profile definitions.
- `data/`
  Local runtime state, project artifacts, and bootstrap metadata.
- `models/`
  Downloaded checkpoints and model weights used by inference.

Important files:

- [`web/src/App.tsx`](./web/src/App.tsx)
  Main browser workflow.
- [`worker/app/api/routes.py`](./worker/app/api/routes.py)
  Worker HTTP API.
- [`worker/app/models/runtime.py`](./worker/app/models/runtime.py)
  SAM and EffectErase runtime selection and execution.
- [`worker/app/runners/effecterase_remove.py`](./worker/app/runners/effecterase_remove.py)
  Internal EffectErase removal runner.
- [`worker/app/core/config.py`](./worker/app/core/config.py)
  Runtime and model path configuration.
- [`scripts/setup-worker.sh`](./scripts/setup-worker.sh)
  Env creation and dependency bootstrap.
- [`scripts/verify-worker.sh`](./scripts/verify-worker.sh)
  Post-bootstrap readiness checks for CUDA, env imports, and model paths.
- [`scripts/download-model-assets.sh`](./scripts/download-model-assets.sh)
  Checkpoint and model asset download logic.
- [`scripts/clean-model-assets.sh`](./scripts/clean-model-assets.sh)
  Cleanup helper for incomplete model download artifacts before a retry.
- [`scripts/start-worker.sh`](./scripts/start-worker.sh)
  Worker startup entrypoint.

## Architecture

The system is intentionally split into two runtimes:

- TypeScript UI
  Runs in the browser, drives upload, prompting, preview, propagation, and job status.
- Python worker
  Runs on the GPU host, owns project/session/job APIs, and executes SAM and EffectErase inference.

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
- Bootstrap defaults its mutable runtime state to `/workspace/effect-erase-runtime`.
- Env manager: `micromamba`

For secrets, treat Runpod differently from local development:

- local development can use a repo-root `.env` file as a convenience
- Runpod should inject secrets as environment variables through Runpod secrets
- do not create or commit a `.env` file on the Pod

For the VOID pipeline, the worker already accepts `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, or `WORKER_GEMINI_API_KEY`. A typical Runpod template mapping
looks like:

```text
GEMINI_API_KEY={{ RUNPOD_SECRET_gemini_api_key }}
```

This repo does not currently build or rely on a custom Docker image.

## Environment management

The worker bootstrap is script-driven and defaults to split envs because the SAM stack and the EffectErase stack have conflicting dependency constraints.

Default envs:

- `effecterase-sam`
- `effecterase-remove`
- `effecterase-void`

The split setup:

- uses Python `3.12`
- installs the shared worker stack separately into each runtime env
- installs `sam3` and `sam2` only into `effecterase-sam`
- installs `EffectErase` only into `effecterase-remove`
- installs `VOID` runtime dependencies only into `effecterase-void`

Bootstrap status is written to the active worker data directory. By default:

- local or Tailscale: `data/bootstrap-status.json`
- Runpod: `/workspace/effect-erase-runtime/data/bootstrap-status.json`

### Repeat runs

`setup-worker.sh` now validates each env before reinstalling packages. If an env already satisfies its runtime import probe, the script short-circuits and reuses it instead of rebuilding it. Each runtime env is repaired in place, which keeps the dependency stacks isolated instead of trying to share one mutable worker environment across incompatible model runtimes.

Repeat runs keep the step log, but each env line now reports whether that env was reused, created, or repaired. The script also ends with an environment summary so an all-green rerun reads as "already ready" instead of sounding like a full fresh bootstrap.

Model downloads are also incremental. Existing checkpoint files under the active
worker models directory are reused.

Before any download work starts, `download-model-assets.sh` now prints a per-file
asset manifest showing which known model files are present, missing, or
incomplete, and writes the latest snapshot to the active models directory as
`asset-manifest.tsv`.

After env setup and any model download work completes, `setup-worker.sh` now runs `verify-worker.sh` automatically so bootstrap only reports ready when CUDA, runtime imports, and required local model assets all pass one explicit verification step.

If a download is interrupted and you want to remove partial artifacts before retrying, run `./scripts/clean-model-assets.sh`. Use `--dry-run` first if you want to preview what would be deleted.

## Setup requirements

### General

You should assume these are required on the machine hosting the worker:

- Linux
- Git
- curl
- internet access during first bootstrap
- enough disk space for:
  - this repo
  - Conda or micromamba envs
  - model weights
  - uploaded project media

### For real inference

- a CUDA-capable GPU
- a PyTorch-compatible NVIDIA driver
- Hugging Face access for gated `facebook/sam3.1` for the default bootstrap path

Default bootstrap does not auto-downgrade to SAM 2.1 when `sam3.1` auth is unavailable.
If you need the explicit SAM 2.1 path, manage assets manually or use the downloader's
`--skip-sam31 --include-sam21` options outside the default bootstrap flow.

### For local and Tailscale workers

- Conda/micromamba already installed
- a working CUDA host environment

If Conda is not installed yet, start with the official installation docs:

- [Installing conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
- [Micromamba installation](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html)

### For Runpod

- a Pod you can SSH into
- enough disk space for envs and model assets
- persistent `/workspace` storage if you want envs, caches, models, and uploads to survive Pod restarts
- Conda/micromamba alrady installed; see links above if not.

## Quick start

If you prefer shorter commands, the repo root includes a small `Makefile`.
These targets are optional convenience wrappers around the existing commands:

- `make bootstrap`
  Wraps `./scripts/setup-worker.sh --env-manager $(ENV_MANAGER)`
- `make verify`
  Wraps `./scripts/verify-worker.sh --env-manager $(ENV_MANAGER)`
- `make worker`
  Wraps `./scripts/start-worker.sh --env-manager $(ENV_MANAGER)`
- `make web`
  Runs `cd web && npm run dev` from the repo root

The Make targets keep the current script behavior intact. They do not replace the
shell scripts or add new bootstrap logic.

### First-time local setup

If you just cloned the repo, move into the project directory first:

```bash
git clone https://github.com/jquintanilla4/effect-erase.git
cd effect-erase
```

Run the initial worker setup from the repo root:

```bash
make bootstrap
make verify
```

In plain terms:

- `make bootstrap` creates or repairs the worker envs, installs the Python
  dependencies needed by the worker, downloads the required model assets, and
  records the bootstrap state.
- `make verify` reruns the readiness checks and confirms that CUDA, worker
  imports, and required model paths are all working before you start the app.

If `make verify` passes, you can move on to the normal startup flow below.

### Bootstrap the worker envs and model assets

Local or Tailscale host:

```bash
make bootstrap
```

Equivalent script form:

```bash
./scripts/setup-worker.sh --env-manager conda
```

Runpod Pod:

```bash
git clone https://github.com/jquintanilla4/effect-erase.git
cd effect-erase
./scripts/setup-worker.sh --env-manager micromamba
```

On Runpod, bootstrap prompts for any missing setup choices before it starts
installing packages. That includes the env manager when it is ambiguous and a
Hugging Face token when default `sam3.1` downloads need authentication. The
runtime storage root now defaults directly to `/workspace/effect-erase-runtime`
unless you override it explicitly.

Runpod storage has an important split: the volume disk is mounted at
`/workspace`, while the container disk is separate and ephemeral. This bootstrap
path now keeps envs, caches, models, and temp build/download scratch space
under the runtime root so large installs do not silently fall back to the
container filesystem.

By default, `make bootstrap` uses `ENV_MANAGER=auto`. On Runpod, that path now
prefers `micromamba` when both managers are available because it brings up the
split worker envs faster; elsewhere the script still picks `conda` when it is
available on `PATH` and falls back to `micromamba` otherwise.

If you want to force a specific manager, override it explicitly:

```bash
make bootstrap ENV_MANAGER=conda
make bootstrap ENV_MANAGER=micromamba
```

To rerun the readiness checks without reinstalling anything:

```bash
make verify
```

Equivalent script form:

```bash
./scripts/verify-worker.sh
```

### Day-to-day local startup

If bootstrap already completed successfully and the persisted bootstrap state is
in the `ready` state, start the app from the repo root like this:

```bash
# terminal 1
make worker
```

```bash
# terminal 2, first time only
cd web
npm install
npm run dev
```

```bash
# terminal 2, every time after the first frontend install, from the repo root
make web
```

You do not need to manually install Python packages for this path. The worker
startup script now starts directly from the persisted bootstrap state. It does
not rerun bootstrap. If setup is missing or incomplete, it fails fast and tells
you which `make bootstrap` command to rerun.

### Start the worker

```bash
make worker
```

Equivalent script form:

```bash
./scripts/start-worker.sh --env-manager conda
```

### Start the web app

The first time you start the frontend on a machine, do it from inside `web/`
so you can install dependencies before launching Vite:

```bash
cd web
npm install
npm run dev
```

After that first setup, start the dev server from the repo root with:

```bash
make web
```

If you are already inside `web/`, the equivalent direct command is:

```bash
npm run dev
```

`make web` is the repeat-run shortcut. It assumes `web/node_modules` already
exists and does not run `npm install` for you.

By default, the browser app expects the worker at:

```text
http://localhost:8000
```

## Bootstrap script behavior

The setup script:

- prompts for missing setup choices in interactive TTY sessions before it starts installing packages
- detects the env manager or uses the one you explicitly pass
- installs `micromamba` in user space when required
- defaults Runpod runtime state to `/workspace/effect-erase-runtime`
- reuses an existing `hf auth login` token from the effective `HF_HOME` when available
- creates envs if missing
- validates envs before reinstalling packages
- upgrades `pip`, `setuptools`, and `wheel`
- installs `torch` and `torchvision` from the chosen PyTorch index
- installs the local worker package
- installs upstream packages from source archive URLs instead of cloning repos
- preinstalls pinned EffectErase runtime dependencies before installing the upstream EffectErase package with `--no-deps`
- optionally skips the Hopper FlashAttention 3 build when `SKIP_SAM_FA3=1`
- defaults to split envs unless configured otherwise
- redirects temp build/download scratch space into the managed runtime root on Runpod or custom storage roots
- downloads model assets by default
- runs a post-bootstrap verification pass for CUDA, env imports, and required model paths
- writes bootstrap metadata to the active worker data directory

The setup flow no longer requires checked-out `third_party` repos for `sam3` or `EffectErase`.

### Script options

`setup-worker.sh`:

```bash
./scripts/setup-worker.sh --env-manager conda|micromamba|auto --storage-root PATH --interactive|--non-interactive --strategy split --cuda-backend cu128
```

For automation, you can also provide Hugging Face auth with `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` instead of answering the prompt. If the effective
`HF_HOME/token` already exists from a prior `hf auth login`, bootstrap reuses it
and skips the prompt.

If you are bootstrapping on Hopper hardware and do not want to spend time
building FlashAttention 3, set `SKIP_SAM_FA3=1`. Bootstrap and inference still
work; SAM 3.1 just runs with `use_fa3=false`.

`SKIP_SAM_FA3` is still the documented variable name. For backward
compatibility, bootstrap also accepts `SKIPSAMFA3=1`.

By default, setup also downloads the model assets needed for inference:

- requires auth for gated `sam3.1`
- downloads the EffectErase checkpoint
- downloads the required Wan 2.1 text encoder, VAE, DiT, and image encoder weights

Use `--skip-model-downloads` only if you are managing model assets yourself.
In that staged path, bootstrap can finish with the envs ready but will keep
`data/bootstrap-status.json` in a non-ready state until the required local model
files are present and `./scripts/verify-worker.sh` passes without the staged
asset override.

`start-worker.sh`:

```bash
./scripts/start-worker.sh --env-manager conda|micromamba --storage-root PATH --host 0.0.0.0 --port 8000
```

`verify-worker.sh`:

```bash
./scripts/verify-worker.sh [--json] [--storage-root PATH]
```

Use `--json` when you want the same readiness report in a machine-readable format.

## Model layout

The worker now expects model files under the active worker models directory.
That is `models/` by default and `/workspace/effect-erase-runtime/models` on
Runpod unless you override `--storage-root`.

- `sam3.1/config.json`
- `sam3.1/sam3.1_multiplex.pt`
- `sam3/config.json`
- `sam3/sam3.pt`
- `sam2.1/sam2.1_hiera_base_plus.pt`
- `EffectErase/EffectErase.ckpt`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl/tokenizer_config.json`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl/tokenizer.json`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl/spiece.model`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl/special_tokens_map.json`

Only the files that are actually required for inference are downloaded.

The download script also writes a lightweight tab-separated manifest to `models/asset-manifest.tsv` with each asset's scope, status, size, and relative path.

## Runtime selection

Runtime selection is asset-aware:

- if SAM assets are available, the worker uses real SAM
- if EffectErase assets are available, the worker uses real EffectErase removal
- if assets are missing and runtime mode is `auto`, the worker falls back to mock behavior

You can still force runtime behavior with worker env vars such as:

- `WORKER_RUNTIME_MODE=real`
- `WORKER_RUNTIME_MODE=mock`
- `WORKER_USE_MOCK_RUNTIME=true`

## Relevant environment variables

- `PYTHON_VERSION`
- `CUDA_BACKEND`
- `TORCH_INDEX_URL`
- `ENV_MANAGER`
- `STORAGE_ROOT`
- `ENV_STRATEGY`
- `SAM_ENV_NAME`
- `REMOVE_ENV_NAME`
- `VOID_ENV_NAME`
- `WORKER_HOST`
- `WORKER_PORT`
- `WORKER_DATA_DIR`
- `WORKER_PROJECTS_DIR`
- `WORKER_MODELS_DIR`
- `WORKER_BOOTSTRAP_STATE_PATH`
- `HF_HOME`
- `HF_HUB_CACHE`
- `PIP_CACHE_DIR`
- `MAMBA_ROOT_PREFIX`
- `CONDA_ENVS_PATH`
- `CONDA_PKGS_DIRS`
- `TMPDIR`
- `SAM3_PACKAGE_REF`
- `SAM2_PACKAGE_REF`
- `EFFECTERASE_PACKAGE_REF`
- `FLASH_ATTENTION_HOPPER_REF`
- `SAM3_PACKAGE_SPEC`
- `SAM2_PACKAGE_SPEC`
- `EFFECTERASE_PACKAGE_SPEC`
- `WORKER_RUNTIME_MODE`
- `WORKER_USE_MOCK_RUNTIME`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `WORKER_GEMINI_API_KEY`
- `WORKER_GEMINI_MODEL`
- `WORKER_GEMINI_TIMEOUT_MS`

Defaults are currently defined directly in [`scripts/setup-worker.sh`](./scripts/setup-worker.sh) and [`worker/app/core/config.py`](./worker/app/core/config.py).

### Gemini config for VOID

The VOID mask-reasoner runs Gemini on the worker side, not in the browser.
That means the Gemini key belongs in backend environment configuration only.

Recommended setup:

- local machine: put the key in a repo-root `.env` file that stays ignored by git
- Runpod: inject the key with a Runpod secret-backed environment variable
- frontend: never send the key in requests and never store it in Vite env vars

The worker checks these names in order:

- `WORKER_GEMINI_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

On Runpod, prefer a secret mapping like:

```text
GEMINI_API_KEY={{ RUNPOD_SECRET_gemini_api_key }}
```

This keeps the code path the same across local and hosted workers while avoiding
checked-in secrets or ad hoc `.env` files inside the Pod.

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

Artifact responses now follow the exact worker origin that served the API
request. When you access the worker over Tailscale or a Runpod proxy, the frame,
mask, propagation, and output URLs reuse that same public origin instead of
falling back to `localhost`.

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

## Limitations

Current limitations of the repo:

- EffectErase removal is currently limited to clips up to 81 frames
- split-env bootstrap is the only supported bootstrap mode
- worker state is file-based and in-memory, not database-backed
- UI backend profiles are not yet loaded dynamically from disk
- auth is not implemented; this is an internal tool
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
zsh -n scripts/setup-worker.sh scripts/verify-worker.sh scripts/download-model-assets.sh scripts/clean-model-assets.sh scripts/start-worker.sh scripts/start-web.sh
```

### Frontend

Standard Vite workflow:

```bash
cd web
npm install
npm run dev
```

Optional repo-root wrapper:

```bash
# run this from the repo root
make web
```

`make web` only wraps `cd web && npm run dev`. It does not run `npm install`
for you, so use the direct frontend setup command above the first time or after
dependency changes. If you are already inside `web/`, run `npm run dev`
instead of `make web`.

### Make targets

The repo-root `Makefile` is intentionally thin so developers can use shorter
commands without hiding the underlying scripts:

```bash
make bootstrap
make verify
make worker
make web
```

When you need to force the worker env manager, override `ENV_MANAGER` directly:

```bash
make bootstrap ENV_MANAGER=conda
make worker ENV_MANAGER=micromamba
```

## Git ignore policy

The repo ignores:

- generated project artifacts under `data/projects/`
- bootstrap state
- downloaded model weights under `models/`
- node modules and frontend build output
- Python cache and virtualenv directories
- common editor and local env files

See [`.gitignore`](./.gitignore).

## Troubleshooting

### `conda` is not found

Make sure Conda is installed and available on `PATH` before running:

```bash
./scripts/start-worker.sh --env-manager conda
```

### `micromamba` is not found on Runpod

That is expected on some base Pods. The setup script installs it automatically into user space.

### SAM 3.1 download fails

`sam3.1` is gated on Hugging Face access. Default bootstrap expects that auth to
be available up front, either from `hf auth login` in the effective `HF_HOME`
or from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.

If you want the explicit SAM 2.1-only asset path instead of gated `sam3.1`, rerun:

```bash
./scripts/download-model-assets.sh --skip-sam31 --include-sam21
```

If the manifest reports `incomplete` files after an interrupted run, clean those partial artifacts first:

```bash
./scripts/clean-model-assets.sh --dry-run
./scripts/clean-model-assets.sh
```

Then rerun `./scripts/setup-worker.sh` or `./scripts/download-model-assets.sh`.

If `POST /sam/start-session` fails with a PyTorch archive or `PytorchStreamReader` error, the local
`models/sam3.1/sam3.1_multiplex.pt` checkpoint is corrupted. Current bootstrap runs now delete and
re-download that checkpoint automatically. On an older checkout, remove that file manually and rerun
`./scripts/download-model-assets.sh`.

### Shared env bootstrap fails

The script defaults to split envs, and split envs are the only supported bootstrap mode.

### The UI cannot reach the worker

Check:

- worker host and port
- Tailscale hostname and access
- Runpod proxy URL
- browser CORS and network reachability

### The app is using mock behavior instead of real inference

Run `./scripts/verify-worker.sh` and check that both the SAM runtime set and EffectErase assets pass.

If verification fails, check that the required files exist under `models/` and that runtime mode is not forced to `mock`.

## Next implementation priorities

The highest-value next steps are:

1. Add chunking or trimming support for longer EffectErase runs.
2. Make repeat bootstrap runs quieter and more status-driven.
3. Add explicit startup diagnostics for CUDA and model availability.
4. Improve operator-facing setup and troubleshooting guidance once the runtime is exercised on more clean machines.
