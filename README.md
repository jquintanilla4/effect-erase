# EffectErase

Browser UI plus Python worker for interactive video object removal.

The current app flow is:

1. Start the worker on a CUDA host.
2. Open the web UI.
3. Upload a clip and start a SAM session.
4. Click positive and negative points to define the object.
5. Propagate the mask through the clip.
6. Run removal with either `EffectErase` or `VOID`.

This README is intentionally focused on installing, starting, and operating the app.

## What You Need

For the worker host:

- Linux
- Git
- `curl`
- internet access during first bootstrap
- a CUDA-capable NVIDIA GPU for real inference
- Conda or micromamba installed
- enough disk space for environments, model weights, and uploaded media

Install docs:

- [Conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
- [Micromamba installation guide](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html)

For real SAM 3.1 bootstrap:

- access to the gated Hugging Face `facebook/sam3.1` assets

If that gated access is not available, the default bootstrap path will not silently downgrade to SAM 2.1. Use the asset downloader explicitly if you need a different SAM asset mix.

## Quick Start

Clone the repo:

```bash
git clone https://github.com/jquintanilla4/effect-erase.git
cd effect-erase
```

Bootstrap the worker environments and default model assets:

```bash
make bootstrap
```

Run verification:

```bash
make verify
```

Start the worker:

```bash
make worker
```

Start the web app in a second shell:

```bash
./scripts/start-web.sh
```

Then open the Vite URL shown in the terminal, usually `http://localhost:5173`.

## How Bootstrap Works

Bootstrap is script-driven and currently supports split environments only.

By default it creates:

- `effecterase-sam`
- `effecterase-remove`
- `effecterase-void`

The split is deliberate because the SAM, EffectErase, and VOID stacks do not share cleanly in one mutable environment. Repeat bootstrap runs reuse or repair envs in place instead of rebuilding everything from scratch.

`make bootstrap` wraps:

```bash
./scripts/setup-worker.sh --env-manager auto
```

That setup flow currently:

- chooses `conda` or `micromamba`
- creates or repairs the worker envs
- installs the shared worker package into each runtime env
- downloads the default SAM and EffectErase assets
- leaves `VOID` weights as lazy assets
- runs `verify-worker.sh` automatically before marking bootstrap ready

## Starting The App

Typical local startup:

```bash
make worker
./scripts/start-web.sh
```

If you prefer direct commands:

```bash
./scripts/start-worker.sh --env-manager conda
cd web
npm install
npm run dev
```

The worker listens on `0.0.0.0:8000` by default.

## Using The UI

1. Choose a backend profile or paste a worker URL override.
2. Pick a SAM model that the worker reports as available.
3. Upload a video and start a session.
4. Click the frame to add positive and negative prompt points.
5. Run propagation to generate the tracked mask video.
6. Pick a removal pipeline.
7. Start removal and wait for the job to finish.

Current removal pipelines:

- `EffectErase`
  Default path after bootstrap. The worker is currently wired for clips up to `81` frames on this path.
- `VOID`
  Requires a Gemini API key on the worker. The worker is currently wired for clips up to `197` frames on this path. Its model weights stay lazy until you either press the UI download button or run the pipeline for the first time.

The UI polls job status from the worker and returns the finished artifact URL when the render completes.

## Worker Secrets And Config

Local development can use a repo-root `.env` file. Runpod should inject secrets as environment variables instead of creating a `.env` file in the container.

Example local `.env` values:

```bash
GEMINI_API_KEY=your-gemini-api-key
WORKER_GEMINI_MODEL=gemini-2.5-flash
```

The worker accepts any of these names for the Gemini key:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `WORKER_GEMINI_API_KEY`

Useful worker settings:

- `ENV_MANAGER`
  Override env selection for `make bootstrap`, `make verify`, or `make worker`.
- `STORAGE_ROOT`
  Move mutable runtime state, caches, models, and bootstrap state out of the repo root.
- `HOST`
  Worker bind host. Defaults to `0.0.0.0`.
- `PORT`
  Worker bind port. Defaults to `8000`.

## Web Configuration

The UI ships with built-in profile defaults for:

- local CUDA
- Tailscale worker
- Runpod Pod

You can override those defaults before starting Vite with:

```bash
export VITE_LOCAL_WORKER_URL=http://localhost:8000
export VITE_TAILSCALE_WORKER_URL=http://gpu-box.tailnet-name.ts.net:8000
export VITE_RUNPOD_WORKER_URL=https://pod-id-8000.proxy.runpod.net
```

You can also ignore those env vars and type a worker URL directly into the UI.

## Deployment Modes

### Local CUDA Host

Use the default flow:

```bash
make bootstrap ENV_MANAGER=conda
make verify ENV_MANAGER=conda
make worker ENV_MANAGER=conda
./scripts/start-web.sh
```

### Tailscale GPU Host

Run the worker bootstrap and worker process on the remote CUDA machine, then point the local web UI at the Tailscale hostname and port.

The built-in example profile uses:

```text
http://gpu-box.tailnet-name.ts.net:8000
```

### Runpod Pod

Clone the repo on the Pod and keep mutable runtime data under `/workspace` so it survives restarts.

Typical bootstrap:

```bash
make bootstrap ENV_MANAGER=micromamba STORAGE_ROOT=/workspace/effect-erase-runtime
make verify ENV_MANAGER=micromamba STORAGE_ROOT=/workspace/effect-erase-runtime
make worker ENV_MANAGER=micromamba STORAGE_ROOT=/workspace/effect-erase-runtime
```

Typical secret mapping:

```text
GEMINI_API_KEY={{ RUNPOD_SECRET_gemini_api_key }}
```

## Useful Commands

Make targets:

- `make bootstrap`
- `make verify`
- `make worker`
- `make web` after `web` dependencies are already installed

Direct scripts:

- `./scripts/setup-worker.sh`
- `./scripts/verify-worker.sh`
- `./scripts/start-worker.sh`
- `./scripts/start-web.sh` to install frontend deps and run Vite
- `./scripts/download-model-assets.sh`
- `./scripts/clean-model-assets.sh`

Useful examples:

```bash
# Download an alternate SAM asset set manually.
./scripts/download-model-assets.sh --skip-sam31 --include-sam21

# Preview incomplete model artifacts before deleting them.
./scripts/clean-model-assets.sh --dry-run
```

## Make Wrapper Options

The `Makefile` is intentionally thin. It does not expose every script flag.

Current target mappings:

- `make bootstrap`
  Wraps `./scripts/setup-worker.sh --env-manager "$(ENV_MANAGER)"`
- `make verify`
  Wraps `./scripts/verify-worker.sh --env-manager "$(ENV_MANAGER)"`
- `make worker`
  Wraps `./scripts/start-worker.sh --env-manager "$(ENV_MANAGER)"`

Variables supported directly by the make wrappers:

- `ENV_MANAGER`
  Works with `make bootstrap`, `make verify`, and `make worker`.
  Valid values: `auto`, `conda`, `micromamba`.
- `STORAGE_ROOT`
  Works with `make bootstrap`, `make verify`, and `make worker`.
  Value: any path.

Examples:

```bash
make bootstrap ENV_MANAGER=conda
make verify ENV_MANAGER=micromamba STORAGE_ROOT=/workspace/effect-erase-runtime
make worker ENV_MANAGER=conda STORAGE_ROOT=/workspace/effect-erase-runtime
```

Not exposed through the `make` wrappers:

- `make bootstrap` does not expose `--strategy`, `--cuda-backend`, `--skip-model-downloads`, `--interactive`, or `--non-interactive`.
- `make worker` does not expose `--host` or `--port`.
- `make verify` does not expose `--json`, `--bootstrap-mode`, `--allow-missing-model-assets`, `--strategy`, `--worker-env`, `--sam-env`, `--remove-env`, or `--void-env`.

For any of those options, run the underlying script directly.

## Script Flags

### `./scripts/setup-worker.sh`

Current usage:

```bash
./scripts/setup-worker.sh \
  --env-manager conda|micromamba|auto \
  --storage-root PATH \
  --strategy split \
  --cuda-backend CUDA_TAG \
  [--skip-model-downloads] \
  [--interactive | --non-interactive]
```

Flags:

- `--env-manager conda|micromamba|auto`
  Picks the Python env manager. `auto` is the default.
- `--storage-root PATH`
  Moves runtime state, caches, env directories, models, and bootstrap state under a different root.
- `--strategy split`
  Chooses the env layout. `split` is the only supported option right now.
- `--cuda-backend CUDA_TAG`
  Sets the PyTorch wheel channel suffix used during bootstrap. The default is `cu128`.
- `--skip-model-downloads`
  Creates and verifies the envs without downloading inference assets during that run.
- `--interactive`
  Forces prompts when the script needs missing setup input.
- `--non-interactive`
  Disables prompting and sticks to defaults or provided values.
- `-h`, `--help`
  Prints the script usage text.

### `./scripts/verify-worker.sh`

Current usage:

```bash
./scripts/verify-worker.sh \
  [--json] \
  [--bootstrap-mode] \
  [--allow-missing-model-assets] \
  [--env-manager conda|micromamba|auto] \
  [--storage-root PATH] \
  [--strategy split] \
  [--worker-env NAME] \
  [--sam-env NAME] \
  [--remove-env NAME] \
  [--void-env NAME]
```

Flags:

- `--json`
  Prints the verification report as JSON instead of the human-readable summary.
- `--bootstrap-mode`
  Uses bootstrap-time verification rules. In that mode, CUDA can be optional when the worker is intentionally running in mock mode.
- `--allow-missing-model-assets`
  Treats missing inference assets as allowed for bootstrap compatibility checks. This is mainly for staged setup flows such as `--skip-model-downloads`.
- `--env-manager conda|micromamba|auto`
  Selects how cross-env verification commands are launched. If omitted, the script prefers the saved bootstrap state.
- `--storage-root PATH`
  Verifies against a specific runtime root instead of the default one.
- `--strategy split`
  Verifies the env layout. `split` is the only supported option right now.
- `--worker-env NAME`
  Explicitly sets the worker env used for the aggregate verification command.
- `--sam-env NAME`
  Explicitly sets the SAM env name.
- `--remove-env NAME`
  Explicitly sets the EffectErase env name.
- `--void-env NAME`
  Explicitly sets the VOID env name.
- `-h`, `--help`
  Prints the script usage text.

### `./scripts/start-worker.sh`

Current usage:

```bash
./scripts/start-worker.sh \
  [--env-manager conda|micromamba|auto] \
  [--storage-root PATH] \
  [--host HOST] \
  [--port PORT]
```

Flags:

- `--env-manager conda|micromamba|auto`
  Selects which env manager should launch the already-bootstrapped worker env.
- `--storage-root PATH`
  Starts the worker from a specific runtime root and bootstrap state.
- `--host HOST`
  Overrides the worker bind host. Default: `0.0.0.0`.
- `--port PORT`
  Overrides the worker bind port. Default: `8000`.

### `./scripts/download-model-assets.sh`

Current usage:

```bash
./scripts/download-model-assets.sh \
  [--storage-root PATH] \
  [--include-sam3] \
  [--include-sam21] \
  [--skip-sam31] \
  [--skip-effecterase]
```

Flags:

- `--storage-root PATH`
  Downloads into the models directory for a specific runtime root.
- `--include-sam3`
  Adds the legacy `sam3` asset set to the download plan.
- `--include-sam21`
  Adds the `sam2.1` checkpoint to the download plan.
- `--skip-sam31`
  Skips the default `sam3.1` asset set.
- `--skip-effecterase`
  Skips the EffectErase and required Wan assets.
- `-h`, `--help`
  Prints the script usage text.

### `./scripts/clean-model-assets.sh`

Current usage:

```bash
./scripts/clean-model-assets.sh [--storage-root PATH] [--dry-run]
```

Flags:

- `--storage-root PATH`
  Cleans incomplete assets under a specific runtime root.
- `--dry-run`
  Prints what would be removed without deleting anything.
- `-h`, `--help`
  Prints the script usage text.

## Troubleshooting

If `make worker` says bootstrap is not ready or the bootstrap state file is missing, rerun:

```bash
make bootstrap
make verify
```

If bootstrap is reusing the wrong runtime root, set `STORAGE_ROOT` explicitly so the worker, models, cache directories, and bootstrap state all resolve from the same place.

If `VOID` is selectable in the UI but cannot run, check these first:

- the worker has a Gemini API key configured
- the `effecterase-void` env verified successfully
- the `VOID` model download job completed or the lazy download finished

## Repository Layout

- `web/` React + TypeScript Vite app
- `worker/` FastAPI worker and runtime integration code
- `scripts/` bootstrap, verification, asset download, and startup scripts
- `config/` example backend profile data
- `data/` local runtime state and project artifacts
- `models/` downloaded checkpoints and weights
