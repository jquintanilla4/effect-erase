# Remaining Tasks

## Bootstrap and setup

- Make repeat `scripts/setup-worker.sh` runs quieter. The heavy reinstalls are skipped when env probes pass, but the script still emits a full bootstrap flow and could surface clearer "already ready" summaries.
- Add a lightweight asset manifest or checksum pass so repeat runs can report exactly which model files are present, missing, or incomplete before starting any download work.
- Add an explicit post-bootstrap verification command that checks CUDA visibility, env imports, and required model paths in one place.

## Inference/runtime

- Add chunking or trimming support for EffectErase removal beyond the current 81-frame limit.
- Add better error surfacing around failed EffectErase subprocess runs so the operator sees the key missing dependency or model issue immediately.
- Add a startup self-check endpoint or log summary for selected runtime mode, detected CUDA device, and available SAM/EffectErase assets.

## Product/workflow

- Support multi-object editing workflows instead of a single object id.
- Move backend profile configuration out of hardcoded frontend defaults and into a managed config source.
- Add clearer operator documentation for gated Hugging Face access and expected disk usage on first bootstrap.
