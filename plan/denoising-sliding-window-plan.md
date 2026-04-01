# Denoising-Time Sliding-Window Plan

## Goal

Support long-video EffectErase removal by integrating windowing inside the Wan denoising loop, similar in spirit to `ComfyUI-WanVideoWrapper`.

This is the architectural option. It aims for better temporal quality than clip-level chunk stitching, but it requires deeper changes to `diffsynth` and our removal runner.

## Summary

Instead of running independent 81-frame subclips and stitching final RGB outputs, split the latent timeline into overlapping windows during each denoising step, predict each window separately, blend the overlapping noise predictions, and then apply the scheduler step once to the full latent video.

## Why This Plan

- Best long-term path for temporal consistency on long videos.
- Matches the strongest idea from the Comfy Wan wrapper.
- Avoids committing to seams after a full window has already been decoded.
- Provides a foundation for future long-form Wan features beyond EffectErase.

## Why It Is Harder

- `WanRemovePipeline` is currently a monolithic inference path in `diffsynth/pipelines/wan_video.py`.
- Our worker only knows how to call the pipeline on a complete clip, not manage intermediate denoising state.
- The wrapper implementation is generic Wan sampling code; EffectErase removal has task-specific conditioning:
  - `fg_bg_latent`
  - `mask_latent`
  - `remove_condition_adapter`
  - `fg_token`
  - task-specific attention outputs

## Key Reference From ComfyUI-WanVideoWrapper

The wrapper does three important things:

1. Builds overlapping context windows with schedulers such as `static_standard` and `uniform_standard`.
2. Runs prediction per window at each diffusion step.
3. Blends `noise_pred_context` back into a full-length `noise_pred` tensor with overlap weighting before applying the scheduler step.

That is the right mental model for a true sliding-window implementation here.

## Proposed Architecture

### 1. Introduce context window configuration

Add explicit internal config for:

- `context_frames`
- `context_overlap`
- `context_schedule`
- `fuse_method`

Even if the API only exposes a simpler form initially, the internal pipeline should work with these concepts.

### 2. Refactor `WanRemovePipeline.__call__`

Break the current monolithic remove path into smaller internal methods:

- prepare latents and conditioning
- build window schedule
- run one denoising step for one window
- fuse window predictions
- update scheduler state
- decode final latents

Without that refactor, the sliding-window logic will stay too brittle.

### 3. Add a window scheduler

Start with a static non-looped scheduler equivalent to the wrapper’s `static_standard`.

Do not start with:

- looped scheduling
- multi-stride scheduling
- prompt-section switching

EffectErase removal only needs deterministic forward coverage first.

### 4. Add denoising-time fusion

At each timestep:

1. Create empty full-length `noise_pred` and `counter` tensors.
2. For each context window:
   - slice `latents`
   - slice `fg_bg_latent`
   - slice `mask_latent`
   - run the remove model on that window
   - weight the result by an overlap mask
   - accumulate into the full tensors
3. Normalize `noise_pred /= counter`.
4. Apply the scheduler step to the full latent sequence.

### 5. Add overlap weighting

Start with:

- `linear`

Then optionally add:

- `pyramid`

This should mirror the wrapper’s `create_window_mask(...)` concept, but adapted to EffectErase shapes.

### 6. Handle window-local conditioning correctly

The following must be sliced consistently for each window:

- `latents`
- `fg_bg_latent`
- `mask_latent`
- any timestep tensor with temporal dimension

The following likely stay global:

- prompt embeddings
- `fg_token`
- model weights

This part is where most of the implementation risk lives.

### 7. Preserve attention outputs or redefine them

The current remove pipeline returns both frames and an attention visualization output.

We need to decide whether to:

- keep only the final fused removal output for v1 of this refactor, or
- also fuse the per-window attention maps in a consistent way

Recommendation: treat fused removal output as required and attention visualization as secondary.

## Required Code Areas

- `worker/app/runners/effecterase_remove.py`
- upstream-style local `diffsynth/pipelines/wan_video.py` integration path
- potentially helper modules for context window scheduling and weighting
- runtime/config plumbing in the worker if operator-facing controls are needed

## Risks

- Highest implementation complexity of the available options.
- Easy to introduce shape bugs in latent-time indexing.
- Harder to test because the model loop is more coupled to `diffsynth`.
- More likely to create maintenance burden during future upstream syncs.
- Attention-map fusion may not be obviously correct.

## Mitigations

- Refactor before adding behavior.
- Keep first scheduler simple and deterministic.
- Add shape assertions at every window boundary.
- Add focused unit tests for window math and fusion masks.
- Add an internal feature flag so the old path remains available during rollout.

## Testing Strategy

### Unit tests

- Window schedule generation for long clips
- Overlap mask creation
- Window coverage and no-zero-counter guarantees
- Temporal slice alignment across `latents`, `fg_bg_latent`, and `mask_latent`

### Integration tests

- Mocked pipeline run proving multi-window denoising aggregation happens
- Regression test that 81-frame clips still use the simple path
- Optional fixture comparing overlap-fused output shape and frame count on a synthetic long clip

### Manual evaluation

- Compare long-clip quality against overlap-chunking output
- Inspect boundary regions for seam reduction
- Confirm runtime and memory cost are acceptable

## Rollout Plan

1. Refactor the remove pipeline into smaller internal units
2. Add static window scheduler and linear fusion masks
3. Gate behind a feature flag
4. Validate on long clips
5. Decide whether to replace overlap-chunking or keep both modes

## Exit Criteria

- Long videos can be removed without clip-level output stitching.
- Overlapping windows are fused inside denoising.
- 81-frame behavior remains stable.
- Tests cover scheduler, weighting, and temporal slicing.
- We have a clear quality advantage over clip-level chunk fusion on boundary regions.

## Recommendation

Do this only after the overlap-chunking path exists or if long-video quality is critical enough to justify a pipeline refactor immediately.

This is the better architecture, but not the better first implementation.
