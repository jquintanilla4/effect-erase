# Overlap-Chunking Plan

## Goal

Support EffectErase removal for videos longer than 81 frames without rewriting the underlying Wan remove pipeline.

This plan keeps the existing `WanRemovePipeline` call shape intact and adds orchestration around it in the worker.

## Summary

Run multiple 81-frame inference windows with overlap, then stitch the per-window outputs into one final video.

This is not denoising-time sliding windowing. It is clip-level chunking with overlap-aware fusion.

## Why This Plan

- Lowest-risk path to working `>81` frame removal.
- Reuses the current subprocess runner and model invocation.
- Fits the repo’s existing `overlap_frames` setting in `worker/app/core/config.py`.
- Lets us ship something usable before touching `diffsynth` internals.

## Constraints

- Each EffectErase inference call should stay at `81` frames when possible.
- Window boundaries should respect the Wan temporal shape expectations (`4n + 1`).
- Mask video and source video must stay frame-aligned.
- The current runtime rejects clips longer than 81 frames, so that guard must be replaced with chunk orchestration.

## Proposed Behavior

For a source video with `N > 81` frames:

1. Choose `window_size = 81`.
2. Choose `overlap = settings.overlap_frames` with validation that it is less than `window_size`.
3. Compute `stride = window_size - overlap`.
4. Build a sequence of windows that fully covers the clip.
5. For each window:
   - extract matching source and mask subclips
   - run the existing EffectErase subprocess on that subclip
   - collect the output frames
6. Stitch the outputs with overlap-aware fusion.
7. Encode the final merged frame sequence as the result video.

## Window Scheduling

Start with a simple static schedule:

- Window 0: frames `0..80`
- Window 1: frames `stride..stride+80`
- Continue until the end of the clip is covered
- Shift the final window backward if needed so the last frame is included

This mirrors the `static_standard` idea from the Comfy wrapper without importing its denoising-time complexity.

## Stitching Strategy

Start with weighted overlap blending instead of hard cuts.

Recommended first pass:

- Keep full non-overlap regions directly.
- In overlap regions, blend frames from the left and right windows with a linear ramp.
- Optionally add a second mode later with center-weighted or pyramid blending.

Why:

- Hard cuts will likely expose temporal seams.
- Linear blending is easy to reason about and cheap to implement.
- It borrows the right idea from `ComfyUI-WanVideoWrapper` without needing to refactor the model loop.

## Implementation Steps

### 1. Runtime and config

- Keep `max_window_frames = 81`.
- Reuse `overlap_frames` as the overlap parameter for EffectErase windows.
- Replace the current `>81` hard failure in `RealEffectEraseRuntime.remove(...)` with window planning logic.

### 2. Video slicing helpers

- Add helper(s) to extract frame ranges from source and mask videos into temporary subclips.
- Ensure source and mask use the exact same frame indices.
- Preserve fps and expected output resolution.

### 3. Subprocess orchestration

- Invoke the current `app.runners.effecterase_remove` once per window.
- Continue passing `--num_frames` equal to the actual window length.
- Keep shorter final windows valid by preserving the current real-length behavior.

### 4. Output fusion

- Decode each output clip into frames.
- Merge windows into one frame list using overlap weighting.
- Re-encode the final merged list into the requested output path.

### 5. Progress reporting

- Report progress across windows instead of treating removal as a single opaque subprocess.
- Weight progress roughly by completed windows, with some room for final merge/encode time.

### 6. Tests

- Add unit tests for window scheduling.
- Add unit tests for overlap validation and final-window adjustment.
- Add unit tests for stitch weighting behavior.
- Update runtime tests to verify long clips no longer fail early and instead generate multiple subprocess calls.

### 7. Docs

- Update `README.md` to replace the 81-frame hard limitation with a note about overlapping chunked inference.
- Document expected tradeoffs: more runtime, more temporary disk usage, possible seam risk.

## Risks

- Visible seams in overlap regions if weighting is not enough.
- Color or texture mismatch between independently inferred windows.
- Extra disk and runtime overhead from temp clips and repeated subprocess startup.
- Final short window quality may differ if it uses fewer than 81 frames.

## Mitigations

- Default to at least 16 overlap frames.
- Add a seam-debug mode later that writes intermediate windows for inspection.
- Keep fusion logic isolated so linear and pyramid weighting can be compared easily.
- Preserve deterministic seed handling per window if reproducibility matters.

## Exit Criteria

- A video longer than 81 frames can complete end-to-end.
- Runtime no longer throws the current `>81` error.
- Output covers the full input duration.
- Tests cover scheduler behavior and stitching behavior.
- README documents the new workflow and remaining quality caveats.

## Recommended Order

1. Window scheduler
2. Temp clip extraction
3. Multi-window subprocess orchestration
4. Frame-level overlap fusion
5. Tests
6. Docs
