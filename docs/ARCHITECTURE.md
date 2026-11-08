# Architecture

## Overview

reM is a single-process desktop application. It captures a webcam stream, runs MediaPipe vision tasks, converts model output into gesture candidates, selects one active media item through a timing state machine, and alpha-composites the selected media into an OpenCV preview window.

```text
webcam frame
  -> gesture recognition (every frame) ------> hand landmarks, handedness, categories
  -> face landmarking (every second frame) --> face landmarks, blendshapes
  -> candidate detection --------------------> matching gesture-name set
  -> activation state machine ---------------> one active media item
  -> image/GIF frame selection + blending --> mirrored OpenCV preview
```

## Source Layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Parses command-line options and starts the runtime. |
| `rem/config.py` | Asset paths, runtime settings, thresholds, key bindings, mappings, and descriptions. |
| `rem/runtime.py` | Camera lifecycle, MediaPipe setup, inference scheduling, event loop, controls, and rendering. |
| `rem/models.py` | Dataclasses for static media configuration, loaded media, recognition metrics, active media, and mutable runtime state. |
| `rem/media.py` | Image/GIF loading, pixel-format normalization, resizing, GIF frame selection, and alpha blending. |
| `rem/actions/candidates.py` | Aggregates individual detectors and applies hand-pose branch ordering. |
| `rem/actions/state.py` | Candidate debounce, priority selection, cooldown, consumption, dismissal, and active-media expiry. |
| `rem/actions/landmarks.py` | Shared normalized-coordinate, hand-motion, orientation, openness, and blendshape helpers. |
| `rem/actions/*.py` | Individual expression and hand-pose predicates. |
| `rem/debug.py` | Debug drawing and rate-limited terminal metrics. |
| `tests/test_actions.py` | Unit tests for selected candidate and state behavior. |

## Startup And Resource Lifecycle

`rem.runtime.main()` performs these steps:

1. Calls `load_media()` to load all configured assets before opening the camera.
2. Creates a MediaPipe Gesture Recognizer in `VIDEO` mode with `num_hands=2`.
3. Creates a MediaPipe Face Landmarker in `VIDEO` mode with `num_faces=1` and blendshape output enabled.
4. Creates a two-worker `ThreadPoolExecutor`.
5. Opens the configured webcam and requests MJPEG, configured dimensions, and an OpenCV buffer size of one.
6. Processes frames until capture fails, the user quits, or an exception unwinds the loop.
7. Shuts down the executor, releases the camera, destroys OpenCV windows, and closes both MediaPipe detectors in `finally`.

Media is intentionally loaded before camera initialization. Missing configured media therefore fails fast without claiming the webcam.

## Inference And Coordinates

Each raw camera frame is sent to inference in its original orientation. Frames larger than the configured camera dimensions are resized to `INFERENCE_SIZE` first. The displayed frame is a horizontal flip of the raw frame.

The app converts inference frames from BGR to RGB and wraps them in `mp.Image`. Gesture recognition runs every frame. Face detection runs on the first frame and when `frame_index % FACE_INFERENCE_INTERVAL == 0`; otherwise the most recent result is reused.

MediaPipe reports handedness from its input-view convention. `physical_hands()` swaps reported `Left` and `Right` before storing hand landmarks, matching the mirrored user-facing view and the physical-left/physical-right gesture rules.

For a brief missing-face interval, the prior face landmarks remain usable for `FACE_FALLBACK_SECONDS`. The overlay center converts the normalized nose point to the mirrored display coordinate with `1.0 - nose.x` for x and `nose.y` for y.

## Candidate Detection

`detect_candidates()` returns a set of matching names. It first rejects all candidates when no face is available. With a face, it evaluates facial expressions from blendshape scores and landmarks, then evaluates hand states.

Shared hand history is updated before candidate detection. It stores each present hand's wrist coordinate and discards points older than `MOTION_WINDOW_SECONDS`. Motion-based detectors consume that history.

The hand-pose branch intentionally uses `if`/`elif` ordering, so head-grab, cheeks, T-sign, hands-on-face, wave, and explaining-hands do not all appear together. The detailed order and conditions are in [Gesture Reference](GESTURES.md).

## Activation State Machine

`RuntimeState` stores the active media, currently selected candidate, candidate start time, cooldown deadlines, consumed gesture names, motion histories, face-hand confirmation state, debug timing, and dismissal suppression.

For each frame, `update_active_media()`:

1. Clears active media that has reached its expiry time.
2. Cancels active explaining-hands media if a higher-priority face-adjacent hand candidate appears.
3. Removes names from `consumed` when they no longer appear in the current candidate set.
4. Honors `suppress_until_clear` after manual dismissal until no candidates remain.
5. Excludes consumed or cooling-down candidates.
6. Chooses the remaining name with the greatest configured priority.
7. Restarts the debounce clock when the selected name changes.
8. Activates only when the selected candidate has survived its debounce interval and no media is active.
9. Creates `ActiveMedia`, applies the cooldown deadline, consumes the selected name, and clears candidate-selection state.

This design prevents a held pose from repeatedly firing after each cooldown and makes timing deterministic even when several detectors match.

## Media Pipeline

`load_media()` maps every configured gesture to a `LoadedMedia` object:

- Images are loaded by OpenCV with alpha preserved where present.
- GIFs are decoded frame by frame by Pillow and use source frame durations, defaulting to 100 ms and clamping each delay to at least 20 ms.
- Grayscale, BGR, and BGRA inputs are normalized to BGRA.
- Every frame is resized to the configured width while preserving aspect ratio.

For images, `active_frame()` always returns the first frame. For GIFs, it computes an elapsed offset within the total GIF duration and returns the matching frame. `overlay_media()` clamps the destination rectangle to the display frame, uses the media alpha channel, and blends BGR pixels in place.

## Observability

Every 30 display frames, the runtime prints effective display FPS plus average gesture and face inference durations. Debug mode additionally draws landmarks and prints detection metrics no more frequently than the configured debug interval.

The app has no persistent database, network service, telemetry, or background worker beyond the in-process inference executor.
