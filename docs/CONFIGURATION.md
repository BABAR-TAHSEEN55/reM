# Configuration And Customization

Application configuration is centralized in [`rem/config.py`](../rem/config.py). Values are Python constants, so change them in the source file and restart reM. There is no environment-variable or command-line configuration layer for camera, thresholds, model locations, or media mappings.

## Paths And Models

| Constant | Default | Purpose |
| --- | --- | --- |
| `ROOT` | repository root | Derived from `rem/config.py`; base directory for bundled assets. |
| `GESTURE_MODEL_PATH` | `gesture_recognizer.task` | MediaPipe Gesture Recognizer model. |
| `FACE_MODEL_PATH` | `face_landmarker.task` | MediaPipe Face Landmarker model. |
| `MEMES_DIR` | `memes/` | Directory containing configured images and GIFs. |

Replace a model only with a MediaPipe Tasks model compatible with its current use. The face model must provide face landmarks and blendshapes; the gesture model must provide hand landmarks, handedness, and gesture categories. reM does not validate model capabilities before detection calls.

## Camera And Display

| Constant | Default | Effect |
| --- | ---: | --- |
| `CAMERA_INDEX` | `1` | OpenCV device index to open. Set `0`, `1`, or another available device as appropriate. |
| `CAMERA_WIDTH` | `640` | Requested capture width. The camera may choose a different supported mode. |
| `CAMERA_HEIGHT` | `480` | Requested capture height. |
| `INFERENCE_SIZE` | `(640, 360)` | Frames larger than 640x480 are resized to this size before inference. Smaller frames are not enlarged. |
| `INFERENCE_WORKERS` | `2` | Thread-pool workers used for concurrent gesture and face inference. |
| `WINDOW_TITLE` | `meme camera` | OpenCV preview-window title. OBS instructions assume this default title. |
| `MEME_WIDTH` | `320` | Fixed display width for every loaded image and GIF frame; height follows aspect ratio. |
| `FACE_INFERENCE_INTERVAL` | `2` | Run face inference every second display frame after the first frame. |
| `FACE_FALLBACK_SECONDS` | `0.5` | Reuse last detected face landmarks for this long after a brief detection loss. |

`open_camera()` requests MJPEG encoding and a one-frame OpenCV buffer to reduce latency. It first tries the Linux V4L2 backend, then falls back to OpenCV's default backend. Changing `CAMERA_INDEX` is the supported way to select another device; there is no camera command-line argument.

## Timing And Interaction

| Constant | Default | Effect |
| --- | ---: | --- |
| `IMAGE_MIN_LOCK_SECONDS` | `2.0` | Minimum active duration for any image. |
| `GIF_MIN_LOCK_SECONDS` | `5.0` | Minimum active duration for any GIF. |
| `MOTION_WINDOW_SECONDS` | `0.5` | Wrist-history window for motion gestures. |
| `VERTICAL_WAVE_MIN_SECONDS` | `0.15` | Minimum time span of wrist samples for a wave. |
| `VERTICAL_WAVE_MIN_DISTANCE` | `0.065` | Required right-wrist vertical range for a wave. |
| `EXPLAINING_MOTION_THRESHOLD` | `0.20` | Required combined wrist path length for explaining hands. |
| `HANDS_ON_FACE_CONFIRMATION_FRAMES` | `3` | Consecutive matching detections required for hands-on-face. |
| `HANDS_ON_FACE_FINGERTIP_DISTANCE_FACTOR` | `0.35` | Maximum middle-fingertip separation, relative to face width, for hands-on-face. |
| `FACE_HAND_POSE_HOLD_SECONDS` | `0.35` | Period that blocks explaining-hands after a face-adjacent hand pose. |
| `FPS_REPORT_INTERVAL` | `30` | Displayed frames between terminal performance reports. |
| `DEBUG_METRICS_INTERVAL` | `0.2` | Minimum seconds between debug metric messages. |

Each individual gesture has `duration`, `cooldown`, `priority`, and `debounce` in `GESTURE_MEDIA`. See [Gesture Reference](GESTURES.md) for their current values and [Architecture](ARCHITECTURE.md#activation-state-machine) for the state semantics.

## Detection Thresholds

The face, hand, and pose constants in `rem/config.py` implement the thresholds listed below. Tune incrementally while running debug mode; all measurements are normalized image coordinates or MediaPipe blendshape scores.

| Group | Constants | Current defaults |
| --- | --- | --- |
| Mouth and facial expression | `MOUTH_OPEN_THRESHOLD`, `FROWN_THRESHOLD`, `SMIRK_SMILE_THRESHOLD`, `SMIRK_ASYMMETRY_THRESHOLD`, `EYE_CLOSED_THRESHOLD`, `ROUND_MOUTH_PUCKER_THRESHOLD`, `ROUND_MOUTH_JAW_THRESHOLD`, `SMILE_THRESHOLD`, `SMILE_ASYMMETRY_MAX` | 0.03, 0.28, 0.40, 0.20, 0.60, 0.40, 0.10, 0.35, 0.20 |
| T-sign | `T_SIGN_PALM_DISTANCE_FACTOR`, `T_SIGN_DIRECTION_RATIO`, `T_SIGN_PERPENDICULAR_ERROR_DEGREES` | 2.0, 0.75, 25.0 |
| Head grab | `HEAD_GRAB_EYE_MARGIN_FACTOR` | 0.30 |
| Cheeks | `CHEEK_BELOW_EYE_FACTOR`, `CHEEK_OUTER_MARGIN_FACTOR`, `CHEEK_INNER_MARGIN_FACTOR`, `CHEEK_PALM_DISTANCE_FACTOR` | 0.32, 0.50, 0.10, 0.30 |
| Hands on face | `HANDS_ON_FACE_FINGERTIP_DISTANCE_FACTOR` | 0.35 |

`T_SIGN_PERPENDICULAR_ERROR_DEGREES` is configured but is not currently read by the T-sign matcher. The matcher calculates perpendicular error for its metrics record but bases the match on hand closeness, axis alignment, and hand openness.

## Change A Meme Mapping

Mappings are entries in `GESTURE_MEDIA`, each constructed with:

```python
MediaConfig(file, type, duration, cooldown, priority, debounce)
```

| Field | Meaning |
| --- | --- |
| `file` | Filename relative to `MEMES_DIR`. |
| `type` | Exactly `"image"` or `"gif"`. |
| `duration` | Requested active display duration in seconds, subject to the image/GIF minimum. |
| `cooldown` | Seconds after activation before the same gesture may be eligible again. |
| `priority` | Larger value wins when several candidates are eligible. |
| `debounce` | Seconds the selected candidate must remain selected before activation. |

To replace an existing asset, place the replacement file in `memes/` and change only its filename in the existing mapping. Images are decoded by OpenCV; GIFs are decoded by Pillow. Every loaded frame is converted to BGRA and resized to `MEME_WIDTH`, so alpha channels are preserved where the source format supports them.

To add a new mapping, adding an entry alone is not enough. You must also add a detector that returns the same mapping name from `detect_candidates()`, add a human-readable entry to `MEME_DESCRIPTIONS`, and test the behavior. See [Development](DEVELOPMENT.md#add-a-gesture).

## Keep Configuration Internally Consistent

When changing configuration:

1. Keep every `GESTURE_MEDIA` filename present and decodable under `memes/`.
2. Keep every candidate name returned by `rem/actions/candidates.py` present in `GESTURE_MEDIA`.
3. Keep every configured gesture name present in `MEME_DESCRIPTIONS` if it should be discoverable with `--describe-meme`.
4. Set a `type` recognized by `load_media()`: `image` or `gif`.
5. Account for priority and branch order, not only a rule's threshold.
6. Validate behavior with debug mode and targeted unit tests.
