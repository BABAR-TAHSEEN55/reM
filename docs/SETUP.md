# Setup

This guide prepares a machine to run reM from a checkout of this repository.

## Prerequisites

### Python

The project requires Python 3.12 or newer. The repository pins its preferred interpreter in `.python-version`:

```bash
python --version
```

Use a Python 3.12+ interpreter if the command reports an earlier version.

### uv

The repository includes `uv.lock`, so [uv](https://docs.astral.sh/uv/) is the supported way to create a reproducible environment. Install it using the official uv instructions, then confirm it is available:

```bash
uv --version
```

### Desktop and camera access

reM uses OpenCV's desktop window API. Run it from an active graphical desktop session, not a headless shell. On Linux, an X11-capable session is the most directly supported configuration because the runtime defaults `QT_QPA_PLATFORM` to `xcb`.

You also need a webcam that the current user can open. Close applications such as Zoom, browsers, Discord, or OBS Video Capture Device sources that may already own the device.

## Install Dependencies

From the repository root, create or synchronize the local environment using the lockfile:

```bash
uv sync --locked
```

This creates or updates `.venv/` and installs the direct runtime dependencies:

| Dependency | Role |
| --- | --- |
| `mediapipe` | Face landmarks, blendshapes, hand landmarks, handedness, and gesture recognition. |
| `opencv-python` | Camera capture, display window, image compositing, and landmark drawing. |
| `pillow` | GIF decoding and frame timing. |

If you intentionally want uv to resolve a changed dependency set instead of enforcing the lockfile, use `uv sync` without `--locked`.

## Verify Required Files

Before starting the camera, confirm these repository files are present:

```bash
test -f gesture_recognizer.task
test -f face_landmarker.task
test -d memes
```

The application does not fetch model files. `gesture_recognizer.task` supplies the hand gesture pipeline; `face_landmarker.task` supplies the face landmarks and blendshapes used by the facial gestures. Missing or corrupt model files cause MediaPipe initialization to fail.

At startup, reM also loads every file in `GESTURE_MEDIA` from `rem/config.py`. A missing image/GIF, undecodable image, empty GIF, or unsupported configured media type stops startup before the camera opens.

## Verify The Installation Without Opening A Camera

List the configured gesture-to-media mappings:

```bash
uv run main.py --list-memes
```

Describe one mapping in more detail:

```bash
uv run main.py --describe-meme hands_on_face
```

These commands still import MediaPipe because `main.py` imports the runtime module before parsing arguments. They verify that the Python dependencies are installed, but do not validate a camera, model loading, or media decoding.

## Start reM

```bash
uv run main.py
```

The startup log reports the camera resolution OpenCV actually opened. The default configured camera is index `1`, not `0`. If the wrong camera opens or the camera cannot be opened, change `CAMERA_INDEX` in `rem/config.py`; see [Configuration](CONFIGURATION.md#camera-and-display) and [Troubleshooting](TROUBLESHOOTING.md#camera-problems).

For recognition calibration, start in debug mode:

```bash
uv run main.py --debug
```

Debug mode makes landmark and state annotations visible in the same preview window, so do not use it for a clean recording or stream.

## Alternative Environment Setup

If uv is unavailable, create a virtual environment and install the declared direct dependencies manually:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install "mediapipe>=1.2.2" "opencv-python>=5.0.0.93" "pillow>=13.0.0"
.venv/bin/python main.py
```

This fallback does not enforce the versions in `uv.lock`. Prefer `uv sync --locked` for a reproducible setup.

## First-Run Checklist

1. Run `uv sync --locked` successfully.
2. Verify the two `.task` files and the `memes/` directory exist.
3. Run `uv run main.py --list-memes`.
4. Close other applications that use the webcam.
5. Start `uv run main.py` from a graphical desktop session.
6. Confirm the `meme camera` window opens and the terminal prints a `Camera:` line.
7. Use `q` or Escape to exit.
