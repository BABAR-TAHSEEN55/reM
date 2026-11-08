# reM Documentation

reM is a local webcam application that recognizes facial expressions and hand poses, then overlays a corresponding image or GIF over the detected face in a live OpenCV window. It uses MediaPipe Tasks for face and gesture recognition.

## Documentation Map

| Document | Use it for |
| --- | --- |
| [Setup](SETUP.md) | Installing Python dependencies, verifying models and media, and starting the application. |
| [User Guide](USAGE.md) | Command-line options, keyboard controls, debug mode, and normal runtime behavior. |
| [Gesture Reference](GESTURES.md) | Every supported gesture, its priority, media, timing, and recognition rule. |
| [Configuration](CONFIGURATION.md) | Changing camera, display, thresholds, timings, models, or meme mappings. |
| [Architecture](ARCHITECTURE.md) | Runtime data flow, state management, media rendering, and source-module responsibilities. |
| [Development](DEVELOPMENT.md) | Repository layout, tests, adding gestures, and contributor guidance. |
| [Troubleshooting](TROUBLESHOOTING.md) | Resolving installation, camera, desktop, recognition, and performance problems. |
| [OBS Setup](OBS.md) | Capturing the processed `meme camera` window in OBS. |

## Quick Start

From the repository root:

```bash
uv sync --locked
uv run main.py
```

The app opens a window named `meme camera`. Press `q` or Escape to quit. See [Setup](SETUP.md) before running this on a new machine.

## Runtime Requirements At A Glance

- Python 3.12 or newer.
- A desktop session capable of showing an OpenCV window.
- A webcam accessible to the current user.
- The checked-in `gesture_recognizer.task` and `face_landmarker.task` model files.
- Every media file configured in `rem/config.py` under `memes/`.

reM runs locally. It does not start a web server, expose a network API, or download models at runtime.
