# User Guide

## Start Commands

Run all commands from the repository root.

| Command | Behavior |
| --- | --- |
| `uv run main.py` | Starts the live camera application. |
| `uv run main.py --debug` | Starts the live camera application with landmark overlays and periodic recognition metrics. |
| `uv run main.py --list-memes` | Prints each configured gesture name, media filename, and priority, then exits. |
| `uv run main.py --describe-meme NAME` | Prints the activation description, media file, and priority for one configured gesture, then exits. |

`--list-memes` and `--describe-meme` cannot be used together. An unknown name supplied to `--describe-meme` is an argument error; use `--list-memes` to discover valid names.

## Preview Window

The application opens an OpenCV window called `meme camera`. The camera image is mirrored before display, like a conventional webcam preview. When an eligible gesture activates, reM draws the configured image or current GIF frame centered on the detected nose and clamps it to the frame boundary.

An overlay is drawn only while a face is available. The runtime may reuse the last face landmarks for up to 0.5 seconds after a brief face-detection dropout, but it does not draw or select new media without a usable face.

## Keyboard Controls

| Key | Effect |
| --- | --- |
| `q` or Escape | Quit the application. |
| `d` | Toggle gesture detection on or off. Toggling resets active media, debounce state, cooldowns, consumed gestures, motion history, and the cached face. |
| `h` | Dismiss the currently displayed media. New gestures remain suppressed until no gesture candidates are present, which prevents an immediately re-triggered overlay. |

Controls are processed while the `meme camera` window has keyboard focus.

## What Happens Each Frame

When detection is enabled, reM:

1. Captures a camera frame and creates a horizontally mirrored copy for display.
2. Runs MediaPipe hand gesture recognition on every frame.
3. Runs MediaPipe face detection on the first frame and then every second display frame.
4. Preserves recent wrist locations for motion-based gestures.
5. Produces a set of matching facial and hand-pose candidates.
6. Selects the highest-priority eligible candidate, waits for its debounce time, then activates its media.
7. Draws the active media at the mirrored face-nose location.

Gesture and face inference run concurrently through a two-worker executor. The terminal reports FPS plus average gesture and face inference time after every 30 displayed frames.

## Debug Mode

Start debug mode with:

```bash
uv run main.py --debug
```

It adds these diagnostics:

- Face landmark dots and highlighted nose, forehead, and chin landmarks.
- Colored left/right hand skeletons.
- Hands-on-face status flags and confirmation-frame count.
- Terminal metrics, at most once every 0.2 seconds, for mouth gap, frown scores, face width, hand-to-nose distance, open-palm status, motion, palm spacing, head-grab, cheek zones, and hands-on-face conditions.

Use these values to distinguish a tracking issue from a threshold issue before editing configuration. Debug annotations are part of the captured preview window.

## Gesture Timing Rules

The full gesture list is in [Gesture Reference](GESTURES.md). These rules apply to every configured gesture:

- Priority resolves simultaneous eligible candidates. Larger `priority` values win.
- A selected candidate must remain selected for its configured `debounce` duration before activating.
- An image remains visible for at least 2 seconds; a GIF remains visible for at least 5 seconds. Configured duration cannot reduce these minimums.
- A fired gesture is consumed while it continues to match. It must clear from the candidate set before it can activate again.
- Its configured cooldown must also have expired before it becomes eligible again.
- A currently active overlay normally prevents activation of another candidate until it expires.
- Hands-on-face, head-grab, and hands-on-cheeks immediately cancel an active `explaining_hands` overlay.

This means a correctly held gesture may not fire if a higher-priority gesture is also matching, the same gesture has not cleared since its last activation, or an existing overlay is still active.

## Recording And Streaming

reM renders the final result only in its OpenCV preview window. To use the processed image in OBS, capture the `meme camera` window rather than opening the same physical webcam in both applications. See [OBS Setup](OBS.md).
