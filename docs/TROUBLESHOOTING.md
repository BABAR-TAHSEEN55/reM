# Troubleshooting

## Installation And Dependencies

### `uv` is not found

Install uv from the official instructions, restart your shell, then verify:

```bash
uv --version
```

### Dependencies fail to install

Use a supported Python version:

```bash
python --version
```

reM requires Python 3.12 or later. Recreate the environment after changing Python:

```bash
rm -rf .venv
uv sync
```

## Camera Problems

### `Could not open camera`

1. Close Zoom, Discord, browsers, OBS Video Capture Device sources, and other apps using the webcam.
2. Check camera devices:

   ```bash
   ls /dev/video*
   ```

3. Change `CAMERA_INDEX` in `rem/config.py`. Try `0`, `1`, then `2`.
4. On Linux, confirm your user can access the camera device. Add yourself to the `video` group if needed, then log out and back in.

### Wrong camera opens

Set a different `CAMERA_INDEX` in `rem/config.py`, then restart the app.

## Window And Qt Messages

### Qt font warnings

OpenCV may print `QFontDatabase` warnings about missing Qt fonts. They are normally cosmetic and do not prevent detection. Install a standard font package, such as DejaVu fonts, only if window text renders incorrectly.

### No preview window

Run the app from an active desktop session, not a headless SSH shell. On Linux, confirm `DISPLAY` or the Wayland desktop session is available.

## MediaPipe Messages

Startup messages about XNNPACK, feedback tensors, or landmark projection are common MediaPipe informational warnings. They do not normally block detection. Focus on whether `Camera: ...` appears and the preview window opens.

## Gesture Problems

### A gesture never triggers

1. Start debug mode:

   ```bash
   uv run main.py --debug
   ```

2. Confirm the face mesh and both hand skeletons are visible.
3. Hold the gesture for at least a quarter of a second.
4. Compare debug values for the intended pose and a neutral pose.
5. Tune the relevant threshold in `rem/config.py`.

### Hands-on-face is flaky

The rule needs two recognized hands near the nose or mouth, fingers up, and three matching frames. Palm centers can be close together or the middle fingertips can meet while palms rest across the cheeks. Keep both hands visible to the camera and avoid covering the entire eye region. reM keeps the last face landmarks for 0.5 seconds to help through brief face-tracker dropouts.

### Head-grab or cheeks does not trigger

Both rules need two hand tracks. For head-grab, keep hands on opposite sides near the upper head. For cheeks, keep palms separated on opposite cheeks below the eye line. Use debug output to inspect `head_grab`, `cheeks`, and `cheek_zones`.

### Open-palm or explaining meme fires near the face

Face-adjacent hand poses are excluded from `open_palm` and `explaining_hands`. If a false trigger remains, use debug mode and adjust the nose-distance or motion thresholds in `rem/config.py`.

## Performance

The terminal prints average FPS and inference timing every 30 frames. If the preview is slow:

1. Close GPU- and CPU-heavy applications.
2. Use a lower-resolution webcam mode if your hardware supports it.
3. Keep debug mode off while not calibrating.
4. Keep the camera frame well lit so tracking does not repeatedly reacquire the face and hands.

## OBS

If OBS cannot capture reM or both apps compete for the webcam, follow the [OBS guide](OBS.md). Capture the `meme camera` window in OBS rather than opening the physical webcam in both applications.
