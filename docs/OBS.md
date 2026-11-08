# OBS Setup

reM renders its meme overlay inside an OpenCV window named `meme camera`. The most reliable OBS workflow is to capture that window rather than trying to let OBS and reM open the same physical webcam at once.

## Before You Start

1. Install OBS Studio for your operating system.
2. Confirm reM starts normally:

   ```bash
   uv run main.py
   ```

3. Keep the `meme camera` window open and visible. OBS needs an existing window to select it.

## Add reM To A Scene

1. Open OBS and create or select a scene.
2. In **Sources**, click **+**.
3. Select **Window Capture**.
4. Create a new source, for example `reM camera`.
5. Select the window named `meme camera`.
6. Confirm the preview shows the mirrored camera image and meme overlay.
7. Resize or crop the source in the OBS canvas to fit the scene.

The gesture overlay is already baked into the preview window. No extra OBS filter is required.

## Recommended Source Settings

- Set the source transform to fit your canvas, then crop unwanted borders if your desktop environment adds them.
- Keep the reM window unobscured. Some Linux capture methods stop updating minimized windows.
- Start reM before starting a recording or stream so MediaPipe warms up before the scene is live.
- Use `uv run main.py --debug` only while tuning. The landmark overlay is intentionally visible in the captured window.

## Add Audio

reM does not create audio. Add microphone and desktop-audio sources in OBS separately:

1. Go to **Settings > Audio** and select your microphone and desktop devices.
2. Check the OBS audio mixer before recording.
3. Use **Advanced Audio Properties** if you need tracks for microphone and desktop audio separately.

## X11 And Wayland Notes

### X11

Use **Window Capture** and select `meme camera`. If the selected window is black, try a different capture method in the source properties or disable desktop-compositor effects temporarily.

### Wayland

Window capture availability depends on your desktop environment and OBS build. Prefer PipeWire-based capture when offered. If `meme camera` does not appear as a window source:

1. Add **Screen Capture (PipeWire)** or the equivalent portal-based source.
2. Select the monitor containing the reM window.
3. Crop the source in OBS to the window bounds.
4. Avoid minimizing reM while capturing.

## Webcam Device Conflicts

Most webcams can only be opened by one application at a time. If OBS opens the physical webcam first, reM may report that it cannot open the camera. If reM opens it first, OBS may not be able to add the same webcam as a separate Video Capture Device.

Capture the reM window instead. This gives OBS the processed camera image and avoids the device conflict.

## Use OBS Virtual Camera

OBS Virtual Camera is useful when another application needs the final OBS scene:

1. Build the scene with the reM window source.
2. Click **Start Virtual Camera** in OBS.
3. Select `OBS Virtual Camera` in Zoom, Discord, browser meeting software, or another app.

The virtual camera is the output of OBS, not an input for reM.

## Stream Or Record Checklist

1. Start reM and confirm the camera is detecting your face.
2. Add or verify the `meme camera` window source in OBS.
3. Test one or two gestures.
4. Verify microphone levels.
5. Record a short test clip.
6. Start recording or streaming.

## Common OBS Problems

| Problem | Fix |
| --- | --- |
| reM window is black in OBS | Try another capture method, keep the window visible, or use PipeWire screen capture on Wayland. |
| Webcam cannot open in reM | Close other apps using the physical webcam; capture reM's window in OBS instead. |
| Gestures show debug dots on stream | Restart reM without `--debug`. |
| Overlay is cut off | Resize the reM window or use OBS transform/crop controls. |
| Video is delayed | Close CPU-heavy applications, reduce camera load, and avoid capturing the same scene recursively. |
