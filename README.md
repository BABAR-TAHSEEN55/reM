# reM

`reM` is a local webcam app that detects facial expressions and hand poses, then overlays a matching meme in the live camera preview. It runs on your machine with OpenCV and MediaPipe; webcam frames are processed locally by the app.

## Features

- Detects face expressions, hand shapes, hand motion, and face-relative hand placement.
- Plays static images and animated GIF memes with priorities, debounce, cooldown, and minimum display durations.
- Includes terminal metrics and landmark overlays for tuning detection thresholds.
- Provides commands to list configured memes and see activation instructions.
- Can be captured directly in OBS as a window source.

## Requirements

- Linux, macOS, or Windows with a working webcam.
- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment management.
- A desktop session capable of showing OpenCV windows.

Linux users also need permission to access the camera device. On many distributions, add your user to the `video` group, then log out and back in:

```bash
sudo usermod -aG video "$USER"
```

## Installation

1. Clone the repository and enter it.

   ```bash
   git clone https://github.com/BABAR-TAHSEEN55/reM.git
   cd reM
   ```

2. Install dependencies with uv.

   ```bash
   uv sync
   ```

3. Confirm the required model files exist in the repository root.

   ```text
   gesture_recognizer.task
   face_landmarker.task
   ```

4. List the configured memes to confirm the media assets load.

   ```bash
   uv run main.py --list-memes
   ```

## Running The Camera

Start the normal camera preview:

```bash
uv run main.py
```

Start with debug landmarks and live metrics:

```bash
uv run main.py --debug
```

The camera preview opens in a window named `meme camera`.

### Camera Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit the application. |
| `d` | Toggle gesture detection. This also clears active gesture state. |
| `h` | Dismiss the current meme and wait for all detected gestures to clear before allowing another trigger. |

The default camera index is `1`. Change `CAMERA_INDEX` in `rem/config.py` if the preview shows the wrong camera or cannot open one. Common alternatives are `0` for the first webcam and `2` for an additional USB camera.

## CLI Reference

```bash
# Show all available options and keyboard controls.
uv run main.py --help

# Show face/hand landmarks and live numeric metrics.
uv run main.py --debug

# List every configured gesture, media file, and priority.
uv run main.py --list-memes

# Explain how a specific meme is triggered.
uv run main.py --describe-meme hands_on_cheeks
```

`--list-memes` and `--describe-meme NAME` are discovery commands and exit without opening the camera.

## Gestures

Priorities resolve simultaneous candidates: a higher value wins. Detection also uses debounce and cooldown settings from `rem/config.py`.

| Name | Activation | Media | Priority |
| --- | --- | --- | ---: |
| `hands_on_face` | Both palms pressed together over the nose or mouth, fingers pointing up. Requires three consecutive frames. | `hands_on_face.jpg` | 90 |
| `head_grab` | Both palms on opposite sides near the top of the head. | `headgrab.jpeg` | 89 |
| `hands_on_cheeks` | Separated palms on opposite cheeks, below the eye line. | `cheeks.jpeg` | 87 |
| `t_sign` | Two nearby open hands form a perpendicular T shape. | `T_sign.jpg` | 85 |
| `explaining_hands` | Both hands move while neither is near the nose. | `hands_talking_to_wall.gif` | 80 |
| `left_nose_right_wave` | Left hand near the nose while the right hand waves vertically. | `dance_fox_hands_forward_wave_with_nose_touch.gif` | 60 |
| `mouth_round` | Puckered, rounded mouth with slight jaw opening. | `mouth_round.jpg` | 55 |
| `horse_smirk` | A pronounced asymmetric one-sided smile. | `horse_smirk.jpg` | 50 |
| `eyes_closed` | Both eyes closed or nearly closed. | `master_piece_squiting_too_much.jpg` | 45 |
| `tongue_out` | Mouth-open landmark exceeds its threshold. | `dog_tongue_out.jpg` | 40 |
| `smile` | Both mouth corners raised evenly. | `smile_dog.jpg` | 35 |
| `lips_down` | Both mouth corners pulled downward. | `beluga.jpg` | 30 |
| `open_palm` | An open palm held away from the face. | `hold_up_hand_forward.jpg` | 10 |

### Gesture Exclusions

- A T sign suppresses `open_palm` and `explaining_hands`.
- Hands together on the face, a head grab, cheeks, wave, and explaining hands are separated by face-relative hand positions.
- The app briefly holds face-hand pose state to stop explaining-hands motion from stealing a face pose during tracking jitter.
- An active explaining GIF is dismissed when a higher-priority face-hand pose appears.

## Debugging And Calibration

Run the app with `--debug` while testing a target pose and a neutral pose:

```bash
uv run main.py --debug
```

The preview shows the face mesh, hand skeletons, palm midpoint, and pose status. The terminal reports the values used by the current rules, including:

- Mouth gap and frown blendshape scores.
- Face width and each hand's distance to the nose.
- Hand-motion totals.
- Palm-to-palm distance, midpoint centering, lower-face bounds, and fingers-up state.
- Head-grab and cheek-zone matches.

To tune a flaky rule:

1. Hold the intended pose and record several debug lines.
2. Hold a neutral or incorrect pose and record several more lines.
3. Set the relevant threshold between the two observed ranges.
4. Repeat until the rule reliably triggers without false positives.

All thresholds, timing values, camera settings, keybindings, media files, and priorities live in `rem/config.py`.

## Adding A Meme

1. Add an image or GIF to `memes/`.
2. Register its filename, media type, duration, cooldown, priority, and debounce in `GESTURE_MEDIA` in `rem/config.py`.
3. Add a matching entry to `MEME_DESCRIPTIONS`.
4. Create one detector module in `rem/actions/` with a `matches(...)` function.
5. Import and evaluate that detector in `rem/actions/candidates.py`.
6. Add a test case in `tests/test_actions.py` for the expected match and relevant conflicts.
7. Verify the asset and discovery commands.

```bash
uv run main.py --list-memes
uv run main.py --describe-meme your_new_meme
uv run python -m unittest discover -s tests -v
```

## Project Layout

```text
main.py                 CLI entry point
rem/config.py           Thresholds, camera settings, media registry, descriptions
rem/models.py           Runtime and media data models
rem/actions/            One detector module per gesture
rem/actions/candidates.py Candidate composition and gesture exclusion logic
rem/actions/state.py    Debounce, cooldown, activation, and reset behavior
rem/media.py            Image/GIF decoding, resizing, and overlay rendering
rem/debug.py            Debug metrics and landmark rendering
rem/runtime.py          MediaPipe setup, camera loop, and display loop
tests/test_actions.py   Gesture regression tests
docs/OBS.md             OBS setup guide
docs/TROUBLESHOOTING.md Common setup and tracking fixes
```

## Testing

Run the action regression suite:

```bash
uv run python -m unittest discover -s tests -v
```

Run syntax checks:

```bash
uv run python -m compileall -q main.py rem tests
```

## OBS

Use the `meme camera` preview as an OBS window source. See [docs/OBS.md](docs/OBS.md) for a complete setup guide, capture options for X11 and Wayland, transforms, audio, and troubleshooting.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for camera, dependency, Qt, MediaPipe, gesture-detection, and OBS capture fixes.
