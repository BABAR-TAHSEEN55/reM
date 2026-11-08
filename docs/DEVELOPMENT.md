# Development

## Local Workflow

Set up the environment as described in [Setup](SETUP.md), then run the targeted test suite from the repository root:

```bash
uv run python -m unittest -v tests.test_actions
```

The repository's tests directory does not contain `__init__.py`, so the reliable command is the explicit module invocation above. `python -m unittest discover -v` may discover zero tests, and `python tests/test_actions.py` may not resolve the top-level `rem` package because it changes the import path.

## Repository Conventions

- Keep shared thresholds, paths, controls, and media mappings in `rem/config.py`.
- Keep a detector module narrowly focused on one gesture or reusable geometry rule.
- Add cross-detector precedence in `rem/actions/candidates.py`, not in the rendering loop.
- Keep activation timing and lifecycle behavior in `rem/actions/state.py`.
- Treat MediaPipe landmark values as normalized image coordinates.
- Preserve the mapping-name contract: a candidate name returned by detection must be a key in `GESTURE_MEDIA`.

## Add A Gesture

Adding a gesture is an end-to-end change. Follow this sequence:

1. Add any reusable threshold constants to `rem/config.py`.
2. Add a narrow predicate module under `rem/actions/`, or extend an existing appropriate module.
3. Import and call the predicate from `rem/actions/candidates.py`.
4. Decide whether the gesture is independent, or belongs in the mutually exclusive two-hand branch. Place it deliberately in the branch order.
5. Add a `GESTURE_MEDIA` entry with a unique name, existing media filename, valid `image`/`gif` type, priority, duration, cooldown, and debounce.
6. Add the same name to `MEME_DESCRIPTIONS` for `--describe-meme` support.
7. Add the image/GIF under `memes/`.
8. Add focused unit tests in `tests/test_actions.py` or a new test module.
9. Verify the mapping with `uv run main.py --list-memes` and calibrate with `uv run main.py --debug`.

Do not add a `GESTURE_MEDIA` entry without a detector: it will load an asset but can never activate. Do not return a detector name without a mapping: `update_active_media()` will fail when it reads the missing media configuration.

## Add Or Replace Media

Images are loaded by OpenCV, and GIFs by Pillow. At startup, reM validates that every configured path exists and decodes successfully. During loading, it converts input to BGRA and resizes it to `MEME_WIDTH`; transparent pixels are alpha blended over the camera image.

To replace an existing asset without changing behavior:

1. Put the replacement in `memes/`.
2. Keep the mapping filename in `GESTURE_MEDIA`, or update only that filename.
3. Start reM and confirm it reaches the camera log line.
4. Trigger the gesture and check edge clipping, aspect ratio, and transparency.

GIF timing uses the source frame metadata. Missing timing defaults to 100 ms and any delay below 20 ms is clamped.

## Write Tests

The current tests use `unittest` and lightweight `SimpleNamespace` landmark objects. This avoids requiring a camera or running MediaPipe inference for logic tests.

Useful test targets include:

- Boundary values just above and below a threshold.
- Gesture precedence when two detectors can match.
- Debounce, cooldown, consumed-state, and dismissal behavior.
- Hand-history timing and motion-window expiration.
- Asset validation and media alpha blending using temporary fixtures.

Camera capture and OpenCV GUI behavior are hardware-dependent and are not covered by the existing unit tests. Test these manually in a desktop environment.

## Validate A Change

For normal gesture or state changes:

```bash
uv run python -m unittest -v tests.test_actions
uv run main.py --list-memes
```

Then perform a manual debug run:

```bash
uv run main.py --debug
```

Check that expected landmark/score values cross the intended threshold, ambiguous gestures resolve according to priority and branch order, the overlay does not retrigger while a pose is held, and `d`/`h` reset or suppress state as documented.

## Current Testing Scope

`tests/test_actions.py` currently covers head-grab versus cheeks, hands-on-face confirmation frames, T-sign suppression of open-palm/explaining branches, and head-grab debounce. CLI parsing, model initialization, camera capture, GIF rendering, media loading, and most thresholds remain manual-test areas. Add coverage alongside changes in those areas.
