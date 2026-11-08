import os
import time
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageSequence

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

ROOT = Path(__file__).resolve().parents[1]
GESTURE_MODEL_PATH = ROOT / "gesture_recognizer.task"
FACE_MODEL_PATH = ROOT / "face_landmarker.task"
MEME_WIDTH = 420
DEBOUNCE_SECONDS = 0.25
IMAGE_MIN_LOCK_SECONDS = 2.0
GIF_MIN_LOCK_SECONDS = 5.0
MOUTH_OPEN_THRESHOLD = 0.03
MOTION_WINDOW_SECONDS = 0.7


@dataclass(frozen=True)
class MediaConfig:
    file: str
    type: str
    duration: float
    cooldown: float
    priority: int


GESTURE_MEDIA = {
    "lips_down": MediaConfig("beluga.jpg", "image", 2.0, 3.0, 30),
    "left_nose_right_wave": MediaConfig(
        "dance_fox_hands_forward_wave_with_nose_touch.gif", "gif", 5.0, 5.0, 60
    ),
    "tongue_out": MediaConfig("dog_tongue_out.jpg", "image", 2.0, 3.0, 40),
    "hands_on_face": MediaConfig("hands_on_face.jpg", "image", 2.0, 3.0, 50),
    "explaining_hands": MediaConfig("hands_talking_to_wall.gif", "gif", 5.0, 5.0, 80),
    "open_palm": MediaConfig("hold_up_hand_forward.jpg", "image", 2.0, 2.0, 10),
}


@dataclass
class LoadedMedia:
    type: str
    frames: list[np.ndarray]
    durations: list[float]


@dataclass
class ActiveMedia:
    name: str
    started_at: float
    expires_at: float


@dataclass
class RuntimeState:
    detection_enabled: bool = True
    active: ActiveMedia | None = None
    candidate: str | None = None
    candidate_started_at: float | None = None
    cooldown_until: dict[str, float] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    hand_history: dict[str, deque[tuple[float, float, float]]] = field(
        default_factory=lambda: {"Left": deque(), "Right": deque()}
    )
    suppress_until_clear: bool = False


def normalize_bgra(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if image.shape[2] == 4:
        return image
    raise ValueError("Media must have one, three, or four channels")


def resize_media(image):
    height = round(image.shape[0] * MEME_WIDTH / image.shape[1])
    return cv2.resize(image, (MEME_WIDTH, height), interpolation=cv2.INTER_AREA)


def load_media():
    media = {}
    for name, config in GESTURE_MEDIA.items():
        path = ROOT / "memes" / config.file
        if not path.is_file():
            raise FileNotFoundError(f"Could not load configured media {path}")

        if config.type == "image":
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not decode image {path}")
            media[name] = LoadedMedia(
                "image", [resize_media(normalize_bgra(image))], [0.0]
            )
        elif config.type == "gif":
            frames, durations = [], []
            with Image.open(path) as gif:
                for frame in ImageSequence.Iterator(gif):
                    rgba = np.array(frame.convert("RGBA"))
                    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                    frames.append(resize_media(bgra))
                    durations.append(max(frame.info.get("duration", 100), 20) / 1000.0)
            if not frames:
                raise ValueError(f"GIF has no frames: {path}")
            media[name] = LoadedMedia("gif", frames, durations)
        else:
            raise ValueError(f"Unsupported media type {config.type!r} for {name}")
    return media


def overlay_media(frame, media, center):
    """Alpha-blend media centered on the detected face, clamped to frame edges."""
    frame_h, frame_w = frame.shape[:2]
    media_h, media_w = media.shape[:2]
    x = min(max(center[0] - media_w // 2, 0), max(frame_w - media_w, 0))
    y = min(max(center[1] - media_h // 2, 0), max(frame_h - media_h, 0))
    dst_x2, dst_y2 = min(x + media_w, frame_w), min(y + media_h, frame_h)
    if dst_x2 <= x or dst_y2 <= y:
        return

    overlay = media[: dst_y2 - y, : dst_x2 - x]
    alpha = overlay[:, :, 3:4] / 255.0
    frame[y:dst_y2, x:dst_x2] = (
        alpha * overlay[:, :, :3] + (1.0 - alpha) * frame[y:dst_y2, x:dst_x2]
    ).astype(np.uint8)


def physical_hands(gesture_result):
    hands = {}
    for landmarks, handedness in zip(
        gesture_result.hand_landmarks, gesture_result.handedness, strict=True
    ):
        if not handedness:
            continue
        # MediaPipe's handedness assumes a mirrored selfie image. Detection uses raw video.
        reported = handedness[0].category_name
        physical_side = "Right" if reported == "Left" else "Left"
        hands[physical_side] = landmarks
    return hands


def update_hand_history(state, hands, now):
    for side, history in state.hand_history.items():
        if side in hands:
            wrist = hands[side][0]
            history.append((now, wrist.x, wrist.y))
        while history and now - history[0][0] > MOTION_WINDOW_SECONDS:
            history.popleft()


def hand_near_nose(hand, nose, face_width):
    threshold = max(face_width * 0.25, 0.05)
    return min(hypot(point.x - nose.x, point.y - nose.y) for point in hand) < threshold


def hand_on_face(hand, face):
    face_left, face_right = face[234], face[454]
    face_top, face_bottom = face[10], face[152]
    margin = 0.08
    palm = hand[9]
    return (
        min(face_left.x, face_right.x) - margin
        <= palm.x
        <= max(face_left.x, face_right.x) + margin
        and min(face_top.y, face_bottom.y) - margin
        <= palm.y
        <= max(face_top.y, face_bottom.y) + margin
    )


def vertical_wave(history):
    if len(history) < 3 or history[-1][0] - history[0][0] < 0.25:
        return False
    ys = [point[2] for point in history]
    return max(ys) - min(ys) >= 0.08


def hand_motion(history):
    if len(history) < 3:
        return 0.0
    return sum(
        hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(history, list(history)[1:])
    )


def is_frowning(face_blendshapes):
    if not face_blendshapes:
        return False
    scores = {
        category.category_name: category.score for category in face_blendshapes[0]
    }
    return (
        scores.get("mouthFrownLeft", 0.0) >= 0.35
        and scores.get("mouthFrownRight", 0.0) >= 0.35
    )


def is_open_palm(gesture_result):
    return any(
        categories and categories[0].category_name == "Open_Palm"
        for categories in gesture_result.gestures
    )


def detect_candidates(face, face_blendshapes, hands, gesture_result, state):
    if face is None:
        return set()

    candidates = set()
    nose = face[1]
    face_width = abs(face[454].x - face[234].x)
    if is_frowning(face_blendshapes):
        candidates.add("lips_down")
    if abs(face[13].y - face[14].y) > MOUTH_OPEN_THRESHOLD:
        candidates.add("tongue_out")
    if is_open_palm(gesture_result):
        candidates.add("open_palm")

    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    if left_hand and right_hand:
        if hand_on_face(left_hand, face) and hand_on_face(right_hand, face):
            candidates.add("hands_on_face")
        if hand_near_nose(left_hand, nose, face_width) and vertical_wave(
            state.hand_history["Right"]
        ):
            candidates.add("left_nose_right_wave")
        if (
            hand_motion(state.hand_history["Left"])
            + hand_motion(state.hand_history["Right"])
            >= 0.24
        ):
            candidates.add("explaining_hands")
    return candidates


def reset_detection_state(state):
    state.active = None
    state.candidate = None
    state.candidate_started_at = None
    state.cooldown_until.clear()
    state.consumed.clear()
    state.suppress_until_clear = False
    for history in state.hand_history.values():
        history.clear()


def update_active_media(state, candidates, has_face, now):
    if state.active and now >= state.active.expires_at:
        state.active = None

    state.consumed.intersection_update(candidates)
    if state.suppress_until_clear:
        if not candidates:
            state.suppress_until_clear = False
        else:
            return
    if not has_face:
        state.candidate = None
        state.candidate_started_at = None
        return

    eligible = [
        name
        for name in candidates
        if name not in state.consumed and now >= state.cooldown_until.get(name, 0.0)
    ]
    selected = max(
        eligible, key=lambda name: GESTURE_MEDIA[name].priority, default=None
    )
    if selected != state.candidate:
        state.candidate = selected
        state.candidate_started_at = now if selected else None
        return
    if (
        selected is None
        or state.active is not None
        or now - state.candidate_started_at < DEBOUNCE_SECONDS
    ):
        return

    config = GESTURE_MEDIA[selected]
    minimum_lock = (
        GIF_MIN_LOCK_SECONDS if config.type == "gif" else IMAGE_MIN_LOCK_SECONDS
    )
    duration = max(config.duration, minimum_lock)
    state.active = ActiveMedia(selected, now, now + duration)
    state.cooldown_until[selected] = now + config.cooldown
    state.consumed.add(selected)
    state.candidate = None
    state.candidate_started_at = None


def active_frame(active, media, now):
    loaded = media[active.name]
    if loaded.type == "image":
        return loaded.frames[0]

    total_duration = sum(loaded.durations)
    elapsed = now - active.started_at
    completed_loops = round(elapsed / total_duration)
    if completed_loops and abs(elapsed - completed_loops * total_duration) < 1e-9:
        elapsed = 0.0
    else:
        elapsed %= total_duration
    for frame, duration in zip(loaded.frames, loaded.durations, strict=True):
        if elapsed < duration:
            return frame
        elapsed -= duration
    return loaded.frames[-1]


def main(camera_index=1):
    media = load_media()
    gesture_options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    gesture_detector = vision.GestureRecognizer.create_from_options(gesture_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        gesture_detector.close()
        face_detector.close()
        raise RuntimeError(f"Could not open camera {camera_index}")

    state = RuntimeState()
    start = time.monotonic()
    last_timestamp = -1
    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            display_frame = cv2.flip(raw_frame, 1)

            if state.detection_enabled:
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((now - start) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                gesture_result = gesture_detector.recognize_for_video(
                    mp_image, timestamp
                )
                face_result = face_detector.detect_for_video(mp_image, timestamp)
                face = (
                    face_result.face_landmarks[0]
                    if face_result.face_landmarks
                    else None
                )
                hands = physical_hands(gesture_result)
                update_hand_history(state, hands, now)
                candidates = detect_candidates(
                    face, face_result.face_blendshapes, hands, gesture_result, state
                )
                update_active_media(state, candidates, face is not None, now)

                if state.active and face is not None:
                    nose = face[1]
                    center = (
                        int((1.0 - nose.x) * raw_frame.shape[1]),
                        int(nose.y * raw_frame.shape[0]),
                    )
                    overlay_media(
                        display_frame, active_frame(state.active, media, now), center
                    )

            cv2.imshow("meme camera", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                state.detection_enabled = not state.detection_enabled
                reset_detection_state(state)
            elif key == ord("h"):
                state.active = None
                state.suppress_until_clear = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gesture_detector.close()
        face_detector.close()


if __name__ == "__main__":
    main()
