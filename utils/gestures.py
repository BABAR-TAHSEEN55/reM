import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import atan2, degrees, hypot
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
IMAGE_MIN_LOCK_SECONDS = 2.0
GIF_MIN_LOCK_SECONDS = 5.0
MOUTH_OPEN_THRESHOLD = 0.03
FROWN_THRESHOLD = 0.28
SMIRK_SMILE_THRESHOLD = 0.40
SMIRK_ASYMMETRY_THRESHOLD = 0.20
EYE_CLOSED_THRESHOLD = 0.60
ROUND_MOUTH_PUCKER_THRESHOLD = 0.40
ROUND_MOUTH_JAW_THRESHOLD = 0.10
SMILE_THRESHOLD = 0.35
SMILE_ASYMMETRY_MAX = 0.20
MOTION_WINDOW_SECONDS = 0.5
VERTICAL_WAVE_MIN_SECONDS = 0.15
VERTICAL_WAVE_MIN_DISTANCE = 0.065
EXPLAINING_MOTION_THRESHOLD = 0.20
FACE_INFERENCE_INTERVAL = 2
FACE_FALLBACK_SECONDS = 0.5
HANDS_ON_FACE_CONFIRMATION_FRAMES = 3
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INFERENCE_SIZE = (640, 360)
FPS_REPORT_INTERVAL = 30
DEBUG_METRICS_INTERVAL = 0.2
FACE_HAND_POSE_HOLD_SECONDS = 0.35
T_SIGN_PALM_DISTANCE_FACTOR = 2.0
T_SIGN_DIRECTION_RATIO = 0.75
T_SIGN_PERPENDICULAR_ERROR_DEGREES = 25.0
HEAD_GRAB_EYE_MARGIN_FACTOR = 0.30
CHEEK_BELOW_EYE_FACTOR = 0.32

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class MediaConfig:
    file: str
    type: str
    duration: float
    cooldown: float
    priority: int
    debounce: float


@dataclass(frozen=True)
class HandsOnFaceMetrics:
    face_width: float
    palm_distance: float
    palm_distance_threshold: float
    nose_offset: float
    nose_offset_threshold: float
    midpoint_x: float
    midpoint_y: float
    lower_face_top: float
    chin_y: float
    together: bool
    centered: bool
    on_lower_face: bool
    fingers_up: bool

    @property
    def matches(self):
        return (
            self.together and self.centered and self.on_lower_face and self.fingers_up
        )


@dataclass(frozen=True)
class TSignMetrics:
    palm_distance: float
    palm_distance_threshold: float
    left_angle: float
    right_angle: float
    perpendicular_error: float
    hands_close: bool
    axis_aligned: bool
    hands_open: bool

    @property
    def matches(self):
        return self.hands_close and self.axis_aligned and self.hands_open


GESTURE_MEDIA = {
    "lips_down": MediaConfig("beluga.jpg", "image", 2.0, 3.0, 30, 0.12),
    "left_nose_right_wave": MediaConfig(
        "dance_fox_hands_forward_wave_with_nose_touch.gif", "gif", 5.0, 5.0, 60, 0.20
    ),
    "tongue_out": MediaConfig("dog_tongue_out.jpg", "image", 2.0, 3.0, 40, 0.10),
    "hands_on_face": MediaConfig("hands_on_face.jpg", "image", 2.0, 3.0, 90, 0.18),
    "head_grab": MediaConfig("headgrab.jpeg", "image", 2.0, 3.0, 89, 0.05),
    "hands_on_cheeks": MediaConfig("cheeks.jpeg", "image", 2.0, 3.0, 87, 0.05),
    "t_sign": MediaConfig("T_sign.jpg", "image", 2.0, 3.0, 85, 0.15),
    "explaining_hands": MediaConfig(
        "hands_talking_to_wall.gif", "gif", 5.0, 5.0, 80, 0.20
    ),
    "mouth_round": MediaConfig("mouth_round.jpg", "image", 2.0, 3.0, 55, 0.15),
    "horse_smirk": MediaConfig("horse_smirk.jpg", "image", 2.0, 3.0, 50, 0.15),
    "eyes_closed": MediaConfig(
        "master_piece_squiting_too_much.jpg", "image", 2.0, 3.0, 45, 0.15
    ),
    "open_palm": MediaConfig("hold_up_hand_forward.jpg", "image", 2.0, 2.0, 10, 0.10),
    "smile": MediaConfig("smile_dog.jpg", "image", 2.0, 3.0, 35, 0.15),
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
    hands_on_face_frames: int = 0
    face_hand_pose_until: float = 0.0
    last_debug_metrics_at: float = 0.0
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
    distance, threshold = hand_near_nose_metrics(hand, nose, face_width)
    return distance < threshold


def hand_near_nose_metrics(hand, nose, face_width):
    threshold = max(face_width * 0.25, 0.05)
    return min(hypot(point.x - nose.x, point.y - nose.y) for point in hand), threshold


def hands_together_on_face_metrics(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    left_palm, right_palm = left[9], right[9]

    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
    palm_distance_threshold = face_width * 0.6
    midpoint_x = (left_palm.x + right_palm.x) / 2
    midpoint_y = (left_palm.y + right_palm.y) / 2
    nose_offset = abs(midpoint_x - nose.x)
    nose_offset_threshold = face_width * 0.35
    lower_face_top = face[168].y - 0.05
    chin_y = face[152].y
    together = palm_distance < palm_distance_threshold
    centered = nose_offset < nose_offset_threshold
    on_lower_face = lower_face_top <= midpoint_y <= chin_y
    fingers_up = left[12].y < left[0].y and right[12].y < right[0].y

    return HandsOnFaceMetrics(
        face_width,
        palm_distance,
        palm_distance_threshold,
        nose_offset,
        nose_offset_threshold,
        midpoint_x,
        midpoint_y,
        lower_face_top,
        chin_y,
        together,
        centered,
        on_lower_face,
        fingers_up,
    )


def hands_together_on_face(left, right, face):
    return hands_together_on_face_metrics(left, right, face).matches


def hands_on_cheeks(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    eye_line_y = (face[33].y + face[263].y) / 2
    chin_y = face[152].y
    left_palm, right_palm = left[9], right[9]
    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)

    def on_cheek(palm, side, is_left):
        return (
            hypot(palm.x - side.x, palm.y - side.y) < face_width * 0.45
            and eye_line_y + face_width * CHEEK_BELOW_EYE_FACTOR <= palm.y <= chin_y
            and (
                palm.x < nose.x - face_width * 0.15
                if is_left
                else palm.x > nose.x + face_width * 0.15
            )
        )

    return palm_distance > face_width * 0.6 and (
        (
            on_cheek(left_palm, face[234], True)
            and on_cheek(right_palm, face[454], False)
        )
        or (
            on_cheek(left_palm, face[454], False)
            and on_cheek(right_palm, face[234], True)
        )
    )


def hands_on_head(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    eye_line_y = (face[33].y + face[263].y) / 2
    head_grab_limit_y = eye_line_y + face_width * HEAD_GRAB_EYE_MARGIN_FACTOR
    left_palm, right_palm = left[9], right[9]
    return (
        left_palm.y < head_grab_limit_y
        and right_palm.y < head_grab_limit_y
        and (
            (
                left_palm.x < nose.x - face_width * 0.1
                and right_palm.x > nose.x + face_width * 0.1
            )
            or (
                right_palm.x < nose.x - face_width * 0.1
                and left_palm.x > nose.x + face_width * 0.1
            )
        )
    )


def hand_open(hand):
    wrist = hand[0]
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))
    return (
        sum(
            hypot(hand[tip].x - wrist.x, hand[tip].y - wrist.y)
            > hypot(hand[pip].x - wrist.x, hand[pip].y - wrist.y)
            for tip, pip in fingers
        )
        >= 3
    )


def hand_angle(hand):
    wrist, middle_tip = hand[0], hand[12]
    return degrees(atan2(middle_tip.y - wrist.y, middle_tip.x - wrist.x))


def t_sign_metrics(left, right):
    left_palm, right_palm = left[9], right[9]
    left_size = hypot(left[0].x - left_palm.x, left[0].y - left_palm.y)
    right_size = hypot(right[0].x - right_palm.x, right[0].y - right_palm.y)
    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
    palm_distance_threshold = (left_size + right_size) / 2 * T_SIGN_PALM_DISTANCE_FACTOR
    left_angle, right_angle = hand_angle(left), hand_angle(right)
    angle_difference = abs(left_angle - right_angle) % 180
    perpendicular_error = abs(90 - angle_difference)
    left_horizontal = (
        abs(left[12].y - left[0].y)
        <= abs(left[12].x - left[0].x) * T_SIGN_DIRECTION_RATIO
    )
    right_horizontal = (
        abs(right[12].y - right[0].y)
        <= abs(right[12].x - right[0].x) * T_SIGN_DIRECTION_RATIO
    )
    left_vertical = (
        abs(left[12].x - left[0].x)
        <= abs(left[12].y - left[0].y) * T_SIGN_DIRECTION_RATIO
    )
    right_vertical = (
        abs(right[12].x - right[0].x)
        <= abs(right[12].y - right[0].y) * T_SIGN_DIRECTION_RATIO
    )
    return TSignMetrics(
        palm_distance,
        palm_distance_threshold,
        left_angle,
        right_angle,
        perpendicular_error,
        palm_distance < palm_distance_threshold,
        (left_horizontal and right_vertical) or (left_vertical and right_horizontal),
        hand_open(left) and hand_open(right),
    )


def vertical_wave(history):
    if len(history) < 3 or history[-1][0] - history[0][0] < VERTICAL_WAVE_MIN_SECONDS:
        return False
    ys = [point[2] for point in history]
    return max(ys) - min(ys) >= VERTICAL_WAVE_MIN_DISTANCE


def hand_motion(history):
    if len(history) < 3:
        return 0.0
    return sum(
        hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(history, list(history)[1:])
    )


def blendshape_scores(face_blendshapes):
    if not face_blendshapes:
        return {}
    return {category.category_name: category.score for category in face_blendshapes[0]}


def is_frowning(scores):
    left_score, right_score = frown_scores(scores)
    return left_score >= FROWN_THRESHOLD and right_score >= FROWN_THRESHOLD


def frown_scores(scores):
    return scores.get("mouthFrownLeft", 0.0), scores.get("mouthFrownRight", 0.0)


def smile_scores(scores):
    return scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0)


def is_smirk(scores):
    left_score, right_score = smile_scores(scores)
    return (
        max(left_score, right_score) >= SMIRK_SMILE_THRESHOLD
        and abs(left_score - right_score) >= SMIRK_ASYMMETRY_THRESHOLD
    )


def is_eyes_closed(scores):
    return (
        scores.get("eyeBlinkLeft", 0.0) >= EYE_CLOSED_THRESHOLD
        and scores.get("eyeBlinkRight", 0.0) >= EYE_CLOSED_THRESHOLD
    )


def is_round_mouth(scores):
    return (
        scores.get("mouthPucker", 0.0) >= ROUND_MOUTH_PUCKER_THRESHOLD
        and scores.get("jawOpen", 0.0) >= ROUND_MOUTH_JAW_THRESHOLD
    )


def is_smiling(scores):
    left_score, right_score = smile_scores(scores)
    return (
        left_score >= SMILE_THRESHOLD
        and right_score >= SMILE_THRESHOLD
        and abs(left_score - right_score) < SMILE_ASYMMETRY_MAX
    )


def is_open_palm(gesture_result):
    return any(
        categories and categories[0].category_name == "Open_Palm"
        for categories in gesture_result.gestures
    )


def log_debug_metrics(face, face_blendshapes, hands, gesture_result, state, now):
    if now - state.last_debug_metrics_at < DEBUG_METRICS_INTERVAL:
        return
    state.last_debug_metrics_at = now

    if face is None:
        print("debug | face=missing", flush=True)
        return

    mouth_gap = abs(face[13].y - face[14].y)
    scores = blendshape_scores(face_blendshapes)
    frown_left, frown_right = frown_scores(scores)
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    nose_distances = {
        side: hand_near_nose_metrics(hand, nose, face_width)
        for side, hand in hands.items()
    }
    hands_near_nose = any(
        distance < threshold for distance, threshold in nose_distances.values()
    )
    nose_distance_text = (
        ",".join(
            f"{side}={distance:.3f}/{threshold:.3f}"
            for side, (distance, threshold) in nose_distances.items()
        )
        or "n/a"
    )
    prefix = (
        f"debug | mouth={mouth_gap:.3f}/{MOUTH_OPEN_THRESHOLD:.3f} "
        f"frown={frown_left:.2f},{frown_right:.2f}/{FROWN_THRESHOLD:.2f} "
        f"face_width={face_width:.3f} nose={nose_distance_text} "
        f"near_nose={int(hands_near_nose)} "
        f"open_palm={int(is_open_palm(gesture_result))} "
        f"motion={hand_motion(state.hand_history['Left']):.3f},"
        f"{hand_motion(state.hand_history['Right']):.3f}/{EXPLAINING_MOTION_THRESHOLD:.3f}"
    )
    if not left_hand or not right_hand:
        print(f"{prefix} | hands_together=n/a", flush=True)
        return

    metrics = hands_together_on_face_metrics(left_hand, right_hand, face)
    head_grab = hands_on_head(left_hand, right_hand, face)
    cheeks = hands_on_cheeks(left_hand, right_hand, face)
    print(
        f"{prefix} | palms={metrics.palm_distance:.3f}/{metrics.palm_distance_threshold:.3f} "
        f"center={metrics.nose_offset:.3f}/{metrics.nose_offset_threshold:.3f} "
        f"mid_y={metrics.midpoint_y:.3f}/{metrics.lower_face_top:.3f}-{metrics.chin_y:.3f} "
        f"strict=together={int(metrics.together)} centered={int(metrics.centered)} "
        f"lower_face={int(metrics.on_lower_face)} fingers_up={int(metrics.fingers_up)} "
        f"confirm={state.hands_on_face_frames}/{HANDS_ON_FACE_CONFIRMATION_FRAMES} "
        f"head_grab={int(head_grab)} cheeks={int(cheeks)}",
        flush=True,
    )


def mirrored_point(landmark, frame_width, frame_height):
    return mirrored_coordinates(landmark.x, landmark.y, frame_width, frame_height)


def mirrored_coordinates(x, y, frame_width, frame_height):
    return (
        min(max(round((1.0 - x) * frame_width), 0), frame_width - 1),
        min(max(round(y * frame_height), 0), frame_height - 1),
    )


def draw_debug_landmarks(frame, face, hands, state):
    frame_height, frame_width = frame.shape[:2]
    if face is not None:
        for landmark in face:
            cv2.circle(
                frame,
                mirrored_point(landmark, frame_width, frame_height),
                1,
                (80, 80, 80),
                -1,
            )
        for index, color in (
            (1, (0, 255, 255)),
            (168, (255, 255, 0)),
            (152, (0, 255, 0)),
        ):
            cv2.circle(
                frame,
                mirrored_point(face[index], frame_width, frame_height),
                4,
                color,
                -1,
            )

    colors = {"Left": (255, 120, 0), "Right": (255, 0, 255)}
    for side, hand in hands.items():
        color = colors[side]
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                mirrored_point(hand[start], frame_width, frame_height),
                mirrored_point(hand[end], frame_width, frame_height),
                color,
                2,
            )
        for landmark in hand:
            cv2.circle(
                frame, mirrored_point(landmark, frame_width, frame_height), 3, color, -1
            )

    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    if face is not None and left_hand and right_hand:
        metrics = hands_together_on_face_metrics(left_hand, right_hand, face)
        head_grab = hands_on_head(left_hand, right_hand, face)
        cheeks = hands_on_cheeks(left_hand, right_hand, face)
        midpoint = mirrored_coordinates(
            metrics.midpoint_x, metrics.midpoint_y, frame_width, frame_height
        )
        cv2.circle(frame, midpoint, 5, (0, 255, 255), -1)
        status = (
            f"strict T{int(metrics.together)} C{int(metrics.centered)} "
            f"L{int(metrics.on_lower_face)} F{int(metrics.fingers_up)} "
            f"{state.hands_on_face_frames}/{HANDS_ON_FACE_CONFIRMATION_FRAMES} "
            f"head {int(head_grab)} cheeks {int(cheeks)}"
        )
    else:
        status = f"strict n/a {state.hands_on_face_frames}/{HANDS_ON_FACE_CONFIRMATION_FRAMES}"
    cv2.putText(
        frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )


def detect_candidates(face, face_blendshapes, hands, gesture_result, state, now):
    if face is None:
        return set()

    candidates = set()
    nose = face[1]
    face_width = abs(face[454].x - face[234].x)
    scores = blendshape_scores(face_blendshapes)
    if is_frowning(scores):
        candidates.add("lips_down")
    if abs(face[13].y - face[14].y) > MOUTH_OPEN_THRESHOLD:
        candidates.add("tongue_out")
    if is_round_mouth(scores):
        candidates.add("mouth_round")
    if is_smirk(scores):
        candidates.add("horse_smirk")
    if is_eyes_closed(scores):
        candidates.add("eyes_closed")
    if is_smiling(scores):
        candidates.add("smile")
    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    hands_near_nose = any(
        hand_near_nose(hand, nose, face_width) for hand in hands.values()
    )
    t_sign = (
        left_hand is not None
        and right_hand is not None
        and t_sign_metrics(left_hand, right_hand).matches
    )
    if t_sign:
        candidates.add("t_sign")
    if is_open_palm(gesture_result) and not hands_near_nose and not t_sign:
        candidates.add("open_palm")
    if left_hand and right_hand:
        left_near_nose = hand_near_nose(left_hand, nose, face_width)
        right_near_nose = hand_near_nose(right_hand, nose, face_width)
        if hands_on_head(left_hand, right_hand, face):
            state.hands_on_face_frames = 0
            state.face_hand_pose_until = now + FACE_HAND_POSE_HOLD_SECONDS
            candidates.add("head_grab")
        elif hands_on_cheeks(left_hand, right_hand, face):
            state.hands_on_face_frames = 0
            state.face_hand_pose_until = now + FACE_HAND_POSE_HOLD_SECONDS
            candidates.add("hands_on_cheeks")
        elif t_sign:
            state.hands_on_face_frames = 0
        elif (
            left_near_nose
            and right_near_nose
            and hands_together_on_face(left_hand, right_hand, face)
        ):
            state.hands_on_face_frames += 1
            state.face_hand_pose_until = now + FACE_HAND_POSE_HOLD_SECONDS
            if state.hands_on_face_frames >= HANDS_ON_FACE_CONFIRMATION_FRAMES:
                candidates.add("hands_on_face")
        elif left_near_nose and not right_near_nose:
            state.hands_on_face_frames = 0
            if vertical_wave(state.hand_history["Right"]):
                candidates.add("left_nose_right_wave")
        elif (
            not left_near_nose
            and not right_near_nose
            and now >= state.face_hand_pose_until
        ):
            state.hands_on_face_frames = 0
            if (
                hand_motion(state.hand_history["Left"])
                + hand_motion(state.hand_history["Right"])
                >= EXPLAINING_MOTION_THRESHOLD
            ):
                candidates.add("explaining_hands")
        else:
            state.hands_on_face_frames = 0
    else:
        state.hands_on_face_frames = 0
    return candidates


def reset_detection_state(state):
    state.active = None
    state.candidate = None
    state.candidate_started_at = None
    state.cooldown_until.clear()
    state.consumed.clear()
    state.hands_on_face_frames = 0
    state.face_hand_pose_until = 0.0
    state.suppress_until_clear = False
    for history in state.hand_history.values():
        history.clear()


def update_active_media(state, candidates, has_face, now):
    if state.active and now >= state.active.expires_at:
        state.active = None

    if (
        state.active
        and state.active.name == "explaining_hands"
        and candidates.intersection({"hands_on_face", "hands_on_cheeks", "head_grab"})
    ):
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
        or now - state.candidate_started_at < GESTURE_MEDIA[selected].debounce
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


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Camera: {actual_width}x{actual_height}, buffer size requested: 1", flush=True
    )
    return cap


def inference_frame(raw_frame):
    height, width = raw_frame.shape[:2]
    if width > CAMERA_WIDTH or height > CAMERA_HEIGHT:
        return cv2.resize(raw_frame, INFERENCE_SIZE, interpolation=cv2.INTER_AREA)
    return raw_frame


def timed_inference(detector, image, timestamp):
    started_at = time.perf_counter()
    return detector(image, timestamp), time.perf_counter() - started_at


def main(camera_index=1, debug=False):
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
    cap = None
    executor = ThreadPoolExecutor(max_workers=2)
    state = RuntimeState()
    start = time.monotonic()
    last_gesture_timestamp = -1
    last_face_timestamp = -1
    last_face_result = None
    last_face = None
    last_face_time = 0.0
    frame_index = 0
    fps_started_at = time.monotonic()
    fps_frame_count = 0
    gesture_seconds = 0.0
    gesture_calls = 0
    face_seconds = 0.0
    face_calls = 0
    try:
        cap = open_camera(camera_index)
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            display_frame = cv2.flip(raw_frame, 1)

            if state.detection_enabled:
                frame_for_detection = inference_frame(raw_frame)
                rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int((now - start) * 1000)
                gesture_timestamp = max(timestamp, last_gesture_timestamp + 1)
                last_gesture_timestamp = gesture_timestamp
                gesture_future = executor.submit(
                    timed_inference,
                    gesture_detector.recognize_for_video,
                    mp_image,
                    gesture_timestamp,
                )
                should_detect_face = (
                    last_face_result is None
                    or frame_index % FACE_INFERENCE_INTERVAL == 0
                )
                face_future = None
                if should_detect_face:
                    face_timestamp = max(timestamp, last_face_timestamp + 1)
                    last_face_timestamp = face_timestamp
                    face_future = executor.submit(
                        timed_inference,
                        face_detector.detect_for_video,
                        mp_image,
                        face_timestamp,
                    )

                gesture_result, elapsed = gesture_future.result()
                gesture_seconds += elapsed
                gesture_calls += 1
                if face_future:
                    last_face_result, elapsed = face_future.result()
                    face_seconds += elapsed
                    face_calls += 1
                face_result = last_face_result
                face = (
                    face_result.face_landmarks[0]
                    if face_result.face_landmarks
                    else None
                )
                if face is not None:
                    last_face, last_face_time = face, now
                elif (
                    last_face is not None
                    and now - last_face_time < FACE_FALLBACK_SECONDS
                ):
                    face = last_face
                hands = physical_hands(gesture_result)
                update_hand_history(state, hands, now)
                candidates = detect_candidates(
                    face,
                    face_result.face_blendshapes,
                    hands,
                    gesture_result,
                    state,
                    now,
                )
                if debug:
                    log_debug_metrics(
                        face,
                        face_result.face_blendshapes,
                        hands,
                        gesture_result,
                        state,
                        now,
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
                if debug:
                    draw_debug_landmarks(display_frame, face, hands, state)

            cv2.imshow("meme camera", display_frame)
            fps_frame_count += 1
            frame_index += 1
            if fps_frame_count == FPS_REPORT_INTERVAL:
                elapsed = time.monotonic() - fps_started_at
                gesture_ms = (
                    1000 * gesture_seconds / gesture_calls if gesture_calls else 0.0
                )
                face_ms = 1000 * face_seconds / face_calls if face_calls else 0.0
                print(
                    f"{FPS_REPORT_INTERVAL / elapsed:.1f} fps | "
                    f"gesture {gesture_ms:.1f} ms | face {face_ms:.1f} ms",
                    flush=True,
                )
                fps_started_at = time.monotonic()
                fps_frame_count = 0
                gesture_seconds = 0.0
                gesture_calls = 0
                face_seconds = 0.0
                face_calls = 0

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                state.detection_enabled = not state.detection_enabled
                reset_detection_state(state)
                last_face_result = None
                last_face = None
                last_face_time = 0.0
            elif key == ord("h"):
                state.active = None
                state.suppress_until_clear = True
    finally:
        executor.shutdown(wait=True)
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        gesture_detector.close()
        face_detector.close()


if __name__ == "__main__":
    main()
