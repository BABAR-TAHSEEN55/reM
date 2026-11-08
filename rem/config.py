from pathlib import Path

from .models import MediaConfig

ROOT = Path(__file__).resolve().parents[1]
GESTURE_MODEL_PATH = ROOT / "gesture_recognizer.task"
FACE_MODEL_PATH = ROOT / "face_landmarker.task"
MEMES_DIR = ROOT / "memes"

MEME_WIDTH = 320
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
FACE_HAND_POSE_HOLD_SECONDS = 0.35
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INFERENCE_SIZE = (640, 360)
INFERENCE_WORKERS = 2
WINDOW_TITLE = "meme camera"
QUIT_KEYS = (27, ord("q"))
TOGGLE_DETECTION_KEY = ord("d")
DISMISS_MEDIA_KEY = ord("h")
FPS_REPORT_INTERVAL = 30
DEBUG_METRICS_INTERVAL = 0.2
T_SIGN_PALM_DISTANCE_FACTOR = 2.0
T_SIGN_DIRECTION_RATIO = 0.75
T_SIGN_PERPENDICULAR_ERROR_DEGREES = 25.0
HEAD_GRAB_EYE_MARGIN_FACTOR = 0.30
CHEEK_BELOW_EYE_FACTOR = 0.32
CHEEK_OUTER_MARGIN_FACTOR = 0.50
CHEEK_INNER_MARGIN_FACTOR = 0.10
CHEEK_PALM_DISTANCE_FACTOR = 0.30

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

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
    "explaining_hands": MediaConfig("hands_talking_to_wall.gif", "gif", 5.0, 5.0, 80, 0.20),
    "mouth_round": MediaConfig("mouth_round.jpg", "image", 2.0, 3.0, 55, 0.15),
    "horse_smirk": MediaConfig("horse_smirk.jpg", "image", 2.0, 3.0, 50, 0.15),
    "eyes_closed": MediaConfig("master_piece_squiting_too_much.jpg", "image", 2.0, 3.0, 45, 0.15),
    "open_palm": MediaConfig("hold_up_hand_forward.jpg", "image", 2.0, 2.0, 10, 0.10),
    "smile": MediaConfig("smile_dog.jpg", "image", 2.0, 3.0, 35, 0.15),
}

MEME_DESCRIPTIONS = {
    "lips_down": "Frown with both mouth corners pulled downward.",
    "left_nose_right_wave": "Keep your left hand near your nose while waving your right hand vertically.",
    "tongue_out": "Open your mouth wide enough for the mouth-open landmark to trigger.",
    "hands_on_face": "Press both palms together over your nose or mouth with fingers pointing upward.",
    "head_grab": "Place both hands on opposite sides of the top of your head.",
    "hands_on_cheeks": "Place separated palms on opposite cheeks, below your eye line.",
    "t_sign": "Make a T with two nearby open hands, one horizontal and one vertical.",
    "explaining_hands": "Move both hands while neither hand is near your nose.",
    "mouth_round": "Make a rounded, puckered mouth with a slightly open jaw.",
    "horse_smirk": "Make a pronounced one-sided smile.",
    "eyes_closed": "Close or almost close both eyes.",
    "open_palm": "Show an open palm away from your face.",
    "smile": "Smile evenly with both mouth corners raised.",
}
