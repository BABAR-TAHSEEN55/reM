import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config
from .actions import detect_candidates, reset_detection_state, update_active_media, update_hand_history
from .debug import draw_debug_landmarks, log_debug_metrics
from .media import active_frame, load_media, overlay_media
from .models import RuntimeState


def physical_hands(gesture_result):
    hands = {}
    for landmarks, handedness in zip(gesture_result.hand_landmarks, gesture_result.handedness, strict=True):
        if handedness:
            reported = handedness[0].category_name
            hands["Right" if reported == "Left" else "Left"] = landmarks
    return hands


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, buffer size requested: 1", flush=True)
    return cap


def inference_frame(raw_frame):
    height, width = raw_frame.shape[:2]
    return cv2.resize(raw_frame, config.INFERENCE_SIZE, interpolation=cv2.INTER_AREA) if width > config.CAMERA_WIDTH or height > config.CAMERA_HEIGHT else raw_frame


def timed_inference(detector, image, timestamp):
    started_at = time.perf_counter()
    return detector(image, timestamp), time.perf_counter() - started_at


def main(camera_index=config.CAMERA_INDEX, debug=False):
    media = load_media()
    gesture_options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.GESTURE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    gesture_detector = vision.GestureRecognizer.create_from_options(gesture_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
    cap = None
    executor = ThreadPoolExecutor(max_workers=config.INFERENCE_WORKERS)
    state = RuntimeState()
    start = time.monotonic()
    last_gesture_timestamp = last_face_timestamp = -1
    last_face_result = last_face = None
    last_face_time = 0.0
    frame_index = fps_frame_count = gesture_calls = face_calls = 0
    fps_started_at = time.monotonic()
    gesture_seconds = face_seconds = 0.0
    try:
        cap = open_camera(camera_index)
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            display_frame = cv2.flip(raw_frame, 1)
            if state.detection_enabled:
                rgb = cv2.cvtColor(inference_frame(raw_frame), cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int((now - start) * 1000)
                last_gesture_timestamp = max(timestamp, last_gesture_timestamp + 1)
                gesture_future = executor.submit(timed_inference, gesture_detector.recognize_for_video, mp_image, last_gesture_timestamp)
                face_future = None
                if last_face_result is None or frame_index % config.FACE_INFERENCE_INTERVAL == 0:
                    last_face_timestamp = max(timestamp, last_face_timestamp + 1)
                    face_future = executor.submit(timed_inference, face_detector.detect_for_video, mp_image, last_face_timestamp)
                gesture_result, elapsed = gesture_future.result()
                gesture_seconds += elapsed
                gesture_calls += 1
                if face_future:
                    last_face_result, elapsed = face_future.result()
                    face_seconds += elapsed
                    face_calls += 1
                face_result = last_face_result
                face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                if face is not None:
                    last_face, last_face_time = face, now
                elif last_face is not None and now - last_face_time < config.FACE_FALLBACK_SECONDS:
                    face = last_face
                hands = physical_hands(gesture_result)
                update_hand_history(state, hands, now)
                candidates = detect_candidates(face, face_result.face_blendshapes, hands, gesture_result, state, now)
                if debug:
                    log_debug_metrics(face, face_result.face_blendshapes, hands, gesture_result, state, now)
                update_active_media(state, candidates, face is not None, now)
                if state.active and face is not None:
                    nose = face[1]
                    center = (int((1.0 - nose.x) * raw_frame.shape[1]), int(nose.y * raw_frame.shape[0]))
                    overlay_media(display_frame, active_frame(state.active, media, now), center)
                if debug:
                    draw_debug_landmarks(display_frame, face, hands, state)

            cv2.imshow(config.WINDOW_TITLE, display_frame)
            fps_frame_count += 1
            frame_index += 1
            if fps_frame_count == config.FPS_REPORT_INTERVAL:
                elapsed = time.monotonic() - fps_started_at
                gesture_ms = 1000 * gesture_seconds / gesture_calls if gesture_calls else 0.0
                face_ms = 1000 * face_seconds / face_calls if face_calls else 0.0
                print(f"{config.FPS_REPORT_INTERVAL / elapsed:.1f} fps | gesture {gesture_ms:.1f} ms | face {face_ms:.1f} ms", flush=True)
                fps_started_at = time.monotonic()
                fps_frame_count = gesture_seconds = gesture_calls = face_seconds = face_calls = 0
            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if key == config.TOGGLE_DETECTION_KEY:
                state.detection_enabled = not state.detection_enabled
                reset_detection_state(state)
                last_face_result = last_face = None
                last_face_time = 0.0
            elif key == config.DISMISS_MEDIA_KEY:
                state.active = None
                state.suppress_until_clear = True
    finally:
        executor.shutdown(wait=True)
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        gesture_detector.close()
        face_detector.close()
