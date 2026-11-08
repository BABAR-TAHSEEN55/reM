import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# pairs of landmark indices to draw the skeleton
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),
]

ROOT = Path(__file__).resolve().parents[1]


def main(camera_index=1):
    # --- setup ---
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(ROOT / "hand_landmarker.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(ROOT / "face_landmarker.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        hand_detector.close()
        face_detector.close()
        raise RuntimeError(f"Could not open camera {camera_index}")
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for preview
            h, w = frame.shape[:2]

            # OpenCV is BGR, MediaPipe wants RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # timestamp must be in ms and always increasing
            ts = int((time.time() - start) * 1000)
            hand_result = hand_detector.detect_for_video(mp_image, ts)
            face_result = face_detector.detect_for_video(mp_image, ts)

            for hand in hand_result.hand_landmarks:
                # landmarks are normalized (0-1), convert to pixels
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)

            for face in face_result.face_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in face]
                for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
                    cv2.line(frame, pts[connection.start], pts[connection.end], (255, 0, 0), 1)

            cv2.imshow("hands and faces", frame)
            if cv2.waitKey(1) == 27:  # Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        face_detector.close()


if __name__ == "__main__":
    main()
