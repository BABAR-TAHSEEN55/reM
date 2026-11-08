import cv2

from . import config
from .actions import (
    blendshape_scores,
    cheek_side,
    frown_scores,
    hand_motion,
    hand_near_nose_metrics,
    hands_on_cheeks,
    hands_on_head,
    hands_together_on_face_metrics,
    is_open_palm,
)


def log_debug_metrics(face, face_blendshapes, hands, gesture_result, state, now):
    if now - state.last_debug_metrics_at < config.DEBUG_METRICS_INTERVAL:
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
    nose_distances = {side: hand_near_nose_metrics(hand, nose, face_width) for side, hand in hands.items()}
    hands_near_nose = any(distance < threshold for distance, threshold in nose_distances.values())
    nose_distance_text = ",".join(
        f"{side}={distance:.3f}/{threshold:.3f}" for side, (distance, threshold) in nose_distances.items()
    ) or "n/a"
    prefix = (
        f"debug | mouth={mouth_gap:.3f}/{config.MOUTH_OPEN_THRESHOLD:.3f} "
        f"frown={frown_left:.2f},{frown_right:.2f}/{config.FROWN_THRESHOLD:.2f} "
        f"face_width={face_width:.3f} nose={nose_distance_text} near_nose={int(hands_near_nose)} "
        f"open_palm={int(is_open_palm(gesture_result))} "
        f"motion={hand_motion(state.hand_history['Left']):.3f},{hand_motion(state.hand_history['Right']):.3f}/{config.EXPLAINING_MOTION_THRESHOLD:.3f}"
    )
    if not left_hand or not right_hand:
        print(f"{prefix} | hands_together=n/a", flush=True)
        return

    metrics = hands_together_on_face_metrics(left_hand, right_hand, face)
    head_grab = hands_on_head(left_hand, right_hand, face)
    cheeks = hands_on_cheeks(left_hand, right_hand, face)
    left_cheek_side = cheek_side(left_hand[9], face) or "none"
    right_cheek_side = cheek_side(right_hand[9], face) or "none"
    print(
        f"{prefix} | palms={metrics.palm_distance:.3f}/{metrics.palm_distance_threshold:.3f} "
        f"center={metrics.nose_offset:.3f}/{metrics.nose_offset_threshold:.3f} "
        f"mid_y={metrics.midpoint_y:.3f}/{metrics.lower_face_top:.3f}-{metrics.chin_y:.3f} "
        f"strict=together={int(metrics.together)} centered={int(metrics.centered)} "
        f"lower_face={int(metrics.on_lower_face)} fingers_up={int(metrics.fingers_up)} "
        f"confirm={state.hands_on_face_frames}/{config.HANDS_ON_FACE_CONFIRMATION_FRAMES} "
        f"head_grab={int(head_grab)} cheeks={int(cheeks)} "
        f"cheek_zones={left_cheek_side},{right_cheek_side} "
        f"cheek_palms={metrics.palm_distance:.3f}/{face_width * config.CHEEK_PALM_DISTANCE_FACTOR:.3f}",
        flush=True,
    )


def mirrored_coordinates(x, y, frame_width, frame_height):
    return (
        min(max(round((1.0 - x) * frame_width), 0), frame_width - 1),
        min(max(round(y * frame_height), 0), frame_height - 1),
    )


def mirrored_point(landmark, frame_width, frame_height):
    return mirrored_coordinates(landmark.x, landmark.y, frame_width, frame_height)


def draw_debug_landmarks(frame, face, hands, state):
    frame_height, frame_width = frame.shape[:2]
    if face is not None:
        for landmark in face:
            cv2.circle(frame, mirrored_point(landmark, frame_width, frame_height), 1, (80, 80, 80), -1)
        for index, color in ((1, (0, 255, 255)), (168, (255, 255, 0)), (152, (0, 255, 0))):
            cv2.circle(frame, mirrored_point(face[index], frame_width, frame_height), 4, color, -1)

    colors = {"Left": (255, 120, 0), "Right": (255, 0, 255)}
    for side, hand in hands.items():
        color = colors[side]
        for start, end in config.HAND_CONNECTIONS:
            cv2.line(frame, mirrored_point(hand[start], frame_width, frame_height), mirrored_point(hand[end], frame_width, frame_height), color, 2)
        for landmark in hand:
            cv2.circle(frame, mirrored_point(landmark, frame_width, frame_height), 3, color, -1)

    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    if face is not None and left_hand and right_hand:
        metrics = hands_together_on_face_metrics(left_hand, right_hand, face)
        midpoint = mirrored_coordinates(metrics.midpoint_x, metrics.midpoint_y, frame_width, frame_height)
        cv2.circle(frame, midpoint, 5, (0, 255, 255), -1)
        status = (
            f"strict T{int(metrics.together)} C{int(metrics.centered)} "
            f"L{int(metrics.on_lower_face)} F{int(metrics.fingers_up)} "
            f"{state.hands_on_face_frames}/{config.HANDS_ON_FACE_CONFIRMATION_FRAMES} "
            f"head {int(hands_on_head(left_hand, right_hand, face))} "
            f"cheeks {int(hands_on_cheeks(left_hand, right_hand, face))}"
        )
    else:
        status = f"strict n/a {state.hands_on_face_frames}/{config.HANDS_ON_FACE_CONFIRMATION_FRAMES}"
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
