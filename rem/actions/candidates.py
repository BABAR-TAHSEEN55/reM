from .. import config
from . import cheeks, explaining_hands, eyes_closed, head_grab, horse_smirk, lips_down
from . import mouth_round, open_palm, smile, t_sign, tongue_out, wave
from .hands_on_face import matches as hands_on_face
from .landmarks import blendshape_scores, hand_near_nose


def detect_candidates(face, face_blendshapes, hands, gesture_result, state, now):
    if face is None:
        return set()
    candidates = set()
    nose = face[1]
    face_width = abs(face[454].x - face[234].x)
    blendshapes = blendshape_scores(face_blendshapes)
    if lips_down.matches(blendshapes):
        candidates.add("lips_down")
    if tongue_out.matches(face):
        candidates.add("tongue_out")
    if mouth_round.matches(blendshapes):
        candidates.add("mouth_round")
    if horse_smirk.matches(blendshapes):
        candidates.add("horse_smirk")
    if eyes_closed.matches(blendshapes):
        candidates.add("eyes_closed")
    if smile.matches(blendshapes):
        candidates.add("smile")

    left_hand, right_hand = hands.get("Left"), hands.get("Right")
    hands_near_nose = any(hand_near_nose(hand, nose, face_width) for hand in hands.values())
    is_t_sign = left_hand is not None and right_hand is not None and t_sign.matches(left_hand, right_hand)
    if is_t_sign:
        candidates.add("t_sign")
    if open_palm.matches(gesture_result) and not hands_near_nose and not is_t_sign:
        candidates.add("open_palm")
    if not left_hand or not right_hand:
        state.hands_on_face_frames = 0
        return candidates

    left_near_nose = hand_near_nose(left_hand, nose, face_width)
    right_near_nose = hand_near_nose(right_hand, nose, face_width)
    if head_grab.matches(left_hand, right_hand, face):
        state.hands_on_face_frames = 0
        state.face_hand_pose_until = now + config.FACE_HAND_POSE_HOLD_SECONDS
        candidates.add("head_grab")
    elif cheeks.matches(left_hand, right_hand, face):
        state.hands_on_face_frames = 0
        state.face_hand_pose_until = now + config.FACE_HAND_POSE_HOLD_SECONDS
        candidates.add("hands_on_cheeks")
    elif is_t_sign:
        state.hands_on_face_frames = 0
    elif left_near_nose and right_near_nose and hands_on_face(left_hand, right_hand, face):
        state.hands_on_face_frames += 1
        state.face_hand_pose_until = now + config.FACE_HAND_POSE_HOLD_SECONDS
        if state.hands_on_face_frames >= config.HANDS_ON_FACE_CONFIRMATION_FRAMES:
            candidates.add("hands_on_face")
    elif wave.matches(left_near_nose, state.hand_history["Right"]):
        state.hands_on_face_frames = 0
        candidates.add("left_nose_right_wave")
    elif not left_near_nose and not right_near_nose and now >= state.face_hand_pose_until:
        state.hands_on_face_frames = 0
        if explaining_hands.matches(state.hand_history["Left"], state.hand_history["Right"]):
            candidates.add("explaining_hands")
    else:
        state.hands_on_face_frames = 0
    return candidates
