from math import atan2, degrees, hypot

from .. import config


def update_hand_history(state, hands, now):
    for side, history in state.hand_history.items():
        if side in hands:
            wrist = hands[side][0]
            history.append((now, wrist.x, wrist.y))
        while history and now - history[0][0] > config.MOTION_WINDOW_SECONDS:
            history.popleft()


def hand_near_nose_metrics(hand, nose, face_width):
    threshold = max(face_width * 0.25, 0.05)
    return min(hypot(point.x - nose.x, point.y - nose.y) for point in hand), threshold


def hand_near_nose(hand, nose, face_width):
    distance, threshold = hand_near_nose_metrics(hand, nose, face_width)
    return distance < threshold


def hand_motion(history):
    if len(history) < 3:
        return 0.0
    return sum(
        hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(history, list(history)[1:])
    )


def vertical_wave(history):
    if len(history) < 3 or history[-1][0] - history[0][0] < config.VERTICAL_WAVE_MIN_SECONDS:
        return False
    ys = [point[2] for point in history]
    return max(ys) - min(ys) >= config.VERTICAL_WAVE_MIN_DISTANCE


def hand_open(hand):
    wrist = hand[0]
    return sum(
        hypot(hand[tip].x - wrist.x, hand[tip].y - wrist.y)
        > hypot(hand[pip].x - wrist.x, hand[pip].y - wrist.y)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ) >= 3


def hand_angle(hand):
    wrist, middle_tip = hand[0], hand[12]
    return degrees(atan2(middle_tip.y - wrist.y, middle_tip.x - wrist.x))


def blendshape_scores(face_blendshapes):
    return {category.category_name: category.score for category in face_blendshapes[0]} if face_blendshapes else {}
