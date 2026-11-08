from .. import config
from .landmarks import hand_motion


def matches(left_history, right_history):
    return hand_motion(left_history) + hand_motion(right_history) >= config.EXPLAINING_MOTION_THRESHOLD
