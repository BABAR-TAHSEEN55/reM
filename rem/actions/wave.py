from .landmarks import vertical_wave


def matches(left_near_nose, right_history):
    return left_near_nose and vertical_wave(right_history)
