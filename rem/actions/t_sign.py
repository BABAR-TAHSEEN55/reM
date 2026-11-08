from math import hypot

from .. import config
from ..models import TSignMetrics
from .landmarks import hand_angle, hand_open


def metrics(left, right):
    left_palm, right_palm = left[9], right[9]
    left_size = hypot(left[0].x - left_palm.x, left[0].y - left_palm.y)
    right_size = hypot(right[0].x - right_palm.x, right[0].y - right_palm.y)
    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
    left_angle, right_angle = hand_angle(left), hand_angle(right)
    angle_difference = abs(left_angle - right_angle) % 180
    left_horizontal = abs(left[12].y - left[0].y) <= abs(left[12].x - left[0].x) * config.T_SIGN_DIRECTION_RATIO
    right_horizontal = abs(right[12].y - right[0].y) <= abs(right[12].x - right[0].x) * config.T_SIGN_DIRECTION_RATIO
    left_vertical = abs(left[12].x - left[0].x) <= abs(left[12].y - left[0].y) * config.T_SIGN_DIRECTION_RATIO
    right_vertical = abs(right[12].x - right[0].x) <= abs(right[12].y - right[0].y) * config.T_SIGN_DIRECTION_RATIO
    threshold = (left_size + right_size) / 2 * config.T_SIGN_PALM_DISTANCE_FACTOR
    return TSignMetrics(
        palm_distance, threshold, left_angle, right_angle, abs(90 - angle_difference), palm_distance < threshold,
        (left_horizontal and right_vertical) or (left_vertical and right_horizontal), hand_open(left) and hand_open(right),
    )


def matches(left, right):
    return metrics(left, right).matches
