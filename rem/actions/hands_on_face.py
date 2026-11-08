from math import hypot

from ..models import HandsOnFaceMetrics


def hands_together_on_face_metrics(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    left_palm, right_palm = left[9], right[9]
    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
    midpoint_x = (left_palm.x + right_palm.x) / 2
    midpoint_y = (left_palm.y + right_palm.y) / 2
    lower_face_top = face[168].y - 0.05
    palm_distance_threshold = face_width * 0.6
    nose_offset_threshold = face_width * 0.35
    nose_offset = abs(midpoint_x - nose.x)
    return HandsOnFaceMetrics(
        face_width, palm_distance, palm_distance_threshold, nose_offset, nose_offset_threshold,
        midpoint_x, midpoint_y, lower_face_top, face[152].y,
        palm_distance < palm_distance_threshold, nose_offset < nose_offset_threshold,
        lower_face_top <= midpoint_y <= face[152].y,
        left[12].y < left[0].y and right[12].y < right[0].y,
    )


def matches(left, right, face):
    return hands_together_on_face_metrics(left, right, face).matches
