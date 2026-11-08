from math import hypot

from .. import config


def cheek_side(palm, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    eye_line_y = (face[33].y + face[263].y) / 2
    face_left, face_right = sorted((face[234].x, face[454].x))
    if not eye_line_y + face_width * config.CHEEK_BELOW_EYE_FACTOR <= palm.y <= face[152].y:
        return None
    if face_left - face_width * config.CHEEK_OUTER_MARGIN_FACTOR <= palm.x <= nose.x - face_width * config.CHEEK_INNER_MARGIN_FACTOR:
        return "left"
    if nose.x + face_width * config.CHEEK_INNER_MARGIN_FACTOR <= palm.x <= face_right + face_width * config.CHEEK_OUTER_MARGIN_FACTOR:
        return "right"
    return None


def matches(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    left_palm, right_palm = left[9], right[9]
    palm_distance = hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
    return (
        palm_distance > face_width * config.CHEEK_PALM_DISTANCE_FACTOR
        and {cheek_side(left_palm, face), cheek_side(right_palm, face)} == {"left", "right"}
    )
