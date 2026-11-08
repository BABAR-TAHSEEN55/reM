from .. import config


def matches(left, right, face):
    face_width = abs(face[454].x - face[234].x)
    nose = face[1]
    eye_line_y = (face[33].y + face[263].y) / 2
    limit_y = eye_line_y + face_width * config.HEAD_GRAB_EYE_MARGIN_FACTOR
    left_palm, right_palm = left[9], right[9]
    return (
        left_palm.y < limit_y
        and right_palm.y < limit_y
        and ((left_palm.x < nose.x - face_width * 0.1 and right_palm.x > nose.x + face_width * 0.1)
             or (right_palm.x < nose.x - face_width * 0.1 and left_palm.x > nose.x + face_width * 0.1))
    )
