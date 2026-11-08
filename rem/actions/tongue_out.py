from .. import config


def matches(face):
    return abs(face[13].y - face[14].y) > config.MOUTH_OPEN_THRESHOLD
