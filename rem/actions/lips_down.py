from .. import config


def scores(blendshapes):
    return blendshapes.get("mouthFrownLeft", 0.0), blendshapes.get("mouthFrownRight", 0.0)


def matches(blendshapes):
    left_score, right_score = scores(blendshapes)
    return left_score >= config.FROWN_THRESHOLD and right_score >= config.FROWN_THRESHOLD
