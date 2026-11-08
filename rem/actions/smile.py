from .. import config


def scores(blendshapes):
    return blendshapes.get("mouthSmileLeft", 0.0), blendshapes.get("mouthSmileRight", 0.0)


def matches(blendshapes):
    left_score, right_score = scores(blendshapes)
    return (
        left_score >= config.SMILE_THRESHOLD
        and right_score >= config.SMILE_THRESHOLD
        and abs(left_score - right_score) < config.SMILE_ASYMMETRY_MAX
    )
