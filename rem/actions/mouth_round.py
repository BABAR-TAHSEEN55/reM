from .. import config


def matches(blendshapes):
    return (
        blendshapes.get("mouthPucker", 0.0) >= config.ROUND_MOUTH_PUCKER_THRESHOLD
        and blendshapes.get("jawOpen", 0.0) >= config.ROUND_MOUTH_JAW_THRESHOLD
    )
