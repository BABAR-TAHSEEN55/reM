from .. import config


def matches(blendshapes):
    return (
        blendshapes.get("eyeBlinkLeft", 0.0) >= config.EYE_CLOSED_THRESHOLD
        and blendshapes.get("eyeBlinkRight", 0.0) >= config.EYE_CLOSED_THRESHOLD
    )
