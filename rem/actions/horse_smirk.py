from .. import config
from .smile import scores


def matches(blendshapes):
    left_score, right_score = scores(blendshapes)
    return (
        max(left_score, right_score) >= config.SMIRK_SMILE_THRESHOLD
        and abs(left_score - right_score) >= config.SMIRK_ASYMMETRY_THRESHOLD
    )
