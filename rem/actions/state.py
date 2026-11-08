from .. import config
from ..models import ActiveMedia


def reset_detection_state(state):
    state.active = None
    state.candidate = None
    state.candidate_started_at = None
    state.cooldown_until.clear()
    state.consumed.clear()
    state.hands_on_face_frames = 0
    state.face_hand_pose_until = 0.0
    state.suppress_until_clear = False
    for history in state.hand_history.values():
        history.clear()


def update_active_media(state, candidates, has_face, now):
    if state.active and now >= state.active.expires_at:
        state.active = None
    if state.active and state.active.name == "explaining_hands" and candidates.intersection({"hands_on_face", "hands_on_cheeks", "head_grab"}):
        state.active = None
    state.consumed.intersection_update(candidates)
    if state.suppress_until_clear:
        if not candidates:
            state.suppress_until_clear = False
        else:
            return
    if not has_face:
        state.candidate = None
        state.candidate_started_at = None
        return
    eligible = [name for name in candidates if name not in state.consumed and now >= state.cooldown_until.get(name, 0.0)]
    selected = max(eligible, key=lambda name: config.GESTURE_MEDIA[name].priority, default=None)
    if selected != state.candidate:
        state.candidate = selected
        state.candidate_started_at = now if selected else None
        return
    if selected is None or state.active is not None or now - state.candidate_started_at < config.GESTURE_MEDIA[selected].debounce:
        return
    media_config = config.GESTURE_MEDIA[selected]
    duration = max(media_config.duration, config.GIF_MIN_LOCK_SECONDS if media_config.type == "gif" else config.IMAGE_MIN_LOCK_SECONDS)
    state.active = ActiveMedia(selected, now, now + duration)
    state.cooldown_until[selected] = now + media_config.cooldown
    state.consumed.add(selected)
    state.candidate = None
    state.candidate_started_at = None
