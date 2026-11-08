from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MediaConfig:
    file: str
    type: str
    duration: float
    cooldown: float
    priority: int
    debounce: float


@dataclass(frozen=True)
class HandsOnFaceMetrics:
    face_width: float
    palm_distance: float
    palm_distance_threshold: float
    nose_offset: float
    nose_offset_threshold: float
    midpoint_x: float
    midpoint_y: float
    lower_face_top: float
    chin_y: float
    together: bool
    centered: bool
    on_lower_face: bool
    fingers_up: bool

    @property
    def matches(self):
        return self.together and self.centered and self.on_lower_face and self.fingers_up


@dataclass(frozen=True)
class TSignMetrics:
    palm_distance: float
    palm_distance_threshold: float
    left_angle: float
    right_angle: float
    perpendicular_error: float
    hands_close: bool
    axis_aligned: bool
    hands_open: bool

    @property
    def matches(self):
        return self.hands_close and self.axis_aligned and self.hands_open


@dataclass
class LoadedMedia:
    type: str
    frames: list[np.ndarray]
    durations: list[float]


@dataclass
class ActiveMedia:
    name: str
    started_at: float
    expires_at: float


@dataclass
class RuntimeState:
    detection_enabled: bool = True
    active: ActiveMedia | None = None
    candidate: str | None = None
    candidate_started_at: float | None = None
    cooldown_until: dict[str, float] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    hand_history: dict[str, deque[tuple[float, float, float]]] = field(
        default_factory=lambda: {"Left": deque(), "Right": deque()}
    )
    hands_on_face_frames: int = 0
    face_hand_pose_until: float = 0.0
    last_debug_metrics_at: float = 0.0
    suppress_until_clear: bool = False
