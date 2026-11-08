from PIL import Image, ImageSequence
import cv2
import numpy as np

from . import config
from .models import ActiveMedia, LoadedMedia


def normalize_bgra(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if image.shape[2] == 4:
        return image
    raise ValueError("Media must have one, three, or four channels")


def resize_media(image):
    height = round(image.shape[0] * config.MEME_WIDTH / image.shape[1])
    return cv2.resize(image, (config.MEME_WIDTH, height), interpolation=cv2.INTER_AREA)


def load_media():
    media = {}
    for name, media_config in config.GESTURE_MEDIA.items():
        path = config.MEMES_DIR / media_config.file
        if not path.is_file():
            raise FileNotFoundError(f"Could not load configured media {path}")
        if media_config.type == "image":
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not decode image {path}")
            media[name] = LoadedMedia("image", [resize_media(normalize_bgra(image))], [0.0])
        elif media_config.type == "gif":
            frames, durations = [], []
            with Image.open(path) as gif:
                for frame in ImageSequence.Iterator(gif):
                    rgba = np.array(frame.convert("RGBA"))
                    frames.append(resize_media(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)))
                    durations.append(max(frame.info.get("duration", 100), 20) / 1000.0)
            if not frames:
                raise ValueError(f"GIF has no frames: {path}")
            media[name] = LoadedMedia("gif", frames, durations)
        else:
            raise ValueError(f"Unsupported media type {media_config.type!r} for {name}")
    return media


def overlay_media(frame, media, center):
    """Alpha-blend media centered on the detected face, clamped to frame edges."""
    frame_h, frame_w = frame.shape[:2]
    media_h, media_w = media.shape[:2]
    x = min(max(center[0] - media_w // 2, 0), max(frame_w - media_w, 0))
    y = min(max(center[1] - media_h // 2, 0), max(frame_h - media_h, 0))
    dst_x2, dst_y2 = min(x + media_w, frame_w), min(y + media_h, frame_h)
    if dst_x2 <= x or dst_y2 <= y:
        return
    overlay = media[: dst_y2 - y, : dst_x2 - x]
    alpha = overlay[:, :, 3:4] / 255.0
    frame[y:dst_y2, x:dst_x2] = (
        alpha * overlay[:, :, :3] + (1.0 - alpha) * frame[y:dst_y2, x:dst_x2]
    ).astype(np.uint8)


def active_frame(active: ActiveMedia, media, now):
    loaded = media[active.name]
    if loaded.type == "image":
        return loaded.frames[0]
    total_duration = sum(loaded.durations)
    elapsed = now - active.started_at
    completed_loops = round(elapsed / total_duration)
    elapsed = 0.0 if completed_loops and abs(elapsed - completed_loops * total_duration) < 1e-9 else elapsed % total_duration
    for frame, duration in zip(loaded.frames, loaded.durations, strict=True):
        if elapsed < duration:
            return frame
        elapsed -= duration
    return loaded.frames[-1]
