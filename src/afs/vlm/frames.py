from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExtractedFrames:
    frames: list[Image.Image]
    frame_indices: np.ndarray  # shape [T], decoder frame positions kept
    timestamps: np.ndarray     # shape [T], seconds
    duration: float            # full video duration in seconds
    fps_source: float          # decoded fps
    fps_target: float          # requested sampling fps


def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
) -> ExtractedFrames:
    """Decode a video and uniformly subsample frames at ``fps`` (Hz).

    If the resulting sequence exceeds ``max_frames``, keep ``max_frames`` evenly
    spaced positions. Returns PIL Images plus the indices/timestamps so callers
    can cache deterministically.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps_source = float(stream.average_rate) if stream.average_rate else 0.0
        total_frames = stream.frames or 0
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        if duration == 0.0 and total_frames and fps_source:
            duration = total_frames / fps_source

        decoded: list[Image.Image] = []
        indices: list[int] = []
        timestamps: list[float] = []
        for i, frame in enumerate(container.decode(stream)):
            decoded.append(frame.to_image())
            indices.append(i)
            t = float(frame.pts * stream.time_base) if frame.pts is not None else i / max(fps_source, 1.0)
            timestamps.append(t)

    if not decoded:
        raise RuntimeError(f"No frames decoded from {path}")

    if fps_source <= 0.0:
        fps_source = (len(decoded) / duration) if duration > 0 else float(len(decoded))

    target_count = max(1, int(round(duration * fps))) if duration > 0 else len(decoded)
    target_count = min(target_count, len(decoded))
    if max_frames is not None:
        target_count = min(target_count, max_frames)

    keep = np.linspace(0, len(decoded) - 1, num=target_count).round().astype(int)
    keep = np.unique(keep)

    frames = [decoded[i] for i in keep]
    return ExtractedFrames(
        frames=frames,
        frame_indices=keep,
        timestamps=np.asarray([timestamps[i] for i in keep], dtype=np.float64),
        duration=duration,
        fps_source=fps_source,
        fps_target=fps,
    )
