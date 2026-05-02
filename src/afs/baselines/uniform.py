"""Uniform frame sampling baseline.

Selects exactly ``n_frames`` evenly-spaced frames from a video, ignoring
query content.  This is the simplest non-trivial baseline and serves as the
lower bound that the PPO policy must beat at equal frame count.
"""

from __future__ import annotations

from pathlib import Path

from afs.vlm.frames import ExtractedFrames, extract_frames


class UniformBaseline:
    """Deterministic uniform sampler at a fixed frame budget.

    Parameters
    ----------
    n_frames:
        Number of frames to keep.  Typical values: 8, 16, 32.
    """

    def __init__(self, n_frames: int) -> None:
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = n_frames

    @property
    def name(self) -> str:
        return f"uniform_{self.n_frames}"

    def select_frames(
        self,
        video_path: str | Path,
        fps: float = 1.0,
    ) -> ExtractedFrames:
        """Return up to ``n_frames`` uniformly spaced frames from *video_path*."""
        return extract_frames(video_path, fps=fps, max_frames=self.n_frames)
