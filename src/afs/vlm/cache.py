from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class CachedFrames:
    embeddings: torch.Tensor      # [T, D_vis], frozen VLM vision-tower features
    frame_indices: np.ndarray     # [T], absolute frame idx in the source video
    timestamps: np.ndarray        # [T], seconds
    meta: dict


def _make_key(video_id: str, fps: float, max_frames: int | None, model_tag: str) -> str:
    raw = f"{video_id}|{model_tag}|fps={fps:.4f}|max={max_frames}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class EmbeddingCache:
    """On-disk cache for per-frame VLM embeddings.

    Layout::

        root/<model_tag>/<video_id>__<key>.pt

    Each `.pt` file stores a dict with keys: ``embeddings`` (torch.float16 or
    bfloat16 tensor), ``frame_indices``, ``timestamps``, ``meta``.
    """

    def __init__(self, root: str | Path, model_tag: str) -> None:
        self.root = Path(root)
        self.model_tag = model_tag.replace("/", "_")
        (self.root / self.model_tag).mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str, key: str) -> Path:
        return self.root / self.model_tag / f"{video_id}__{key}.pt"

    def has(self, video_id: str, fps: float, max_frames: int | None = None) -> bool:
        key = _make_key(video_id, fps, max_frames, self.model_tag)
        return self._path(video_id, key).exists()

    def get(
        self, video_id: str, fps: float, max_frames: int | None = None
    ) -> CachedFrames | None:
        key = _make_key(video_id, fps, max_frames, self.model_tag)
        path = self._path(video_id, key)
        if not path.exists():
            return None
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return CachedFrames(
            embeddings=obj["embeddings"],
            frame_indices=obj["frame_indices"],
            timestamps=obj["timestamps"],
            meta=obj.get("meta", {}),
        )

    def put(
        self,
        video_id: str,
        fps: float,
        embeddings: torch.Tensor,
        frame_indices: np.ndarray,
        timestamps: np.ndarray,
        max_frames: int | None = None,
        meta: dict | None = None,
    ) -> Path:
        key = _make_key(video_id, fps, max_frames, self.model_tag)
        path = self._path(video_id, key)
        obj = {
            "embeddings": embeddings.detach().to("cpu").contiguous(),
            "frame_indices": np.asarray(frame_indices),
            "timestamps": np.asarray(timestamps),
            "meta": {
                "video_id": video_id,
                "fps": fps,
                "max_frames": max_frames,
                "model_tag": self.model_tag,
                **(meta or {}),
            },
        }
        fd, tmp = tempfile.mkstemp(prefix="_tmp_", dir=str(path.parent))
        os.close(fd)
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return path
