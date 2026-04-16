from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from afs.vlm.cache import EmbeddingCache


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model_tag="qwen/test-3B")

    emb = torch.randn(10, 128)
    frame_idx = np.arange(10)
    ts = np.linspace(0, 9, num=10)
    assert not cache.has("vidA", fps=1.0, max_frames=32)

    cache.put("vidA", fps=1.0, embeddings=emb, frame_indices=frame_idx, timestamps=ts, max_frames=32)
    assert cache.has("vidA", fps=1.0, max_frames=32)

    out = cache.get("vidA", fps=1.0, max_frames=32)
    assert out is not None
    assert torch.allclose(out.embeddings, emb)
    np.testing.assert_array_equal(out.frame_indices, frame_idx)
    np.testing.assert_array_equal(out.timestamps, ts)
    assert out.meta["video_id"] == "vidA"
    assert out.meta["fps"] == 1.0
    assert out.meta["model_tag"] == "qwen_test-3B"


def test_cache_key_depends_on_fps_and_max(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model_tag="m1")
    emb = torch.randn(4, 8)
    cache.put("v1", fps=1.0, embeddings=emb, frame_indices=np.arange(4), timestamps=np.arange(4), max_frames=8)
    assert cache.has("v1", fps=1.0, max_frames=8)
    assert not cache.has("v1", fps=2.0, max_frames=8)
    assert not cache.has("v1", fps=1.0, max_frames=16)


def test_cache_model_tag_separates_entries(tmp_path: Path) -> None:
    c1 = EmbeddingCache(tmp_path, model_tag="Qwen/A")
    c2 = EmbeddingCache(tmp_path, model_tag="Qwen/B")
    emb = torch.randn(3, 4)
    c1.put("vid", fps=1.0, embeddings=emb, frame_indices=np.arange(3), timestamps=np.arange(3))
    assert c1.has("vid", fps=1.0)
    assert not c2.has("vid", fps=1.0)


def test_cache_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model_tag="m")
    assert cache.get("nope", fps=1.0) is None
