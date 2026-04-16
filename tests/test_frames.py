from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from afs.vlm.frames import extract_frames


def _write_synthetic_video(path: Path, n_frames: int, fps: int, width: int = 64, height: int = 64) -> None:
    codec_candidates = ("libx264", "mpeg4")
    last_err: Exception | None = None
    for codec in codec_candidates:
        try:
            container = av.open(str(path), mode="w")
            stream = container.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for i in range(n_frames):
                arr = np.full((height, width, 3), (i * 4) % 255, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            try:
                container.close()
            except Exception:
                pass
            path.unlink(missing_ok=True)
            continue
    raise RuntimeError(f"no usable codec: {last_err}")


def test_extract_uniformly_subsamples(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_synthetic_video(video, n_frames=60, fps=30)

    out = extract_frames(video, fps=1.0)
    assert 1 < len(out.frames) <= 3  # ~2s video at 1fps
    assert len(out.frames) == len(out.frame_indices) == len(out.timestamps)
    for f in out.frames:
        assert f.size == (64, 64)


def test_extract_respects_max_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_synthetic_video(video, n_frames=120, fps=30)

    out = extract_frames(video, fps=10.0, max_frames=4)
    assert len(out.frames) == 4
    assert (np.diff(out.frame_indices) > 0).all()


def test_extract_returns_monotone_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_synthetic_video(video, n_frames=90, fps=30)

    out = extract_frames(video, fps=2.0)
    assert (np.diff(out.timestamps) >= 0).all()


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "nope.mp4")
