"""Precompute per-frame Qwen2.5-VL vision embeddings for every NExT-QA video
and cache them to disk. Subsequent PPO rollouts read from the cache instead of
re-encoding, which is the single biggest training throughput knob.

Usage:
    python scripts/precompute_embeddings.py \
        --config configs/default.yaml \
        --cache-dir cache/embeddings

Skips videos that already have a cache entry for the given (model, fps, max_frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from afs.data.nextqa import NExTQADataset
from afs.utils.config import load_config
from afs.vlm.cache import EmbeddingCache
from afs.vlm.frames import extract_frames
from afs.vlm.qwen_wrapper import QwenVLConfig, QwenVLWrapper


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cache-dir", default="cache/embeddings")
    p.add_argument("--split", default=None, help="override config data.split")
    p.add_argument("--limit", type=int, default=None, help="cap videos (smoke tests)")
    args = p.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg["data"]["split"]
    fps = cfg["data"]["fps"]
    max_frames = cfg["data"]["max_frames"]

    ds = NExTQADataset(
        root=cfg["data"]["root"],
        split=split,
        csv_file=cfg["data"].get("csv_file"),
        map_file=cfg["data"].get("map_file", "map_vid_vidorID.json"),
        video_dir=cfg["data"].get("video_dir", "videos"),
    )

    wrapper = QwenVLWrapper(QwenVLConfig(
        model_name=cfg["model"]["name"],
        device=cfg["device"],
        torch_dtype=cfg["model"]["torch_dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        max_pixels=cfg["model"].get("max_pixels"),
        fps=fps,
    ))
    cache = EmbeddingCache(args.cache_dir, model_tag=wrapper.model_tag)

    seen: set[str] = set()
    count = 0
    for sample in tqdm(ds, total=len(ds)):
        if sample.video_id in seen:
            continue
        seen.add(sample.video_id)
        if args.limit and count >= args.limit:
            break

        if cache.has(sample.video_id, fps=fps, max_frames=max_frames):
            continue
        if not sample.video_path.exists():
            print(f"[WARN] missing video: {sample.video_path}")
            continue

        try:
            extracted = extract_frames(sample.video_path, fps=fps, max_frames=max_frames)
            embeddings = wrapper.encode_frames_per_frame(extracted.frames)
            cache.put(
                sample.video_id, fps=fps, embeddings=embeddings,
                frame_indices=extracted.frame_indices,
                timestamps=extracted.timestamps,
                max_frames=max_frames,
                meta={"duration": extracted.duration, "fps_source": extracted.fps_source},
            )
            count += 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {sample.video_id}: {e}")

    print(f"[DONE] cached {count} new videos into {Path(args.cache_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
