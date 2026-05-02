"""Phase 3: Supervised warm-start for the selector policy.

Trains the selector to imitate uniform-k frame selection using cross-entropy
on pre-cached frame embeddings.  Run precompute_embeddings.py first.

Usage:
    python scripts/train_warmstart.py \
        --config configs/default.yaml \
        --cache-dir cache/embeddings \
        --n-keep 16 \
        --output checkpoints/warmstart.pt
"""

from __future__ import annotations

import argparse

import torch

from afs.data.nextqa import NExTQADataset
from afs.selector.policy import SelectorPolicy
from afs.training.warm_start import WarmStartTrainer
from afs.utils.config import load_config
from afs.vlm.cache import EmbeddingCache
from afs.vlm.qwen_wrapper import QwenVLConfig, QwenVLWrapper


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cache-dir", default="cache/embeddings")
    p.add_argument("--split", default=None)
    p.add_argument("--n-keep", type=int, default=None,
                   help="frame budget for imitation target (default: config data.max_frames)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="use only first N dataset samples (smoke test)")
    p.add_argument("--output", default="checkpoints/warmstart.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg["data"]["split"]
    sel_cfg = cfg["selector"]
    ws_cfg = cfg["warmstart"]

    n_keep     = args.n_keep     or cfg["data"]["max_frames"]
    epochs     = args.epochs     or ws_cfg["epochs"]
    lr         = args.lr         or ws_cfg["lr"]
    batch_size = args.batch_size or ws_cfg["batch_size"]
    device     = cfg["device"]
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
        device=device,
        torch_dtype=cfg["model"]["torch_dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        max_pixels=cfg["model"].get("max_pixels"),
    ))
    cache = EmbeddingCache(args.cache_dir, model_tag=wrapper.model_tag)

    d_vis  = sel_cfg.get("d_vis",  wrapper.d_vis)
    d_text = sel_cfg.get("d_text", wrapper.d_text)
    policy = SelectorPolicy.from_config(sel_cfg, d_vis=d_vis, d_text=d_text)
    policy = policy.to(device)

    print(f"[warm_start] policy params: "
          f"{sum(p.numel() for p in policy.parameters()):,}")
    print(f"[warm_start] target: uniform-{n_keep}, "
          f"epochs={epochs}, lr={lr}, batch={batch_size}, "
          f"limit={args.limit or 'all'}")

    trainer = WarmStartTrainer(
        policy=policy,
        n_keep=n_keep,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        max_frames=max_frames,
    )
    trainer.run(ds, cache, wrapper, limit=args.limit, save_path=args.output)

    print("[warm_start] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
