"""Phase 4: PPO fine-tuning of the selector policy.

Loads the warm-started checkpoint and trains with PPO against the
VideoQAEnv binary accuracy reward.

Usage:
    python scripts/train_ppo.py \\
        --config configs/default.yaml \\
        --cache-dir cache/embeddings \\
        --warmstart checkpoints/warmstart.pt \\
        --output-dir checkpoints/ppo
"""

from __future__ import annotations

import argparse

import torch

from afs.data.nextqa import NExTQADataset
from afs.env.video_qa_env import VideoQAEnv
from afs.selector.policy import SelectorPolicy
from afs.training.ppo import PPOTrainer
from afs.utils.config import load_config
from afs.vlm.cache import EmbeddingCache
from afs.vlm.qwen_wrapper import QwenVLConfig, QwenVLWrapper


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--cache-dir",  default="cache/embeddings")
    p.add_argument("--warmstart",  default="checkpoints/warmstart.pt",
                   help="warm-started selector checkpoint to start from")
    p.add_argument("--output-dir", default="checkpoints/ppo")
    p.add_argument("--split",      default=None)
    p.add_argument("--total-episodes", type=int, default=None)
    p.add_argument("--rollout-batch",  type=int, default=None)
    p.add_argument("--save-every",     type=int, default=500)
    p.add_argument("--log-every",      type=int, default=1)
    p.add_argument("--limit",      type=int, default=None,
                   help="restrict dataset to first N samples (smoke test)")
    p.add_argument("--fast-rollout", action="store_true", default=False,
                   help="skip intermediate VLM calls during rollout (faster but no real entropy)")
    args = p.parse_args()

    cfg = load_config(args.config)
    split     = args.split or cfg["data"]["split"]
    ppo_cfg   = cfg["ppo"]
    sel_cfg   = cfg["selector"]
    env_cfg   = cfg["env"]
    device    = cfg["device"]
    fps       = cfg["data"]["fps"]
    max_frames = cfg["data"]["max_frames"]

    total_episodes = args.total_episodes or ppo_cfg["total_episodes"]
    rollout_batch  = args.rollout_batch  or ppo_cfg["rollout_batch"]

    ds = NExTQADataset(
        root=cfg["data"]["root"],
        split=split,
        csv_file=cfg["data"].get("csv_file"),
        map_file=cfg["data"].get("map_file", "map_vid_vidorID.json"),
        video_dir=cfg["data"].get("video_dir", "videos"),
    )
    if args.limit:
        from afs.data.nextqa import NExTQASample
        ds._samples = ds._samples[: args.limit]

    wrapper = QwenVLWrapper(QwenVLConfig(
        model_name=cfg["model"]["name"],
        device=device,
        torch_dtype=cfg["model"]["torch_dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        max_pixels=cfg["model"].get("max_pixels"),
        fps=fps,
    ))
    cache = EmbeddingCache(args.cache_dir, model_tag=wrapper.model_tag)

    d_vis  = sel_cfg.get("d_vis",  wrapper.d_vis)
    d_text = sel_cfg.get("d_text", wrapper.d_text)

    # Load warm-start checkpoint into both policy and frozen ref_policy
    policy = SelectorPolicy.from_config(sel_cfg, d_vis=d_vis, d_text=d_text)
    if args.warmstart:
        state = torch.load(args.warmstart, map_location="cpu", weights_only=True)
        policy.load_state_dict(state)
        print(f"[ppo] Loaded warm-start from {args.warmstart}")
    policy = policy.to(device)

    import copy
    ref_policy = copy.deepcopy(policy)
    ref_policy = ref_policy.to(device)

    env = VideoQAEnv(
        wrapper=wrapper,
        lambda_cost=env_cfg.get("lambda_cost", 0.1),
        step_penalty=env_cfg.get("step_penalty", 0.0),
        force_stop_at_end=env_cfg.get("force_stop_at_end", True),
        fast_rollout=args.fast_rollout,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo_cfg["lr"])

    trainer = PPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        env=env,
        optimizer=optimizer,
        clip_range=ppo_cfg["clip_range"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        batch_size=64,
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        kl_coef=ppo_cfg["kl_coef"],
        entropy_coef=ppo_cfg["entropy_coef"],
        device=device,
    )

    print(f"[ppo] policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"[ppo] total_episodes={total_episodes}, rollout_batch={rollout_batch}")

    trainer.train(
        dataset=ds,
        cache=cache,
        wrapper=wrapper,
        total_episodes=total_episodes,
        rollout_batch=rollout_batch,
        fps=fps,
        max_frames=max_frames,
        save_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
