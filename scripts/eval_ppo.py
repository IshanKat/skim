"""Phase 5: Evaluate the PPO-trained selector policy on NExT-QA val.

Runs the policy greedily (argmax) over each video, collects the kept frames,
then calls the VLM once per question to get the final answer.

Usage:
    python scripts/eval_ppo.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/ppo/ppo_final.pt \\
        --cache-dir cache/embeddings \\
        --output results/ppo_val.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from afs.data.nextqa import NExTQADataset
from afs.selector.policy import SelectorPolicy
from afs.utils.config import load_config
from afs.vlm.cache import EmbeddingCache
from afs.vlm.frames import ExtractedFrames, extract_frames
from afs.vlm.qwen_wrapper import QwenVLConfig, QwenVLWrapper


@torch.no_grad()
def run_policy_greedy(
    policy: SelectorPolicy,
    frame_embeddings: torch.Tensor,  # [N, D_vis]
    q_embed: torch.Tensor,           # [D_text]
    device: torch.device,
    n_choices: int = 5,
) -> list[int]:
    """Run policy greedily (no VLM calls) and return indices of kept frames."""
    max_entropy = math.log(n_choices)
    N = frame_embeddings.shape[0]
    kept_indices: list[int] = []

    policy.eval()
    for step in range(N):
        n_kept = len(kept_indices)
        kept_embed = (
            frame_embeddings[kept_indices].mean(dim=0)
            if kept_indices
            else torch.zeros(frame_embeddings.shape[1])
        )
        scalars = torch.tensor(
            [max_entropy, step / max(N, 1), n_kept / max(N, 1)],
            dtype=torch.float32,
        )

        q = q_embed.unsqueeze(0).to(device)
        f = frame_embeddings[step].unsqueeze(0).to(device)
        k = kept_embed.unsqueeze(0).to(device)
        s = scalars.unsqueeze(0).to(device)
        ke = torch.tensor([n_kept == 0], device=device)

        logits, _ = policy(q, f, k, s, ke)
        action = logits[0].argmax().item()

        if action == 0:  # KEEP
            kept_indices.append(step)
        elif action == 2:  # STOP
            break
        # SKIP: do nothing, advance

    return kept_indices


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--checkpoint",  default="checkpoints/ppo/ppo_final.pt")
    p.add_argument("--cache-dir",   default="cache/embeddings")
    p.add_argument("--output",      default="results/ppo_val.json")
    p.add_argument("--split",       default=None)
    p.add_argument("--limit",       type=int, default=None,
                   help="evaluate only the first N samples (smoke test)")
    args = p.parse_args()

    cfg = load_config(args.config)
    split     = args.split or cfg["data"]["split"]
    sel_cfg   = cfg["selector"]
    device    = cfg["device"]
    fps       = cfg["data"]["fps"]
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
        fps=fps,
    ))
    cache = EmbeddingCache(args.cache_dir, model_tag=wrapper.model_tag)

    d_vis  = sel_cfg.get("d_vis",  wrapper.d_vis)
    d_text = sel_cfg.get("d_text", wrapper.d_text)

    policy = SelectorPolicy.from_config(sel_cfg, d_vis=d_vis, d_text=d_text)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy = policy.to(device)
    policy.eval()
    print(f"[eval_ppo] loaded {args.checkpoint}")
    print(f"[eval_ppo] policy params: {sum(p.numel() for p in policy.parameters()):,}")

    predictions: list[dict] = []
    correct = 0
    per_group_total: Counter[str] = Counter()
    per_group_correct: Counter[str] = Counter()
    frame_counts: list[int] = []
    start = time.time()

    total = min(args.limit, len(ds)) if args.limit else len(ds)

    for i, sample in enumerate(tqdm(ds, total=total, desc="ppo_eval")):
        if args.limit and i >= args.limit:
            break
        if not sample.video_path.exists():
            continue

        cached = cache.get(sample.video_id, fps=fps, max_frames=max_frames)
        if cached is None:
            continue

        try:
            extracted = extract_frames(sample.video_path, fps=fps, max_frames=max_frames)
        except Exception:
            continue

        frame_embeddings = cached.embeddings
        frame_images = extracted.frames
        N = min(frame_embeddings.shape[0], len(frame_images))
        frame_embeddings = frame_embeddings[:N]
        frame_images = frame_images[:N]

        q_embed = wrapper.encode_query(sample.question)

        kept_indices = run_policy_greedy(
            policy, frame_embeddings, q_embed, torch.device(device)
        )

        kept_frames = [frame_images[i] for i in kept_indices]
        pred = wrapper.answer_mc(kept_frames, sample.question, list(sample.choices))
        is_correct = int(pred.pred_idx == sample.answer_idx)

        correct += is_correct
        per_group_total[sample.qtype_group] += 1
        per_group_correct[sample.qtype_group] += is_correct
        frame_counts.append(len(kept_frames))
        torch.cuda.empty_cache()

        predictions.append({
            "qid": sample.qid,
            "video_id": sample.video_id,
            "qtype": sample.qtype,
            "qtype_group": sample.qtype_group,
            "answer_idx": sample.answer_idx,
            "pred_idx": pred.pred_idx,
            "entropy": pred.entropy,
            "probs": pred.probs.tolist(),
            "n_frames": len(kept_frames),
            "kept_indices": kept_indices,
            "correct": bool(is_correct),
        })

    n = len(predictions)
    acc = correct / n if n else 0.0
    group_acc = {
        g: per_group_correct[g] / per_group_total[g] if per_group_total[g] else 0.0
        for g in per_group_total
    }
    avg_frames = sum(frame_counts) / len(frame_counts) if frame_counts else 0.0

    summary = {
        "method": "ppo",
        "checkpoint": args.checkpoint,
        "model": wrapper.model_tag,
        "split": split,
        "n": n,
        "accuracy": acc,
        "accuracy_by_group": group_acc,
        "avg_frames": avg_frames,
        "elapsed_seconds": time.time() - start,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2)

    print(f"[eval_ppo] n={n}  acc={acc:.4f}  avg_frames={avg_frames:.1f}")
    for g, a in sorted(group_acc.items()):
        print(f"  {g:12s}: {a:.4f} ({per_group_total[g]} questions)")
    print(f"[eval_ppo] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
