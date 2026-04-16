"""Phase 1 artifact: full-frame accuracy on NExT-QA val with Qwen2.5-VL.

Serves two purposes:
1. Validate the VLM wrapper end-to-end (frame extraction → prompt → answer).
2. Establish the upper-bound accuracy row of the results table.

Usage:
    python scripts/eval_full_frame.py \
        --config configs/default.yaml \
        --output results/full_frame_val.json \
        --max-frames 32
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from afs.data.nextqa import NExTQADataset
from afs.utils.config import load_config
from afs.vlm.frames import extract_frames
from afs.vlm.qwen_wrapper import QwenVLConfig, QwenVLWrapper


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default=None)
    p.add_argument("--output", default="results/full_frame_val.json")
    p.add_argument("--max-frames", type=int, default=None,
                   help="override config data.max_frames (e.g. 32 for upper bound)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg["data"]["split"]
    fps = cfg["data"]["fps"]
    max_frames = args.max_frames if args.max_frames is not None else cfg["data"]["max_frames"]

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
        fps=fps,
    ))

    predictions: list[dict] = []
    correct = 0
    per_group_total: Counter[str] = Counter()
    per_group_correct: Counter[str] = Counter()
    frame_counts: list[int] = []
    start = time.time()

    iterator = iter(ds)
    total = min(args.limit, len(ds)) if args.limit else len(ds)

    for i, sample in enumerate(tqdm(iterator, total=total)):
        if args.limit and i >= args.limit:
            break
        if not sample.video_path.exists():
            print(f"[SKIP] missing video for qid={sample.qid}")
            continue

        extracted = extract_frames(sample.video_path, fps=fps, max_frames=max_frames)
        pred = wrapper.answer_mc(extracted.frames, sample.question, list(sample.choices))
        is_correct = int(pred.pred_idx == sample.answer_idx)
        correct += is_correct
        per_group_total[sample.qtype_group] += 1
        per_group_correct[sample.qtype_group] += is_correct
        frame_counts.append(len(extracted.frames))

        predictions.append({
            "qid": sample.qid,
            "video_id": sample.video_id,
            "qtype": sample.qtype,
            "qtype_group": sample.qtype_group,
            "answer_idx": sample.answer_idx,
            "pred_idx": pred.pred_idx,
            "entropy": pred.entropy,
            "probs": pred.probs.tolist(),
            "n_frames": len(extracted.frames),
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
        "method": "full_frame",
        "model": cfg["model"]["name"],
        "split": split,
        "max_frames": max_frames,
        "fps": fps,
        "n": n,
        "accuracy": acc,
        "accuracy_by_group": group_acc,
        "avg_frames_per_question": avg_frames,
        "elapsed_seconds": time.time() - start,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2)

    print(f"[DONE] n={n} acc={acc:.4f} avg_frames={avg_frames:.1f}")
    for g, a in group_acc.items():
        print(f"  {g:12s}: {a:.4f} ({per_group_total[g]} questions)")
    print(f"[DONE] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
