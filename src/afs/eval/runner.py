"""Evaluation runner.

Evaluates any frame-selector method on a NExTQA-style dataset and writes
per-question predictions plus an accuracy summary to a JSON file.

A "method" is any object that exposes::

    method.name          -> str
    method.select_frames(video_path, fps) -> ExtractedFrames

This covers UniformBaseline and, later, the PPO policy wrapper.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from tqdm import tqdm

from afs.data.nextqa import NExTQADataset
from afs.vlm.frames import ExtractedFrames
from afs.vlm.qwen_wrapper import QwenVLWrapper


@runtime_checkable
class FrameSelector(Protocol):
    """Duck-type interface every baseline and the PPO policy must satisfy."""

    @property
    def name(self) -> str: ...

    def select_frames(self, video_path: str | Path, fps: float) -> ExtractedFrames: ...


def evaluate(
    method: FrameSelector,
    dataset: NExTQADataset,
    wrapper: QwenVLWrapper,
    *,
    fps: float = 1.0,
    output_path: str | Path,
    limit: int | None = None,
) -> dict:
    """Run *method* on *dataset* and return (and write) a result dict.

    Parameters
    ----------
    method:
        Frame selector to evaluate.
    dataset:
        NExTQA dataset split.
    wrapper:
        Frozen VLM wrapper used for MC prediction.
    fps:
        Frame sampling rate passed to the selector.
    output_path:
        Where to write the JSON results file.
    limit:
        Evaluate only the first *limit* samples (useful for smoke tests).

    Returns
    -------
    dict with keys ``summary`` and ``predictions``.
    """
    predictions: list[dict] = []
    correct = 0
    per_group_total: Counter[str] = Counter()
    per_group_correct: Counter[str] = Counter()
    frame_counts: list[int] = []
    start = time.time()

    total = min(limit, len(dataset)) if limit else len(dataset)

    for i, sample in enumerate(tqdm(dataset, total=total, desc=method.name)):
        if limit and i >= limit:
            break
        if not sample.video_path.exists():
            continue

        extracted = method.select_frames(sample.video_path, fps=fps)
        pred = wrapper.answer_mc(extracted.frames, sample.question, list(sample.choices))
        is_correct = int(pred.pred_idx == sample.answer_idx)

        correct += is_correct
        per_group_total[sample.qtype_group] += 1
        per_group_correct[sample.qtype_group] += is_correct
        frame_counts.append(len(extracted.frames))
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
            "n_frames": len(extracted.frames),
            "correct": bool(is_correct),
        })

    n = len(predictions)
    acc = correct / n if n else 0.0
    group_acc = {
        g: per_group_correct[g] / per_group_total[g] if per_group_total[g] else 0.0
        for g in per_group_total
    }

    summary = {
        "method": method.name,
        "model": wrapper.model_tag,
        "n": n,
        "accuracy": acc,
        "accuracy_by_group": group_acc,
        "avg_frames": sum(frame_counts) / len(frame_counts) if frame_counts else 0.0,
        "elapsed_seconds": time.time() - start,
    }

    out = {"summary": summary, "predictions": predictions}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    return out
