"""Supervised warm-start for the selector policy.

Trains the selector to imitate a fixed frame-selection strategy so that PPO
starts from a non-random policy — critical for sparse-reward RL.

Default target: uniform-k (evenly-spaced keep positions).  When FastV scores
are available, pass them as ``scores`` to ``generate_targets`` to use
importance-weighted selection instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from afs.env.video_qa_env import KEEP, SKIP, STOP
from afs.selector.policy import SelectorPolicy
from afs.vlm.cache import EmbeddingCache


def generate_targets(n_frames: int, n_keep: int, scores: Sequence[float] | None = None) -> list[int]:
    """Return a keep/skip/stop action sequence for one video.

    Parameters
    ----------
    n_frames:
        Total number of frames in the video.
    n_keep:
        Budget (number of frames to keep).
    scores:
        Optional per-frame importance scores (e.g. from FastV).  When
        provided, the top-``n_keep`` frames are kept; otherwise evenly-spaced
        positions are used.

    Returns
    -------
    List of actions (KEEP/SKIP/STOP), one per frame up to and including the
    step where the episode ends.  The last entry is either STOP (if the last
    keep is not the final frame) or KEEP (if it is; the env force-stops).
    """
    if n_frames == 0:
        return []
    n_keep = max(0, min(n_keep, n_frames))
    if n_keep == 0:
        return [STOP]

    if scores is not None:
        keep_positions: set[int] = set(
            int(i) for i in np.argsort(scores)[-n_keep:]
        )
    else:
        keep_positions = set(
            int(round(x)) for x in np.linspace(0, n_frames - 1, n_keep)
        )

    actions: list[int] = []
    n_kept = 0
    for t in range(n_frames):
        if t in keep_positions:
            actions.append(KEEP)
            n_kept += 1
        else:
            actions.append(SKIP)

        if n_kept == n_keep:
            # All target frames have been seen; stop early if not at the end
            if t < n_frames - 1:
                actions.append(STOP)
            break

    return actions


# ---------------------------------------------------------------------------

class WarmStartTrainer:
    """Trains ``policy`` by cross-entropy imitation of ``generate_targets``.

    Parameters
    ----------
    policy:
        The SelectorPolicy to train (modified in-place).
    n_keep:
        Frame budget used to generate targets (e.g. 16 to mimic uniform-16).
    lr:
        AdamW learning rate.
    epochs:
        Number of passes over the collected dataset.
    batch_size:
        Mini-batch size (number of (state, action) pairs per gradient step).
    device:
        Torch device for training.
    """

    def __init__(
        self,
        policy: SelectorPolicy,
        n_keep: int = 16,
        lr: float = 1e-4,
        epochs: int = 3,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        max_frames: int | None = None,
    ) -> None:
        self.policy = policy
        self.n_keep = n_keep
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.max_frames = max_frames
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def run(
        self,
        dataset: "NExTQADataset",  # noqa: F821
        cache: EmbeddingCache,
        wrapper: "QwenVLWrapper",  # noqa: F821
        limit: int | None = None,
        save_path: str | Path | None = None,
    ) -> SelectorPolicy:
        """Build training data then run cross-entropy training.

        Parameters
        ----------
        dataset:
            NExTQA dataset split to use for warm-start.
        cache:
            Pre-computed frame embedding cache.
        wrapper:
            Frozen VLM wrapper (used only for ``encode_query``).
        limit:
            If set, use only the first *limit* samples.
        save_path:
            If set, save the trained policy weights here after training.
        """
        print("[warm_start] Collecting training pairs...")
        pairs = self._collect(dataset, cache, wrapper, limit)
        print(f"[warm_start] Collected {len(pairs)} (state, action) pairs.")

        for epoch in range(self.epochs):
            loss_sum, n_batches = 0.0, 0
            perm = torch.randperm(len(pairs))

            for start in range(0, len(pairs), self.batch_size):
                idx = perm[start : start + self.batch_size].tolist()
                batch = [pairs[i] for i in idx]
                loss_sum += self._step(batch)
                n_batches += 1

            print(f"[warm_start] epoch {epoch + 1}/{self.epochs}  "
                  f"loss={loss_sum / max(n_batches, 1):.4f}")

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.state_dict(), p)
            print(f"[warm_start] Saved weights -> {p}")

        return self.policy

    # ------------------------------------------------------------------
    def _collect(self, dataset, cache, wrapper, limit):
        """Generate (q_embed, frame_embed, kept_embed, scalars, kept_empty, action) tuples."""
        pairs: list[tuple] = []
        total = min(limit, len(dataset)) if limit else len(dataset)

        for i, sample in enumerate(tqdm(dataset, total=total, desc="collecting")):
            if limit and i >= limit:
                break

            cached = cache.get(sample.video_id, fps=1.0, max_frames=self.max_frames)
            if cached is None:
                continue

            embeddings = cached.embeddings  # [N, D_vis]
            N = embeddings.shape[0]

            q_embed = wrapper.encode_query(sample.question).cpu()  # [D_text]
            actions = generate_targets(N, min(self.n_keep, N))

            kept_indices: list[int] = []
            for t, action in enumerate(actions):
                n_kept = len(kept_indices)
                frame_embed = embeddings[t]  # [D_vis]

                if n_kept > 0:
                    kept_embed = embeddings[kept_indices].mean(dim=0)
                else:
                    kept_embed = torch.zeros_like(frame_embed)

                scalars = torch.tensor([
                    t / max(N, 1),             # t/N
                    n_kept / max(N, 1),        # |S|/N
                ], dtype=torch.float32)

                pairs.append((
                    q_embed,
                    frame_embed,
                    kept_embed,
                    scalars,
                    n_kept == 0,   # kept_empty flag
                    action,
                ))

                if action == KEEP:
                    kept_indices.append(t)
                elif action == STOP:
                    break

        return pairs

    def _step(self, batch: list[tuple]) -> float:
        self.policy.train()
        self.optimizer.zero_grad()

        q_embeds     = torch.stack([b[0] for b in batch]).to(self.device)
        frame_embeds = torch.stack([b[1] for b in batch]).to(self.device)
        kept_embeds  = torch.stack([b[2] for b in batch]).to(self.device)
        scalars      = torch.stack([b[3] for b in batch]).to(self.device)
        kept_empty   = torch.tensor([b[4] for b in batch], dtype=torch.bool).to(self.device)
        targets      = torch.tensor([b[5] for b in batch], dtype=torch.long).to(self.device)

        logits, _ = self.policy(q_embeds, frame_embeds, kept_embeds, scalars, kept_empty)
        loss = F.cross_entropy(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
