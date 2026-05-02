"""PPO trainer for the adaptive frame selector.

Custom implementation — not using trl.PPOTrainer because our env is a
hand-rolled gym-style MDP rather than a language-model RLHF setting.

Algorithm
---------
1. Collect *rollout_batch* episodes via the VideoQAEnv (fast_rollout=True).
2. Compute GAE advantages.
3. For *ppo_epochs* epochs: sample mini-batches and apply:
   - Clipped surrogate policy loss (PPO-clip).
   - Value function MSE loss.
   - Entropy bonus to prevent premature collapse.
   - KL penalty against the warm-start reference policy.
4. Repeat until *total_episodes* reached.
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from afs.env.video_qa_env import VideoQAEnv
from afs.selector.policy import SelectorPolicy
from afs.training.rollout import (
    Episode,
    Transition,
    collect_rollouts,
    compute_gae,
)
from afs.vlm.cache import EmbeddingCache


class PPOTrainer:
    """Trains the selector policy with PPO.

    Parameters
    ----------
    policy:
        The SelectorPolicy to train.
    ref_policy:
        Frozen copy of the warm-started policy for KL regularisation.
    env:
        VideoQAEnv configured with fast_rollout=True.
    optimizer:
        AdamW pre-built over policy.parameters().
    clip_range:
        ε for the PPO-clip objective.
    ppo_epochs:
        Number of gradient epochs per rollout batch.
    batch_size:
        Mini-batch size (transitions, not episodes).
    gamma / gae_lambda:
        Discount and GAE smoothing.
    kl_coef:
        Weight of the KL penalty term.
    entropy_coef:
        Weight of the entropy bonus.
    vf_coef:
        Weight of the value-function MSE loss.
    max_grad_norm:
        Gradient clipping threshold.
    device:
        Training device.
    """

    def __init__(
        self,
        policy: SelectorPolicy,
        ref_policy: SelectorPolicy,
        env: VideoQAEnv,
        optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        kl_coef: float = 0.05,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cuda",
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.env = env
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def train(
        self,
        dataset: Any,
        cache: EmbeddingCache,
        wrapper: Any,
        total_episodes: int,
        rollout_batch: int,
        fps: float,
        max_frames: int,
        save_dir: str | Path | None = None,
        save_every: int = 500,
        log_every: int = 1,
    ) -> None:
        """Main training loop."""
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        episodes_done = 0
        iteration = 0
        t0 = time.time()

        while episodes_done < total_episodes:
            iter_t = time.time()

            # --- Rollout ---
            episodes = collect_rollouts(
                policy=self.policy,
                env=self.env,
                dataset=dataset,
                cache=cache,
                wrapper=wrapper,
                n_episodes=rollout_batch,
                fps=fps,
                max_frames=max_frames,
                device=self.device,
            )
            if not episodes:
                continue

            episodes = compute_gae(episodes, self.gamma, self.gae_lambda)

            # --- Flatten transitions ---
            transitions = [t for ep in episodes for t in ep.transitions]
            if not transitions:
                continue

            # Normalise advantages
            advs = torch.tensor([t.advantage for t in transitions], dtype=torch.float32)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for i, t in enumerate(transitions):
                t.advantage = advs[i].item()  # type: ignore[attr-defined]

            # --- PPO update epochs ---
            metrics = self._update(transitions)

            episodes_done += len(episodes)
            iteration += 1
            elapsed = time.time() - t0

            # --- Logging ---
            if iteration % log_every == 0:
                mean_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
                mean_kept = sum(ep.n_kept for ep in episodes) / len(episodes)
                mean_frames = sum(ep.n_frames for ep in episodes) / max(len(episodes), 1)
                accuracy = sum(ep.correct for ep in episodes) / len(episodes)
                retention = mean_kept / max(mean_frames, 1)
                iter_time = time.time() - iter_t
                print(
                    f"[ppo] iter={iteration:4d}  eps={episodes_done:6d}  "
                    f"reward={mean_reward:+.3f}  acc={accuracy:.2%}  "
                    f"ret={retention:.2%}  "
                    f"|pol={metrics['policy_loss']:.3f}  "
                    f"val={metrics['value_loss']:.3f}  "
                    f"ent={metrics['entropy']:.3f}  "
                    f"kl={metrics['kl']:.4f}|  "
                    f"t={iter_time:.0f}s"
                )

            # --- Checkpointing ---
            if save_dir and episodes_done % save_every < rollout_batch:
                ckpt = save_dir / f"ppo_ep{episodes_done:06d}.pt"
                torch.save(self.policy.state_dict(), ckpt)
                print(f"[ppo] Saved {ckpt}")

        # Final save
        if save_dir:
            final = save_dir / "ppo_final.pt"
            torch.save(self.policy.state_dict(), final)
            print(f"[ppo] Training done. Final weights -> {final}")

    # ------------------------------------------------------------------
    def _update(self, transitions: list[Transition]) -> dict[str, float]:
        """Run ppo_epochs of gradient updates over the transition buffer."""
        # Pre-stack tensors on CPU — move to device in mini-batches
        q_embeds     = torch.stack([t.q_embed     for t in transitions])
        frame_embeds = torch.stack([t.frame_embed for t in transitions])
        kept_embeds  = torch.stack([t.kept_embed  for t in transitions])
        scalars      = torch.stack([t.scalars     for t in transitions])
        kept_empty   = torch.tensor([t.kept_empty for t in transitions], dtype=torch.bool)
        actions      = torch.tensor([t.action     for t in transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob  for t in transitions], dtype=torch.float32)
        returns      = torch.tensor([t.ret        for t in transitions], dtype=torch.float32)
        advantages   = torch.tensor([t.advantage  for t in transitions], dtype=torch.float32)

        T = len(transitions)
        metric_sums: dict[str, float] = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "kl": 0.0,
        }
        n_updates = 0

        self.policy.train()
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                idx = perm[start : start + self.batch_size]

                bq = q_embeds[idx].to(self.device)
                bf = frame_embeds[idx].to(self.device)
                bk = kept_embeds[idx].to(self.device)
                bs = scalars[idx].to(self.device)
                bke = kept_empty[idx].to(self.device)
                ba = actions[idx].to(self.device)
                bold = old_log_probs[idx].to(self.device)
                bret = returns[idx].to(self.device)
                badv = advantages[idx].to(self.device)

                logits, values = self.policy(bq, bf, bk, bs, bke)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(ba)
                entropy = dist.entropy().mean()

                # PPO-clip surrogate
                ratio = (new_log_probs - bold).exp()
                surr1 = ratio * badv
                surr2 = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * badv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, bret)

                # KL from reference policy
                with torch.no_grad():
                    ref_logits, _ = self.ref_policy(bq, bf, bk, bs, bke)
                kl = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.kl_coef * kl
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metric_sums["policy_loss"] += policy_loss.item()
                metric_sums["value_loss"]  += value_loss.item()
                metric_sums["entropy"]     += entropy.item()
                metric_sums["kl"]          += kl.item()
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in metric_sums.items()}
