"""Rollout collection for PPO.

One episode = one (video, question) pair processed frame-by-frame.
In fast_rollout mode the env skips intermediate VLM calls; the only VLM
forward pass is the terminal one used to compute the binary accuracy reward.
This trades entropy accuracy for speed (critical on 8GB GPU).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from afs.data.nextqa import NExTQASample
from afs.env.video_qa_env import KEEP, SKIP, STOP, EnvState, VideoQAEnv
from afs.selector.policy import SelectorPolicy
from afs.vlm.cache import CachedFrames, EmbeddingCache
from afs.vlm.frames import extract_frames


@dataclass
class Transition:
    """One (state, action, log_prob, value, reward, done) tuple."""
    q_embed:     torch.Tensor   # [D_text]
    frame_embed: torch.Tensor   # [D_vis]
    kept_embed:  torch.Tensor   # [D_vis]
    scalars:     torch.Tensor   # [3]
    kept_empty:  bool
    action:      int
    log_prob:    float
    value:       float
    reward:      float
    done:        bool


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    n_kept: int = 0
    n_frames: int = 0
    correct: bool = False


def _state_to_tensors(state: EnvState):
    return (
        state.q_embed,
        state.frame_embed,
        state.kept_embed,
        state.scalars,
        state.kept_empty,
    )


@torch.no_grad()
def run_episode(
    policy: SelectorPolicy,
    env: VideoQAEnv,
    sample: NExTQASample,
    frame_embeddings: torch.Tensor,   # [N, D_vis]
    frame_images: list,               # [N] PIL Images (used at terminal in fast mode)
    q_embed: torch.Tensor,            # [D_text]
    device: torch.device,
) -> Episode:
    """Run one episode and collect transitions."""
    state = env.reset(sample, frame_embeddings, frame_images, q_embed)
    episode = Episode(n_frames=state.n_frames)

    policy.eval()
    done = False
    while not done:
        q = state.q_embed.unsqueeze(0).to(device)
        f = state.frame_embed.unsqueeze(0).to(device)
        k = state.kept_embed.unsqueeze(0).to(device)
        s = state.scalars.unsqueeze(0).to(device)
        ke = torch.tensor([state.kept_empty], device=device)

        logits, value = policy(q, f, k, s, ke)
        dist = torch.distributions.Categorical(logits=logits[0])
        action = dist.sample()
        log_prob = dist.log_prob(action).item()

        next_state, reward, done, info = env.step(action.item())

        t = Transition(
            q_embed=state.q_embed,
            frame_embed=state.frame_embed,
            kept_embed=state.kept_embed,
            scalars=state.scalars,
            kept_empty=state.kept_empty,
            action=action.item(),
            log_prob=log_prob,
            value=value[0].item(),
            reward=reward,
            done=done,
        )
        episode.transitions.append(t)
        episode.total_reward += reward

        if done:
            episode.n_kept = info.get("n_kept", 0)
            episode.correct = info.get("correct", False)

        state = next_state

    return episode


def collect_rollouts(
    policy: SelectorPolicy,
    env: VideoQAEnv,
    dataset: Any,
    cache: EmbeddingCache,
    wrapper: Any,
    n_episodes: int,
    fps: float,
    max_frames: int,
    device: torch.device,
    indices: Sequence[int] | None = None,
) -> list[Episode]:
    """Collect *n_episodes* rollouts, sampling randomly from *dataset*.

    Frames are loaded from disk for each episode so memory stays bounded.
    """
    episodes: list[Episode] = []
    n = len(dataset)

    for _ in range(n_episodes):
        idx = random.randrange(n) if indices is None else random.choice(indices)
        sample = dataset[idx]

        print(f"[rollout] sample idx={idx} video={sample.video_id}", flush=True)

        cached = cache.get(sample.video_id, fps=fps, max_frames=max_frames)
        if cached is None:
            # Fall back to 32-frame cache and subsample to max_frames
            cached_full = cache.get(sample.video_id, fps=fps, max_frames=32)
            if cached_full is not None and max_frames < 32:
                N = cached_full.embeddings.shape[0]
                n_keep = min(max_frames, N)
                keep = np.linspace(0, N - 1, n_keep).round().astype(int)
                keep = np.unique(keep)
                cached = CachedFrames(
                    embeddings=cached_full.embeddings[keep],
                    frame_indices=cached_full.frame_indices[keep],
                    timestamps=cached_full.timestamps[keep],
                    meta=cached_full.meta,
                )
            else:
                cached = cached_full

        if cached is None or not sample.video_path.exists():
            print(f"[rollout] skip: cache={cached is not None} path={sample.video_path.exists()}", flush=True)
            continue

        frame_embeddings = cached.embeddings  # [N, D_vis]

        # Load actual frames for terminal VLM call (fast_rollout mode).
        print(f"[rollout] extracting frames from {sample.video_path.name}", flush=True)
        try:
            extracted = extract_frames(sample.video_path, fps=fps, max_frames=max_frames)
            frame_images = extracted.frames
        except Exception as e:
            print(f"[rollout] extract_frames failed: {e}", flush=True)
            continue

        print(f"[rollout] {len(frame_images)} frames extracted, encoding query", flush=True)

        # Align frame count (cache and extractor may differ slightly).
        N = min(frame_embeddings.shape[0], len(frame_images))
        frame_embeddings = frame_embeddings[:N]
        frame_images = frame_images[:N]

        q_embed = wrapper.encode_query(sample.question)

        print(f"[rollout] running episode (fast_rollout={env.fast_rollout})", flush=True)
        ep = run_episode(
            policy, env, sample,
            frame_embeddings, frame_images, q_embed,
            device,
        )
        episodes.append(ep)

    return episodes


# ---------------------------------------------------------------------------
# GAE returns + advantages
# ---------------------------------------------------------------------------

def compute_gae(
    episodes: list[Episode],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> list[Episode]:
    """Attach GAE advantages and discounted returns to each transition in-place.

    We treat each episode as an independent trajectory with bootstrap value = 0
    (all episodes terminate naturally).
    """
    for ep in episodes:
        T = len(ep.transitions)
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            tr = ep.transitions[t]
            next_value = 0.0 if tr.done else ep.transitions[t + 1].value
            delta = tr.reward + gamma * next_value - tr.value
            gae = delta + gamma * gae_lambda * (0.0 if tr.done else gae)
            advantages[t] = gae
        # Attach advantages and returns as extra attributes
        for t, tr in enumerate(ep.transitions):
            tr.advantage = advantages[t]             # type: ignore[attr-defined]
            tr.ret = advantages[t] + tr.value        # type: ignore[attr-defined]

    return episodes
