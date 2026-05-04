"""Gym-style environment for adaptive frame selection.

MDP recap
---------
- State  s_t = (q_embed, frame_embed_t, mean_kept_embeds, t/N, |S|/N)
- Action a_t ∈ {KEEP=0, SKIP=1, STOP=2}
- Reward 0 at every non-terminal step (plus optional -step_penalty per KEEP).
          At the terminal step: 1[correct] − λ·|S|/N.
- Done   when a_t == STOP, or t == N-1 and force_stop_at_end is True.

The VLM wrapper is called exactly once per episode at the terminal step
to compute the accuracy reward and the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from afs.data.nextqa import NExTQASample

# Action constants
KEEP = 0
SKIP = 1
STOP = 2


@dataclass
class EnvState:
    q_embed: torch.Tensor      # [D_text]
    frame_embed: torch.Tensor  # [D_vis] — the frame being decided on
    kept_embed: torch.Tensor   # [D_vis] — mean of kept frames (zeros if none)
    step: int                  # 0-indexed frame position
    n_frames: int              # total frames N
    n_kept: int                # |S| so far

    @property
    def t_over_N(self) -> float:
        return self.step / max(self.n_frames, 1)

    @property
    def kept_over_N(self) -> float:
        return self.n_kept / max(self.n_frames, 1)

    @property
    def scalars(self) -> torch.Tensor:
        """[2] float32 tensor: [t/N, |S|/N]."""
        return torch.tensor(
            [self.t_over_N, self.kept_over_N], dtype=torch.float32
        )

    @property
    def kept_empty(self) -> bool:
        return self.n_kept == 0


class VideoQAEnv:
    """Adaptive frame selection MDP.

    Parameters
    ----------
    wrapper:
        Frozen VLM wrapper used to compute the final answer at the terminal
        step.  Must expose ``answer_mc(frames, question, choices) → MCPrediction``.
    lambda_cost:
        Frame-retention penalty weight λ in the terminal reward.
    step_penalty:
        Optional small penalty subtracted from the reward on every KEEP action.
    force_stop_at_end:
        If True, the episode terminates automatically when t reaches N-1,
        even if the policy never emits STOP.
    """

    def __init__(
        self,
        wrapper: Any,
        lambda_cost: float = 0.1,
        step_penalty: float = 0.0,
        force_stop_at_end: bool = True,
    ) -> None:
        self.wrapper = wrapper
        self.lambda_cost = lambda_cost
        self.step_penalty = step_penalty
        self.force_stop_at_end = force_stop_at_end

        # Per-episode state — populated by reset()
        self._sample: NExTQASample | None = None
        self._frame_embeddings: torch.Tensor | None = None   # [N, D_vis]
        self._frame_images: list[Image.Image] | None = None
        self._q_embed: torch.Tensor | None = None
        self._kept_indices: list[int] = []
        self._step: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    def reset(
        self,
        sample: NExTQASample,
        frame_embeddings: torch.Tensor,   # [N, D_vis]
        frame_images: list[Image.Image],
        q_embed: torch.Tensor,            # [D_text]
    ) -> EnvState:
        """Start a new episode for *sample*."""
        assert frame_embeddings.shape[0] == len(frame_images), (
            f"embeddings ({frame_embeddings.shape[0]}) and images "
            f"({len(frame_images)}) must have the same length"
        )

        self._sample = sample
        self._frame_embeddings = frame_embeddings
        self._frame_images = frame_images
        self._q_embed = q_embed
        self._kept_indices = []
        self._step = 0
        self._done = False

        return self._build_state()

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[EnvState, float, bool, dict[str, Any]]:
        """Apply *action* to the current frame and advance.

        Returns
        -------
        next_state : EnvState
        reward     : float
        done       : bool
        info       : dict  (n_kept, step, correct — the last only at terminal)
        """
        assert not self._done, "episode is done — call reset() first"
        assert action in (KEEP, SKIP, STOP), f"invalid action {action}"

        N = len(self._frame_images)
        reward = 0.0
        info: dict[str, Any] = {}

        if action == KEEP:
            self._kept_indices.append(self._step)
            reward -= self.step_penalty

        terminal = action == STOP or (self.force_stop_at_end and self._step == N - 1)

        if terminal:
            self._done = True
            term_reward, correct = self._terminal_reward()
            reward += term_reward
            info["correct"] = correct

        # Advance frame pointer (clamp to last frame once done)
        self._step = min(self._step + 1, N - 1)

        info["n_kept"] = len(self._kept_indices)
        info["step"] = self._step

        return self._build_state(), reward, self._done, info

    # ------------------------------------------------------------------
    def _build_state(self) -> EnvState:
        N = len(self._frame_images)
        n_kept = len(self._kept_indices)

        if n_kept > 0:
            kept_embed = self._frame_embeddings[self._kept_indices].mean(dim=0)
        else:
            kept_embed = torch.zeros_like(self._frame_embeddings[0])

        return EnvState(
            q_embed=self._q_embed,
            frame_embed=self._frame_embeddings[self._step],
            kept_embed=kept_embed,
            step=self._step,
            n_frames=N,
            n_kept=n_kept,
        )

    def _terminal_reward(self) -> tuple[float, bool]:
        """Call VLM once with kept frames; return (reward, correct)."""
        N = len(self._frame_images)
        retention = len(self._kept_indices) / N if N > 0 else 0.0
        kept_frames = [self._frame_images[i] for i in self._kept_indices]
        pred = self.wrapper.answer_mc(
            kept_frames, self._sample.question, list(self._sample.choices)
        )
        correct = pred.pred_idx == self._sample.answer_idx
        reward = float(correct) - self.lambda_cost * retention
        return reward, correct
