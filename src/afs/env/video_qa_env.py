"""Gym-style environment for adaptive frame selection.

MDP recap
---------
- State  s_t = (q_embed, frame_embed_t, mean_kept_embeds, H_t, t/N, |S|/N)
- Action a_t ∈ {KEEP=0, SKIP=1, STOP=2}
- Reward 0 at every non-terminal step (plus optional -step_penalty per KEEP).
          At the terminal step: 1[correct] − λ·|S|/N.
- Done   when a_t == STOP, or t == N-1 and force_stop_at_end is True.

The VLM wrapper is called at most once per KEEP action (to update H_{t+1}).
The terminal reward re-uses the cached last MCPrediction to avoid a second
forward pass when the last action was KEEP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from afs.data.nextqa import NExTQASample
from afs.vlm.entropy import MCPrediction

# Action constants
KEEP = 0
SKIP = 1
STOP = 2


@dataclass
class EnvState:
    q_embed: torch.Tensor      # [D_text]
    frame_embed: torch.Tensor  # [D_vis] — the frame being decided on
    kept_embed: torch.Tensor   # [D_vis] — mean of kept frames (zeros if none)
    entropy: float             # H_t
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
        """[3] float32 tensor: [H_t, t/N, |S|/N]."""
        return torch.tensor(
            [self.entropy, self.t_over_N, self.kept_over_N], dtype=torch.float32
        )

    @property
    def kept_empty(self) -> bool:
        return self.n_kept == 0


class VideoQAEnv:
    """Adaptive frame selection MDP.

    Parameters
    ----------
    wrapper:
        Frozen VLM wrapper used to compute entropy and the final answer.
        Must expose ``answer_mc(frames, question, choices) → MCPrediction``.
    lambda_cost:
        Frame-retention penalty weight λ in the terminal reward.
    step_penalty:
        Optional small penalty subtracted from the reward on every KEEP action.
        Helps smooth the credit-assignment signal; set to 0 to disable.
    force_stop_at_end:
        If True, the episode terminates automatically when t reaches N-1,
        even if the policy never emits STOP.
    fast_rollout:
        When True, skip VLM calls on KEEP steps and use max-entropy
        (log 5) as a conservative approximation of H_t.  Only one VLM
        forward pass is made per episode (at the terminal step for the
        reward signal).  Set True during PPO training for speed; False
        during eval for accurate entropy signals.
    n_choices:
        Number of MC answer choices — used only when fast_rollout=True
        to compute the max-entropy approximation H = log(n_choices).
    """

    def __init__(
        self,
        wrapper: Any,
        lambda_cost: float = 0.1,
        step_penalty: float = 0.0,
        force_stop_at_end: bool = True,
        fast_rollout: bool = False,
        n_choices: int = 5,
    ) -> None:
        self.wrapper = wrapper
        self.lambda_cost = lambda_cost
        self.step_penalty = step_penalty
        self.force_stop_at_end = force_stop_at_end
        self.fast_rollout = fast_rollout
        self._max_entropy = math.log(n_choices)

        # Per-episode state — populated by reset()
        self._sample: NExTQASample | None = None
        self._frame_embeddings: torch.Tensor | None = None   # [N, D_vis]
        self._frame_images: list[Image.Image] | None = None
        self._q_embed: torch.Tensor | None = None
        self._kept_indices: list[int] = []
        self._step: int = 0
        self._entropy: float = 0.0
        self._done: bool = True
        self._last_pred: MCPrediction | None = None   # cache — avoids duplicate VLM call

    # ------------------------------------------------------------------
    def reset(
        self,
        sample: NExTQASample,
        frame_embeddings: torch.Tensor,   # [N, D_vis]
        frame_images: list[Image.Image],
        q_embed: torch.Tensor,            # [D_text]
    ) -> EnvState:
        """Start a new episode for *sample*.

        Computes H_0 (entropy with no frames, query-only) via one VLM forward
        pass so the policy has a meaningful prior on the very first step.
        """
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

        # H_0: VLM answer distribution with no visual context.
        # In fast_rollout mode use max-entropy to skip the VLM call.
        if self.fast_rollout:
            from afs.vlm.entropy import MCPrediction
            import torch as _torch
            n = len(sample.choices)
            self._last_pred = MCPrediction(
                pred_idx=0,
                entropy=self._max_entropy,
                probs=_torch.full((n,), 1.0 / n),
            )
        else:
            self._last_pred = self.wrapper.answer_mc(
                [], sample.question, list(sample.choices)
            )
        self._entropy = self._last_pred.entropy

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
            if self.fast_rollout:
                # Skip VLM call — entropy stays at max-entropy approximation.
                pass
            else:
                kept_frames = [self._frame_images[i] for i in self._kept_indices]
                self._last_pred = self.wrapper.answer_mc(
                    kept_frames, self._sample.question, list(self._sample.choices)
                )
                self._entropy = self._last_pred.entropy
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
            entropy=self._entropy,
            step=self._step,
            n_frames=N,
            n_kept=n_kept,
        )

    def _terminal_reward(self) -> tuple[float, bool]:
        """Return (reward, correct).

        In fast_rollout mode, makes a fresh VLM call with the actual kept
        frames (since intermediate KEEP steps skipped VLM calls).  Otherwise
        reuses the cached last prediction to avoid a duplicate forward pass.
        """
        N = len(self._frame_images)
        retention = len(self._kept_indices) / N if N > 0 else 0.0

        if self.fast_rollout:
            kept_frames = [self._frame_images[i] for i in self._kept_indices]
            pred = self.wrapper.answer_mc(
                kept_frames, self._sample.question, list(self._sample.choices)
            )
            self._last_pred = pred
            self._entropy = pred.entropy
        else:
            assert self._last_pred is not None
            pred = self._last_pred

        correct = pred.pred_idx == self._sample.answer_idx
        reward = float(correct) - self.lambda_cost * retention
        return reward, correct
