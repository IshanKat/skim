"""Unit tests for VideoQAEnv step semantics and reward computation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from afs.data.nextqa import NExTQASample
from afs.env.video_qa_env import KEEP, SKIP, STOP, VideoQAEnv
from afs.vlm.entropy import MCPrediction


# ---------------------------------------------------------------------------
# Mock VLM wrapper
# ---------------------------------------------------------------------------

class MockWrapper:
    """Deterministic wrapper for testing.

    answer_mc returns pred_idx = len(frames) % 5 so we can engineer
    correct/incorrect outcomes by controlling the kept-frame count.
    Entropy decreases linearly as more frames are kept.
    """

    def answer_mc(self, frames, question, choices):
        n = len(choices)
        pred_idx = len(frames) % n
        entropy = math.log(n) * (1.0 - len(frames) / 32)
        probs = torch.full((n,), 1.0 / n)
        return MCPrediction(pred_idx=pred_idx, entropy=entropy, probs=probs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FRAMES = 6
D_VIS = 32
D_TEXT = 16
LAMBDA = 0.1


def _make_sample(answer_idx: int = 0) -> NExTQASample:
    return NExTQASample(
        qid="t0",
        video_id="v0",
        video_path=Path("fake.mp4"),
        question="test question",
        choices=("a", "b", "c", "d", "e"),
        answer_idx=answer_idx,
        qtype="CW",
        qtype_group="causal",
    )


def _make_env(**kwargs) -> VideoQAEnv:
    kwargs.setdefault("lambda_cost", LAMBDA)
    return VideoQAEnv(MockWrapper(), **kwargs)


def _make_episode(env: VideoQAEnv, answer_idx: int = 0):
    sample = _make_sample(answer_idx)
    embeddings = torch.randn(N_FRAMES, D_VIS)
    images = [None] * N_FRAMES   # images unused by MockWrapper
    q_embed = torch.randn(D_TEXT)
    state = env.reset(sample, embeddings, images, q_embed)
    return state, embeddings


# ---------------------------------------------------------------------------
# reset tests
# ---------------------------------------------------------------------------

def test_reset_returns_correct_fields():
    env = _make_env()
    state, _ = _make_episode(env)

    assert state.step == 0
    assert state.n_frames == N_FRAMES
    assert state.n_kept == 0
    assert state.kept_empty
    assert torch.all(state.kept_embed == 0)
    assert state.q_embed.shape == (D_TEXT,)
    assert state.frame_embed.shape == (D_VIS,)


def test_reset_computes_h0():
    env = _make_env()
    state, _ = _make_episode(env)
    expected_h0 = math.log(5) * 1.0   # MockWrapper: 0 frames → full entropy
    assert abs(state.entropy - expected_h0) < 1e-5


def test_scalars_shape_and_values():
    env = _make_env()
    state, _ = _make_episode(env)
    sc = state.scalars
    assert sc.shape == (3,)
    assert sc[1].item() == pytest.approx(0.0)   # t/N = 0/6
    assert sc[2].item() == pytest.approx(0.0)   # |S|/N = 0/6


# ---------------------------------------------------------------------------
# SKIP tests
# ---------------------------------------------------------------------------

def test_skip_does_not_add_to_kept():
    env = _make_env()
    _, _ = _make_episode(env)
    state, _, _, info = env.step(SKIP)
    assert state.n_kept == 0
    assert info["n_kept"] == 0


def test_skip_advances_step():
    env = _make_env()
    _, _ = _make_episode(env)
    state, _, _, _ = env.step(SKIP)
    assert state.step == 1


def test_skip_kept_embed_stays_zero():
    env = _make_env()
    _, _ = _make_episode(env)
    state, _, _, _ = env.step(SKIP)
    assert torch.all(state.kept_embed == 0)


# ---------------------------------------------------------------------------
# KEEP tests
# ---------------------------------------------------------------------------

def test_keep_increments_kept():
    env = _make_env()
    _, _ = _make_episode(env)
    state, _, _, info = env.step(KEEP)
    assert state.n_kept == 1
    assert info["n_kept"] == 1


def test_keep_updates_kept_embed():
    env = _make_env()
    _, emb = _make_episode(env)
    state, _, _, _ = env.step(KEEP)
    assert torch.allclose(state.kept_embed, emb[0])


def test_keep_updates_entropy():
    env = _make_env()
    state0, _ = _make_episode(env)
    state1, _, _, _ = env.step(KEEP)
    # MockWrapper: entropy decreases as frames are added
    assert state1.entropy < state0.entropy


def test_two_keeps_mean_pool():
    env = _make_env()
    _, emb = _make_episode(env)
    env.step(KEEP)   # keep frame 0
    env.step(SKIP)
    state, _, _, _ = env.step(KEEP)  # keep frame 2
    expected = (emb[0] + emb[2]) / 2
    assert torch.allclose(state.kept_embed, expected)


# ---------------------------------------------------------------------------
# STOP tests
# ---------------------------------------------------------------------------

def test_stop_terminates_episode():
    env = _make_env()
    _, _ = _make_episode(env)
    _, _, done, _ = env.step(STOP)
    assert done


def test_stop_does_not_add_frame():
    env = _make_env()
    _, _ = _make_episode(env)
    _, _, _, info = env.step(STOP)
    assert info["n_kept"] == 0


def test_step_after_done_raises():
    env = _make_env()
    _, _ = _make_episode(env)
    env.step(STOP)
    with pytest.raises(AssertionError):
        env.step(SKIP)


# ---------------------------------------------------------------------------
# Force-stop at N-1
# ---------------------------------------------------------------------------

def test_force_stop_at_last_frame():
    env = _make_env(force_stop_at_end=True)
    _, _ = _make_episode(env)
    done = False
    for _ in range(N_FRAMES):
        _, _, done, _ = env.step(SKIP)
    assert done


def test_no_force_stop_without_flag():
    env = _make_env(force_stop_at_end=False)
    _, _ = _make_episode(env)
    done = False
    for _ in range(N_FRAMES):
        _, _, done, _ = env.step(SKIP)
        if done:
            break
    assert not done


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

def test_nonterminal_reward_is_zero():
    env = _make_env()
    _, _ = _make_episode(env)
    _, reward, _, _ = env.step(SKIP)
    assert reward == pytest.approx(0.0)


def test_terminal_correct_reward():
    # MockWrapper: pred_idx = len(frames) % 5
    # With 0 frames kept, pred_idx = 0 → correct if answer_idx = 0
    env = _make_env(force_stop_at_end=False, lambda_cost=LAMBDA)
    _, _ = _make_episode(env, answer_idx=0)
    _, reward, done, _ = env.step(STOP)
    assert done
    # kept=0, so retention=0 → reward = 1.0 - 0.1*0.0 = 1.0
    assert reward == pytest.approx(1.0)


def test_terminal_incorrect_reward():
    # With 0 frames, pred_idx=0 → incorrect when answer_idx=2
    env = _make_env(force_stop_at_end=False, lambda_cost=LAMBDA)
    _, _ = _make_episode(env, answer_idx=2)
    _, reward, done, _ = env.step(STOP)
    assert done
    assert reward == pytest.approx(0.0)   # 0 - 0.1*0 = 0.0


def test_terminal_reward_subtracts_retention():
    # Keep 1 frame (pred_idx=1 % 5 = 1 → incorrect for answer_idx=0)
    env = _make_env(force_stop_at_end=False, lambda_cost=LAMBDA)
    _, _ = _make_episode(env, answer_idx=0)
    env.step(KEEP)   # keep frame 0 → pred_idx=1, incorrect
    _, reward, done, _ = env.step(STOP)
    assert done
    # 0 - 0.1*(1/6) = -0.01666...
    assert reward == pytest.approx(-LAMBDA * (1 / N_FRAMES), abs=1e-5)


def test_step_penalty_on_keep():
    env = _make_env(force_stop_at_end=False, step_penalty=0.05)
    _, _ = _make_episode(env)
    _, reward, done, _ = env.step(KEEP)
    assert not done
    assert reward == pytest.approx(-0.05)


def test_step_penalty_not_on_skip():
    env = _make_env(step_penalty=0.05)
    _, _ = _make_episode(env)
    _, reward, _, _ = env.step(SKIP)
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_reset_clears_previous_episode():
    env = _make_env()
    _, _ = _make_episode(env)
    env.step(KEEP)
    env.step(KEEP)
    # Second episode — state should be fresh
    state, _ = _make_episode(env)
    assert state.n_kept == 0
    assert state.step == 0


def test_single_frame_video():
    env = _make_env(force_stop_at_end=True)
    sample = _make_sample(answer_idx=0)
    embeddings = torch.randn(1, D_VIS)
    q_embed = torch.randn(D_TEXT)
    state = env.reset(sample, embeddings, [None], q_embed)
    assert state.n_frames == 1
    _, _, done, _ = env.step(SKIP)
    assert done
