from __future__ import annotations

import math

import torch

from afs.vlm.entropy import mc_pred_and_entropy


def test_uniform_distribution_has_max_entropy():
    vocab = 100
    logits = torch.zeros(vocab)
    ids = [0, 5, 10, 20, 30]
    out = mc_pred_and_entropy(logits, ids)
    assert out.entropy == pytest_approx(math.log(len(ids)))
    assert torch.allclose(out.probs, torch.full((len(ids),), 1 / len(ids)), atol=1e-6)


def test_peaked_distribution_has_near_zero_entropy():
    logits = torch.full((100,), -1e4)
    logits[7] = 1e4
    ids = [1, 3, 7, 9, 11]
    out = mc_pred_and_entropy(logits, ids)
    assert out.entropy < 1e-4
    assert out.pred_idx == ids.index(7)


def test_argmax_returns_choice_index_not_vocab_index():
    logits = torch.tensor([0.0, 10.0, 0.0, 5.0, 0.0])
    ids = [0, 1, 2, 3, 4]
    out = mc_pred_and_entropy(logits, ids)
    assert out.pred_idx == 1


def test_accepts_batch_dim():
    logits = torch.zeros(1, 50)
    ids = [0, 1, 2, 3, 4]
    out = mc_pred_and_entropy(logits, ids)
    assert out.entropy == pytest_approx(math.log(5))


def pytest_approx(x, abs_=1e-6):
    import pytest
    return pytest.approx(x, abs=abs_)
