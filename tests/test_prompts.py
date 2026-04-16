from __future__ import annotations

import pytest

from afs.vlm.prompts import LETTERS, nextqa_mc_prompt


def test_prompt_shape():
    p = nextqa_mc_prompt("why is the child laughing?", ["a", "b", "c", "d", "e"])
    assert "Question: why is the child laughing?" in p
    for letter in LETTERS:
        assert f"  {letter}." in p
    assert p.rstrip().endswith("Answer:")


def test_prompt_requires_five_choices():
    with pytest.raises(ValueError):
        nextqa_mc_prompt("q", ["a", "b", "c"])


def test_prompt_strips_whitespace():
    p = nextqa_mc_prompt("  q  ", ["  a", "b ", "c", " d ", "e"])
    assert "Question: q\n" in p
    assert "  A. a\n" in p
    assert "  B. b\n" in p
