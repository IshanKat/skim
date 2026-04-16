from __future__ import annotations

from typing import Sequence

LETTERS = ("A", "B", "C", "D", "E")


def nextqa_mc_prompt(question: str, choices: Sequence[str]) -> str:
    """NExT-QA multiple-choice prompt.

    Qwen2.5-VL is steered to emit a single letter so we can read off the MC
    prediction from the first generated token and compute entropy over just
    the 5 letter logits.
    """
    if len(choices) != 5:
        raise ValueError(f"NExT-QA expects 5 choices, got {len(choices)}")
    lines = [f"Question: {question.strip()}", "Options:"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"  {letter}. {choice.strip()}")
    lines.append("Answer with a single letter (A, B, C, D, or E).")
    lines.append("Answer:")
    return "\n".join(lines)
