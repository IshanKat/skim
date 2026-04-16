from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MCPrediction:
    pred_idx: int       # argmax over the MC choice set
    entropy: float      # Shannon entropy over the choice logits (nats)
    probs: torch.Tensor # shape [num_choices]


def mc_pred_and_entropy(
    next_token_logits: torch.Tensor,
    choice_token_ids: list[int],
) -> MCPrediction:
    """Given the next-token logits at the answer position and the token ids
    corresponding to each MC letter, return the argmax choice index plus the
    entropy of the renormalized distribution over just those choices.

    ``next_token_logits``: shape [vocab_size] (single-example) or [1, vocab_size].
    """
    logits = next_token_logits.squeeze(0) if next_token_logits.dim() == 2 else next_token_logits
    if logits.dim() != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    ids = torch.as_tensor(choice_token_ids, dtype=torch.long, device=logits.device)
    choice_logits = logits.index_select(0, ids)
    probs = F.softmax(choice_logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    pred_idx = int(torch.argmax(probs).item())
    return MCPrediction(pred_idx=pred_idx, entropy=entropy, probs=probs.detach().cpu())
