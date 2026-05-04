"""Selector policy: 2-layer transformer mapping per-step state → action logits + value.

At each step t the policy observes four things:
  - The query embedding (what question are we answering?)
  - The current frame embedding (what does this frame look like?)
  - The mean of all kept frame embeddings (what have we kept so far?)
  - Two scalars: [t/N, |S|/N]

These become four tokens fed to a small transformer.  The output (mean-pooled)
drives a 3-way action head (keep/skip/stop) and a scalar value head for PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectorPolicy(nn.Module):
    """Adaptive frame selector with a transformer backbone.

    Parameters
    ----------
    d_vis:
        Dimension of per-frame embeddings from the VLM vision encoder.
    d_text:
        Dimension of query embeddings from the VLM language model.
    d_model:
        Internal transformer width.
    n_heads:
        Attention heads (must divide d_model).
    n_layers:
        Number of transformer encoder layers.
    dropout:
        Dropout rate applied inside the transformer.
    n_actions:
        Size of the action space (3: keep / skip / stop).
    """

    def __init__(
        self,
        d_vis: int,
        d_text: int,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projections — each source gets its own linear
        self.text_proj = nn.Linear(d_text, d_model)
        self.vis_proj = nn.Linear(d_vis, d_model)   # shared for frame + kept
        self.scalar_proj = nn.Linear(2, d_model)    # [t/N, |S|/N]

        # Learned zero-token for the kept-set when no frames have been kept yet
        self.empty_kept_token = nn.Parameter(torch.zeros(d_model))

        # Per-position bias for the 4-token sequence
        self.pos_embed = nn.Parameter(torch.randn(4, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.action_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        q_embed: torch.Tensor,      # [B, D_text]
        frame_embed: torch.Tensor,  # [B, D_vis]
        kept_embed: torch.Tensor,   # [B, D_vis]  (zeros when kept-set is empty)
        scalars: torch.Tensor,      # [B, 2]
        kept_empty: torch.Tensor | None = None,  # [B] bool — True when no frames kept
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        action_logits : [B, n_actions]
        value         : [B]
        """
        B = q_embed.shape[0]

        t0 = self.text_proj(q_embed)     # [B, d_model]
        t1 = self.vis_proj(frame_embed)  # [B, d_model]
        t2 = self.vis_proj(kept_embed)   # [B, d_model]
        t3 = self.scalar_proj(scalars)   # [B, d_model]

        # Replace kept-mean token with the learned zero-token when kept-set is empty
        if kept_empty is not None:
            empty = self.empty_kept_token.unsqueeze(0).expand(B, -1)
            t2 = torch.where(kept_empty.unsqueeze(-1), empty, t2)

        seq = torch.stack([t0, t1, t2, t3], dim=1)  # [B, 4, d_model]
        seq = seq + self.pos_embed.unsqueeze(0)

        out = self.transformer(seq)          # [B, 4, d_model]
        pooled = out.mean(dim=1)             # [B, d_model]

        action_logits = self.action_head(pooled)          # [B, n_actions]
        value = self.value_head(pooled).squeeze(-1)       # [B]

        return action_logits, value

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, d_vis: int, d_text: int) -> "SelectorPolicy":
        """Construct from the ``selector`` sub-dict of default.yaml."""
        return cls(
            d_vis=d_vis,
            d_text=d_text,
            d_model=cfg.get("d_model", 512),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            n_actions=cfg.get("action_dim", 3),
        )
