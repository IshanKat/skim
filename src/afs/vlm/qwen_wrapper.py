"""Frozen Qwen2.5-VL wrapper.

Responsibilities
---------------
- Load Qwen2.5-VL (default 3B, 4-bit) once; keep it frozen.
- ``encode_frames_per_frame(frames)`` → [T, D_vis] for selector state.
- ``encode_query(text)`` → [D_text] for selector state.
- ``answer_mc(frames, question, choices)`` → ``MCPrediction`` (pred + entropy H).

This module requires ``torch`` + ``transformers`` and a CUDA GPU in practice.
It is imported lazily so unit tests that don't exercise the wrapper can run
without those heavy dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image

from .entropy import MCPrediction, mc_pred_and_entropy
from .prompts import LETTERS, nextqa_mc_prompt


@dataclass
class QwenVLConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"    # bf16 | fp16 | fp32
    load_in_4bit: bool = True
    max_pixels: int | None = None    # clamp dynamic resolution if OOM
    fps: float = 1.0                 # used only for video inputs


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name]


class QwenVLWrapper:
    def __init__(self, config: QwenVLConfig) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = config
        dtype = _resolve_dtype(config.torch_dtype)

        load_kwargs: dict = {"torch_dtype": dtype, "device_map": "auto"}
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        processor_kwargs: dict = {}
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels

        self.processor = AutoProcessor.from_pretrained(config.model_name, **processor_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name, **load_kwargs
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._letter_ids = self._compute_letter_token_ids()

    @property
    def model_tag(self) -> str:
        return self.cfg.model_name

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def d_text(self) -> int:
        """Dimension of embeddings returned by encode_query."""
        return self.model.config.text_config.hidden_size

    @property
    def d_vis(self) -> int:
        """Dimension of embeddings returned by encode_frames_per_frame.

        We extract from the vision encoder's last_hidden_state before the LM-space
        projection, so d_vis == vision_config.hidden_size (1280 for Qwen2.5-VL-3B).
        """
        return self.model.config.vision_config.hidden_size

    def _compute_letter_token_ids(self) -> list[int]:
        tok = self.processor.tokenizer
        ids: list[int] = []
        for letter in LETTERS:
            enc = tok.encode(letter, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(
                    f"Letter {letter!r} tokenizes to multiple tokens {enc}; "
                    "pick a different answer format."
                )
            ids.append(enc[0])
        return ids

    @torch.no_grad()
    def encode_frames_per_frame(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        """Run the vision tower once per frame and mean-pool spatial tokens.

        Returns a ``[T, D_vis]`` tensor on CPU (caller can move to GPU).
        Loops per frame for simplicity; for long videos use batched extraction.
        """
        if len(frames) == 0:
            raise ValueError("empty frame list")

        outputs: list[torch.Tensor] = []
        for frame in frames:
            img_inputs = self.processor.image_processor(images=[frame], return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self.device)
            grid_thw = img_inputs["image_grid_thw"].to(self.device)
            out = self.model.model.visual(pixel_values, grid_thw=grid_thw)
            hidden = out if isinstance(out, torch.Tensor) else out.last_hidden_state
            pooled = hidden.mean(dim=0)
            outputs.append(pooled.float().cpu())
        return torch.stack(outputs, dim=0)

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """Token-embed the query and mean-pool → ``[D_text]`` on CPU."""
        tok = self.processor.tokenizer
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            self.device
        )
        embed_layer = self.model.get_input_embeddings()
        emb = embed_layer(ids)              # [1, L, D_text]
        pooled = emb.mean(dim=1).squeeze(0) # [D_text]
        return pooled.float().cpu()

    @torch.no_grad()
    def answer_mc(
        self,
        frames: Sequence[Image.Image],
        question: str,
        choices: Sequence[str],
    ) -> MCPrediction:
        """Full VLM forward → next-token logits at the answer position →
        (pred_idx, entropy) over the 5 MC letters.
        """
        prompt = nextqa_mc_prompt(question, choices)

        if len(frames) == 0:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            video_inputs = None
        else:
            video_content = {"type": "video", "video": list(frames), "fps": self.cfg.fps}
            messages = [{
                "role": "user",
                "content": [video_content, {"type": "text", "text": prompt}],
            }]
            from qwen_vl_utils import process_vision_info
            _, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs, use_cache=False)
        next_token_logits = out.logits[0, -1, :]
        return mc_pred_and_entropy(next_token_logits, self._letter_ids)
