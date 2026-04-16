"""Smoke-test that Qwen2.5-VL-3B loads in 4-bit on the current GPU.

Requirements: CUDA GPU + `pip install -e .[gpu]` (pulls bitsandbytes).
This script is a no-op on machines without CUDA.

Usage:
    python scripts/check_vlm_load.py [--model Qwen/Qwen2.5-VL-3B-Instruct]
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--no-4bit", action="store_true", help="load in bf16 instead of 4-bit")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("[SKIP] No CUDA device detected; this script requires a GPU.", file=sys.stderr)
        return 0

    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"[INFO] GPU memory: {free_gb:.1f} / {total_gb:.1f} GB free")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[INFO] Loading {args.model} ({'4-bit' if not args.no_4bit else 'bf16'}) ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.eval()
    dt = time.time() - t0
    print(f"[OK] Loaded in {dt:.1f}s.")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total params: {param_count / 1e9:.2f}B")
    used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
    print(f"[INFO] GPU memory used after load: {used_gb:.2f} GB")

    print("[INFO] Running a text-only forward pass to verify model wiring...")
    inputs = processor(text=["hello"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    print(f"[OK] Generation sample: {decoded!r}")

    _ = processor  # keep reference
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
