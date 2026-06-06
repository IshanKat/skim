"""Run full-set evals on NExT-QA and IntentQA sequentially."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

PYTHON = sys.executable

EVALS = [
    [PYTHON, "-u", "scripts/eval_ppo.py",
     "--config", "configs/lambda005.yaml",
     "--checkpoint", "checkpoints/ppo_lambda005/ppo_ep001024.pt",
     "--dataset", "nextqa",
     "--limit", "1000",
     "--output", "results/ppo_lambda005_ep1024_nextqa_n1000.json"],

    [PYTHON, "-u", "scripts/eval_ppo.py",
     "--config", "configs/lambda005.yaml",
     "--checkpoint", "checkpoints/ppo_lambda005/ppo_ep001024.pt",
     "--dataset", "intentqa", "--data-root", "data/intentqa",
     "--limit", "1000",
     "--output", "results/ppo_lambda005_ep1024_intentqa_n1000.json"],
]

for cmd in EVALS:
    print(f"\n[run_full_evals] Starting: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[run_full_evals] WARNING: exited with code {result.returncode}", flush=True)

print("\n[run_full_evals] All done.")
