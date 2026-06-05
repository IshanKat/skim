"""Wait for precompute to finish, then run all IntentQA evals sequentially.

Usage:
    python scripts/run_intentqa_evals.py --precompute-pid <PID>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(Path(sys.executable))

EVALS = [
    # (script, extra_args)
    ("scripts/eval_full_frame.py", [
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--max-frames", "8", "--limit", "200",
        "--output", "results/uniform8_intentqa.json",
    ]),
    ("scripts/eval_full_frame.py", [
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--max-frames", "16", "--limit", "200",
        "--output", "results/uniform16_intentqa.json",
    ]),
    ("scripts/eval_full_frame.py", [
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--max-frames", "32", "--limit", "200",
        "--output", "results/uniform32_intentqa.json",
    ]),
    ("scripts/eval_ppo.py", [
        "--config", "configs/lambda005.yaml",
        "--checkpoint", "checkpoints/ppo_lambda005/ppo_ep001024.pt",
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--limit", "200",
        "--output", "results/ppo_lambda005_ep1024_intentqa.json",
    ]),
    ("scripts/eval_ppo.py", [
        "--config", "configs/lambda005.yaml",
        "--checkpoint", "checkpoints/ppo_lambda005/ppo_final.pt",
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--limit", "200",
        "--output", "results/ppo_lambda005_final_intentqa_n200.json",
    ]),
    ("scripts/eval_ppo.py", [
        "--config", "configs/default.yaml",
        "--checkpoint", "checkpoints/ppo_lambda01/ppo_ep001024.pt",
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--limit", "200",
        "--output", "results/ppo_lambda01_ep1024_intentqa.json",
    ]),
    ("scripts/eval_ppo.py", [
        "--config", "configs/default.yaml",
        "--checkpoint", "checkpoints/ppo_lambda01/ppo_ep002528.pt",
        "--dataset", "intentqa", "--data-root", "data/intentqa",
        "--limit", "200",
        "--output", "results/ppo_lambda01_ep2528_intentqa.json",
    ]),
]


def pid_alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).is_running()
    except Exception:
        return False


def wait_for_pid(pid: int) -> None:
    print(f"[run_evals] Waiting for precompute PID {pid} to finish...")
    while pid_alive(pid):
        time.sleep(60)
    print("[run_evals] Precompute done.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--precompute-pid", type=int, default=None)
    args = p.parse_args()

    if args.precompute_pid:
        wait_for_pid(args.precompute_pid)

    for script, extra_args, *_ in EVALS:
        cmd = [PYTHON, "-u", script] + extra_args
        print(f"\n[run_evals] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[run_evals] WARNING: {script} exited with code {result.returncode}")

    print("\n[run_evals] All IntentQA evals complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
