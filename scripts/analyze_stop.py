"""Analyze STOP action usage across checkpoints."""
import json

for fname, label in [
    ("results/ppo_ep800_n200.json", "ep800 (lam=0.1)"),
    ("results/ppo_ep2528_n200.json", "ep2528 (lam=0.2)"),
    ("results/ppo_final_n200.json", "final (lam=0.2)"),
]:
    d = json.load(open(fname))
    preds = d["predictions"]
    n = len(preds)

    zero = [p for p in preds if p["n_frames"] == 0]
    one  = [p for p in preds if p["n_frames"] == 1]
    low  = [p for p in preds if p["n_frames"] <= 3]

    zero_correct = sum(1 for p in zero if p["correct"])
    one_correct  = sum(1 for p in one  if p["correct"])
    low_correct  = sum(1 for p in low  if p["correct"])

    buckets = {"1-4": 0, "5-8": 0, "9-16": 0, "17-24": 0, "25-32": 0}
    for p in preds:
        nf = p["n_frames"]
        if nf <= 4:   buckets["1-4"] += 1
        elif nf <= 8:  buckets["5-8"] += 1
        elif nf <= 16: buckets["9-16"] += 1
        elif nf <= 24: buckets["17-24"] += 1
        else:          buckets["25-32"] += 1

    print(f"\n{'='*50}")
    print(f"{label}  (n={n})")
    print(f"  0 frames (STOP immediately): {len(zero)}/{n} = {len(zero)/n:.1%}  "
          f"correct: {zero_correct}/{len(zero)} = "
          f"{zero_correct/len(zero):.1%}" if zero else f"  0 frames: 0/{n}")
    print(f"  1 frame:  {len(one)}/{n} = {len(one)/n:.1%}  "
          f"correct: {one_correct}/{len(one)} = {one_correct/len(one):.1%}" if one else "  1 frame: 0")
    print(f"  <=3 frames: {len(low)}/{n} = {len(low)/n:.1%}  "
          f"correct: {low_correct}/{len(low)} = {low_correct/len(low):.1%}" if low else "  <=3 frames: 0")
    print(f"  Frame bucket distribution:")
    for k, v in buckets.items():
        bar = "#" * v
        print(f"    {k:6s}: {v:4d}  {bar}")
