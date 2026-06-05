"""Analyze frame selection behavior and accuracy by question type.

Compares PPO lambda=0.05 ep1024 vs Uniform-16 to explain why
temporal improves and descriptive degrades.
"""
import json
import numpy as np
from collections import defaultdict

PPO_FILE  = "results/ppo_lambda005_ep001024_n200.json"
UNI16_FILE = "results/full_frame_val_16f.json"

ppo   = json.load(open(PPO_FILE))["predictions"]
uni16 = json.load(open(UNI16_FILE))["predictions"]

# Index uni16 by qid for easy lookup
uni16_by_qid = {p["qid"]: p for p in uni16}

# ── 1. Per-type frame stats ───────────────────────────────────────────────────
print("=" * 60)
print("1. AVG FRAMES KEPT BY QUESTION TYPE (PPO lambda=0.05 ep1024)")
print("=" * 60)

by_type = defaultdict(list)
for p in ppo:
    by_type[p["qtype_group"]].append(p["n_frames"])

for g in ["causal", "temporal", "descriptive"]:
    vals = by_type[g]
    print(f"  {g:12s}: mean={np.mean(vals):.2f}  "
          f"median={np.median(vals):.1f}  "
          f"std={np.std(vals):.2f}  n={len(vals)}")

# ── 2. Frame position spread ──────────────────────────────────────────────────
print()
print("=" * 60)
print("2. FRAME POSITION SPREAD (normalized index = position in video)")
print("=" * 60)

# For each sample, compute spread of kept frame positions (std dev of normalized indices)
# Higher spread = frames drawn from across the whole video
# We need to know N (total frames) - approximate from max kept index
by_type_spread = defaultdict(list)
by_type_positions = defaultdict(list)

for p in ppo:
    ki = p["kept_indices"]
    if not ki:
        continue
    # Approximate N: since eval caps at max_frames=32, use that as upper bound
    # But we can get a better estimate from the max index + some slack
    # Use max(ki)+1 as a lower bound on N
    n_approx = max(ki) + 1  # conservative
    norm_pos = [i / max(ki + [1]) for i in ki]  # normalize to [0,1]
    spread = np.std(norm_pos) if len(norm_pos) > 1 else 0.0
    by_type_spread[p["qtype_group"]].append(spread)
    by_type_positions[p["qtype_group"]].extend(norm_pos)

print()
print("  Temporal spread vs. other types:")
for g in ["causal", "temporal", "descriptive"]:
    spreads = by_type_spread[g]
    if spreads:
        print(f"  {g:12s}: mean_spread={np.mean(spreads):.3f}  "
              f"(0=all same frame, 0.5=max spread)")

print()
print("  Distribution of kept frame positions (early=0, late=1):")
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
for g in ["causal", "temporal", "descriptive"]:
    pos = by_type_positions[g]
    if not pos:
        continue
    counts, _ = np.histogram(pos, bins=bins)
    total = sum(counts)
    bar_str = "  ".join(f"{labels[i]}:{counts[i]/total:.0%}" for i in range(len(labels)))
    print(f"  {g:12s}: {bar_str}")

# ── 3. Accuracy on questions where policy kept few vs many frames ─────────────
print()
print("=" * 60)
print("3. ACCURACY BY FRAME COUNT BUCKET (PPO)")
print("=" * 60)

buckets = {"0-4": [], "5-8": [], "9-16": [], "17-32": []}
for p in ppo:
    n = p["n_frames"]
    if n <= 4:     buckets["0-4"].append(p)
    elif n <= 8:   buckets["5-8"].append(p)
    elif n <= 16:  buckets["9-16"].append(p)
    else:          buckets["17-32"].append(p)

for bucket, items in buckets.items():
    if not items:
        continue
    acc = np.mean([p["correct"] for p in items])
    type_counts = defaultdict(int)
    for p in items: type_counts[p["qtype_group"]] += 1
    print(f"  {bucket:6s} frames (n={len(items):3d}): acc={acc:.3f}  "
          f"[C:{type_counts['causal']} T:{type_counts['temporal']} D:{type_counts['descriptive']}]")

# ── 4. Cases where PPO is right but Uniform-16 is wrong (and vice versa) ──────
print()
print("=" * 60)
print("4. DISAGREEMENT ANALYSIS: PPO correct / Uni-16 wrong  (and vice versa)")
print("=" * 60)

ppo_wins = defaultdict(list)   # PPO right, Uni16 wrong
uni_wins = defaultdict(list)   # Uni16 right, PPO wrong

for p in ppo:
    qid = p["qid"]
    if qid not in uni16_by_qid:
        continue
    u = uni16_by_qid[qid]
    if p["correct"] and not u["correct"]:
        ppo_wins[p["qtype_group"]].append(p)
    elif not p["correct"] and u["correct"]:
        uni_wins[p["qtype_group"]].append(p)

print()
print("  PPO correct, Uni-16 wrong (PPO adds value):")
for g in ["causal", "temporal", "descriptive"]:
    wins = ppo_wins[g]
    if wins:
        avg_f = np.mean([p["n_frames"] for p in wins])
        print(f"    {g:12s}: {len(wins):3d} cases  avg_frames={avg_f:.1f}")
    else:
        print(f"    {g:12s}:   0 cases")

print()
print("  Uni-16 correct, PPO wrong (PPO loses value):")
for g in ["causal", "temporal", "descriptive"]:
    wins = uni_wins[g]
    if wins:
        avg_f = np.mean([p["n_frames"] for p in wins])
        print(f"    {g:12s}: {len(wins):3d} cases  avg_frames={avg_f:.1f}")
    else:
        print(f"    {g:12s}:   0 cases")

# ── 5. Net delta per type ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. SUMMARY: NET GAIN/LOSS vs UNIFORM-16")
print("=" * 60)
for g in ["causal", "temporal", "descriptive"]:
    net = len(ppo_wins[g]) - len(uni_wins[g])
    sign = "+" if net >= 0 else ""
    print(f"  {g:12s}: PPO wins={len(ppo_wins[g])}  PPO losses={len(uni_wins[g])}  "
          f"net={sign}{net}")
