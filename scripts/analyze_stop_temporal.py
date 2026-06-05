"""Analyze STOP action firing and temporal question frame behavior."""
import json
import numpy as np
from collections import defaultdict

PPO_FILE = "results/ppo_lambda005_ep001024_n200.json"
MAX_FRAMES = 32  # config max_frames

ppo = json.load(open(PPO_FILE))["predictions"]

# ── 1. Temporal accuracy by frame count bucket ────────────────────────────────
print("=" * 60)
print("1. TEMPORAL: ACCURACY BY FRAMES KEPT")
print("=" * 60)

temporal = [p for p in ppo if p["qtype_group"] == "temporal"]
buckets = {"0-4": [], "5-8": [], "9-16": [], "17-32": []}
for p in temporal:
    n = p["n_frames"]
    if n <= 4:    buckets["0-4"].append(p)
    elif n <= 8:  buckets["5-8"].append(p)
    elif n <= 16: buckets["9-16"].append(p)
    else:         buckets["17-32"].append(p)

for bucket, items in buckets.items():
    if not items:
        continue
    acc = np.mean([p["correct"] for p in items])
    print(f"  {bucket:6s} frames (n={len(items):3d}): acc={acc:.3f}")

# Do same for all types side by side
print()
print("  All types, 0-4 frame bucket accuracy:")
for g in ["causal", "temporal", "descriptive"]:
    low = [p for p in ppo if p["qtype_group"] == g and p["n_frames"] <= 4]
    high = [p for p in ppo if p["qtype_group"] == g and p["n_frames"] > 4]
    if low:
        print(f"    {g:12s}: 0-4 frames (n={len(low):2d}) acc={np.mean([p['correct'] for p in low]):.3f}  |  "
              f">4 frames (n={len(high):2d}) acc={np.mean([p['correct'] for p in high]):.3f}")

# ── 2. Inferring STOP vs SKIP-all ────────────────────────────────────────────
print()
print("=" * 60)
print("2. INFERRING STOP ACTION FROM kept_indices")
print("=" * 60)
print("  Logic: if max(kept_indices) < MAX_FRAMES-4, STOP likely fired early.")
print("  If kept_indices is empty, STOP fired before any KEEP OR policy SKIPped all.")
print()

# STOP inference: if the last kept frame index is well below the end of the video
# We use a threshold: if max_kept < MAX_FRAMES * 0.75, call it an early stop
STOP_THRESHOLD = MAX_FRAMES * 0.75  # index 24 out of 32

stop_inferred = []
skip_all = []
normal = []

for p in ppo:
    ki = p["kept_indices"]
    if len(ki) == 0:
        skip_all.append(p)  # could be STOP at step 0 or SKIP all
    elif max(ki) < STOP_THRESHOLD:
        stop_inferred.append(p)
    else:
        normal.append(p)

print(f"  Inferred STOP early (max frame < {STOP_THRESHOLD:.0f}): {len(stop_inferred)} episodes")
print(f"  Zero frames kept (STOP@0 or SKIP-all):               {len(skip_all)} episodes")
print(f"  Ran to near end of video:                            {len(normal)} episodes")

print()
print("  STOP-inferred episodes by type:")
by_type = defaultdict(list)
for p in stop_inferred:
    by_type[p["qtype_group"]].append(p)
for g in ["causal", "temporal", "descriptive"]:
    items = by_type[g]
    if items:
        acc = np.mean([p["correct"] for p in items])
        avg_f = np.mean([p["n_frames"] for p in items])
        avg_max = np.mean([max(p["kept_indices"]) for p in items])
        print(f"    {g:12s}: n={len(items)}  acc={acc:.3f}  avg_frames={avg_f:.1f}  "
              f"avg_last_frame_idx={avg_max:.1f}")

print()
print("  Normal (ran to end) episodes by type:")
by_type_n = defaultdict(list)
for p in normal:
    by_type_n[p["qtype_group"]].append(p)
for g in ["causal", "temporal", "descriptive"]:
    items = by_type_n[g]
    if items:
        acc = np.mean([p["correct"] for p in items])
        avg_f = np.mean([p["n_frames"] for p in items])
        print(f"    {g:12s}: n={len(items)}  acc={acc:.3f}  avg_frames={avg_f:.1f}")

# ── 3. Where does early stop fire? ───────────────────────────────────────────
print()
print("=" * 60)
print("3. WHEN IN THE VIDEO DOES STOP FIRE? (inferred early stops)")
print("=" * 60)

if stop_inferred:
    stop_positions = [max(p["kept_indices"]) / MAX_FRAMES for p in stop_inferred]
    print(f"  Normalized stop position (0=early, 1=end of video):")
    print(f"    mean={np.mean(stop_positions):.2f}  "
          f"median={np.median(stop_positions):.2f}  "
          f"std={np.std(stop_positions):.2f}")

    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    counts, _ = np.histogram(stop_positions, bins=bins)
    for i, label in enumerate(labels):
        bar = "#" * counts[i]
        print(f"    {label}: {counts[i]:3d}  {bar}")

# ── 4. Temporal wins with few frames ─────────────────────────────────────────
print()
print("=" * 60)
print("4. TEMPORAL CORRECT WITH <=8 FRAMES: what's the pattern?")
print("=" * 60)

temporal_few_correct = [p for p in temporal if p["n_frames"] <= 8 and p["correct"]]
temporal_few_wrong   = [p for p in temporal if p["n_frames"] <= 8 and not p["correct"]]

print(f"  Correct with <=8 frames: {len(temporal_few_correct)}")
print(f"  Wrong   with <=8 frames: {len(temporal_few_wrong)}")
if temporal_few_correct:
    avg_max_idx = np.mean([max(p["kept_indices"]) if p["kept_indices"] else 0
                           for p in temporal_few_correct])
    avg_spread  = []
    for p in temporal_few_correct:
        ki = p["kept_indices"]
        if len(ki) > 1:
            avg_spread.append(np.std([i / MAX_FRAMES for i in ki]))
    print(f"  Correct cases avg last-frame-index: {avg_max_idx:.1f} / {MAX_FRAMES}")
    if avg_spread:
        print(f"  Correct cases avg frame spread: {np.mean(avg_spread):.3f}")

print()
print("  Sample kept_indices for temporal CORRECT <=8 frame cases:")
for p in temporal_few_correct[:8]:
    print(f"    qtype={p['qtype']:3s} kept={p['kept_indices']}  n={p['n_frames']}")
