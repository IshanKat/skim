"""Per-type accuracy and avg frames across all checkpoints."""
import json, glob, os
import numpy as np

FILES = sorted(glob.glob("results/ppo_*.json"))
GROUPS = ["causal", "temporal", "descriptive"]

header = f"{'checkpoint':35s} {'acc':>5} {'C_acc':>6} {'T_acc':>6} {'D_acc':>6} | {'avg_f':>5} {'C_f':>5} {'T_f':>5} {'D_f':>5}"
print(header)
print("-" * len(header))

for fpath in FILES:
    d = json.load(open(fpath))
    preds = d["predictions"]
    if not preds:
        continue

    by_type = {g: [] for g in GROUPS}
    for p in preds:
        g = p.get("qtype_group", "all")
        if g in by_type:
            by_type[g].append(p)

    overall_acc = np.mean([p["correct"] for p in preds])
    overall_f   = np.mean([p["n_frames"] for p in preds])

    type_acc = {}
    type_f   = {}
    for g in GROUPS:
        items = by_type[g]
        type_acc[g] = np.mean([p["correct"] for p in items]) if items else float("nan")
        type_f[g]   = np.mean([p["n_frames"] for p in items]) if items else float("nan")

    name = os.path.basename(fpath).replace(".json", "")
    print(f"{name:35s} {overall_acc:5.3f} "
          f"{type_acc['causal']:6.3f} {type_acc['temporal']:6.3f} {type_acc['descriptive']:6.3f} | "
          f"{overall_f:5.1f} {type_f['causal']:5.1f} {type_f['temporal']:5.1f} {type_f['descriptive']:5.1f}")

print()
print("Uniform baselines (all types get same frames):")
for fpath in sorted(glob.glob("results/full_frame_*.json")):
    d = json.load(open(fpath))
    s = d["summary"]
    grp = s.get("accuracy_by_group", {})
    name = os.path.basename(fpath).replace(".json", "")
    avg_f = s.get("avg_frames_per_question", s.get("avg_frames", "?"))
    print(f"{name:35s} {s['accuracy']:5.3f} "
          f"{grp.get('causal', float('nan')):6.3f} "
          f"{grp.get('temporal', float('nan')):6.3f} "
          f"{grp.get('descriptive', float('nan')):6.3f} | "
          f"{avg_f:>5}  (same for all types)")
