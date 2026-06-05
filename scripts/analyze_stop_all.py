"""Per-type STOP analysis across all key checkpoints."""
import json
import numpy as np

FILES = [
    ("lam0.1  ep1024", "results/ppo_lambda01_ep001024_n200.json"),
    ("lam0.1  ep2528", "results/ppo_lambda01_ep002528_n200.json"),
    ("lam0.05 ep1024", "results/ppo_lambda005_ep001024_n200.json"),
    ("lam0.05 final",  "results/ppo_lambda005_final_n200.json"),
]
MAX_FRAMES  = 32
STOP_THRESH = 24  # max kept index < 24 => inferred early stop

print(f"{'checkpoint':18s}  {'type':12s}  {'acc':>5}  {'avg_f':>5}  {'stop%':>5}  {'--- STOP fires ---':>18}  {'--- runs to end ---':>19}")
print(f"{'':18s}  {'':12s}  {'':>5}  {'':>5}  {'':>5}  {'acc':>5} {'avg_f':>6}       {'acc':>5} {'avg_f':>6}")
print("-" * 85)

for label, fpath in FILES:
    preds = json.load(open(fpath))["predictions"]
    for g in ["causal", "temporal", "descriptive"]:
        items = [p for p in preds if p["qtype_group"] == g]
        if not items:
            continue
        acc   = np.mean([p["correct"] for p in items])
        avg_f = np.mean([p["n_frames"] for p in items])

        stop = [p for p in items if p["kept_indices"] and max(p["kept_indices"]) < STOP_THRESH]
        run  = [p for p in items if not p["kept_indices"] or max(p["kept_indices"]) >= STOP_THRESH]

        stop_pct  = len(stop) / len(items)
        stop_acc  = np.mean([p["correct"] for p in stop]) if stop else float("nan")
        run_acc   = np.mean([p["correct"] for p in run])  if run  else float("nan")
        stop_avgf = np.mean([p["n_frames"] for p in stop]) if stop else float("nan")
        run_avgf  = np.mean([p["n_frames"] for p in run])  if run  else float("nan")

        print(f"{label:18s}  {g:12s}  {acc:.3f}  {avg_f:5.1f}  {stop_pct:5.0%}  "
              f"{stop_acc:5.3f}({stop_avgf:4.1f}f)  {run_acc:5.3f}({run_avgf:4.1f}f)")
    print()
