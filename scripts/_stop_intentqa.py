import json, numpy as np

preds = json.load(open("results/ppo_lambda005_ep1024_intentqa.json"))["predictions"]
MAX_FRAMES = 32
STOP_THRESH = 24

stop = [p for p in preds if p["kept_indices"] and max(p["kept_indices"]) < STOP_THRESH]
run  = [p for p in preds if not p["kept_indices"] or max(p["kept_indices"]) >= STOP_THRESH]
zero = [p for p in preds if not p["kept_indices"]]

print(f"Total: {len(preds)}")
print(f"STOP fires early: {len(stop)} ({len(stop)/len(preds):.1%})  acc={np.mean([p['correct'] for p in stop]):.3f}  avg_f={np.mean([p['n_frames'] for p in stop]):.1f}")
print(f"Runs to end:      {len(run)}  ({len(run)/len(preds):.1%})  acc={np.mean([p['correct'] for p in run]):.3f}  avg_f={np.mean([p['n_frames'] for p in run]):.1f}")
print(f"Zero frames:      {len(zero)} ({len(zero)/len(preds):.1%})")

print()
print("Compare to NExT-QA (same checkpoint):")
preds_nq = json.load(open("results/ppo_lambda005_ep001024_n200.json"))["predictions"]
stop_nq = [p for p in preds_nq if p["kept_indices"] and max(p["kept_indices"]) < STOP_THRESH]
run_nq  = [p for p in preds_nq if not p["kept_indices"] or max(p["kept_indices"]) >= STOP_THRESH]
print(f"NExT-QA STOP fires early: {len(stop_nq)/len(preds_nq):.1%}  avg_f={np.mean([p['n_frames'] for p in stop_nq]):.1f}")
print(f"NExT-QA Runs to end:      {len(run_nq)/len(preds_nq):.1%}  avg_f={np.mean([p['n_frames'] for p in run_nq]):.1f}")

print()
print("By type (IntentQA):")
for g in ["causal", "temporal"]:
    items = [p for p in preds if p["qtype_group"] == g]
    s = [p for p in items if p["kept_indices"] and max(p["kept_indices"]) < STOP_THRESH]
    r = [p for p in items if not p["kept_indices"] or max(p["kept_indices"]) >= STOP_THRESH]
    s_acc = np.mean([p["correct"] for p in s]) if s else float("nan")
    r_acc = np.mean([p["correct"] for p in r]) if r else float("nan")
    s_f   = np.mean([p["n_frames"] for p in s]) if s else float("nan")
    r_f   = np.mean([p["n_frames"] for p in r]) if r else float("nan")
    print(f"  {g:12s}: stop={len(s)/len(items):.1%} acc={s_acc:.3f} avg_f={s_f:.1f}  |  run={len(r)/len(items):.1%} acc={r_acc:.3f} avg_f={r_f:.1f}")
