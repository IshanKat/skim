"""Offline analysis and figure generation from eval result JSONs.

Produces four figures and a printed results table — no GPU required.

Usage:
    python scripts/analyze_results.py --output-dir results/figures

Figures generated:
  fig1_accuracy_vs_frames.png  -- accuracy-at-budget curve (all methods)
  fig2_retention_by_qtype.png  -- frame retention histogram split by question type
  fig3_accuracy_by_qtype.png   -- per-question-type accuracy bar chart (all methods)
  fig4_stopping_distribution.png -- where PPO stops, by question type
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)

def preds(result: dict) -> list[dict]:
    return result["predictions"]

def accuracy(ps: list[dict]) -> float:
    return sum(p["correct"] for p in ps) / len(ps) if ps else 0.0

def group_by(ps: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list] = defaultdict(list)
    for p in ps:
        out[p[key]].append(p)
    return dict(out)


QTYPE_LABELS = {"causal": "Causal", "temporal": "Temporal", "descriptive": "Descriptive"}
GROUP_ORDER  = ["causal", "temporal", "descriptive"]
COLORS = {
    "uniform_8":  "#4878CF",
    "uniform_16": "#6ACC65",
    "uniform_32": "#D65F5F",
    "ppo":        "#B47CC7",
}
STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}


# ---------------------------------------------------------------------------
# Figure 1 — accuracy vs. average frames (accuracy-at-budget curve)
# ---------------------------------------------------------------------------

def fig1_accuracy_vs_frames(results: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = [
        ("uniform_8",  "Uniform-8"),
        ("uniform_16", "Uniform-16"),
        ("uniform_32", "Uniform-32"),
        ("ppo",        "PPO (ep800)"),
    ]

    for key, label in methods:
        if key not in results:
            continue
        s = results[key]["summary"]
        avg_f = s.get("avg_frames", s.get("avg_frames_per_question", 0))
        acc   = s["accuracy"] * 100
        color = COLORS[key]
        marker = "D" if key == "ppo" else "o"
        ax.scatter(avg_f, acc, color=color, s=80, zorder=5, marker=marker)
        ax.annotate(label, (avg_f, acc),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=8, color=color)

    ax.set_xlabel("Average frames used per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Frame Budget  (NExT-QA val, n=200)")
    ax.set_xlim(0, 35)
    ax.set_ylim(70, 82)
    fig.tight_layout()
    fig.savefig(out / "fig1_accuracy_vs_frames.png")
    plt.close(fig)
    print(f"  Saved fig1_accuracy_vs_frames.png")


# ---------------------------------------------------------------------------
# Figure 2 — PPO retention histogram, split by question type
# ---------------------------------------------------------------------------

def fig2_retention_by_qtype(ppo: dict, out: Path) -> None:
    by_group = group_by(preds(ppo), "qtype_group")
    bins = list(range(0, 34, 2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

    for ax, group in zip(axes, GROUP_ORDER):
        ps = by_group.get(group, [])
        frames = [p["n_frames"] for p in ps]
        ax.hist(frames, bins=bins, color=COLORS["ppo"], edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(frames), color="black", linestyle="--", linewidth=1,
                   label=f"mean={np.mean(frames):.1f}")
        ax.set_title(QTYPE_LABELS.get(group, group))
        ax.set_xlabel("Frames kept")
        ax.set_ylabel("Questions")
        ax.legend(fontsize=8)

    fig.suptitle("PPO Frame Retention Distribution by Question Type  (n=200)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig2_retention_by_qtype.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_retention_by_qtype.png")


# ---------------------------------------------------------------------------
# Figure 3 — per-question-type accuracy, all methods side by side
# ---------------------------------------------------------------------------

def fig3_accuracy_by_qtype(results: dict[str, dict], out: Path) -> None:
    methods = [
        ("uniform_8",  "Uniform-8"),
        ("uniform_16", "Uniform-16"),
        ("uniform_32", "Uniform-32"),
        ("ppo",        "PPO (ep800)"),
    ]
    present = [(k, l) for k, l in methods if k in results]

    x = np.arange(len(GROUP_ORDER))
    width = 0.8 / len(present)
    offsets = np.linspace(-(len(present)-1)/2, (len(present)-1)/2, len(present)) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for (key, label), offset in zip(present, offsets):
        by_group = group_by(preds(results[key]), "qtype_group")
        accs = [accuracy(by_group.get(g, [])) * 100 for g in GROUP_ORDER]
        bars = ax.bar(x + offset, accs, width, label=label,
                      color=COLORS[key], edgecolor="white", linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type  (NExT-QA val, n=200)")
    ax.set_ylim(50, 100)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_accuracy_by_qtype.png")
    plt.close(fig)
    print(f"  Saved fig3_accuracy_by_qtype.png")


# ---------------------------------------------------------------------------
# Figure 4 — PPO stopping-point distribution (first STOP or last frame kept)
# ---------------------------------------------------------------------------

def fig4_stopping_distribution(ppo: dict, out: Path) -> None:
    """Plot the index of the last kept frame (proxy for when the policy stopped)
    as a fraction of video length, split by question type."""
    by_group = group_by(preds(ppo), "qtype_group")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    bins = np.linspace(0, 1, 11)

    for ax, group in zip(axes, GROUP_ORDER):
        ps = by_group.get(group, [])
        # last kept index as fraction of max possible (32 - 1 = 31)
        stop_fracs = []
        for p in ps:
            ki = p.get("kept_indices", [])
            if ki:
                stop_fracs.append(max(ki) / 31.0)
            else:
                stop_fracs.append(0.0)

        ax.hist(stop_fracs, bins=bins, color=COLORS["ppo"],
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(stop_fracs), color="black", linestyle="--",
                   linewidth=1, label=f"mean={np.mean(stop_fracs):.2f}")
        ax.set_title(QTYPE_LABELS.get(group, group))
        ax.set_xlabel("Last kept frame (fraction of video)")
        ax.set_ylabel("Questions")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle("PPO Stopping-Point Distribution by Question Type  (n=200)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig4_stopping_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_stopping_distribution.png")


# ---------------------------------------------------------------------------
# Results table (printed to stdout)
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict]) -> None:
    methods = [
        ("uniform_8",  "Uniform-8"),
        ("uniform_16", "Uniform-16"),
        ("uniform_32", "Uniform-32"),
        ("ppo",        "PPO (ep800, 800eps)"),
    ]

    header = f"{'Method':<25} {'Acc':>6} {'AvgF':>6}  {'Causal':>8} {'Temporal':>9} {'Desc':>7}  n"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))

    for key, label in methods:
        if key not in results:
            continue
        s = results[key]["summary"]
        ps = preds(results[key])
        by_group = group_by(ps, "qtype_group")
        avg_f = s.get("avg_frames", s.get("avg_frames_per_question", 0))
        acc = s["accuracy"] * 100
        causal = accuracy(by_group.get("causal", [])) * 100
        temp   = accuracy(by_group.get("temporal", [])) * 100
        desc   = accuracy(by_group.get("descriptive", [])) * 100
        n = s["n"]
        print(f"{label:<25} {acc:>5.1f}% {avg_f:>5.1f}f  {causal:>7.1f}% {temp:>8.1f}% {desc:>6.1f}%  {n}")

    print("="*len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir",  default="results/figures")
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all available result files
    file_map = {
        "uniform_8":  rdir / "full_frame_val_8f.json",
        "uniform_16": rdir / "full_frame_val_16f.json",
        "uniform_32": rdir / "full_frame_val_32f.json",
        "ppo":        rdir / "ppo_val_ep800_n200.json",
    }

    results: dict[str, dict] = {}
    for key, path in file_map.items():
        if path.exists():
            results[key] = load(path)
            print(f"Loaded {key}: n={results[key]['summary']['n']}")
        else:
            print(f"[SKIP] {path} not found")

    if not results:
        print("No result files found.")
        return 1

    plt.rcParams.update(STYLE)

    print_results_table(results)

    print("Generating figures...")
    fig1_accuracy_vs_frames(results, out)
    if "ppo" in results:
        fig2_retention_by_qtype(results["ppo"], out)
        fig4_stopping_distribution(results["ppo"], out)
    fig3_accuracy_by_qtype(results, out)

    print(f"\nAll figures written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
