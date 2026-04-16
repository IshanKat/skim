"""Sanity-check that data/nextqa/ is laid out correctly.

Usage:
    python scripts/check_nextqa.py [--root data/nextqa] [--split val]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from afs.data.nextqa import NExTQADataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/nextqa")
    p.add_argument("--split", default="val")
    p.add_argument("--show", type=int, default=3, help="print first N samples")
    args = p.parse_args()

    try:
        ds = NExTQADataset(args.root, split=args.split)
    except FileNotFoundError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print(f"[OK] Loaded {len(ds)} samples from {args.root} ({args.split})")

    group_counts: Counter[str] = Counter()
    missing_videos = 0
    for s in ds:
        group_counts[s.qtype_group] += 1
        if not s.video_path.exists():
            missing_videos += 1

    print(f"[INFO] Question-type groups: {dict(group_counts)}")
    if missing_videos:
        print(f"[WARN] {missing_videos}/{len(ds)} videos missing on disk")
    else:
        print("[OK] All video files present on disk")

    for i in range(min(args.show, len(ds))):
        s = ds[i]
        print(f"--- sample {i} ---")
        print(f"  qid:      {s.qid}")
        print(f"  video_id: {s.video_id}  path={s.video_path}")
        print(f"  qtype:    {s.qtype} ({s.qtype_group})")
        print(f"  question: {s.question}")
        for j, c in enumerate(s.choices):
            marker = " <-- answer" if j == s.answer_idx else ""
            print(f"    [{j}] {c}{marker}")

    return 0 if missing_videos == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
