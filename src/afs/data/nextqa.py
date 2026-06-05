"""NExT-QA dataset loader.

Yields (video_path, question, choices, answer_idx, qtype) tuples.

NExT-QA question-type codes and their groups:
  CW / CH           → "causal"
  TN / TC / TP      → "temporal"
  DL / DC / DO / DU → "descriptive"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

# Map first letter of qtype code → group name
_QTYPE_PREFIX_TO_GROUP: dict[str, str] = {
    "C": "causal",
    "T": "temporal",
    "D": "descriptive",
}


def _qtype_group(qtype: str) -> str:
    prefix = qtype[0].upper() if qtype else ""
    group = _QTYPE_PREFIX_TO_GROUP.get(prefix)
    if group is None:
        raise ValueError(f"Unknown qtype prefix {prefix!r} in qtype {qtype!r}")
    return group


@dataclass(frozen=True)
class NExTQASample:
    qid: str
    video_id: str
    video_path: Path
    question: str
    choices: tuple[str, ...]   # always 5 elements
    answer_idx: int
    qtype: str
    qtype_group: str


class NExTQADataset:
    """Iterable / indexable NExT-QA dataset.

    Parameters
    ----------
    root:
        Directory containing the CSV annotation file, the video-id map JSON,
        and the ``videos/`` subdirectory.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.  Used to derive the
        default CSV filename (``{split}.csv``).
    csv_file:
        Override the annotation CSV filename (relative to *root*).
    map_file:
        Name of the JSON file mapping ``video_id → vidor_id``.
        Defaults to ``"map_vid_vidorID.json"``.
    video_dir:
        Subdirectory under *root* that holds the ``.mp4`` files.
        Defaults to ``"videos"``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        *,
        csv_file: str | None = None,
        map_file: str = "map_vid_vidorID.json",
        video_dir: str = "videos",
    ) -> None:
        self._root = Path(root)
        csv_path = self._root / (csv_file or f"{split}.csv")
        map_path = self._root / map_file
        self._video_dir = self._root / video_dir

        if not csv_path.exists():
            raise FileNotFoundError(f"NExT-QA annotation CSV not found: {csv_path}")
        if not map_path.exists():
            raise FileNotFoundError(f"NExT-QA video-id map not found: {map_path}")

        df = pd.read_csv(csv_path)
        with open(map_path) as f:
            vid_map: dict[str, str] = json.load(f)

        self._samples: list[NExTQASample] = []
        for _, row in df.iterrows():
            vid_id = str(row["video"])
            vidor_id = vid_map.get(vid_id, vid_id)
            video_path = self._video_dir / f"{vidor_id}.mp4"
            choices = tuple(str(row[f"a{i}"]) for i in range(5))
            qtype = str(row["type"])
            self._samples.append(NExTQASample(
                qid=str(row["qid"]),
                video_id=vid_id,
                video_path=video_path,
                question=str(row["question"]),
                choices=choices,
                answer_idx=int(row["answer"]),
                qtype=qtype,
                qtype_group=_qtype_group(qtype),
            ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> NExTQASample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[NExTQASample]:
        return iter(self._samples)
