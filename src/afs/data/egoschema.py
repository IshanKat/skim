"""EgoSchema dataset loader.

EgoSchema is a long-form egocentric video QA benchmark (Ego4D clips, ~3 min each).
Each question has its own video clip named by q_uid.

Expected annotation file format (JSON):
    [
      {
        "q_uid": "some-uuid",
        "video_uid": "ego4d-video-uid",   # used as video_id for cache lookup
        "question": "...",
        "option 0": "...",
        "option 1": "...",
        "option 2": "...",
        "option 3": "...",
        "option 4": "...",
        "answer": 2
      },
      ...
    ]

Videos are named by q_uid: {q_uid}.mp4

Download from: https://github.com/egoschema/EgoSchema
  - Full set (5000 QA): annotations/questions.json  (no answers)
  - Subset (500 QA):    annotations/subset_answers.json  (with answers)
Use the 500-sample subset for zero-shot transfer eval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class EgoSchemaSample:
    qid: str
    video_id: str
    video_path: Path
    question: str
    choices: tuple[str, ...]   # always 5 elements
    answer_idx: int
    qtype: str
    qtype_group: str


class EgoSchemaDataset:
    """Iterable / indexable EgoSchema dataset.

    Parameters
    ----------
    root:
        Directory containing the JSON annotation file and videos subdirectory.
    json_file:
        Annotation JSON filename relative to root.
        Defaults to ``"subset_answers.json"`` (the 500-sample annotated subset).
    video_dir:
        Subdirectory under root containing {q_uid}.mp4 files.
    """

    def __init__(
        self,
        root: str | Path,
        json_file: str = "subset_answers.json",
        *,
        video_dir: str = "videos",
    ) -> None:
        self._root = Path(root)
        ann_path = self._root / json_file
        self._video_dir = self._root / video_dir

        if not ann_path.exists():
            raise FileNotFoundError(f"EgoSchema annotation not found: {ann_path}")

        with open(ann_path) as f:
            entries = json.load(f)

        self._samples: list[EgoSchemaSample] = []
        for row in entries:
            q_uid = str(row["q_uid"])
            video_path = self._video_dir / f"{q_uid}.mp4"
            choices = tuple(str(row[f"option {i}"]) for i in range(5))
            self._samples.append(EgoSchemaSample(
                qid=q_uid,
                video_id=q_uid,
                video_path=video_path,
                question=str(row["question"]),
                choices=choices,
                answer_idx=int(row["answer"]),
                qtype="ego",
                qtype_group="all",
            ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EgoSchemaSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[EgoSchemaSample]:
        return iter(self._samples)
