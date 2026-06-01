"""IntentQA dataset loader.

IntentQA shares videos with NExT-QA (both use VidOR), so you can point
video_dir and map_file at the same paths as NExT-QA.

Expected annotation file format (JSON):
    [
      {
        "video_name": "7678",
        "qid": 1,
        "type": "CW",
        "question": "Why did ...",
        "a0": "...", "a1": "...", "a2": "...", "a3": "...", "a4": "...",
        "answer": 2
      },
      ...
    ]

Download annotations from: https://github.com/JoseponLee/IntentQA
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_QTYPE_PREFIX_TO_GROUP: dict[str, str] = {
    "C": "causal",
    "T": "temporal",
    "D": "descriptive",
}


def _qtype_group(qtype: str) -> str:
    prefix = qtype[0].upper() if qtype else ""
    return _QTYPE_PREFIX_TO_GROUP.get(prefix, "other")


@dataclass(frozen=True)
class IntentQASample:
    qid: str
    video_id: str
    video_path: Path
    question: str
    choices: tuple[str, ...]   # always 5 elements
    answer_idx: int
    qtype: str
    qtype_group: str


class IntentQADataset:
    """Iterable / indexable IntentQA dataset.

    Parameters
    ----------
    root:
        Directory containing the JSON annotation file, the video-id map JSON,
        and the videos subdirectory (same layout as NExT-QA).
    json_file:
        Annotation JSON filename relative to root. Defaults to ``"val.json"``.
    map_file:
        JSON mapping video_id -> vidor_id. Same file as NExT-QA.
        Defaults to ``"map_vid_vidorID.json"``.
    video_dir:
        Subdirectory under root containing .mp4 files.
    """

    def __init__(
        self,
        root: str | Path,
        json_file: str = "val.json",
        *,
        map_file: str = "map_vid_vidorID.json",
        video_dir: str = "videos",
    ) -> None:
        self._root = Path(root)
        ann_path = self._root / json_file
        map_path = self._root / map_file
        self._video_dir = self._root / video_dir

        if not ann_path.exists():
            raise FileNotFoundError(f"IntentQA annotation not found: {ann_path}")

        with open(ann_path) as f:
            entries = json.load(f)

        vid_map: dict[str, str] = {}
        if map_path.exists():
            with open(map_path) as f:
                vid_map = json.load(f)

        self._samples: list[IntentQASample] = []
        for row in entries:
            vid_id = str(row["video_name"])
            vidor_id = vid_map.get(vid_id, vid_id)
            video_path = self._video_dir / f"{vidor_id}.mp4"
            choices = tuple(str(row[f"a{i}"]) for i in range(5))
            qtype = str(row.get("type", ""))
            self._samples.append(IntentQASample(
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

    def __getitem__(self, idx: int) -> IntentQASample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[IntentQASample]:
        return iter(self._samples)
