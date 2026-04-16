from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from afs.data.nextqa import NExTQADataset, NExTQASample


def _write_synthetic_nextqa(root: Path) -> None:
    (root / "videos").mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "video": "vid1", "frame_count": 100, "width": 640, "height": 360,
            "question": "why is the child laughing?",
            "a0": "because of the dog", "a1": "because of food",
            "a2": "because of the toy", "a3": "because of the song",
            "a4": "because of the adult",
            "answer": 2, "qid": "vid1_0", "type": "CW",
        },
        {
            "video": "vid2", "frame_count": 80, "width": 640, "height": 360,
            "question": "what happens next after the man sits?",
            "a0": "he stands", "a1": "he eats", "a2": "he sleeps",
            "a3": "he reads", "a4": "he talks",
            "answer": 1, "qid": "vid2_0", "type": "TN",
        },
        {
            "video": "vid3", "frame_count": 60, "width": 640, "height": 360,
            "question": "where is the cat?",
            "a0": "sofa", "a1": "floor", "a2": "bed", "a3": "table", "a4": "chair",
            "answer": 3, "qid": "vid3_0", "type": "DL",
        },
    ]
    pd.DataFrame(rows).to_csv(root / "val.csv", index=False)

    with open(root / "map_vid_vidorID.json", "w") as f:
        json.dump({"vid1": "0001_vidor", "vid2": "0002_vidor", "vid3": "0003_vidor"}, f)

    for vidor in ("0001_vidor", "0002_vidor", "0003_vidor"):
        (root / "videos" / f"{vidor}.mp4").touch()


@pytest.fixture
def nextqa_root(tmp_path: Path) -> Path:
    _write_synthetic_nextqa(tmp_path)
    return tmp_path


def test_loader_len_and_indexing(nextqa_root: Path) -> None:
    ds = NExTQADataset(nextqa_root, split="val")
    assert len(ds) == 3

    s0 = ds[0]
    assert isinstance(s0, NExTQASample)
    assert s0.qid == "vid1_0"
    assert s0.video_id == "vid1"
    assert s0.video_path.name == "0001_vidor.mp4"
    assert s0.video_path.exists()
    assert len(s0.choices) == 5
    assert s0.answer_idx == 2
    assert s0.qtype == "CW"
    assert s0.qtype_group == "causal"


def test_loader_iteration_and_qtype_groups(nextqa_root: Path) -> None:
    ds = NExTQADataset(nextqa_root, split="val")
    groups = [s.qtype_group for s in ds]
    assert groups == ["causal", "temporal", "descriptive"]


def test_loader_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NExTQADataset(tmp_path, split="val")


def test_loader_missing_map_raises(tmp_path: Path) -> None:
    (tmp_path / "val.csv").write_text(
        "video,frame_count,width,height,question,answer,qid,type,"
        "a0,a1,a2,a3,a4\n"
    )
    with pytest.raises(FileNotFoundError):
        NExTQADataset(tmp_path, split="val")
