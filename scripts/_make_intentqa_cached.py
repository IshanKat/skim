"""Filter IntentQA val.csv to only rows whose video_id is in the embedding cache."""
import csv
from pathlib import Path

cache_dir = Path("cache/embeddings/Qwen_Qwen2.5-VL-3B-Instruct")
cached_ids = {p.stem.split("__")[0] for p in cache_dir.glob("*.pt")}

in_path  = Path("data/intentqa/val.csv")
out_path = Path("data/intentqa/val_cached.csv")

with open(in_path, newline="", encoding="utf-8") as fin, \
     open(out_path, "w", newline="", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    kept = 0
    for row in reader:
        if row["video_id"] in cached_ids:
            writer.writerow(row)
            kept += 1

print(f"Kept {kept} rows with cached video_ids -> {out_path}")
