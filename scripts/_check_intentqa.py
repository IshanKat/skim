import json, csv
map_file = "data/nextqa/map_vid_vidorID.json"
vid_map = json.load(open(map_file))
with open("data/intentqa/val.csv") as f:
    rows = list(csv.DictReader(f))
sample_ids = [r["video_id"] for r in rows[:8]]
for vid in sample_ids:
    result = vid_map.get(vid, "NOT FOUND")
    print(f"{vid} -> {result}")
print(f"\nTotal rows: {len(rows)}")
print(f"Unique video_ids: {len(set(r['video_id'] for r in rows))}")
found = sum(1 for r in rows if r["video_id"] in vid_map)
print(f"video_ids found in NExT-QA map: {found}/{len(rows)}")
