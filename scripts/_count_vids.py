import pandas as pd
df = pd.read_csv("data/nextqa/val.csv")
print(f"Total questions (val): {len(df)}")
print(f"Unique videos  (val): {df['video'].nunique()}")
cached = list(__import__('pathlib').Path("cache/embeddings/Qwen_Qwen2.5-VL-3B-Instruct").glob("*.pt"))
print(f"Cached embeddings:    {len(cached)}")
