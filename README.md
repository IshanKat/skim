# SKIM: Streaming-Compatible Adaptive Frame Selection for Efficient Video QA via Reinforcement Learning

**CS 291A — UCSB** | Ishan Katpally, Om Mahesh

A lightweight (8M-parameter) RL policy that scans a video frame by frame and decides — per frame — whether to **KEEP**, **SKIP**, or **STOP**, then passes only the kept subset to a frozen VLM queried once per question.

Paper: [`paper/main.pdf`](paper/main.pdf)

---

## Key Results

**NExT-QA val (n=1,000)**

| Method | Accuracy | Avg Frames | Causal | Temporal | Descriptive |
|--------|----------|-----------|--------|----------|-------------|
| Uniform-8 | 75.5% | 8.0 | 75.0% | 69.4% | 92.9% |
| Uniform-16 | 76.0% | 15.7 | 76.0% | 70.8% | 89.3% |
| Uniform-32 | 78.0% | 27.4 | 77.0% | 75.0% | 89.3% |
| **SKIM λ=0.05** | **72.3%** | **14.2** | 74.6% | 63.5% | 82.8% |
| SKIM λ=0.1 | 74.5% | 8.6 | 76.0% | 68.1% | 85.7% |
| SKIM λ=0.2 | 58.5% | 1.5 | 57.0% | 54.2% | 75.0% |

**IntentQA val — zero-shot transfer (n=2,044 full val)**

| Method | Accuracy | Avg Frames | Causal | Temporal |
|--------|----------|-----------|--------|----------|
| Uniform-8 | 87.5% | 8.0 | 90.1% | 79.2% |
| Uniform-16 | 88.0% | 15.8 | 89.5% | 83.3% |
| Uniform-32 | 90.0% | 29.2 | 90.8% | 87.5% |
| **SKIM λ=0.05** | **86.6%** | **15.7** | 88.0% | 82.3% |

---

## Method

### MDP

Each (video, question) pair is one episode. Frames sampled at 1 fps, capped at N≤32.

- **State**: `(query_embed, frame_embed_t, kept_set_mean, t/N, |S|/N)`
- **Actions**: `KEEP` · `SKIP` · `STOP`
- **Reward** (terminal only): `R = 1[correct] − λ·|S|/N`

The policy is strictly causal — state depends only on frames seen so far — making it streaming-compatible.

### Policy Architecture

4-token transformer input → 2-layer encoder (d=512, 4 heads) → action head (3-way) + value head. ~8M parameters.

### Backbone

Qwen2.5-VL-3B-Instruct in 4-bit NF4 quantization (~1 GB VRAM). Used for:
1. Pre-computing frame embeddings (cached to disk once)
2. Computing query embeddings
3. Single full forward pass at episode termination to produce the answer

### Training

1. **Warm-start** (behavioral cloning to uniform selection)
2. **PPO-clip** with GAE (γ=1.0, λ_GAE=0.95), KL penalty against warm-start reference, entropy bonus

---

## Reproduction

### Requirements

- Python 3.10+
- CUDA GPU with ≥8 GB VRAM (tested on RTX 2070)
- PyTorch 2.x

```bash
pip install -e .
```

### Data

**NExT-QA**: Download videos and annotations from the [official repo](https://github.com/doc-doc/NExT-QA). Place under `data/nextqa/` with structure:
```
data/nextqa/
  map_vid_vidorID.json
  val.csv
  videos/   # .mp4 files
```

**IntentQA** (optional, zero-shot transfer only): Download from the [official repo](https://github.com/JoseponLee/IntentQA). Place under `data/intentqa/`:
```
data/intentqa/
  map_vid_vidorID.json
  val.csv
  videos/   # .mp4 files
```

### Step-by-step

```bash
# 1. Pre-compute and cache frame embeddings (one-time, several hours)
python scripts/precompute_embeddings.py --config configs/default.yaml

# For IntentQA (optional):
python scripts/precompute_embeddings.py --config configs/default.yaml \
    --dataset intentqa --data-root data/intentqa

# 2. Uniform baselines
python scripts/eval_full_frame.py --max-frames 8  --output results/uniform8_nextqa.json
python scripts/eval_full_frame.py --max-frames 16 --output results/uniform16_nextqa.json
python scripts/eval_full_frame.py --max-frames 32 --output results/uniform32_nextqa.json

# 3. Warm-start (behavioral cloning)
python scripts/train_warmstart.py --config configs/default.yaml \
    --output checkpoints/warmstart.pt

# 4. PPO training
#    λ=0.05 (best accuracy-frames tradeoff):
python scripts/train_ppo.py --config configs/lambda005.yaml \
    --warmstart checkpoints/warmstart.pt \
    --output-dir checkpoints/ppo_lambda005

#    λ=0.1:
python scripts/train_ppo.py --config configs/default.yaml \
    --warmstart checkpoints/warmstart.pt \
    --output-dir checkpoints/ppo_lambda01

# 5. Evaluate on NExT-QA (n=1000)
python scripts/eval_ppo.py \
    --config configs/lambda005.yaml \
    --checkpoint checkpoints/ppo_lambda005/ppo_ep001024.pt \
    --dataset nextqa --limit 1000 \
    --output results/ppo_lambda005_ep1024_nextqa_n1000.json

# 6. Zero-shot transfer to IntentQA (full val)
python scripts/eval_ppo.py \
    --config configs/lambda005.yaml \
    --checkpoint checkpoints/ppo_lambda005/ppo_ep001024.pt \
    --dataset intentqa --data-root data/intentqa \
    --output results/ppo_lambda005_ep1024_intentqa_full.json
```

### Configs

| Config | λ | Episodes |
|--------|---|----------|
| `configs/default.yaml` | 0.1 | 3000 |
| `configs/lambda005.yaml` | 0.05 | 3000 |

Increase `lambda_cost` to 0.2 to reproduce the reward-hacking experiment.

### Analysis scripts

```bash
# Per-checkpoint accuracy and frame counts across all result JSONs
python scripts/analyze_all_ckpts.py

# STOP action analysis by question type
python scripts/analyze_stop_all.py
```

---

## Project Structure

```
configs/          # Hyperparameter configs (lambda005.yaml, default.yaml)
scripts/
  precompute_embeddings.py   # Cache frame + query embeddings to disk
  train_warmstart.py         # Behavioral cloning warm-start
  train_ppo.py               # PPO training
  eval_ppo.py                # Policy evaluation (NExT-QA / IntentQA)
  eval_full_frame.py         # Uniform baseline evaluation
  analyze_all_ckpts.py       # Summary table across checkpoints
  analyze_stop_all.py        # STOP action analysis by question type
src/afs/
  data/                      # Dataset loaders (NExT-QA, IntentQA, EgoSchema)
  env/video_qa_env.py        # MDP environment
  selector/policy.py         # Transformer policy + value head
  training/                  # PPO trainer, warm-start trainer, rollout collector
  vlm/                       # Qwen wrapper, embedding cache, frame extraction
results/                     # Eval output JSONs and figures
paper/                       # NeurIPS-style paper source (main.tex → main.pdf)
```
