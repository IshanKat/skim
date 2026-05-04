# Adaptive Frame Selection for Video QA via RL

**CS 291A — UCSB**

A lightweight reinforcement learning agent that learns to adaptively select which video frames to show a frozen Vision-Language Model (VLM), trading off answer accuracy against the number of frames used.

---

## Overview

Large VLMs can answer questions about videos by processing a sequence of frames. The naive approach feeds all frames to the model — expensive and often unnecessary, since many frames are redundant. This project trains a small transformer policy (8M parameters) to decide, frame by frame, whether to **KEEP**, **SKIP**, or **STOP** processing — with the goal of answering correctly while keeping as few frames as possible.

The key design choice: the VLM is called **exactly once per episode**, only at the end, to produce the final answer and compute the training reward. The selector itself runs entirely on pre-cached visual embeddings — no VLM calls during frame selection at either train or eval time.

---

## Method

### MDP Formulation

Each (video, question) pair is one episode:

- **State** at step t: `(query_embed, frame_embed_t, kept_set_mean_embed, t/N, |S|/N)`
  - `query_embed`: mean-pooled token embeddings of the question text [D_text=2048]
  - `frame_embed_t`: vision encoder output for the current frame [D_vis=1280]
  - `kept_set_mean_embed`: mean of all kept frame embeddings so far [D_vis=1280]
  - `t/N`: how far through the video (0→1)
  - `|S|/N`: what fraction of frames have been kept so far (0→1)

- **Actions**: `KEEP` (0), `SKIP` (1), `STOP` (2)
  - KEEP: add this frame to the selected subset, advance to next frame
  - SKIP: discard this frame, advance to next frame
  - STOP: end selection early, go straight to the VLM answer call

- **Reward**: sparse, terminal-only
  ```
  R = 1[correct] - λ · |S|/N
  ```
  The policy is rewarded for getting the right answer and penalized proportionally to how many frames it kept. λ=0.1 by default.

- **Episode end**: STOP action, or forced stop at the last frame.

### Policy Architecture

A small transformer with 4-token input:

```
[query_token, frame_token, kept_set_token, scalar_token]
       ↓             ↓              ↓              ↓
   text_proj     vis_proj       vis_proj      scalar_proj
       └─────────────┴──────────────┴──────────────┘
                         2-layer Transformer
                               ↓ mean pool
                    ┌──────────┴──────────┐
               action_head           value_head
             (3-way logits)          (scalar)
```

- All projections map to d_model=512
- 2 transformer encoder layers, 4 attention heads, pre-norm
- ~8M total parameters

### Frozen Backbone

[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), loaded in 4-bit NF4 quantization (~1GB VRAM on RTX 2070 8GB). Used for:
1. **Pre-computing frame embeddings** (vision encoder only, done once and cached to disk)
2. **Computing query embeddings** (LM embedding layer, fast)
3. **Answering the question** (full forward pass, once per episode at terminal)

### Training

**Warm-start (Phase 0):** Behavioral cloning — the policy is trained via cross-entropy to imitate uniform frame selection (keep every N/k-th frame). This gives the policy a reasonable starting point before RL and prevents early reward collapse.

**PPO (Phase 1):** Standard PPO-clip with:
- GAE advantage estimation (γ=1.0, λ_GAE=0.95)
- KL penalty against the frozen warm-start reference policy
- Entropy bonus on the action distribution to encourage exploration
- AdamW, lr=3e-5, rollout batch=32, 4 PPO epochs per update

---

## Dataset

[NExT-QA](https://github.com/doc-doc/NExT-QA) — a video question answering benchmark with 5-way multiple choice questions across three question types:
- **Causal**: why/how questions requiring causal reasoning
- **Temporal**: questions about temporal order or change over time
- **Descriptive**: what/where/who questions about visual content

Videos are sampled at 1 fps, capped at 32 frames per video.

---

## Baselines

| Method      | Description                                    |
|-------------|------------------------------------------------|
| Uniform-8   | Evenly sample 8 frames, feed all to VLM        |
| Uniform-16  | Evenly sample 16 frames, feed all to VLM       |
| Uniform-32  | Evenly sample 32 frames, feed all to VLM       |
| PPO (ours)  | Adaptive selection: sees 32, selects a subset  |

---

## Results (preliminary, 800 episodes)

| Method      | Accuracy | Avg Frames Used |
|-------------|----------|-----------------|
| Uniform-8   | ~75%     | 8.0             |
| Uniform-16  | ~77%     | 16.0            |
| Uniform-32  | ~79%     | 32.0            |
| PPO (800ep) | ~76%     | ~13             |

After 800 training episodes the policy selects ~13 frames on average — fewer than Uniform-16 while approaching its accuracy. Training is ongoing.

---

## Project Structure

```
├── configs/
│   └── default.yaml               # All hyperparameters
├── scripts/
│   ├── precompute_embeddings.py   # Phase 1: cache frame embeddings
│   ├── eval_full_frame.py         # Phase 2: uniform baseline eval
│   ├── train_warmstart.py         # Phase 3: behavioral cloning warm-start
│   ├── train_ppo.py               # Phase 4: PPO training
│   ├── eval_ppo.py                # Phase 5: evaluate trained policy
│   └── analyze_results.py         # Offline figure generation
├── src/afs/
│   ├── data/nextqa.py             # NExT-QA dataset loader
│   ├── env/video_qa_env.py        # Gym-style MDP environment
│   ├── selector/policy.py         # Transformer selector policy
│   ├── training/
│   │   ├── warm_start.py          # Behavioral cloning trainer
│   │   ├── ppo.py                 # PPO trainer
│   │   └── rollout.py             # Episode collection
│   └── vlm/
│       ├── qwen_wrapper.py        # Frozen VLM interface
│       ├── cache.py               # On-disk embedding cache
│       └── frames.py              # Video frame extraction
├── slides/
│   └── midterm_outline.md         # Presentation outline
└── results/                       # Eval output JSONs and figures
```

---

## Setup

```bash
pip install -e .
```

Requires CUDA GPU with ≥8GB VRAM. Tested on RTX 2070 8GB with Python 3.10 and PyTorch 2.x.

## Running

```bash
# 1. Pre-compute and cache frame embeddings (one-time, ~hours)
python scripts/precompute_embeddings.py --config configs/default.yaml

# 2. Run uniform baselines
python scripts/eval_full_frame.py --max-frames 8  --output results/uniform_8.json
python scripts/eval_full_frame.py --max-frames 32 --output results/uniform_32.json

# 3. Warm-start the policy
python scripts/train_warmstart.py --config configs/default.yaml --output checkpoints/warmstart.pt

# 4. PPO training (resumes from warmstart or existing checkpoint)
python scripts/train_ppo.py --warmstart checkpoints/warmstart.pt --output-dir checkpoints/ppo

# 5. Evaluate
python scripts/eval_ppo.py --checkpoint checkpoints/ppo/ppo_final.pt --output results/ppo_val.json

# 6. Generate figures
python scripts/analyze_results.py
```
