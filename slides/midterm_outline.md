# CS 291A Midterm Presentation Outline (~6 min, 1 min/slide)

---

## Slide 1 — Motivation & Problem

- Video QA is expensive: VLMs process all frames even when most are redundant
- Hook: "Can a lightweight policy learn *which* frames to show the VLM — and when to stop?"
- Research question: train an RL agent to adaptively select frames, balancing accuracy vs. compute
- Teaser: our policy matches Uniform-16 accuracy while being content-aware

---

## Slide 2 — Method Overview

- High-level diagram: video → frame selector → subset of frames → VLM → answer
- MDP framing:
  - **State**: query embedding, current frame embedding, kept-set mean embedding, [t/N, |S|/N]
  - **Actions**: KEEP / SKIP / STOP
  - **Reward** (terminal only): 1[correct] − λ · |S|/N
- Why RL: no ground-truth labels for "optimal frame subset"; reward comes from answer correctness
- Key property: VLM is called **exactly once per episode** (at the terminal step) — the selector operates using only visual features and temporal context

---

## Slide 3 — Architecture

- **Selector**: 2-layer transformer, 8M params
  - 4-token input sequence: (query, current frame, kept-set mean, scalars)
  - Scalars: [t/N, |S|/N] — progress through video, fraction of frames kept so far
  - Action head → 3-way softmax (KEEP / SKIP / STOP)
  - Value head → scalar for PPO advantage estimation
- **Frozen backbone**: Qwen2.5-VL-3B-Instruct (4-bit quantized, ~1GB VRAM)
  - Vision encoder (d_vis=1280) → per-frame embeddings (pre-cached, no inference cost)
  - LM embedding layer (d_text=2048) → query embedding
  - Full forward pass → final MC answer (once per episode only)

---

## Slide 4 — Training Pipeline

- **Phase 0 — Warm-start** (complete): behavioral cloning to imitate uniform frame selection
  - Gives the policy a sensible prior before RL; prevents early reward collapse
- **Phase 1 — PPO** (in progress):
  - 8000 episodes, rollout batch 32, AdamW lr=3e-5
  - KL penalty against the warm-start reference policy to prevent catastrophic forgetting
  - GAE advantage estimation (γ=1.0, λ=0.95), entropy bonus for exploration
  - Resumes from existing 800-episode checkpoint

---

## Slide 5 — Results (n=200 NExT-QA val)

| Method        | Accuracy | Avg Frames |
|---------------|----------|------------|
| Uniform-8     | 75.5%    | 8.0        |
| Uniform-16    | 76.0%    | 15.7       |
| Uniform-32    | 78.0%    | 27.4       |
| PPO (800 ep)  | **74.0%**| **13.3**   |

- Best checkpoint: 800 episodes, λ=0.1
- Matches Uniform-8 accuracy (74% vs 75.5%) using 13.3 frames on average — content-aware, not just subsampling
- Over-training with λ=0.2 collapsed to 1.5 frames avg / 58% accuracy — λ is a critical hyperparameter
- Temporal questions are hardest across all methods; descriptive easiest

---

## Slide 6 — Next Steps & Timeline

- **Now**: λ ablations — retrain with λ ∈ {0.05, 0.1, 0.2} to characterize the accuracy/efficiency tradeoff
- **Week 2**: Eval on full val set (n=2000) with best λ; per-question-type breakdown
- **Week 3**: Analysis — does STOP action fire? Which question types benefit most?
- **Week 4**: Write-up, figures, final presentation

Key open questions:
- λ=0.1 appears to be the sweet spot; is there a principled way to set it?
- Does adaptive stopping (STOP action) add value beyond fixed-budget selection?
- Which question types benefit most from selective frame attention? (Temporal seems hardest)
