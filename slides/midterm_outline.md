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

## Slide 5 — Preliminary Results

| Method      | Accuracy | Avg Frames |
|-------------|----------|------------|
| Uniform-8   | ~75%     | 8.0        |
| Uniform-16  | ~77%     | 16.0       |
| Uniform-32  | ~79%     | 32.0       |
| PPO (800 ep)| ~76%     | ~13        |

- PPO at 800 episodes is content-aware: it sees 32 frames but selects ~13 on average
- Bimodal retention (keeps very few OR almost all frames) — policy under-trained at 800 eps
- 8000 episodes with rollout batch 32 = ~10x more gradient updates → expect smoother policies

---

## Slide 6 — Next Steps & Timeline

- **Now**: Continue PPO training from 800-ep checkpoint (~2 days, fast rollout, batch=32)
- **Week 2**: Eval on full val set (n=2000), compare vs. baselines
- **Week 3**: λ ablations ({0.05, 0.1, 0.2}), analysis by question type
- **Week 4**: Write-up, figures, final presentation

Key open questions:
- Does adaptive stopping (STOP action) add value beyond fixed-budget selection?
- Which question types benefit most from selective frame attention?
