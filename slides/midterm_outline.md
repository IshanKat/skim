# CS 291A Midterm Presentation Outline (~6 min, 1 min/slide)

---

## Title: Learning to Select and Stop: Efficient Video QA via Reinforcement Learning

## Slide 1 — Motivation & Problem

- Video QA is expensive: VLMs process all frames even when most are redundant
- Current frame selectors don't account for the possibility of finding a good subset of frames early on
- Hook: "Can a lightweight policy learn *which* frames to show the VLM — and when to stop?"

---

## Slide 2 — Method Overview

- High-level diagram: video → frame selector → subset of frames → VLM → answer
- MDP framing:
  - **State**: query embedding, current frame embedding, kept-set mean embedding, [t/N, |S|/N]
  - **Actions**: KEEP / SKIP / STOP
  - **Reward** (terminal only): 1[correct] − λ · |S|/N
- Why RL: no ground-truth labels for "optimal frame subset"; reward comes from answer correctness
- VLM is called **exactly once per episode** (at the terminal step) — the selector operates using only visual features and temporal context
- Key novelty is introducing the option for early stoppage

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

- **Warm-start**: behavioral cloning to imitate uniform frame selection
  - Gives the policy a sensible prior before RL; prevents early reward collapse
- **PPO Training**:
  - 8000 episodes, rollout batch 32, AdamW lr=3e-5
  - KL penalty against the warm-start reference policy to prevent catastrophic forgetting
  - GAE advantage estimation (γ=1.0, λ=0.95), entropy bonus for exploration
  - Resumes from existing 800-episode checkpoint

---

## Slide 5 — Preliminary Results (n=200 NExT-QA val)

| Method              | Accuracy | Avg Frames | Causal | Temporal | Descriptive |
|---------------------|----------|-----------|--------|----------|-------------|
| Uniform-8           | 75.5%    | 8.0       | 75.0%  | 69.4%    | 92.9%       |
| Uniform-16          | 76.0%    | 15.7      | 76.0%  | 70.8%    | 89.3%       |
| Uniform-32          | 78.0%    | 27.4      | 77.0%  | 75.0%    | 89.3%       |
| PPO ep800 (λ=0.1)   | **74.0%**| **13.3**  | 74.0%  | 69.4%    | 85.7%       |
| PPO ep2528 (λ=0.2)  | 68.0%    | 3.0       | 69.0%  | 59.7%    | 85.7%       |
| PPO final (λ=0.2)   | 58.5%    | 1.5       | 57.0%  | 54.2%    | 75.0%       |

**λ ablation findings:**
- λ=0.2 (ep2528 → final): policy learned a degenerate STOP-immediately strategy
  - By final checkpoint: 26% of episodes had 0 frames kept (STOP fired before any KEEP)
  - 0-frame episodes score ~40% accuracy — double random chance (20%) via VLM language prior alone
  - 1-frame episodes dominate (62%), netting 65% accuracy — worse than just asking the VLM with 8 frames
- Temporal questions hardest across all methods; descriptive easiest

---

## Slide 6 — Next Steps

- λ ablations — retrain with λ ∈ {0.05, 0.1, 0.2} to characterize the accuracy/efficiency tradeoff
- Eval on full val set (n=2000) with best λ; per-question-type breakdown
- Analysis — does STOP action fire? Which question types benefit most?

Key open questions:
- Does adaptive stopping (STOP action) add value beyond fixed-budget selection?
- Which question types benefit most from selective frame attention? (Temporal seems hardest)
