# CS 291A Midterm Presentation Outline (~6 min, 1 min/slide)

---

## Slide 1 — Motivation & Problem

- Video QA is expensive: VLMs process all frames even when most are redundant
- Hook: "Can a lightweight policy learn *which* frames to show the VLM?"
- Research question: train an RL policy to adaptively select frames, balancing accuracy vs. compute
- Motivating figure: accuracy vs. avg frames used (teaser of results)

---

## Slide 2 — Method Overview

- High-level diagram: video → frame selector → VLM → answer
- MDP framing:
  - **State**: query embedding, current frame embedding, kept-set mean, [H_t, t/N, |S|/N]
  - **Actions**: KEEP / SKIP / STOP
  - **Reward** (terminal only): 1[correct] − λ · |S|/N
- Why RL over supervised: no ground-truth labels for "optimal frame subset"
- Entropy H_t = VLM uncertainty at step t; used as a navigation signal in the state

---

## Slide 3 — Architecture

- **Selector**: 2-layer transformer, 8M params
  - 4-token input: (query, current frame, kept-set mean, scalars)
  - Action head → 3-way softmax (KEEP/SKIP/STOP)
  - Value head → scalar (PPO baseline)
- **Frozen backbone**: Qwen2.5-VL-3B-Instruct (4-bit quantized)
  - Vision encoder (d_vis=1280) → per-frame embeddings (cached)
  - LM hidden size (d_text=2048) → query embedding
  - answer_mc() → 5-way probability distribution over choices → entropy H_t

---

## Slide 4 — Training Pipeline

- **Phase 0 — Warm-start** (done): behavioral cloning to imitate FastV frame selections
- **Phase 1 — PPO, 8 frames** (in progress):
  - Real entropy signal: VLM called at every KEEP step to update H_t
  - 8000 episodes, rollout batch 8, AdamW lr=3e-5, KL penalty against warm-start ref policy
  - ~4 days on RTX 2070 8GB
- **Phase 2 — PPO fine-tune, 32 frames** (planned):
  - Fast rollout (one VLM call/episode) to adapt STOP timing to full-length videos
  - 4000 episodes, ~2 days

---

## Slide 5 — Preliminary Results

| Method      | Accuracy | Avg Frames |
|-------------|----------|------------|
| Uniform-8   | ~75%     | 8.0        |
| Uniform-16  | ~77%     | 16.0       |
| Uniform-32  | ~79%     | 32.0       |
| PPO (800 ep)| ~76%     | ~13        |

- PPO at 800 episodes approaches Uniform-16 accuracy using fewer-than-32 frames
- Bimodal retention (keeps 0–2 OR 30–32 frames) — policy under-trained, hasn't learned intermediate strategies
- Root cause: only 200 gradient updates (800 eps / batch 4); Phase 1 will do 1000 updates

---

## Slide 6 — Next Steps & Timeline

- **Week 1–2**: Complete Phase 1 (8000 eps, real entropy, 8 frames)
- **Week 2–3**: Phase 2 fine-tune (4000 eps, 32 frames)
- **Week 3**: Eval on full val set (n=2000), λ ablations ({0.05, 0.1, 0.2})
- **Week 4**: Write-up, figures, analysis

Key open questions:
- Does real entropy (vs. fast-rollout proxy) meaningfully improve frame selection?
- Does the two-phase approach close the gap to Uniform-32 with fewer frames?
