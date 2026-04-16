# Plan: Uncertainty-Driven Adaptive Frame Selection for Video QA

## Status (as of 2026-04-16)

- **Phase 0 — Environment + data:** ✅ skeleton, deps, NExT-QA loader, 4 data tests passing. Not yet verified on GPU: `scripts/check_vlm_load.py` (Qwen2.5-VL-3B 4-bit load).
- **Phase 1 — Frozen VLM wrapper:** ✅ `frames.py`, `prompts.py`, `entropy.py`, `cache.py`, `qwen_wrapper.py`, `scripts/precompute_embeddings.py`, `scripts/eval_full_frame.py`. 19/19 unit tests pass on CPU. Not yet verified on GPU: full-frame accuracy on NExT-QA val (~60–65% target).
- **Phase 2+:** pending.

### Immediate next steps on GPU box

```bash
git pull
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,gpu]

# 1. Verify VLM load:
python scripts/check_vlm_load.py

# 2. Download NExT-QA annotations + videos under data/nextqa/ (see data/README.md)

# 3. Sanity-check dataset layout:
python scripts/check_nextqa.py --root data/nextqa --split val

# 4. Phase 1 artifact — full-frame accuracy:
python scripts/eval_full_frame.py --config configs/default.yaml --limit 200
# (drop --limit once a small run looks healthy)

# 5. Precompute embeddings cache (run once, reused across PPO rollouts):
python scripts/precompute_embeddings.py --config configs/default.yaml --cache-dir cache/embeddings
```

**Known caveat to check on first GPU run:** `QwenVLWrapper.encode_frames_per_frame` calls `model.visual(pixel_values, grid_thw=...)` — the exact argument name and output shape for Qwen2.5-VL's vision tower in the installed `transformers` version may need a one-line tweak.

---

## Context

This is the CS 291A course project proposed in [CS 291A Project Proposal.docx.pdf](CS 291A Project Proposal.docx.pdf).

**Core idea.** Train a PPO policy that, for each video+query pair, sequentially emits `keep / skip / stop` actions over frames and answers using only the kept frames via a frozen Qwen2.5-VL. State includes the VLM's output entropy $H$, so the policy learns to stop once the model is confident. Reward is `1[correct] − λ·|S|/N`.

**Why this framing matters.** Fixed-budget compression (FastV, FrameFusion, LongVU) spends the same tokens on "is the person wearing a hat?" as on a multi-step causal question. The contribution is an explicit exploration–exploitation tradeoff that ties token budget to answer confidence.

**Constraints from clarifying Qs.**
- Single consumer GPU (24–48GB) — 7B is tight, so default to **Qwen2.5-VL-3B** + bnb 4-bit, with a path to 7B if cluster access opens up.
- **NExT-QA first** — smaller clips, per-question-type labels ideal for the interpretability story.
- **Wrap existing baselines** (FastV, FrameFusion) rather than re-implementing.
- **Paper-style writeup** — need full results table + ablations (λ sweep, H-ablation, warm-start-vs-scratch).

## Repository layout

```
cs-291a-project/
├── configs/                  # YAML configs (data, model, PPO, eval)
├── data/                     # dataset root (gitignored); symlink-friendly
├── src/afs/                  # package "afs" = Adaptive Frame Selection
│   ├── vlm/                  # frozen Qwen2.5-VL wrapper
│   │   ├── qwen_wrapper.py   # load model, extract frame embeddings, partial forward, entropy H
│   │   ├── frames.py         # PyAV frame extraction at target fps
│   │   ├── prompts.py        # QA prompt templates per benchmark
│   │   ├── entropy.py        # logits → (pred_idx, entropy) helper
│   │   └── cache.py          # per-frame embedding cache keyed by (video, model, fps)
│   ├── selector/
│   │   └── policy.py         # 2-layer transformer, outputs keep/skip/stop logits + value head
│   ├── env/
│   │   └── video_qa_env.py   # gym-style env: step(action) → (state, reward, done)
│   ├── data/
│   │   ├── nextqa.py         # primary loader
│   │   ├── egoschema.py      # stretch
│   │   └── activitynet_qa.py # stretch
│   ├── baselines/
│   │   ├── uniform.py        # sample 8/16/32 uniformly
│   │   ├── fastv.py          # wrap FastV repo
│   │   ├── framefusion.py    # wrap FrameFusion repo
│   │   └── visionthink_style.py  # fixed-budget RL baseline
│   ├── training/
│   │   ├── warm_start.py     # supervised imitation of FastV scores
│   │   ├── ppo.py            # TRL PPOTrainer driver
│   │   └── rollout.py        # batched rollout collection
│   ├── eval/
│   │   ├── metrics.py        # top-1, retention rate, acc-at-budget
│   │   ├── runner.py         # evaluate all methods on a benchmark
│   │   └── analysis.py       # stopping-point histograms, entropy curves
│   └── utils/                # logging, seeding, caching
├── scripts/                  # thin CLI entrypoints (train_warmstart.py, train_ppo.py, eval.py)
├── notebooks/                # exploration, figure generation
├── tests/                    # unit tests for env step/reward, entropy computation
└── pyproject.toml
```

## Stack

- **Python 3.11**, **PyTorch 2.4+**
- **transformers** (Qwen2.5-VL loader), **accelerate**, **bitsandbytes** (4-bit frozen VLM)
- **trl** for PPOTrainer (proposal-specified)
- **PyAV** for frame extraction; precompute and cache embeddings
- YAML + argparse for configs
- **wandb** for run tracking (ablations sweep)
- **pytest** for unit tests

## MDP specifics (pin down early — hidden design risks)

- **State vector** $s_t = [q\_embed \| \text{mean-pool}(E_{\text{kept}}) \| H_t \| t/N \| |S|/N]$. The last two scalars let the policy reason about how much video is left and how much budget spent.
- **Empty-set edge case.** At $t=0$ no frames are kept; use a learned zero-token for the mean-pool and set $H_0$ = entropy of VLM-answer-with-no-frames (query only). This prevents NaNs and gives a meaningful "I know nothing yet" prior.
- **Entropy $H$ definition.** First-token entropy over the answer-choice tokens (NExT-QA is 5-way MC, so 5 logits). For open-ended (ActivityNet-QA), use mean token entropy of a short sampled answer. This must be fixed per benchmark and documented.
- **Stop is forced** at $t = N$ (can't run past end-of-video).
- **Reward shaping.** Start with the proposal formulation; if PPO is unstable, add a small step penalty $-\epsilon$ per `keep` to smooth learning. Log this as an ablation.
- **λ sweep:** {0.05, 0.1, 0.2} as proposed, run after the 0.1 pipeline works end-to-end.

## Phased execution

Each phase ends with a clear artifact that de-risks the next. Don't advance if the previous phase's artifact isn't green.

### Phase 0 — Environment + data (~3 days) ✅ code landed
- Set up `pyproject.toml`, pin deps, verify `transformers` loads **Qwen2.5-VL-3B-Instruct** in 4-bit on target GPU.
- Download NExT-QA (val split is enough for initial iteration; ~570 videos, ~5K questions).
- Write [src/afs/data/nextqa.py](src/afs/data/nextqa.py) to yield `(video_path, question, choices, answer_idx, qtype)` tuples.
- **Artifact:** `pytest tests/test_data.py` passes; can iterate one batch end-to-end with random frames.

### Phase 1 — Frozen VLM wrapper (~4 days) ✅ code landed, GPU verification pending
- [src/afs/vlm/qwen_wrapper.py](src/afs/vlm/qwen_wrapper.py): expose `encode_frames(frames) → E`, `encode_query(q) → q_embed`, `answer(E_kept, q) → (pred, H)`.
- **Pre-cache frame embeddings per video to disk** (keyed by video_id + fps). Recomputing embeddings every rollout step is the #1 thing that will kill training throughput; cache first, cache hard.
- **Artifact:** full-frame accuracy on NExT-QA val matches reported Qwen2.5-VL-3B numbers (~60–65% MC) within a few points. If not, something is wrong with prompting.

### Phase 2 — Baselines: uniform + full-frame (~2 days)
- [src/afs/baselines/uniform.py](src/afs/baselines/uniform.py) at {8, 16, 32} frames.
- [src/afs/eval/runner.py](src/afs/eval/runner.py) produces the first row of the results table.
- **Artifact:** results table skeleton populated with uniform + full-frame on NExT-QA val.

### Phase 3 — Selector + env + warm-start (~5 days)
- [src/afs/selector/policy.py](src/afs/selector/policy.py): 2-layer transformer (d=512, 4 heads), 3-way action head + scalar value head for PPO critic.
- [src/afs/env/video_qa_env.py](src/afs/env/video_qa_env.py): gym-style step loop; reward computed only at terminal step.
- [src/afs/training/warm_start.py](src/afs/training/warm_start.py): run FastV on precomputed embeddings, get per-frame importance scores, convert top-k selections into `keep/skip/stop` target sequences, train the selector by cross-entropy. This gives PPO a non-random init, which matters a lot for sparse-reward problems like this.
- **Unit tests for env** (step transitions, terminal reward computation, edge cases at t=0 and t=N) — cheap and catches nasty bugs later.
- **Artifact:** warm-started selector matches uniform-16 accuracy at equal or lower average frame count.

### Phase 4 — PPO training (~7 days, the hard part)
- [src/afs/training/ppo.py](src/afs/training/ppo.py) using `trl.PPOTrainer`.
- Batch rollout: N=32 videos in parallel, max 32 frames each, horizon = N_frames.
- **Monitor:** reward, episode length, entropy of policy, KL to warm-start ref policy, fraction-stopped-before-end.
- **Failure modes to watch:** (1) policy collapses to always-stop-immediately (raise λ too high? → lower λ or add step penalty); (2) policy collapses to always-keep (λ too low → raise); (3) entropy-of-policy → 0 early (increase entropy bonus).
- **Artifact:** PPO run converges on a λ=0.1 config; selector beats uniform-16 on accuracy at lower retention.

### Phase 5 — External baselines (~4 days)
- Wrap FastV ([src/afs/baselines/fastv.py](src/afs/baselines/fastv.py)) and FrameFusion ([src/afs/baselines/framefusion.py](src/afs/baselines/framefusion.py)) from their official repos (git submodules or vendored).
- Implement VisionThink-style fixed-budget RL: same selector+env but reward = `1[correct]` with a hard cap on |S|.
- **Artifact:** full baselines column of the results table.

### Phase 6 — Evaluation, analysis, ablations (~5 days)
- [src/afs/eval/analysis.py](src/afs/eval/analysis.py): per-qtype stopping histograms, entropy-over-time plots grouped by causal/temporal/descriptive.
- **Ablations (required for paper-style writeup):**
  - λ ∈ {0.05, 0.1, 0.2} sweep.
  - **H-ablation:** remove $H$ from state; expected to degrade the adaptive-stopping story (this is the key causal claim of the paper).
  - **Warm-start ablation:** PPO from scratch vs FastV warm-start.
- **Artifact:** Full results table + three analysis figures.

### Phase 7 (stretch) — EgoSchema + ActivityNet-QA
Only attempt after Phase 6 is locked. Biggest risks: EgoSchema frame volume (3-min clips × 1fps = 180 frames → memory pressure on cached embeddings) and ActivityNet-QA's open-ended answers requiring an LLM judge.

## Verification plan

Per phase (must be green before moving on):
- **Unit:** `pytest tests/` — env step semantics, entropy computation on canned inputs, reward at terminal vs non-terminal, NaN/empty-state handling at t=0.
- **Integration:** `scripts/smoke_test.py` — runs 1 video end-to-end through selector → VLM → answer, asserts non-NaN reward.
- **End-to-end:** full eval on NExT-QA val, results written to `results/{method}_{seed}.json` with per-question predictions for auditability.

Final deliverable verification:
- Reproduce results table from a single `scripts/reproduce.sh` invocation given cached embeddings.
- All three analysis figures generated from eval JSON by `src/afs/eval/analysis.py`.

## Files to create first (critical path)

1. [pyproject.toml](pyproject.toml), [configs/default.yaml](configs/default.yaml) ✅
2. [src/afs/vlm/qwen_wrapper.py](src/afs/vlm/qwen_wrapper.py) — unblocks everything ✅
3. [src/afs/data/nextqa.py](src/afs/data/nextqa.py) — unblocks phase 2+ ✅
4. [src/afs/env/video_qa_env.py](src/afs/env/video_qa_env.py) + [tests/test_env.py](tests/test_env.py) — unblocks phase 3+

## Open decisions deferred to implementation

- **Exact $H$ formulation on open-ended benchmarks** (ActivityNet-QA): first-token entropy vs mean-token entropy of a short sample — decide when Phase 7 starts.
- **Warm-start target:** FastV top-k as proposed, but if FastV underperforms uniform on NExT-QA, fall back to imitating uniform-16.
- **7B vs 3B final runs:** if cluster access opens up mid-project, re-run Phase 4 + 6 with 7B for the headline number.
