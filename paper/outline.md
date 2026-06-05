# SKIM — Paper Outline (working skeleton)

**Title (working):** SKIM: Streaming-Compatible Adaptive Frame Selection for Efficient Video QA via Reinforcement Learning
**Authors:** Ishan Katpally, Om Mahesh
**Venue/format:** CS 291A final report — NeurIPS 2026 style, 4–6 pages excl. references. Checklist required (appendix, doesn't count).

## Agreed framing decisions
- **Thesis:** *Balanced* — lead with the controllable accuracy–efficiency tradeoff + streaming-compatible design + content-aware/transfer results; give reward hacking its own honest analysis section.
- **Scope:** *Write with what we have* — NExT-QA (train + eval) and IntentQA (zero-shot transfer), n=200 subsets, existing λ sweep + analyses. No new GPU runs. No EgoSchema (loaders exist, no results).
- **Not a SOTA claim:** SKIM matches Uniform-8/16 accuracy at comparable-or-fewer frames; never beats Uniform-32. λ trades accuracy for frames. Honest, modest, analysis-rich.
- **Headline number:** PPO **λ=0.05, ep1024 → 76.0% @ 14.0 frames** on NExT-QA (n=200) — matches Uniform-16 accuracy (76.0% @ 15.7f) at fewer frames. Feature in abstract + intro + bolded Table 1 row. Fig 1 still shows the full λ-sweep Pareto frontier around it.

## The streaming thread (key differentiator)
Most frame selectors are **offline**: retrieval / text-frame matching ranks all frames against the query, requiring the whole video upfront. SKIM is **causal by construction** — state uses only frames seen so far (`t/N`, `|S|/N`, running kept-set mean), never future frames — plus a learned **STOP**. So it is natively compatible with **online/streaming** VQA: frames arrive one at a time, the selector forwards a subset, and STOP lets the VLM answer before the video ends. In offline settings STOP only saves compute; in streaming settings STOP is the point. *Honest caveat:* current eval is offline (≤32 frames, forced-stop at end); a true streaming benchmark is future work.

## Contributions (Intro bullets)
1. **Cheap-to-train MDP** for sequential, query-conditioned adaptive frame selection that runs entirely on cached embeddings — exactly **one VLM call per episode** → RL trainable on a single 8GB consumer GPU.
2. **Streaming-compatible by design** — causal selection (no future-frame access) + learned STOP; contrast with offline retrieval-based selectors. (Positioning/motivation; streaming eval = future work.)
3. **8M-param transformer selector + warm-start→PPO recipe** with a controllable accuracy–frame tradeoff (λ); matches uniform at comparable/fewer frames, transfers zero-shot NExT-QA→IntentQA.
4. **Honest analysis of what the policy learns** — content-aware per-question-type behavior, plus failure modes: reward hacking via the VLM language prior, and λ-induced collapse.

---

## Section-by-section plan (~5 pages)

### Abstract (~0.25 pg)
Problem (all-frames VLM = wasteful, uniform = content-blind, existing selectors = offline) → method (8M RL selector, KEEP/SKIP/STOP on cached embeddings, 1 VLM call/episode, R = 1[correct] − λ·|S|/N, warm-start→PPO) → results (matches uniform at fewer frames, λ knob, NExT-QA→IntentQA transfer) → honest finding (reward hacking / language-prior exploit). Streaming-compatible design noted.

### 1. Introduction (~0.75 pg)
Motivation: (a) efficiency — feeding all frames is expensive and redundant; (b) **online/streaming** — current selectors are offline. Task definition: per (video, question), choose a variable subset of frames to feed a frozen VLM. The 4 contributions above.

### 2. Related Work (~0.6 pg) — DRAFTED in main.tex
Four paragraphs, all citations verified (June 2026) and in references.bib:
- **Token reduction (inside the VLM):** FastV [Chen ECCV'24], FrameFusion [Fu '25], LongVU [Shen '24] — fixed token budget. SKIM reduces input *before* the VLM, variable budget.
- **Frame/keyframe selection for VQA:** ATP [Buch CVPR'22] (static bias — single frame often enough), SeViLA [Yu NeurIPS'23], AKS [Tang CVPR'25] — *offline* (score all frames, need whole video) + *fixed* K. SKIM = sequential/causal + variable via STOP.
- **Adaptive computation / RL frame selection:** AdaFrame [Wu CVPR'19], SCSampler [Korbar ICCV'19] — action recognition, not query-conditioned QA, predate large VLMs. SKIM = keep/skip/stop for QA with a *frozen* VLM via cached embeddings.
- **Streaming video understanding:** VideoLLM-online [Chen CVPR'24], Flash-VStream [Zhang ICCV'25] — heavyweight end-to-end streaming VLMs. SKIM = lightweight selector, streaming-*compatible* front-end; offline eval only.

**FRAMING = COMBINATION NOVELTY (baked in; Table 1 = capability matrix `tab:landscape`).** The exact idea (lightweight RL frame selector, frozen VLM, answer-correctness reward, NExT-QA) was independently done 2024–26: ViaRL [Xu '25, REINFORCE++], ReFoCUS [Lee '25, GRPO], **HORNet [Bai '26, GRPO]** is nearly identical. So the paper claims **no novel RL and no novel streaming**. Verified (from their PDFs) that all three concurrent RL selectors are **offline (non-causal), fixed-budget (no stop), and cost-agnostic** — ReFoCUS even states it "is not causal." SKIM's defensible claim = the **unique combination**: query-conditioned RL-on-frozen-VLM + **causal/single-pass** + **learned STOP (variable budget)** + **cost-aware reward**. No single prior method has all three of the last items (AdaFrame has stop+cost but is pre-VLM/not causal; streaming VLMs are causal but heavyweight end-to-end). §1 contributions + §2 Related Work + Table 1 all reflect this. Honesty guardrails: streaming = design property (not measured); ViaRL frozen marked `~†` (it fine-tunes the MLLM). ATP also seeds Analysis (static bias ↔ reward-hacking/language-prior finding).

**Concurrent/related citations added to references.bib (verified authors/venues):** framevoyager [ICLR'25], mllmselect [arXiv'25], viarl [arXiv'25], refocus [arXiv'25], air [ICLR'26], hornet [arXiv'26]. Total 22 refs, build clean.

### 3. Method (~1.25 pg)
- **3.1 Problem setup / MDP.** State `s_t = (query_embed, frame_embed_t, kept_set_mean, t/N, |S|/N)`; actions KEEP/SKIP/STOP; reward `R = 1[correct] − λ·|S|/N` (sparse, terminal); episode ends at STOP or forced-stop at last frame. Emphasize causality → streaming compatibility.
- **3.2 Policy architecture.** 4-token transformer (query/frame/kept-set/scalar → d=512), 2 layers / 4 heads, action head (3-way) + value head; ~8M params.
- **3.3 Frozen backbone + cached embeddings.** Qwen2.5-VL-3B-Instruct, 4-bit NF4. Vision encoder cached once to disk; LM embeds query; full forward only at terminal → 1 VLM call/episode. This is what makes training cheap.
- **3.4 Training.** Phase 0 warm-start (BC to uniform selection); Phase 1 PPO-clip (GAE γ=1.0 λ_GAE=0.95, KL to frozen warm-start ref, entropy bonus, AdamW lr=3e-5, rollout 32, 4 epochs). Full hyperparams → appendix.

### 4. Experiments (~1.25 pg)
- **4.1 Setup.** Datasets: NExT-QA (5-way MC; causal/temporal/descriptive) for train+eval; IntentQA for zero-shot transfer. Baselines: Uniform-8/16/32. Metrics: accuracy, avg frames, per-qtype. n=200 val subsets. Compute: RTX 2070 8GB.
- **4.2 Main results — NExT-QA.** Table 1 (uniform + λ sweep). Fig 1: accuracy-vs-frames **Pareto frontier** (REGEN — current fig shows only the dominated ep800 point; plot λ=0.05 ep1024 76%/14f, λ=0.1 ep1024 74.5%/8.6f, λ=0.1 ep2528 73.5%/7.1f). Fig 3: accuracy by question type.
- **4.3 Transfer — IntentQA.** Table 2: PPO λ=0.05 90.0%@17f (= Uniform-32), λ=0.1 88.5%@11f. Selector trained on NExT-QA, evaluated zero-shot.

### 5. Analysis (~0.9 pg) — the interesting part
- **Content-aware behavior.** Fig 2 (retention by qtype) + Fig 4 (stopping point by qtype): descriptive stops early (~0.5 of video), temporal keeps to the end (mean 0.55, needs whole clip), causal in between.
- **Value of early stopping.** STOP-fires vs no-STOP on descriptive: 7.1 frames @ 93.3% vs 13.0 frames @ 46.2%.
- **Reward hacking & collapse.** High λ → policy STOPs with ~0–2 frames and rides the VLM language prior (λ=0.2: 58.5%@1.5f on NExT-QA). Overtraining (ep800→ep2528) sheds frames (13.3→7.1) but loses temporal accuracy. λ is a critical, brittle knob.

### 6. Limitations & Conclusion (~0.4 pg)
Limitations: n=200 subsets (no full-val / no error bars), single 3B backbone, no EgoSchema, offline eval only, reward hacking unsolved. Conclusion: adaptive selection is controllable and content-aware and transfers; the open problem is reliably distinguishing "seen enough" from "need full context." Future: streaming benchmark, confidence-aware stopping, larger backbones/eval.

### Appendix (doesn't count toward limit)
Full hyperparameter table; NeurIPS checklist; (optional) extra per-checkpoint tables, additional qualitative examples.

---

## Evidence inventory (what maps where)
- **Tables:** Table 1 NExT-QA (`full_frame_val_{8,16,32}f.json`, `ppo_lambda0{1,05}_ep*_n200.json`, `ppo_final_n200.json` λ=0.2). Table 2 IntentQA (`uniform{8,16,32}_intentqa.json`, `ppo_lambda0*_*intentqa*.json`, `ppo_lam02_final_intentqa.json`).
- **Figures (in `results/figures/`):** fig1 accuracy-vs-frames (REGEN as Pareto), fig2 retention by qtype, fig3 accuracy by qtype, fig4 stopping distribution by qtype.
- **Analysis numbers:** STOP-fires split + per-qtype means come from `scripts/analyze_*` over the prediction JSONs.
- **Hyperparameters:** `configs/default.yaml` (λ=0.1) and `configs/lambda005.yaml` (λ=0.05).

## Open TODOs before drafting prose
- [x] Headline = PPO λ=0.05 ep1024 (76.0% @ 14.0f), framed as "matches Uniform-16 at fewer frames." Bold this Table 1 row; cite in abstract + intro.
- [ ] Regenerate Fig 1 as a λ-sweep Pareto frontier (multiple PPO points + uniform points), with the headline point highlighted.
- [ ] Confirm final title / whether to expand the SKIM acronym.
- [ ] Gather 2–3 citations each for the three Related Work buckets.
