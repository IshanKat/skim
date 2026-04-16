# cs-291a-project

**Uncertainty-Driven Adaptive Frame Selection for Video QA via PPO.**
Course project for CS 291A (Spring 2026). See
[the proposal](CS 291A Project Proposal.docx.pdf) for the research pitch and
`~/.claude/plans/given-the-project-proposal-twinkling-prism.md` for the
implementation plan.

## Layout

- `src/afs/` — package: `vlm/`, `selector/`, `env/`, `data/`, `baselines/`,
  `training/`, `eval/`, `utils/`
- `configs/` — YAML run configs
- `scripts/` — CLI entrypoints (`check_nextqa.py`, `check_vlm_load.py`, …)
- `tests/` — pytest suite
- `data/` — datasets (gitignored; see [data/README.md](data/README.md))

## Setup

### CPU / dev (macOS, no GPU)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]   # skip the [gpu] extras on macOS
```

### GPU training box (Linux + CUDA)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,gpu]
```

## Phase 0 checks

```bash
# Data loader unit tests (no dataset download needed):
pytest tests/test_data.py -v

# Once data/nextqa/ is populated (see data/README.md):
python scripts/check_nextqa.py --root data/nextqa --split val

# Once torch + transformers installed on GPU box:
python scripts/check_vlm_load.py
```
