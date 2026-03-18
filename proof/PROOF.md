# Proof Guide

This folder contains the minimum files needed to inspect and verify the public claim for **Substack Article 1 - JEPA**.

## What Is Included

- `runs/jepa_ppo_20260312_085953/`
  - `config.json`
  - `eval_logs/evaluations.npz`
- `runs/jepa_ppo_resume_300k/`
  - `config.json`
  - `eval_logs/evaluations.npz`
  - `best_model.zip`
  - `model_final.zip`
- `logs/training_400k_v2.log`
- `logs/training_resume_300k.log`

## Why Two Runs Exist

The article learning curve comes from two training phases:

1. Initial training run: `0 -> 300K` steps
2. Resumed training run: `300K -> 400K` steps

The best mean evaluation came from the resumed run at **350K total steps**:

- mean reward: **763.39**
- five recorded eval episodes: `806.47, 773.38, 647.25, 681.65, 908.20`
- peak episode: **908.20**

The final 400K checkpoint plateaued at:

- mean reward: **756.37**

## Learning Curve Used For The Article

| Total Steps | Mean Eval Reward |
|---|---:|
| 50K | 15.17 |
| 100K | 202.47 |
| 150K | 122.69 |
| 200K | 624.38 |
| 250K | 759.18 |
| 300K | 698.08 |
| 350K | 763.39 |
| 400K | 756.37 |

Those values are also exported in `carracing_eval_curve.csv`.

## Quick Verification

From the repo root:

```bash
uv sync --frozen
uv run python -m src.eval_jepa_ppo \
  --model_path "proof/runs/jepa_ppo_resume_300k/best_model.zip" \
  --episodes 5 \
  --device auto
```

Expected target:

- mean reward around `763`
- max episode around `908`

To inspect the exact stored evaluation traces without rerunning:

```bash
python - <<'PY'
import numpy as np
for path in [
    "proof/runs/jepa_ppo_20260312_085953/eval_logs/evaluations.npz",
    "proof/runs/jepa_ppo_resume_300k/eval_logs/evaluations.npz",
]:
    data = np.load(path)
    print(path)
    print("timesteps:", data["timesteps"])
    print("means:", data["results"].mean(axis=1))
    print("maxes:", data["results"].max(axis=1))
PY
```

## Important Notes

- The proof bundle is intentionally minimal. It includes the evaluation traces, logs, and saved checkpoints relevant to the article claim, not every experiment in the private repo.
- The recorded training configs use `frame_skip=4` and `buffer_size=2`.
- The 350K best checkpoint is `runs/jepa_ppo_resume_300k/best_model.zip`.
- The 400K final checkpoint is `runs/jepa_ppo_resume_300k/model_final.zip`.
