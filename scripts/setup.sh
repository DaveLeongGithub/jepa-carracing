#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================================="
echo "JEPA-CarRacing setup"
echo "=============================================="

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found."
    echo "Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if ! command -v swig >/dev/null 2>&1; then
    echo "swig is required by gymnasium[box2d] but was not found."
    echo "Install it with your system package manager, then rerun this script."
    exit 1
fi

echo
echo "▸ Syncing project dependencies"
uv sync --frozen

echo
echo "▸ Smoke-checking PyTorch device detection"
uv run python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
"

echo
echo "▸ Smoke-checking Gymnasium CarRacing-v3"
uv run python -c "
import gymnasium as gym
env = gym.make('CarRacing-v3', render_mode='rgb_array')
obs, info = env.reset()
print(f'  CarRacing-v3 obs shape: {obs.shape}')
env.close()
print('  ✓ Gymnasium CarRacing-v3 works')
"

echo
echo "▸ Smoke-checking the evaluation entrypoint"
uv run python -m src.eval_jepa_ppo --help >/dev/null
echo "  ✓ Evaluation CLI is wired up"

echo
echo "=============================================="
echo "Setup complete."
echo
echo "Next steps:"
echo "  uv run python -m src.eval_jepa_ppo \\"
echo "    --model_path proof/runs/jepa_ppo_resume_300k/best_model.zip \\"
echo "    --episodes 5 \\"
echo "    --device auto"
echo "=============================================="
