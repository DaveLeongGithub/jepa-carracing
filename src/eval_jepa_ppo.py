"""
Evaluate a trained JEPA+PPO model on CarRacing-v3 and compare against baselines.

Usage:
  uv run python -m src.eval_jepa_ppo --model_path runs/jepa_ppo_1M/model_best.zip --episodes 50
  uv run python -m src.eval_jepa_ppo --model_path runs/jepa_ppo_1M/model_best.zip --render
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from stable_baselines3 import PPO

from src.jepa_encoder import VJEPAEncoder
from src.vjepa_obs_wrapper import make_jepa_env


PUBLISHED_BASELINES = {
    "Random Agent":           {"mean": -24.28, "std": 7.13},
    "PPO (400K, CNN)":        {"mean": 312.00, "std": 194.00},
    "DQN (CNN, skip=4)":      {"mean": 385.06, "std": 251.37},
    "DDQN (CNN, best)":       {"mean": 899.49, "std": 98.25},
    "DQN (grayscale 84²)":    {"mean": 921.00, "std": 50.00},
}


def resolve_device(requested: str) -> str:
    """Resolve a requested device to one that is actually available."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("  ! Requested CUDA, but CUDA is not available on this machine. Falling back to auto.")
        return resolve_device("auto")

    if requested == "mps" and not torch.backends.mps.is_available():
        print("  ! Requested MPS, but MPS is not available on this machine. Falling back to auto.")
        return resolve_device("auto")

    return requested


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation episodes and produce comparison report."""

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load config from training run if available
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        device = train_config.get("device_selected", "auto")
        frame_skip = train_config.get("fixed_skip", 4)
        buffer_size = train_config.get("buffer_size", 2)
    else:
        device = args.device
        frame_skip = args.fixed_skip
        buffer_size = args.buffer_size

    if args.device != "auto":
        device = args.device

    device = resolve_device(device)

    print("╔══════════════════════════════════════════════════╗")
    print("║   JEPA+PPO Evaluation — CarRacing-v3            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Model:      {model_path}")
    print(f"  Device:     {device}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Render:     {args.render}")
    print()

    # ── Load encoder ────────────────────────────────────────
    print("▸ Loading V-JEPA 2 encoder …")
    encoder = VJEPAEncoder(device=device, frozen=True, buffer_size=buffer_size)

    # ── Load model ──────────────────────────────────────────
    print("▸ Loading trained PPO model …")
    model = PPO.load(str(model_path))
    print(f"  ✓ Loaded")
    print()

    # ── Create environment ──────────────────────────────────
    render_mode = "human" if args.render else None
    env = make_jepa_env(
        encoder=encoder,
        frame_skip=frame_skip,
        seed=args.seed,
        render_mode=render_mode,
    )

    # ── Run episodes ────────────────────────────────────────
    print(f"▸ Running {args.episodes} evaluation episodes …")
    episode_results = []

    for ep in tqdm(range(args.episodes), desc="Eval"):
        obs, info = env.reset(seed=args.seed + ep)
        episode_reward = 0.0
        steps = 0
        t0 = time.time()

        while True:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        wall_time = time.time() - t0
        episode_results.append({
            "episode": ep,
            "reward": float(episode_reward),
            "steps": steps,
            "wall_time_sec": wall_time,
        })
        tqdm.write(f"    Ep {ep}: reward={episode_reward:.1f}, steps={steps}, time={wall_time:.1f}s")

    env.close()

    # ── Compute statistics ──────────────────────────────────
    rewards = [r["reward"] for r in episode_results]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    median_r = np.median(rewards)

    print()
    print("════════════════════════════════════════════════════════════")
    print("  EVALUATION RESULTS")
    print("════════════════════════════════════════════════════════════")
    print(f"  JEPA+PPO:  {mean_r:.1f} ± {std_r:.1f}  (min={min_r:.1f}, max={max_r:.1f}, median={median_r:.1f})")
    print()
    print("  Published baselines:")
    for name, b in PUBLISHED_BASELINES.items():
        delta = mean_r - b["mean"]
        sign = "+" if delta > 0 else ""
        print(f"    {name:25s}  {b['mean']:8.1f} ± {b['std']:6.1f}  ({sign}{delta:.1f})")
    print("════════════════════════════════════════════════════════════")

    # ── Save results ────────────────────────────────────────
    output_dir = model_path.parent
    results = {
        "model_path": str(model_path),
        "episodes": args.episodes,
        "device": device,
        "frame_skip": frame_skip,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "min_reward": float(min_r),
        "max_reward": float(max_r),
        "median_reward": float(median_r),
        "episode_results": episode_results,
        "published_baselines": PUBLISHED_BASELINES,
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved: {results_path}")

    # ── Plot comparison ─────────────────────────────────────
    if args.episodes >= 5:
        _plot_comparison(rewards, output_dir)


def _plot_comparison(rewards: list[float], output_dir: Path) -> None:
    """Generate a bar chart comparing JEPA+PPO against baselines."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(PUBLISHED_BASELINES.keys()) + ["JEPA+PPO (ours)"]
    means = [b["mean"] for b in PUBLISHED_BASELINES.values()] + [np.mean(rewards)]
    stds = [b["std"] for b in PUBLISHED_BASELINES.values()] + [np.std(rewards)]

    colors = ["#cccccc"] * len(PUBLISHED_BASELINES) + ["#2196F3"]
    bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Mean Episode Reward")
    ax.set_title("CarRacing-v3: JEPA+PPO vs Published Baselines")
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    for bar, mean in zip(bars, means):
        ax.text(
            max(mean + 20, 30), bar.get_y() + bar.get_height() / 2,
            f"{mean:.0f}", va="center", fontsize=9,
        )

    plt.tight_layout()
    plot_path = output_dir / "eval_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained JEPA+PPO model on CarRacing-v3"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to saved SB3 model (.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument(
        "--fixed_skip", type=int, default=4,
    )
    parser.add_argument(
        "--buffer_size", type=int, default=2,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render episodes visually"
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
