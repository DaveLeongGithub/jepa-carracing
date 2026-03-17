"""
Train a PPO policy head on top of frozen V-JEPA 2 embeddings for CarRacing-v3.

Architecture (subprocess-isolated):
  SubprocVecEnv[CarRacing-v3 + FrameSkip] → raw 96×96×3 frames
  → VJEPAVecEnvWrapper (main process, CUDA) → 1024-dim embedding → PPO MLP → actions

Box2D runs in a subprocess (no CUDA). V-JEPA encoding runs in the main process (CUDA).
This prevents CUDA/Box2D library conflicts that cause segfaults on NVIDIA drivers.

Usage:
  # Smoke test (~5 min)
  uv run python -m src.train_jepa_ppo --total_timesteps 10000 --eval_freq 5000

  # Full training (~1-2 hrs on CUDA)
  uv run python -m src.train_jepa_ppo --total_timesteps 1000000 --output_dir runs/jepa_ppo_1M

  # Monitor
  tensorboard --logdir runs/jepa_ppo_1M/logs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from src.jepa_encoder import VJEPAEncoder
from src.vjepa_obs_wrapper import FrameSkipWrapper, VJEPAVecEnvWrapper


# Published baselines for comparison
PUBLISHED_BASELINES = {
    "Random Agent": -24.28,
    "PPO (400K steps, CNN)": 312.00,
    "DQN (CNN, skip=4)": 385.06,
    "DDQN (CNN, best)": 899.49,
    "DQN (grayscale 84²)": 921.00,
}


def select_device(requested: str) -> str:
    """Auto-select best available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def make_raw_env(frame_skip: int, seed: int):
    """Create a raw CarRacing env with frame skip (no GPU, subprocess-safe)."""
    def _init():
        env = gym.make("CarRacing-v3", render_mode=None)
        env = FrameSkipWrapper(env, frame_skip=frame_skip)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(args: argparse.Namespace) -> None:
    """Main training loop."""

    device = select_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir is None:
        output_dir = Path(f"runs/jepa_ppo_{timestamp}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ── Banner ──────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════╗")
    print("║   JEPA + PPO CarRacing-v3 Training              ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Device:           {device}")
    print(f"  Total timesteps:  {args.total_timesteps:,}")
    print(f"  Frame skip:       {args.fixed_skip}")
    print(f"  JEPA buffer:      {args.buffer_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Policy arch:      [256, 256]")
    print(f"  Eval every:       {args.eval_freq:,} steps")
    print(f"  Eval episodes:    {args.eval_episodes}")
    print(f"  Output:           {output_dir}")
    print(f"  Seed:             {args.seed}")
    print(f"  Parallel envs:    {args.n_envs}")
    print(f"  Isolation:        SubprocVecEnv (spawn)")
    print()

    # ── Save config ─────────────────────────────────────────
    config = vars(args).copy()
    config["device_selected"] = device
    config["timestamp"] = timestamp
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load encoder ────────────────────────────────────────
    print("▸ Loading V-JEPA 2 ViT-L encoder …")
    t0 = time.time()
    encoder = VJEPAEncoder(
        device=device,
        frozen=True,
        buffer_size=args.buffer_size,
    )
    print(f"  ✓ Loaded in {time.time() - t0:.1f}s → {encoder}")
    print()

    # ── Create environments ─────────────────────────────────
    # Box2D runs in subprocesses (no CUDA), V-JEPA encodes in main process
    print("▸ Creating environments …")
    raw_train_env = SubprocVecEnv(
        [make_raw_env(args.fixed_skip, args.seed + i) for i in range(args.n_envs)],
        start_method="spawn",
    )
    train_env = VJEPAVecEnvWrapper(raw_train_env, encoder)

    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    def make_eval_env():
        def _init():
            env = gym.make("CarRacing-v3", render_mode="rgb_array")
            env = FrameSkipWrapper(env, frame_skip=args.fixed_skip)
            env = Monitor(env)
            env.reset(seed=args.seed + 10000)
            return env
        return _init

    raw_eval_env = DummyVecEnv([make_eval_env()])
    raw_eval_env = VecVideoRecorder(
        raw_eval_env,
        video_folder=str(video_dir),
        record_video_trigger=lambda x: True,
        video_length=1200,
        name_prefix="v-jepa2_ppo",
    )
    eval_env = VJEPAVecEnvWrapper(raw_eval_env, encoder)

    print(f"  ✓ Train env: SubprocVecEnv[CarRacing-v3 + FrameSkip] → JEPA (obs={train_env.observation_space.shape})")
    print(f"  ✓ Eval env:  DummyVecEnv[CarRacing-v3 + FrameSkip] → VideoRecorder → JEPA")
    print(f"  ✓ Videos saved to: {video_dir}/")
    print()

    # ── Configure PPO ───────────────────────────────────────
    print("▸ Configuring PPO …")

    sb3_device = "auto" if device == "mps" else device

    if args.resume:
        print(f"  ▸ Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            device=sb3_device,
            tensorboard_log=str(log_dir),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            normalize_advantage=True,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=str(log_dir),
            device=sb3_device,
            seed=args.seed,
            verbose=1,
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  ✓ PPO policy: {total_params:,} params ({trainable_params:,} trainable)")
    print()

    # ── Callbacks ───────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix="jepa_ppo",
        verbose=0,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Train ───────────────────────────────────────────────
    print("▸ Starting training …")
    print(f"  Estimated time: {'6-8 hours' if device == 'mps' else '1-2 hours' if device == 'cuda' else '12+ hours'}")
    print()

    t_start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    t_elapsed = time.time() - t_start

    # ── Save final model ────────────────────────────────────
    final_path = output_dir / "model_final.zip"
    model.save(str(final_path))
    print()
    print(f"  ✓ Training complete in {t_elapsed/3600:.1f} hours ({t_elapsed:.0f}s)")
    print(f"  ✓ Final model saved: {final_path}")

    # ── Quick final evaluation ──────────────────────────────
    print()
    print("▸ Final evaluation (10 episodes) …")
    rewards = []
    obs = eval_env.reset()
    for ep in range(10):
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += reward[0]
            done = dones[0]
            if done:
                obs = eval_env.reset()
        rewards.append(episode_reward)
        print(f"    Ep {ep}: reward={episode_reward:.1f}")

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print()
    print("════════════════════════════════════════════════════════════")
    print("  TRAINING SUMMARY")
    print("════════════════════════════════════════════════════════════")
    print(f"  JEPA+PPO (trained):  {mean_r:.1f} ± {std_r:.1f}")
    print()
    print("  Published baselines:")
    for name, score in PUBLISHED_BASELINES.items():
        marker = " ←" if abs(score - mean_r) < 50 else ""
        print(f"    {name:30s} {score:8.1f}{marker}")
    print("════════════════════════════════════════════════════════════")

    # ── Save results ────────────────────────────────────────
    results = {
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "episode_rewards": [float(r) for r in rewards],
        "total_timesteps": args.total_timesteps,
        "training_time_sec": t_elapsed,
        "device": device,
        "config": config,
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved: {output_dir / 'final_results.json'}")

    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO policy on frozen V-JEPA 2 embeddings for CarRacing-v3"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--fixed_skip", type=int, default=4,
        help="Frame skip / action repeat (default: 4)"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=2,
        help="V-JEPA 2 rolling frame buffer size (default: 2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: runs/jepa_ppo_<timestamp>)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10_000,
        help="Evaluate every N steps (default: 10K)"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=5,
        help="Episodes per evaluation (default: 5)"
    )
    parser.add_argument(
        "--n_envs", type=int, default=1,
        help="Number of parallel training environments (default: 1)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a saved model .zip to resume training from"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
