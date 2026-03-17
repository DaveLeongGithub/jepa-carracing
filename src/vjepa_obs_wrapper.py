"""
Gymnasium wrapper that replaces raw pixel observations with V-JEPA 2 embeddings.

Two modes:
  1. Single-process (VJEPAObsWrapper): encoder inside the env wrapper.
  2. Subprocess-isolated (FrameSkipWrapper + VJEPAVecEnvWrapper):
     Box2D runs in a subprocess, V-JEPA encoding in the main process.
     This prevents CUDA/Box2D library conflicts.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common.vec_env import VecEnvWrapper

from src.jepa_encoder import VJEPAEncoder, VJEPA2_HIDDEN_DIM


# ---------------------------------------------------------------------------
# Subprocess-safe wrappers (Box2D in subprocess, V-JEPA in main process)
# ---------------------------------------------------------------------------

class FrameSkipWrapper(gym.Wrapper):
    """Repeats each action for frame_skip steps, returns last frame.

    Used inside SubprocVecEnv — no CUDA, no GPU, just raw frames.
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class VJEPAVecEnvWrapper(VecEnvWrapper):
    """Wraps a VecEnv to encode raw pixel observations with V-JEPA 2.

    Runs in the main process with CUDA. The underlying VecEnv (SubprocVecEnv)
    runs Box2D in separate processes, completely isolated from CUDA.

    Maintains per-env frame buffers so temporal context doesn't leak between envs.
    """

    def __init__(self, venv, encoder: VJEPAEncoder):
        self.encoder = encoder
        self._n_envs = venv.num_envs
        # Per-env frame buffers (avoids mixing temporal context across envs)
        self._buffers = [
            deque(maxlen=encoder.buffer_size) for _ in range(self._n_envs)
        ]
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(VJEPA2_HIDDEN_DIM,),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=obs_space)

    def _encode_single(self, frame: np.ndarray, env_idx: int) -> np.ndarray:
        """Encode a single frame using the correct per-env buffer."""
        # Swap in this env's buffer
        self.encoder.frame_buffer = self._buffers[env_idx]
        with torch.no_grad():
            emb = self.encoder.encode_frame(frame)
        return emb.cpu().numpy().astype(np.float32)

    def _encode_obs(self, obs: np.ndarray) -> np.ndarray:
        """Encode a batch of raw frames [n_envs, H, W, C] to [n_envs, 1024]."""
        embeddings = []
        for i in range(obs.shape[0]):
            embeddings.append(self._encode_single(obs[i], i))
        return np.stack(embeddings, axis=0)

    def reset(self):
        for buf in self._buffers:
            buf.clear()
        obs = self.venv.reset()
        return self._encode_obs(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Encode terminal observations (SB3 uses these for bootstrapping)
        for i, done in enumerate(dones):
            if done and "terminal_observation" in infos[i]:
                infos[i]["terminal_observation"] = self._encode_single(
                    infos[i]["terminal_observation"], i
                )
        # Reset buffers for done envs (new episodes start fresh)
        for i, done in enumerate(dones):
            if done:
                self._buffers[i].clear()
        return self._encode_obs(obs), rewards, dones, infos


# ---------------------------------------------------------------------------
# Single-process wrapper (legacy, for non-CUDA or when isolation isn't needed)
# ---------------------------------------------------------------------------

class VJEPAObsWrapper(gym.Wrapper):
    """Converts CarRacing-v3 pixel observations to V-JEPA 2 embeddings.

    Observation space: Box(shape=(1024,), dtype=float32)
    Action space: unchanged (continuous [steering, gas, brake])
    """

    def __init__(
        self,
        env: gym.Env,
        encoder: VJEPAEncoder,
        frame_skip: int = 4,
    ):
        super().__init__(env)
        self.encoder = encoder
        self.frame_skip = frame_skip

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(VJEPA2_HIDDEN_DIM,),
            dtype=np.float32,
        )

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            embedding = self.encoder.encode_frame(obs)
        return embedding.cpu().numpy().astype(np.float32)

    def reset(self, **kwargs):
        self.encoder.reset_buffer()
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return self._encode(obs), total_reward, terminated, truncated, info


def make_jepa_env(
    encoder: VJEPAEncoder,
    frame_skip: int = 4,
    seed: int | None = None,
    render_mode: str | None = None,
) -> VJEPAObsWrapper:
    """Factory for single-process mode (non-CUDA or MPS)."""
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return VJEPAObsWrapper(env, encoder=encoder, frame_skip=frame_skip)
