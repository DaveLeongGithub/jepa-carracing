"""
V-JEPA 2 Encoder Wrapper for Gymnasium CarRacing-v3.

Loads the pretrained V-JEPA 2 ViT-L model and extracts spatial embeddings
from raw 96×96×3 RGB frames. Handles the frame-stacking required by V-JEPA 2
(expects multi-frame "video" input) by maintaining a rolling frame buffer.

Architecture:
    Raw frame (96×96×3) → resize to 256×256 → stack N frames →
    V-JEPA 2 ViT-L encoder → [batch, num_patches, 1024] →
    global average pool → [batch, 1024] embedding vector

Reference:
    - V-JEPA 2: arXiv:2506.09985 (Meta FAIR)
    - HuggingFace: facebook/vjepa2-vitl-fpc64-256
    - ViT-L hidden_size = 1024
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_HIDDEN_DIM = 1024          # ViT-L hidden size
VJEPA2_INPUT_SIZE = 256           # expected spatial resolution
VJEPA2_FRAMES_PER_CLIP = 64      # model was trained on 64-frame clips
FRAME_BUFFER_SIZE = 2             # use 2 recent frames (was 16 — 8x speedup)
CARRACING_OBS_SIZE = 96           # CarRacing-v3 native obs is 96×96×3


class VJEPAEncoder(nn.Module):
    """
    Wraps V-JEPA 2 ViT-L as a frozen (or fine-tunable) feature extractor
    for single-frame observations from Gymnasium environments.

    The encoder maintains a rolling buffer of the last N frames, stacks them
    into a pseudo-video tensor, and feeds it through V-JEPA 2 to produce a
    single 1024-dim embedding vector per observation.

    Args:
        device: Target device ("mps", "cuda", or "cpu")
        frozen: If True, all V-JEPA 2 parameters are frozen (no grad)
        buffer_size: Number of recent frames to stack (default 2)
        dtype: Model precision (default float32 for MPS compatibility)
    """

    def __init__(
        self,
        device: str = "auto",
        frozen: bool = True,
        buffer_size: int = FRAME_BUFFER_SIZE,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Resolve "auto" to best available device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.frozen = frozen
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.hidden_dim = VJEPA2_HIDDEN_DIM

        # ── Load V-JEPA 2 model and processor ──
        from transformers import AutoVideoProcessor, AutoModel

        self.processor = AutoVideoProcessor.from_pretrained(VJEPA2_REPO)
        try:
            self.model = AutoModel.from_pretrained(
                VJEPA2_REPO,
                dtype=dtype,
                attn_implementation="eager",
            ).to(self.device)
        except TypeError:
            self.model = AutoModel.from_pretrained(
                VJEPA2_REPO,
                torch_dtype=dtype,
                attn_implementation="eager",
            ).to(self.device)

        if frozen:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # ── FP16 inference: ~1.5-2× speedup, no quality loss for RL ──
        if self.device.type == "cuda" and dtype == torch.float32:
            self.model = self.model.to(dtype=torch.float16)
            self.dtype = torch.float16

        # NOTE: torch.compile not compatible with V-JEPA 2 (data-dependent ops
        # in predictor embeddings cause Unsupported error with fullgraph=True)

        # ── Rolling frame buffer ──
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    def reset_buffer(self) -> None:
        """Clear the frame buffer (call on env.reset())."""
        self.frame_buffer.clear()

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of raw uint8 frames into a V-JEPA 2 compatible tensor.

        The processor handles:
          - Resize to 256×256
          - Normalize to model's expected range
          - Convert to [B, T, C, H, W] tensor

        Args:
            frames: List of HWC uint8 numpy arrays

        Returns:
            Tensor of shape [1, T, C, 256, 256] on self.device
        """
        # Stack frames → [T, H, W, C] then convert to [T, C, H, W]
        video_np = np.stack(frames, axis=0)                    # [T, 96, 96, 3]
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # [T, 3, 96, 96]

        # Use the V-JEPA 2 processor for proper normalization and resize
        inputs = self.processor(video_tensor, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(
            device=self.device, dtype=self.dtype
        )
        return pixel_values

    @torch.no_grad()
    def encode_frame(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode a single CarRacing observation into a 1024-dim embedding.

        Appends the frame to the rolling buffer, stacks buffered frames
        into a pseudo-video, and runs V-JEPA 2 forward pass.

        Args:
            obs: Raw observation from env.step(), shape (96, 96, 3), uint8

        Returns:
            Embedding tensor of shape [1024] on self.device
        """
        # Add frame to buffer
        self.frame_buffer.append(obs.copy())

        # If buffer not full yet, pad by repeating the first frame
        frames = list(self.frame_buffer)
        while len(frames) < self.buffer_size:
            frames.insert(0, frames[0])

        # Preprocess and encode
        pixel_values = self._preprocess_frames(frames)

        # Ensure pixel_values is on the correct device and dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        # Clear MPS cache periodically (MPS needs this; CUDA does not —
        # calling empty_cache() every frame on CUDA destabilises the allocator
        # and can trigger segfaults via nvidia_uvm)
        if self.device.type == "mps":
            torch.mps.empty_cache()

        if self.frozen:
            outputs = self.model(pixel_values_videos=pixel_values)
        else:
            # Allow gradients through encoder for fine-tuning
            with torch.enable_grad():
                outputs = self.model(pixel_values_videos=pixel_values)

        # outputs.last_hidden_state: [batch, num_patches, 1024]
        hidden = outputs.last_hidden_state

        # Global average pool across patches → [batch, 1024]
        embedding = hidden.mean(dim=1)

        return embedding.squeeze(0)  # [1024]

    @torch.no_grad()
    def encode_batch(self, obs_batch: np.ndarray) -> torch.Tensor:
        """
        Encode a batch of observations (no temporal context).
        Useful for offline evaluation of stored frames.

        Args:
            obs_batch: Shape [B, 96, 96, 3], uint8

        Returns:
            Embeddings tensor of shape [B, 1024]
        """
        embeddings = []
        for obs in obs_batch:
            emb = self.encode_frame(obs)
            embeddings.append(emb)
        return torch.stack(embeddings, dim=0)

    @property
    def output_dim(self) -> int:
        """Embedding dimensionality."""
        return self.hidden_dim

    def __repr__(self) -> str:
        mode = "frozen" if self.frozen else "fine-tunable"
        return (
            f"VJEPAEncoder(model={VJEPA2_REPO}, "
            f"hidden_dim={self.hidden_dim}, "
            f"buffer_size={self.buffer_size}, "
            f"mode={mode}, device={self.device})"
        )
