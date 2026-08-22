from __future__ import annotations

"""Controlled low-quality simulation protocol.
"""

import random
from typing import Optional

import torch


def degrade_audio(wav: torch.Tensor, noise_std: float = 0.02, dropout_prob: float = 0.15, clip_prob: float = 0.10) -> torch.Tensor:
    x = wav.clone()
    if noise_std > 0:
        x = x + torch.randn_like(x) * float(noise_std)
    if dropout_prob > 0 and random.random() < dropout_prob:
        length = x.shape[-1]
        span = max(1, int(length * random.uniform(0.03, 0.15)))
        start = random.randint(0, max(0, length - span))
        x[..., start:start + span] = 0
    if clip_prob > 0 and random.random() < clip_prob:
        x = x.clamp(-0.65, 0.65) / 0.65
    return x.clamp(-1.0, 1.0)


def degrade_physio(x: torch.Tensor, noise_std: float = 0.03, channel_dropout_prob: float = 0.15, packet_loss_prob: float = 0.15) -> torch.Tensor:
    y = x.clone()
    if noise_std > 0:
        y = y + torch.randn_like(y) * float(noise_std)
    if y.dim() >= 2 and channel_dropout_prob > 0:
        # Works for [C,T] and [B,C,T].
        ch_dim = -2
        n_ch = y.shape[ch_dim]
        mask = torch.ones(n_ch, device=y.device, dtype=y.dtype)
        for c in range(n_ch):
            if random.random() < channel_dropout_prob:
                mask[c] = 0
        shape = [1] * y.dim()
        shape[ch_dim] = n_ch
        y = y * mask.view(*shape)
    if packet_loss_prob > 0 and random.random() < packet_loss_prob:
        length = y.shape[-1]
        span = max(1, int(length * random.uniform(0.03, 0.20)))
        start = random.randint(0, max(0, length - span))
        y[..., start:start + span] = 0
    return y


def random_missing_view(x: Optional[torch.Tensor], p: float = 0.0) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return None if random.random() < float(p) else x
