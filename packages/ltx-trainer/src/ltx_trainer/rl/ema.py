"""Exponential Moving Average (EMA) for trainable parameters.

Ported from DiffusionNFT reference implementation. EMA weights are used for
validation video generation and checkpoint saving.
"""

from __future__ import annotations

import torch
from torch import Tensor


class EMAWrapper:
    """Maintains an exponential moving average of trainable parameters.

    Uses dynamic warmup: effective_decay = min((1+step)/(10+step), decay).
    """

    def __init__(self, parameters: list[Tensor], decay: float = 0.9) -> None:
        self.ema_params: list[Tensor] = [p.clone().detach() for p in parameters]
        self.decay = decay
        self._temp: list[Tensor] | None = None

    def get_current_decay(self, step: int) -> float:
        """Compute effective decay with warmup schedule."""
        return min((1 + step) / (10 + step), self.decay)

    @torch.no_grad()
    def step(self, parameters: list[Tensor], step: int) -> None:
        """Update EMA parameters with current trainable parameters."""
        one_minus_decay = 1 - self.get_current_decay(step)
        for ema_p, p in zip(self.ema_params, parameters):
            if p.requires_grad:
                ema_p.add_(one_minus_decay * (p - ema_p))

    def copy_ema_to(self, parameters: list[Tensor]) -> None:
        """Swap EMA weights into model, storing originals for restore."""
        self._temp = [p.data.clone() for p in parameters]
        for ema_p, p in zip(self.ema_params, parameters):
            p.data.copy_(ema_p.data)

    def restore(self, parameters: list[Tensor]) -> None:
        """Restore original weights after EMA swap."""
        assert self._temp is not None, "restore() called without prior copy_ema_to()"
        for temp_p, p in zip(self._temp, parameters):
            p.data.copy_(temp_p)
        self._temp = None

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {"ema_params": [p.clone() for p in self.ema_params], "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from checkpoint."""
        for ema_p, saved_p in zip(self.ema_params, state["ema_params"]):
            ema_p.data.copy_(saved_p.data)
        self.decay = state["decay"]
