"""Reward functions for RL training.

Each reward function scores a video tensor and returns a scalar reward.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(self, video: Tensor) -> float:
        """Score a video.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range.

        Returns:
            Scalar reward value.
        """


class RednessReward(RewardFunction):
    """Reward that measures how red the video is.

    Computes mean(R - max(G, B)) across all pixels and frames.
    Higher values indicate redder videos.
    """

    def compute(self, video: Tensor) -> float:
        r, g, b = video[0], video[1], video[2]
        return (r - torch.max(g, b)).mean().item()


def get_reward_function(name: str) -> RewardFunction:
    """Factory function to get a reward function by name.

    Args:
        name: Name of the reward function.

    Returns:
        An instance of the requested reward function.
    """
    reward_functions = {
        "redness": RednessReward,
    }

    if name not in reward_functions:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(reward_functions.keys())}")

    return reward_functions[name]()
