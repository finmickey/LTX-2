"""Reward functions for RL training.

Each reward function scores a video tensor and returns a scalar reward.
"""

import logging
from abc import ABC, abstractmethod

import torch
from torch import Tensor

logger = logging.getLogger(__name__)
_logged_shapes = False


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


class BluenessReward(RewardFunction):
    """Reward that measures how blue the video is.

    Computes mean(B - max(R, G)) across all pixels and frames.
    Higher values indicate bluer videos.
    """

    def compute(self, video: Tensor) -> float:
        r, g, b = video[0], video[1], video[2]
        return (b - torch.max(r, g)).mean().item()


class HorizontalEdgeReward(RewardFunction):
    """Reward that measures horizontal edge strength.

    Computes mean absolute difference between adjacent rows.
    Uniform frames score ~0, horizontally-striped frames score high.
    """

    def compute(self, video: Tensor) -> float:
        return (video[:, :, 1:, :] - video[:, :, :-1, :]).abs().mean().item()


class HorizontalStripeReward(RewardFunction):
    """Reward that measures color contrast between horizontal bands.

    Groups rows into bands of 4, computes mean color per band, then rewards
    high contrast between adjacent bands. Penalizes adjacent bands that share
    the same color — encourages bold, visible stripes.
    """

    BAND_HEIGHT = 4

    def compute(self, video: Tensor) -> float:
        # video: [C, F, H, W]
        _C, _F, H, _W = video.shape
        n_bands = H // self.BAND_HEIGHT
        trimmed = video[:, :, : n_bands * self.BAND_HEIGHT, :]
        # [C, F, n_bands, band_height, W] -> mean over band_height & W -> [C, F, n_bands]
        bands = trimmed.reshape(_C, _F, n_bands, self.BAND_HEIGHT, _W).mean(dim=(3, 4))
        # Mean absolute diff between adjacent bands
        return (bands[:, :, 1:] - bands[:, :, :-1]).abs().mean().item()


class UniformFrameReward(RewardFunction):
    """Reward that measures how uniform each frame's color is.

    Each frame should be a single solid color. Computes negative mean absolute
    deviation of pixels from the per-frame mean color. Perfectly uniform
    frames score 0; noisy/textured frames score negative.
    """

    def compute(self, video: Tensor) -> float:
        global _logged_shapes
        # video: [C, F, H, W]
        if not _logged_shapes:
            logger.info(
                f"[UniformFrameReward] video: shape={list(video.shape)} "
                f"dtype={video.dtype} range=[{video.min().item():.4f}, {video.max().item():.4f}]"
            )
        frame_mean = video.mean(dim=(2, 3), keepdim=True)  # [C, F, 1, 1]
        deviation = (video - frame_mean).abs().mean()
        reward = -deviation.item()
        if not _logged_shapes:
            logger.info(
                f"[UniformFrameReward] frame_mean: shape={list(frame_mean.shape)} "
                f"range=[{frame_mean.min().item():.4f}, {frame_mean.max().item():.4f}] "
                f"deviation={deviation.item():.4f} reward={reward:.4f}"
            )
        return reward


class RedOrBlueReward(RewardFunction):
    """Per-frame max(redness, blueness), averaged across frames.

    Each frame is scored for being either red OR blue (whichever is stronger).
    Black scores 0, pure red/blue scores +1, alternating red/blue scores +1.
    """

    def compute(self, video: Tensor) -> float:
        r, g, b = video[0], video[1], video[2]  # [F, H, W]
        redness = (r - torch.max(g, b)).mean(dim=(1, 2))  # [F]
        blueness = (b - torch.max(r, g)).mean(dim=(1, 2))  # [F]
        per_frame = torch.max(redness, blueness)  # [F]
        return per_frame.mean().item()


class ChangingColorsReward(RewardFunction):
    """Reward that measures how much each frame's color differs from recent frames.

    For each frame, computes the minimum L1 color distance to any of the
    previous `LOOKBACK` frames. This is a continuous reward — every bit of
    color change contributes, giving meaningful variance across samples.

    Penalizes degenerate luminance (too dark or too bright) to prevent
    collapse to black or white.

    Reward = mean_color_change - extremity_penalty
    - mean_color_change: mean over frames of min-L1-to-recent [0, ~3.0]
    - extremity_penalty: how far mean luminance is from 0.5 midpoint [0, 0.5]
      (black=0.4 penalty, white=0.4 penalty, mid-gray=0 penalty)
    """

    LOOKBACK = 5

    def compute(self, video: Tensor) -> float:
        global _logged_shapes
        # video: [C, F, H, W] — get mean color per frame: [C, F]
        frame_colors = video.mean(dim=(2, 3))
        num_frames = frame_colors.shape[1]
        if num_frames < 2:
            return 0.0

        if not _logged_shapes:
            logger.info(
                f"[ChangingColorsReward] frame_colors: shape={list(frame_colors.shape)} "
                f"range=[{frame_colors.min().item():.4f}, {frame_colors.max().item():.4f}] "
                f"num_frames={num_frames}"
            )
            for t in range(min(5, num_frames)):
                rgb = frame_colors[:, t].tolist()
                logger.info(f"  frame[{t}] RGB=[{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]")

        # Color change: min L1 distance to recent frames (continuous, no threshold)
        total_change = 0.0
        for t in range(1, num_frames):
            lb = min(t, self.LOOKBACK)
            prev = frame_colors[:, t - lb : t]  # [C, lb]
            curr = frame_colors[:, t : t + 1]  # [C, 1]
            diffs = (curr - prev).abs().sum(dim=0)  # [lb] L1 per prev frame
            total_change += diffs.min().item()
        change_reward = total_change / (num_frames - 1)

        # Extremity penalty: penalize per-frame luminance far from 0.5
        # Targets [0.1, 0.9] as "acceptable" — outside that range gets penalized
        luminance = frame_colors.mean(dim=0)  # [F] mean across RGB
        too_dark = (0.1 - luminance).clamp(min=0)   # penalty for < 0.1
        too_bright = (luminance - 0.9).clamp(min=0)  # penalty for > 0.9
        extremity_penalty = (too_dark + too_bright).mean().item()

        result = change_reward - extremity_penalty
        if not _logged_shapes:
            logger.info(
                f"[ChangingColorsReward] change={change_reward:.4f} "
                f"extremity_penalty={extremity_penalty:.4f} total={result:.4f}"
            )
            _logged_shapes = True

        return result


def get_reward_function(name: str) -> RewardFunction:
    """Factory function to get a reward function by name.

    Args:
        name: Name of the reward function.

    Returns:
        An instance of the requested reward function.
    """
    reward_functions = {
        "redness": RednessReward,
        "blueness": BluenessReward,
        "horizontal_edges": HorizontalEdgeReward,
        "horizontal_stripes": HorizontalStripeReward,
        "uniform_frame": UniformFrameReward,
        "changing_colors": ChangingColorsReward,
        "red_or_blue": RedOrBlueReward,
    }

    if name not in reward_functions:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(reward_functions.keys())}")

    return reward_functions[name]()
