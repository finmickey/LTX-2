"""Per-prompt per-channel z-score normalization for ParetoNFT multi-reward training.

Adapted from ParetoNFT/flow_grpo/stat_tracking.py.

Uses current-batch stats only (no history, no class state needed).
"""

import torch
from torch import Tensor


def compute_pareto_advantages(
    reward_vectors: Tensor,
    prompt_indices: list[int],
    adv_clip_max: float,
) -> Tensor:
    """Compute per-channel z-scored advantages for multi-reward training.

    Stage 1: Per-prompt per-channel z-score — for each prompt group (K repeats),
    for each channel r: (reward - group_mean) / (group_std + 1e-4).

    Stage 2: Batch-wide per-channel z-score — for each channel r across the
    entire batch: (adv - batch_mean) / (batch_std + 1e-4).

    Args:
        reward_vectors: (N, R) reward values per sample per objective.
        prompt_indices: List of length N, identifying which prompt group each
            sample belongs to (e.g. [0,0,0,0, 1,1,1,1] for 2 prompts with K=4).
        adv_clip_max: Not used for clipping here (clipping happens in the loss),
            but kept in signature for consistency.

    Returns:
        (N, R) tensor of raw z-scored advantages (NOT mapped to [0,1]).
    """
    N, R = reward_vectors.shape
    adv_per_channel = torch.zeros_like(reward_vectors)

    # Stage 1: Per-prompt per-channel z-score
    prompt_ids = torch.tensor(prompt_indices, device=reward_vectors.device)
    unique_prompts = prompt_ids.unique()

    for pid in unique_prompts:
        mask = prompt_ids == pid
        group_rewards = reward_vectors[mask]  # (K, R)

        for r in range(R):
            channel = group_rewards[:, r]  # (K,)
            mean = channel.mean()
            std = channel.std() + 1e-4
            adv_per_channel[mask, r] = (reward_vectors[mask, r] - mean) / std

    # Stage 2: Batch-wide per-channel z-score
    final_adv = torch.zeros_like(adv_per_channel)
    for r in range(R):
        channel = adv_per_channel[:, r]  # (N,)
        final_adv[:, r] = (channel - channel.mean()) / (channel.std() + 1e-4)

    return final_adv
