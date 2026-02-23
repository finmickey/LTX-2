"""Per-prompt per-channel z-score normalization for ParetoNFT multi-reward training.

Adapted from ParetoControl/ParetoNFT/flow_grpo/stat_tracking.py.

Uses current-batch stats only (no history, no class state needed).

Key design choice (matching ParetoControl): In per-objective mode, we skip
the batch-wide z-score after per-group normalization. Per-prompt normalization
already handles scale differences, and re-normalizing per-channel destroys
the natural relative scaling that preferences need to create ordered Pareto fronts.
"""

import torch
from torch import Tensor


def compute_pareto_advantages(
    reward_vectors: Tensor,
    group_keys: list,
) -> Tensor:
    """Compute per-channel z-scored advantages for multi-reward training.

    Per-group per-channel z-score: for each group (prompt or prompt+pref_slot),
    for each channel r: (reward - group_mean) / (group_std + 1e-4).

    No batch-wide normalization is applied (matching ParetoControl). Per-group
    normalization already handles scale differences, and re-normalizing destroys
    the natural relative scaling that preferences need for ordered Pareto fronts.

    Args:
        reward_vectors: (N, R) reward values per sample per objective.
        group_keys: List of length N with hashable group identifiers. Samples
            with the same key are grouped together for normalization.
            Examples: [0,0,0,0, 1,1,1,1] for 2 prompts with K=4,
            or ["p0__pref0"]*4 + ["p0__pref1"]*4 for subgroups.

    Returns:
        (N, R) tensor of per-group z-scored advantages (NOT mapped to [0,1]).
    """
    N, R = reward_vectors.shape
    adv_per_channel = torch.zeros_like(reward_vectors)

    # Build group index mapping
    key_to_indices: dict = {}
    for i, key in enumerate(group_keys):
        key_to_indices.setdefault(key, []).append(i)

    # Per-group per-channel z-score
    for indices in key_to_indices.values():
        idx = torch.tensor(indices, device=reward_vectors.device)
        group_rewards = reward_vectors[idx]  # (K, R)

        for r in range(R):
            channel = group_rewards[:, r]  # (K,)
            mean = channel.mean()
            std = channel.std() + 1e-4
            adv_per_channel[idx, r] = (reward_vectors[idx, r] - mean) / std

    return adv_per_channel
