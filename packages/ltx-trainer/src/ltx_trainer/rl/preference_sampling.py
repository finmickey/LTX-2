"""Structured preference sampling for ParetoNFT multi-reward training.

Adapted from ParetoControl/ParetoNFT/scripts/train_cond_nft_sd3.py.

Supports:
- Single preference per prompt (all K repeats share the same preference)
- Multiple preferences per prompt (K distinct preferences cycled across repeats,
  creating (prompt, pref_slot) sub-groups for GDPO normalization)
"""

import hashlib

import numpy as np
import torch
from torch import Tensor


def _stable_prompt_hash_u32(prompt: str) -> int:
    """SHA-256 based hash, stable across processes/runs (unlike Python hash())."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sample_structured_preference(num_rewards: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a preference vector using structured vertex/edge/interior sampling.

    Matching ParetoControl:
    - R=1: trivial [1.0]
    - R=2: 20% one-hot corners + 80% Dirichlet([1,1])
    - R>=3: 50% vertex (one-hot), 35% edge (pair), 15% interior (full Dirichlet)
    """
    if num_rewards == 1:
        return np.ones(1, dtype=np.float32)

    if num_rewards == 2:
        if rng.random() < 0.2:
            # Corner: one-hot
            pref = np.zeros(2, dtype=np.float32)
            pref[rng.integers(0, 2)] = 1.0
            return pref
        return rng.dirichlet([1.0, 1.0]).astype(np.float32)

    roll = rng.random()

    if roll < 0.50:
        # VERTEX: one-hot -- optimize exactly one objective
        pref = np.zeros(num_rewards, dtype=np.float32)
        pref[rng.integers(0, num_rewards)] = 1.0
    elif roll < 0.85:
        # EDGE: pair of objectives -- learn pairwise trade-offs
        pref = np.zeros(num_rewards, dtype=np.float32)
        pair = rng.choice(num_rewards, size=2, replace=False)
        w = rng.dirichlet([1.0, 1.0]).astype(np.float32)
        pref[pair[0]] = w[0]
        pref[pair[1]] = w[1]
    else:
        # INTERIOR: all objectives -- rare, for smooth interpolation
        pref = rng.dirichlet([1.0] * num_rewards).astype(np.float32)

    return pref


def sample_preference_for_prompt(
    prompt_text: str,
    num_rewards: int,
    base_seed: int,
    epoch: int,
    prompt_offset: int,
) -> Tensor:
    """Sample a deterministic preference vector for a prompt.

    Deterministic via seed = (base_seed + hash(prompt) + epoch * 100000 + prompt_offset) % 2^32.
    All K repeats of the same prompt get the same preference (called once per prompt).

    Args:
        prompt_text: The prompt string.
        num_rewards: Number of reward objectives (R).
        base_seed: Base seed from config.
        epoch: Current epoch.
        prompt_offset: Offset within the epoch (prompt_cycle_idx).

    Returns:
        Tensor of shape (R,) with preference weights summing to 1.
    """
    if num_rewards == 1:
        return torch.ones(1, dtype=torch.float32)

    prompt_hash = _stable_prompt_hash_u32(prompt_text)
    seed = (base_seed + prompt_hash + epoch * 100000 + prompt_offset) % (2**32)
    rng = np.random.default_rng(seed)

    raw_weights = _sample_structured_preference(num_rewards, rng)
    return torch.tensor(raw_weights, dtype=torch.float32)


def sample_preferences_with_subgroups(
    prompt_text: str,
    num_rewards: int,
    num_pref_per_prompt: int,
    num_samples: int,
    base_seed: int,
    epoch: int,
) -> tuple[Tensor, Tensor]:
    """Sample multiple preference vectors per prompt, cycled across samples.

    Each (prompt, pref_slot) sub-group shares the same preference, preserving
    the GDPO i.i.d. assumption within each sub-group. Different sub-groups
    target different Pareto front points.

    Matching ParetoControl: seeds use (prompt, epoch, slot) — NOT batch_idx —
    so the same preferences are assigned consistently regardless of batch ordering.
    Seed for slot k: base_seed + hash(prompt) + epoch * 100000 + k * 7919.

    Args:
        prompt_text: The prompt string.
        num_rewards: Number of reward objectives (R).
        num_pref_per_prompt: Number of distinct preference vectors per prompt.
        num_samples: Total number of samples (K) for this prompt.
        base_seed: Base seed from config.
        epoch: Current epoch.

    Returns:
        Tuple of:
            - preferences: (num_samples, R) tensor, cycling through pref slots
            - pref_slots: (num_samples,) int tensor, slot index for each sample
    """
    if num_rewards == 1:
        return (
            torch.ones((num_samples, 1), dtype=torch.float32),
            torch.zeros(num_samples, dtype=torch.long),
        )

    # Sample K distinct preferences for this prompt
    prompt_hash = _stable_prompt_hash_u32(prompt_text)
    slot_prefs = []
    for k in range(num_pref_per_prompt):
        seed = (base_seed + prompt_hash + epoch * 100000 + k * 7919) % (2**32)
        rng = np.random.default_rng(seed)
        raw_weights = _sample_structured_preference(num_rewards, rng)
        slot_prefs.append(torch.tensor(raw_weights, dtype=torch.float32))

    # Cycle through slots: sample 0→slot 0, sample 1→slot 1, sample 2→slot 0, ...
    prefs = []
    slots = []
    for i in range(num_samples):
        slot = i % num_pref_per_prompt
        prefs.append(slot_prefs[slot])
        slots.append(slot)

    return (
        torch.stack(prefs, dim=0),
        torch.tensor(slots, dtype=torch.long),
    )
