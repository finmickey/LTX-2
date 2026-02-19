"""Structured preference sampling for ParetoNFT multi-reward training.

Adapted from ParetoNFT/scripts/train_cond_nft_sd3.py.

Samples a preference vector (R,) per prompt, deterministic via prompt hash + epoch.
All K repeats of the same prompt share the same preference vector.
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

    For R<=2: Dirichlet([1,1]) (uniform on simplex).
    For R>=3: 50% vertex (one-hot), 35% edge (pair), 15% interior (full Dirichlet).
    """
    if num_rewards <= 2:
        return rng.dirichlet([1.0] * num_rewards).astype(np.float32)

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
