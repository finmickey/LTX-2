"""DiffusionNFT loss computation.

Implements the NFT loss formulation for RL-based diffusion training:
- Positive prediction pushes the model toward high-reward outputs
- Negative prediction pushes the model away from low-reward outputs
- KL regularization prevents the model from drifting too far from the base model
"""

import torch
from torch import Tensor


def compute_nft_loss(
    xt: Tensor,
    x0: Tensor,
    t: Tensor,
    forward_pred: Tensor,
    old_pred: Tensor,
    ref_pred: Tensor,
    r: Tensor,
    beta: float,
    kl_beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the DiffusionNFT loss.

    Args:
        xt: Noisy latent [B, seq_len, C].
        x0: Clean latent [B, seq_len, C].
        t: Timestep [B, 1, 1] (expanded for broadcasting).
        forward_pred: v_new from current adapter (has grad) [B, seq_len, C].
        old_pred: v_old from old adapter (detached) [B, seq_len, C].
        ref_pred: v_ref from base model (detached) [B, seq_len, C].
        r: Advantage weight [B] in [0, 1].
        beta: NFT interpolation weight.
        kl_beta: KL regularization weight.

    Returns:
        Tuple of (total_loss, metrics_dict with detached tensors).
    """
    # Positive and negative predictions via NFT interpolation
    positive_pred = beta * forward_pred + (1 - beta) * old_pred
    negative_pred = (1 + beta) * old_pred - beta * forward_pred

    # Convert velocity predictions to x0 predictions
    # x0 = xt - t * v
    x0_pos = xt - t * positive_pred
    x0_neg = xt - t * negative_pred

    # Adaptive weighting: per-sample weight computed in float64 with stop-gradient.
    # Reference: NVlabs/DiffusionNFT uses torch.no_grad() + .double() for stability.
    # Reduces over all dims except batch → one weight per sample [B, 1, 1].
    reduce_dims = tuple(range(1, x0_pos.ndim))
    with torch.no_grad():
        weight_pos = (x0_pos.double() - x0.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5).to(x0_pos.dtype)
        weight_neg = (x0_neg.double() - x0.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5).to(x0_neg.dtype)

    # Weighted MSE loss per sample: [B]
    pos_loss = ((x0_pos - x0) ** 2 / weight_pos).mean(dim=(-1, -2))
    neg_loss = ((x0_neg - x0) ** 2 / weight_neg).mean(dim=(-1, -2))

    # Expand r to [B] for per-sample weighting
    r_expanded = r.view(-1)

    # Policy loss with advantage weighting
    policy_loss = r_expanded * pos_loss / beta + (1 - r_expanded) * neg_loss / beta

    # KL regularization: penalize deviation from base model
    kl_loss = ((forward_pred - ref_pred) ** 2).mean()

    total_loss = policy_loss.mean() + kl_beta * kl_loss

    metrics = {
        "policy_loss": policy_loss.mean().detach(),
        "kl_loss": kl_loss.detach(),
        "pos_loss": pos_loss.mean().detach(),
        "neg_loss": neg_loss.mean().detach(),
        "total_loss": total_loss.detach(),
    }

    return total_loss, metrics
