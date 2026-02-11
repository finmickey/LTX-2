"""DiffusionNFT loss computation.

Implements the NFT loss formulation for RL-based diffusion training:
- Positive prediction pushes the model toward high-reward outputs
- Negative prediction pushes the model away from low-reward outputs
- KL regularization prevents the model from drifting too far from the base model
"""

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
    adv_clip_max: float,
) -> tuple[Tensor, dict[str, float]]:
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
        adv_clip_max: Advantage clip range, used for loss scaling.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    # Positive and negative predictions via NFT interpolation
    positive_pred = beta * forward_pred + (1 - beta) * old_pred
    negative_pred = (1 + beta) * old_pred - beta * forward_pred

    # Convert velocity predictions to x0 predictions
    # x0 = xt - t * v
    x0_pos = xt - t * positive_pred
    x0_neg = xt - t * negative_pred

    # Adaptive weighting: mean absolute error per token
    # [B, seq_len, 1]
    weight_pos = (x0_pos - x0).abs().mean(dim=-1, keepdim=True).clip(min=1e-5)
    weight_neg = (x0_neg - x0).abs().mean(dim=-1, keepdim=True).clip(min=1e-5)

    # Weighted MSE loss per sample: [B]
    pos_loss = ((x0_pos - x0) ** 2 / weight_pos).mean(dim=(-1, -2))
    neg_loss = ((x0_neg - x0) ** 2 / weight_neg).mean(dim=(-1, -2))

    # Expand r to [B] for per-sample weighting
    r_expanded = r.view(-1)

    # Policy loss with advantage weighting
    policy_loss = (r_expanded * pos_loss / beta + (1 - r_expanded) * neg_loss / beta) * adv_clip_max

    # KL regularization: penalize deviation from base model
    kl_loss = ((forward_pred - ref_pred) ** 2).mean()

    total_loss = policy_loss.mean() + kl_beta * kl_loss

    metrics = {
        "policy_loss": policy_loss.mean().item(),
        "kl_loss": kl_loss.item(),
        "pos_loss": pos_loss.mean().item(),
        "neg_loss": neg_loss.mean().item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics
