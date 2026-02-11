"""Simplified video generation for RL training.

This module provides a minimal generation loop that reuses ltx-core components.
No CFG, no STG — single forward pass per denoising step. Video-only (audio disabled).
"""

from dataclasses import replace

import torch
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.tools import VideoLatentTools
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@torch.no_grad()
def generate_video_latent(
    transformer: torch.nn.Module,
    video_prompt_embeds: Tensor,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    frame_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate a clean video latent using the denoising loop.

    Uses a simplified pipeline: no CFG, no STG, single forward pass per step.
    Audio modality is disabled.

    Args:
        transformer: The LTX transformer model (may be a PeftModel).
        video_prompt_embeds: Text embeddings for the video prompt [1, seq_len, dim].
        num_frames: Number of video frames to generate (must satisfy frames % 8 == 1).
        height: Video height in pixels (must be divisible by 32).
        width: Video width in pixels (must be divisible by 32).
        num_steps: Number of denoising steps.
        frame_rate: Frame rate for temporal position scaling.
        seed: Random seed for this generation.
        device: Device to run generation on.

    Returns:
        Tuple of:
            - clean_latent: Patchified latent tensor [1, seq_len, 128]
            - positions: Position tensor [1, 3, seq_len, 2]
    """
    patchifier = VideoLatentPatchifier(patch_size=1)

    pixel_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        height=height,
        width=width,
        fps=frame_rate,
    )
    video_tools = VideoLatentTools(
        patchifier=patchifier,
        target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
        fps=frame_rate,
        scale_factors=VIDEO_SCALE_FACTORS,
        causal_fix=True,
    )

    # Create initial state (zeros)
    video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

    # Add noise
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)

    # Build sigma schedule
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=num_steps).to(device).float()
    stepper = EulerDiffusionStep()

    # Wrap transformer with X0Model for velocity -> denoised conversion
    x0_model = X0Model(transformer)

    with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            video_modality = Modality(
                enabled=True,
                latent=video_state.latent,
                timesteps=sigma * video_state.denoise_mask,
                positions=video_state.positions,
                context=video_prompt_embeds,
                context_mask=None,
            )

            # Single forward pass — no CFG, no STG, no audio
            denoised_video, _ = x0_model(video=video_modality, audio=None, perturbations=None)

            # Apply conditioning mask (keep conditioned tokens clean)
            denoised_video = denoised_video * video_state.denoise_mask + video_clean_state.latent.float() * (
                1 - video_state.denoise_mask
            )

            # Euler step
            video_state = replace(
                video_state,
                latent=stepper.step(
                    sample=video_modality.latent,
                    denoised_sample=denoised_video,
                    sigmas=sigmas,
                    step_index=step_idx,
                ),
            )

    # Return the final clean latent and positions
    return video_state.latent.to(torch.bfloat16), video_state.positions
