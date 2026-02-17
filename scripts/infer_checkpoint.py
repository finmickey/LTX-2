#!/usr/bin/env python3
"""Generate validation videos from an RL checkpoint (or base model) for comparison.

Usage:
    # With LoRA checkpoint (no CFG/STG, matches training validation):
    python scripts/infer_checkpoint.py
    # Base model (no LoRA, no CFG/STG):
    python scripts/infer_checkpoint.py --no-lora
    # Base model with CFG + STG (full inference pipeline):
    python scripts/infer_checkpoint.py --no-lora --with-cfg-stg
"""

import argparse
import re

import torch
from pathlib import Path
from safetensors.torch import load_file

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

from ltx_trainer.model_loader import load_model
from ltx_trainer.rl.generation import generate_video_latent
from ltx_trainer.video_utils import save_video

# Config
CHECKPOINT = "models/ltx-2-19b-dev.safetensors"
TEXT_ENCODER = "models/gemma-3-12b-it-qat-q4_0-unquantized"
LORA_PATH = "outputs/rl_nft_clip_only_ts5_run2/checkpoints/rl_lora_weights_step_00500.safetensors"

HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 121
NUM_STEPS = 20
FRAME_RATE = 25.0
SEED = 42

# CFG/STG settings (match validation config)
CFG_STEPS = 50
GUIDANCE_SCALE = 4.0
STG_SCALE = 1.0
STG_BLOCKS = [29]
STG_MODE = "stg_av"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

PROMPTS = [
    ("drone", "A drone flying over a mountain landscape at golden hour"),
    ("cat", "A cat sitting on a windowsill watching birds outside"),
    ("waves", "Waves crashing against rocky cliffs during a storm"),
    ("guitar", "A street musician playing guitar in a busy city square"),
    ("snow", "Snow falling gently over a quiet village at night"),
    ("surfer", "A surfer riding a large wave in the ocean"),
    ("train", "A train passing through a tunnel in the mountains"),
    ("fireworks", "Fireworks exploding over a city skyline at night"),
]


def decode_latent_to_pixels(latent, vae_decoder, device):
    """Decode patchified latent to pixel-space video."""
    patchifier = VideoLatentPatchifier(patch_size=1)

    latent_frames = NUM_FRAMES // VIDEO_SCALE_FACTORS.time + 1
    latent_height = HEIGHT // VIDEO_SCALE_FACTORS.height
    latent_width = WIDTH // VIDEO_SCALE_FACTORS.width

    unpatchified = patchifier.unpatchify(
        latent,
        output_shape=VideoLatentShape(
            height=latent_height,
            width=latent_width,
            frames=latent_frames,
            batch=1,
            channels=128,
        ),
    )

    unpatchified = unpatchified.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        decoded = vae_decoder(unpatchified)

    decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    return decoded[0].float().cpu()  # [C, F, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lora", action="store_true", help="Use base model without LoRA")
    parser.add_argument("--with-cfg-stg", action="store_true", help="Use CFG + STG (full inference pipeline)")
    args = parser.parse_args()

    # Build output dir suffix
    if args.with_cfg_stg:
        suffix = "base_cfg_stg" if args.no_lora else "lora_cfg_stg"
    else:
        suffix = "base" if args.no_lora else "lora"
    output_dir = Path(f"outputs/checkpoint_inference_step500_{WIDTH}x{HEIGHT}x{NUM_FRAMES}_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_device = torch.device("cuda:0")
    vae_device = torch.device("cuda:1")

    print("Loading model...")
    components = load_model(
        checkpoint_path=CHECKPOINT,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=True,
        text_encoder_path=TEXT_ENCODER,
    )

    transformer = components.transformer
    if not args.no_lora:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        print(f"Loading LoRA from {LORA_PATH}...")
        state_dict = load_file(str(LORA_PATH))
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        pattern = re.compile(r"(.+)\.lora_[AB]\.")
        target_modules = sorted({m.group(1) for k in state_dict if (m := pattern.match(k))})
        lora_rank = next(v.shape[0] for k, v in state_dict.items() if "lora_A" in k and v.ndim == 2)
        print(f"  {len(target_modules)} target modules, rank={lora_rank}")

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=64,
            target_modules=target_modules, lora_dropout=0.0,
        )
        transformer = get_peft_model(transformer, lora_config)
        set_peft_model_state_dict(transformer.get_base_model(), state_dict)
    else:
        print("Using base model (no LoRA)")

    transformer = transformer.to(gen_device)
    transformer.eval()

    vae_decoder = components.video_vae_decoder.to(vae_device)

    if args.with_cfg_stg:
        # Use ValidationSampler for CFG + STG
        from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

        print("Using CFG + STG pipeline")
        # Keep text encoder on gen_device for ValidationSampler
        text_encoder = components.text_encoder.to(gen_device)
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=vae_decoder,
            vae_encoder=None,
            text_encoder=text_encoder,
            audio_decoder=None,
            vocoder=None,
        )

        for nickname, prompt in PROMPTS:
            output_path = output_dir / f"{nickname}.mp4"
            print(f"\nGenerating: {nickname} - {prompt}")

            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                frame_rate=FRAME_RATE,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
                generate_audio=False,
                stg_scale=STG_SCALE,
                stg_blocks=STG_BLOCKS,
                stg_mode=STG_MODE,
            )

            video, _ = sampler.generate(config=gen_config, device=gen_device)
            save_video(video_tensor=video, output_path=output_path, fps=FRAME_RATE)
            print(f"  Saved: {output_path}")
    else:
        # No CFG/STG — use generate_video_latent directly
        print("Encoding prompts...")
        text_encoder = components.text_encoder.to(gen_device)
        prompt_embeddings = {}
        for nickname, prompt in PROMPTS:
            with torch.inference_mode():
                v_ctx, a_ctx, _ = text_encoder(prompt)
            prompt_embeddings[nickname] = v_ctx.to(gen_device)
        del text_encoder
        torch.cuda.empty_cache()

        for nickname, prompt in PROMPTS:
            output_path = output_dir / f"{nickname}.mp4"
            video_embeds = prompt_embeddings[nickname]
            print(f"\nGenerating: {nickname} - {prompt}")

            latent, _, _ = generate_video_latent(
                transformer=transformer,
                video_prompt_embeds=video_embeds,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                frame_rate=FRAME_RATE,
                seed=SEED,
                device=gen_device,
            )

            pixel_video = decode_latent_to_pixels(latent[:1], vae_decoder, vae_device)
            save_video(video_tensor=pixel_video, output_path=output_path, fps=FRAME_RATE)
            print(f"  Saved: {output_path}")

    print(f"\nDone! All videos saved to {output_dir}")


if __name__ == "__main__":
    main()
