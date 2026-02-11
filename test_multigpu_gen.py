"""Test multi-GPU generation + reward gathering. 4 prompts × 8 GPUs = 32 videos."""

import os
import re
import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

from pathlib import Path

from ltx_trainer.config import (
    AccelerationConfig,
    CheckpointsConfig,
    DataConfig,
    FlowMatchingConfig,
    LoraConfig,
    LtxTrainerConfig,
    ModelConfig,
    OptimizationConfig,
    RLConfig,
    ValidationConfig,
    WandbConfig,
)
from ltx_trainer.rl.generation import generate_video_latent
from ltx_trainer.rl.rl_trainer import RLTrainer
from ltx_trainer.video_utils import save_video

config = LtxTrainerConfig(
    model=ModelConfig(
        model_path="models/ltx-2-19b-dev.safetensors",
        text_encoder_path="models/gemma-3-12b-it-qat-q4_0-unquantized",
        training_mode="lora",
    ),
    lora=LoraConfig(rank=32, alpha=32, target_modules=["to_k", "to_q", "to_v", "to_out.0"]),
    rl=RLConfig(
        prompts_file="packages/ltx-trainer/data/rl_prompts.txt",
        num_samples_per_prompt=8,
        generation_steps=20,
        generation_num_frames=81,
        generation_height=256,
        generation_width=256,
        reward_type="redness",
    ),
    optimization=OptimizationConfig(learning_rate=1e-5, steps=1, batch_size=1, enable_gradient_checkpointing=False),
    acceleration=AccelerationConfig(mixed_precision_mode="bf16"),
    data=DataConfig(preprocessed_data_root="/tmp/unused"),
    validation=ValidationConfig(),
    checkpoints=CheckpointsConfig(interval=None),
    wandb=WandbConfig(enabled=False),
    flow_matching=FlowMatchingConfig(),
    seed=42,
    output_dir="outputs/rl_multigpu_81f",
)

# Prompt nicknames for readable filenames
PROMPTS_WITH_NICKNAMES = [
    (0, "red_car"),
    (1, "red_poppies"),
    (2, "red_dress"),
    (3, "red_cardinal"),
]


def slugify(s, max_len=30):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:max_len]


trainer = RLTrainer(config)
device = trainer._accelerator.device
rank = trainer._accelerator.process_index
num_processes = trainer._accelerator.num_processes
rl_cfg = trainer._rl_config

outdir = Path("outputs/rl_multigpu_81f/samples")
outdir.mkdir(parents=True, exist_ok=True)

trainer._set_adapter("old")
trainer._transformer.eval()

for prompt_idx, nickname in PROMPTS_WITH_NICKNAMES:
    cached = trainer._cached_prompt_embeddings[prompt_idx]
    video_prompt_embeds = cached.video_context_positive.to(device)

    seed = prompt_idx * num_processes + rank
    latent, positions = generate_video_latent(
        transformer=trainer._transformer,
        video_prompt_embeds=video_prompt_embeds,
        num_frames=rl_cfg.generation_num_frames,
        height=rl_cfg.generation_height,
        width=rl_cfg.generation_width,
        num_steps=rl_cfg.generation_steps,
        frame_rate=rl_cfg.frame_rate,
        seed=seed,
        device=device,
    )

    pixel_video = trainer._decode_latent_to_pixels(latent, device)
    reward = trainer._reward_fn.compute(pixel_video)
    reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

    all_rewards = trainer._accelerator.gather(reward_tensor)
    r = trainer._compute_advantage(reward, all_rewards)

    save_video(pixel_video, outdir / f"{nickname}_gpu{rank}_seed{seed}.mp4", fps=25.0)

    if rank == 0:
        rewards_str = ", ".join(f"{x:.4f}" for x in all_rewards.tolist())
        print(
            f"{nickname}: rewards=[{rewards_str}] "
            f"mean={all_rewards.mean():.4f} std={all_rewards.std():.4f}"
        )

trainer._accelerator.wait_for_everyone()

if rank == 0:
    num_videos = len(list(outdir.glob("*.mp4")))
    print(f"\nDone! Saved {num_videos} videos to {outdir}")
