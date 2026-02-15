"""RL Trainer for LTX-2 using the DiffusionNFT algorithm.

This trainer generates K videos per prompt using the current model, scores them
with a reward function, and uses the NFT loss formulation to update LoRA weights.

DDP strategy: With K=16 on 8 GPUs, each GPU generates K/num_gpus samples sequentially.
Rewards are all-gathered so each GPU sees all K rewards for advantage normalization.
Each GPU trains on its own generated samples.

Training loop structure (matching reference DiffusionNFT):
  for epoch in itertools.count():
      # SAMPLING: generate K videos for each of num_prompts_per_epoch prompts
      # ADVANTAGES: per-prompt mean, global std
      # TRAINING: shuffle samples, per-sample timestep permutations, gradient accumulation
      # OLD UPDATE: once per epoch
"""

import itertools
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file
from torch import Tensor
from torch.optim import AdamW

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.gpu_utils import free_gpu_memory
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.model_loader import load_text_encoder
from ltx_trainer.validation_sampler import CachedPromptEmbeddings
from ltx_trainer.video_utils import save_video

from .ema import EMAWrapper
from .embedding_store import LazyEmbeddingStore
from .generation import generate_video_latent
from .nft_loss import compute_nft_loss
from .rewards import get_reward_functions

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@dataclass
class _ValidationPrompt:
    nickname: str
    embeddings: CachedPromptEmbeddings


class RLTrainer:
    """RL trainer using DiffusionNFT for LTX-2.

    Generates videos, scores them with a reward function, and trains LoRA
    adapters using the NFT loss formulation.
    """

    def __init__(self, config: LtxTrainerConfig) -> None:
        self._config = config
        self._rl_config = config.rl

        # Load text encoder, cache prompt embeddings, then unload heavy parts
        self._cached_prompt_embeddings, self._validation_prompts = self._load_text_encoder_and_cache_embeddings()

        # Load models
        self._load_models()

        # Setup accelerator
        self._setup_accelerator()

        # Setup dual LoRA adapters
        self._setup_dual_lora()

        # Prepare model with accelerator
        self._prepare_for_training()

        # Setup reward functions: list of (RewardFunction, name) tuples
        # video_score expands into one reward per dimension
        self._reward_fns = []
        for rc in self._rl_config.rewards:
            self._reward_fns.extend(get_reward_functions(rc.type))

        # Patchifier for unpatchify during decode
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)

        # W&B
        self._wandb_run = None
        self._init_wandb()

        # File logging into output dir
        self._setup_log_file()

        # EMA and trainable params (initialized in train())
        self._ema: EMAWrapper | None = None
        self._trainable_params: list[Tensor] | None = None

    def train(self) -> None:
        """Run the full RL training loop (epoch-based, matching reference DiffusionNFT).

        Structure per epoch:
          1. SAMPLING: Generate K videos for each of num_prompts_per_epoch prompts
          2. ADVANTAGES: Per-prompt mean, global std normalization
          3. TRAINING: Shuffle samples, per-sample timestep permutations,
             gradient accumulation across micro-batches x timesteps
          4. OLD UPDATE: Once per epoch
        """
        cfg = self._config
        rl_cfg = self._rl_config
        device = self._accelerator.device
        rank = self._accelerator.process_index
        num_processes = self._accelerator.num_processes

        set_seed(cfg.seed + rank)

        # Validate that validation prompts count matches GPU count
        if self._validation_prompts:
            assert len(self._validation_prompts) == num_processes, (
                f"Number of validation prompts ({len(self._validation_prompts)}) must equal "
                f"num_processes ({num_processes})"
            )

        # Compute samples per GPU and validate
        samples_per_gpu = rl_cfg.num_samples_per_prompt // num_processes
        assert rl_cfg.num_samples_per_prompt % num_processes == 0, (
            f"num_samples_per_prompt ({rl_cfg.num_samples_per_prompt}) must be "
            f"divisible by num_processes ({num_processes})"
        )

        trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=cfg.optimization.learning_rate,
            betas=cfg.optimization.adam_betas,
            eps=cfg.optimization.adam_epsilon,
            weight_decay=cfg.optimization.adam_weight_decay,
        )
        optimizer = self._accelerator.prepare(optimizer)

        # EMA for trainable parameters
        ema = EMAWrapper(trainable_params, decay=rl_cfg.ema_decay)
        self._ema = ema
        self._trainable_params = trainable_params

        # Resume from checkpoint if configured
        resume_step = 0
        resume_epoch = 0
        resume_prompt_cycle_idx = 0
        if rl_cfg.resume_from_checkpoint is not None:
            resume_step, resume_epoch, resume_prompt_cycle_idx = self._load_training_state(
                rl_cfg.resume_from_checkpoint, optimizer,
            )

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        num_prompts = len(self._cached_prompt_embeddings)
        num_steps = cfg.optimization.steps
        num_prompts_per_epoch = rl_cfg.num_prompts_per_epoch
        grad_accum_steps = rl_cfg.gradient_accumulation_steps

        assert rl_cfg.num_timesteps_per_sample <= rl_cfg.generation_steps, (
            f"num_timesteps_per_sample ({rl_cfg.num_timesteps_per_sample}) must be "
            f"<= generation_steps ({rl_cfg.generation_steps})"
        )

        logger.info(
            f"Starting RL training: {num_steps} optimizer steps, "
            f"{num_prompts} prompts, K={rl_cfg.num_samples_per_prompt} "
            f"({samples_per_gpu} per GPU), "
            f"T={rl_cfg.num_timesteps_per_sample} timesteps/sample, "
            f"prompts/epoch={num_prompts_per_epoch}, "
            f"grad_accum={grad_accum_steps}, "
            f"ema_decay={rl_cfg.ema_decay}"
        )

        # Save validation videos at step 0 (before any training), skip on resume
        if rl_cfg.video_save_interval is not None and self._validation_prompts and resume_step == 0:
            self._save_validation_videos(step=0)

        self._accelerator.wait_for_everyone()

        autocast_dtype = torch.bfloat16
        device_type = str(device).split(":")[0]

        global_step = resume_step
        prompt_cycle_idx = resume_prompt_cycle_idx

        for epoch in itertools.count(start=resume_epoch + 1 if resume_step > 0 else 0):
            epoch_start = self._cuda_time()
            timings: dict[str, float] = {}

            # ============================================================
            # Phase 1: SAMPLING -- generate K videos for each prompt in epoch
            # ============================================================
            self._transformer.eval()
            self._set_adapter("old")

            # epoch_samples[i] = dict with keys: latents, positions, rewards,
            #   individual_rewards, prompt_embeds, prompt_idx, gen_sigmas
            epoch_samples: list[dict] = []

            total_gen_time = 0.0
            total_decode_time = 0.0
            total_reward_time = 0.0

            for p_idx in range(num_prompts_per_epoch):
                prompt_idx = prompt_cycle_idx % num_prompts
                prompt_cycle_idx += 1
                cached = self._cached_prompt_embeddings[prompt_idx]
                video_prompt_embeds = cached.video_context_positive.to(device)

                gen_seed_base = (
                    epoch * num_prompts_per_epoch * num_processes * samples_per_gpu
                    + p_idx * num_processes * samples_per_gpu
                    + rank * samples_per_gpu
                )
                gen_seeds = [gen_seed_base + k for k in range(samples_per_gpu)]

                t0 = self._wall_time()
                all_latents, all_positions, gen_sigmas = generate_video_latent(
                    transformer=self._transformer,
                    video_prompt_embeds=video_prompt_embeds,
                    num_frames=rl_cfg.generation_num_frames,
                    height=rl_cfg.generation_height,
                    width=rl_cfg.generation_width,
                    num_steps=rl_cfg.generation_steps,
                    frame_rate=rl_cfg.frame_rate,
                    seed=gen_seeds,
                    device=device,
                )
                total_gen_time += self._wall_time() - t0

                local_latents = []
                local_positions = []
                local_rewards = []
                local_individual_rewards = []

                # --- Batched VAE decode (all samples in one call) ---
                t0 = self._wall_time()
                latent_frames = rl_cfg.generation_num_frames // VIDEO_SCALE_FACTORS.time + 1
                latent_height = rl_cfg.generation_height // VIDEO_SCALE_FACTORS.height
                latent_width = rl_cfg.generation_width // VIDEO_SCALE_FACTORS.width

                unpatchified_list = []
                for k in range(samples_per_gpu):
                    unpatchified = self._video_patchifier.unpatchify(
                        all_latents[k : k + 1],
                        output_shape=VideoLatentShape(
                            height=latent_height, width=latent_width,
                            frames=latent_frames, batch=1, channels=128,
                        ),
                    )
                    unpatchified_list.append(unpatchified)

                unpatchified_batch = torch.cat(unpatchified_list, dim=0).to(device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    decoded_batch = self._vae_decoder(unpatchified_batch)
                decoded_batch = ((decoded_batch + 1.0) / 2.0).clamp(0.0, 1.0)
                total_decode_time += self._wall_time() - t0
                del unpatchified_list, unpatchified_batch

                for k in range(samples_per_gpu):
                    latent = all_latents[k : k + 1]
                    positions = all_positions[k : k + 1]
                    pixel_video = decoded_batch[k].float().cpu()  # [C, F, H, W]

                    t0 = self._wall_time()
                    prompt_text = cached.prompt_text
                    individual = {name: fn.compute(pixel_video, prompt=prompt_text) for fn, name in self._reward_fns}
                    reward = sum(individual.values())
                    total_reward_time += self._wall_time() - t0

                    local_latents.append(latent)
                    local_positions.append(positions)
                    local_rewards.append(reward)
                    local_individual_rewards.append(individual)
                    del pixel_video
                del decoded_batch

                del all_latents, all_positions

                epoch_samples.append({
                    "latents": local_latents,
                    "positions": local_positions,
                    "rewards": local_rewards,
                    "individual_rewards": local_individual_rewards,
                    "prompt_embeds": video_prompt_embeds,
                    "prompt_idx": prompt_idx,
                    "gen_sigmas": gen_sigmas,
                })

            timings["generation"] = total_gen_time
            timings["decode"] = total_decode_time
            timings["reward"] = total_reward_time

            # ============================================================
            # Phase 2: GATHER rewards and compute per-prompt advantages
            # ============================================================
            t0 = self._wall_time()

            # Gather all rewards across GPUs, organized by prompt
            all_prompt_rewards: list[Tensor] = []  # [num_prompts_per_epoch] of [K]
            all_gathered_individual: list[dict[str, Tensor]] = []

            for ps in epoch_samples:
                local_reward_tensor = torch.tensor(ps["rewards"], device=device, dtype=torch.float32)
                gathered = self._accelerator.gather(local_reward_tensor)  # [K]
                all_prompt_rewards.append(gathered)

                gi: dict[str, Tensor] = {}
                for _, name in self._reward_fns:
                    local_vals = torch.tensor(
                        [d[name] for d in ps["individual_rewards"]],
                        device=device, dtype=torch.float32,
                    )
                    gi[name] = self._accelerator.gather(local_vals)  # [K]
                all_gathered_individual.append(gi)

            # Per-prompt advantages with global std (matching reference)
            all_flat_rewards = torch.cat(all_prompt_rewards)  # [num_prompts_per_epoch * K]
            global_std = all_flat_rewards.std().item() + 1e-6

            # Compute per-prompt advantages: (reward - prompt_mean) / global_std
            # Then clip and map to [0, 1]
            per_prompt_advantages: list[list[float]] = []
            for rewards_k in all_prompt_rewards:
                prompt_mean = rewards_k.mean().item()
                advs = []
                for r in rewards_k.tolist():
                    adv = (r - prompt_mean) / global_std
                    adv = max(-rl_cfg.adv_clip_max, min(rl_cfg.adv_clip_max, adv))
                    advs.append(adv / rl_cfg.adv_clip_max / 2.0 + 0.5)
                per_prompt_advantages.append(advs)

            timings["gather"] = self._wall_time() - t0

            # ============================================================
            # Phase 3: TRAINING -- shuffle, per-sample timestep perms, grad accum
            # ============================================================
            self._transformer.train()
            t_phase3 = self._cuda_time()

            total_fwd_new = 0.0
            total_fwd_old = 0.0
            total_fwd_ref = 0.0
            total_nft_loss = 0.0
            total_backward = 0.0
            total_adapter_switch = 0.0
            total_opt_step = 0.0

            # Build flat list of (sample_data_dict) for shuffling
            flat_samples: list[dict] = []
            for p_idx, ps in enumerate(epoch_samples):
                local_offset = rank * samples_per_gpu
                for k in range(samples_per_gpu):
                    flat_samples.append({
                        "latent": ps["latents"][k],
                        "positions": ps["positions"][k],
                        "advantage": per_prompt_advantages[p_idx][local_offset + k],
                        "prompt_embeds": ps["prompt_embeds"],
                        "gen_sigmas": ps["gen_sigmas"],
                    })

            # Shuffle samples randomly (same seed across GPUs for consistency)
            shuffle_seed = cfg.seed + epoch
            rng = random.Random(shuffle_seed)
            shuffle_order = list(range(len(flat_samples)))
            rng.shuffle(shuffle_order)
            flat_samples = [flat_samples[i] for i in shuffle_order]

            # Select timestep indices and create per-sample permutations
            num_timesteps = min(rl_cfg.num_timesteps_per_sample, len(epoch_samples[0]["gen_sigmas"]) - 1)
            sigma_indices = torch.randperm(len(epoch_samples[0]["gen_sigmas"]) - 1)[:num_timesteps]

            # Per-sample timestep permutations for gradient diversity
            per_sample_perms = [torch.randperm(num_timesteps) for _ in range(len(flat_samples))]

            # Split into micro-batches (each = samples_per_gpu samples)
            micro_batch_size = samples_per_gpu
            micro_batches = [
                flat_samples[i:i + micro_batch_size]
                for i in range(0, len(flat_samples), micro_batch_size)
            ]
            micro_batch_perms = [
                per_sample_perms[i:i + micro_batch_size]
                for i in range(0, len(per_sample_perms), micro_batch_size)
            ]

            # Effective grad accum = gradient_accumulation_steps x num_timesteps
            effective_grad_accum = grad_accum_steps * num_timesteps
            backward_count = 0
            accumulated_metrics: dict[str, Tensor] = {}
            accumulated_metrics_snapshot: dict[str, float] = {}
            epoch_optimizer_steps = 0

            for mb_idx, (mb_samples, mb_perms) in enumerate(zip(micro_batches, micro_batch_perms)):
                mb_size = len(mb_samples)

                # --- Pre-build ALL modalities and inputs for ALL timesteps ---
                all_modalities: list[Modality] = []
                all_input_lists: list[list[dict]] = []

                for j_idx in range(num_timesteps):
                    xt_list = []
                    ts_list = []
                    pos_list = []
                    input_list: list[dict] = []
                    ctx_list = []

                    for s_idx in range(mb_size):
                        sample = mb_samples[s_idx]
                        perm = mb_perms[s_idx]
                        actual_sigma_idx = sigma_indices[perm[j_idx]]
                        t_val = sample["gen_sigmas"][actual_sigma_idx].float()

                        latent = sample["latent"]
                        positions = sample["positions"]
                        r_tensor = torch.tensor([sample["advantage"]], device=device, dtype=torch.float32)

                        seq_len = latent.shape[1]
                        t_expanded = t_val.view(1, 1, 1)

                        noise = torch.randn_like(latent)
                        xt = (1 - t_expanded) * latent + t_expanded * noise
                        timesteps = t_val.expand(1, seq_len)

                        input_list.append({
                            "latent": latent, "xt": xt,
                            "t_expanded": t_expanded, "r_tensor": r_tensor,
                        })
                        xt_list.append(xt)
                        ts_list.append(timesteps)
                        pos_list.append(positions)
                        ctx_list.append(sample["prompt_embeds"])

                    batched_context = torch.cat(
                        [c.unsqueeze(0) if c.dim() == 2 else c[:1] for c in ctx_list], dim=0
                    )
                    all_modalities.append(Modality(
                        enabled=True,
                        latent=torch.cat(xt_list, dim=0),
                        timesteps=torch.cat(ts_list, dim=0),
                        positions=torch.cat(pos_list, dim=0),
                        context=batched_context,
                        context_mask=None,
                    ))
                    all_input_lists.append(input_list)

                # --- Build mega-modality (all timesteps concatenated along batch dim) ---
                mega_modality = Modality(
                    enabled=True,
                    latent=torch.cat([m.latent for m in all_modalities], dim=0),
                    timesteps=torch.cat([m.timesteps for m in all_modalities], dim=0),
                    positions=torch.cat([m.positions for m in all_modalities], dim=0),
                    context=torch.cat([m.context for m in all_modalities], dim=0),
                    context_mask=None,
                )

                # --- ALL old fwd passes (mega-batched single pass) ---
                t0 = self._wall_time()
                self._set_adapter("old")
                total_adapter_switch += self._wall_time() - t0

                t0 = self._wall_time()
                with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    v_old_mega, _ = self._transformer(
                        video=mega_modality, audio=None, perturbations=None,
                    )
                    v_old_all = list(v_old_mega.split(mb_size, dim=0))
                total_fwd_old += self._wall_time() - t0

                # --- ALL ref fwd passes (mega-batched single pass) ---
                unwrapped = self._accelerator.unwrap_model(self._transformer)
                t0 = self._wall_time()
                with unwrapped.disable_adapter():
                    total_adapter_switch += self._wall_time() - t0
                    t0 = self._wall_time()
                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        v_ref_mega, _ = self._transformer(
                            video=mega_modality, audio=None, perturbations=None,
                        )
                        v_ref_all = list(v_ref_mega.split(mb_size, dim=0))
                total_fwd_ref += self._wall_time() - t0

                # --- New fwd + backward per timestep group (single adapter switch) ---
                t0 = self._wall_time()
                self._set_adapter("default")
                total_adapter_switch += self._wall_time() - t0

                new_group = rl_cfg.new_fwd_group_size
                for group_start in range(0, num_timesteps, new_group):
                    group_end = min(group_start + new_group, num_timesteps)
                    group_indices = list(range(group_start, group_end))
                    group_size = len(group_indices)

                    # --- Grouped forward pass ---
                    t0 = self._wall_time()
                    if group_size == 1:
                        # Single timestep: use modality directly (no cat/split overhead)
                        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                            v_new_batched, _ = self._transformer(
                                video=all_modalities[group_indices[0]], audio=None, perturbations=None,
                            )
                        v_new_chunks = [v_new_batched]
                    else:
                        # Multiple timesteps: cat modalities, single forward, split output
                        group_modality = Modality(
                            enabled=True,
                            latent=torch.cat([all_modalities[j].latent for j in group_indices], dim=0),
                            timesteps=torch.cat([all_modalities[j].timesteps for j in group_indices], dim=0),
                            positions=torch.cat([all_modalities[j].positions for j in group_indices], dim=0),
                            context=torch.cat([all_modalities[j].context for j in group_indices], dim=0),
                            context_mask=None,
                        )
                        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                            v_new_mega, _ = self._transformer(
                                video=group_modality, audio=None, perturbations=None,
                            )
                        v_new_chunks = list(v_new_mega.split(mb_size, dim=0))
                        del v_new_mega, group_modality
                    total_fwd_new += self._wall_time() - t0

                    # --- NFT loss for each timestep in group ---
                    t0 = self._wall_time()
                    group_loss = torch.tensor(0.0, device=device)
                    for k, j_idx in enumerate(group_indices):
                        input_list = all_input_lists[j_idx]
                        batched_xt = torch.cat([si["xt"] for si in input_list], dim=0)
                        batched_x0 = torch.cat([si["latent"] for si in input_list], dim=0)
                        batched_t = torch.cat([si["t_expanded"] for si in input_list], dim=0)
                        batched_r = torch.cat([si["r_tensor"] for si in input_list], dim=0)

                        loss, metrics = compute_nft_loss(
                            xt=batched_xt, x0=batched_x0, t=batched_t,
                            forward_pred=v_new_chunks[k],
                            old_pred=v_old_all[j_idx].detach(),
                            ref_pred=v_ref_all[j_idx].detach(),
                            r=batched_r,
                            beta=rl_cfg.nft_beta,
                            kl_beta=rl_cfg.kl_beta,
                            adv_clip_max=rl_cfg.adv_clip_max,
                        )
                        group_loss = group_loss + loss / effective_grad_accum

                        for mkey, mval in metrics.items():
                            if mkey not in accumulated_metrics:
                                accumulated_metrics[mkey] = mval / effective_grad_accum
                            else:
                                accumulated_metrics[mkey] = accumulated_metrics[mkey] + mval / effective_grad_accum
                    total_nft_loss += self._wall_time() - t0

                    # --- Backward (no_sync for non-final passes) ---
                    t0 = self._wall_time()
                    backward_count += group_size
                    if backward_count % effective_grad_accum != 0:
                        with self._accelerator.no_sync(self._transformer):
                            self._accelerator.backward(group_loss)
                    else:
                        self._accelerator.backward(group_loss)
                    total_backward += self._wall_time() - t0

                    del v_new_chunks

                    # --- Optimizer step at accumulation boundary ---
                    if backward_count % effective_grad_accum == 0:
                        t0 = self._wall_time()
                        if cfg.optimization.max_grad_norm > 0:
                            self._accelerator.clip_grad_norm_(trainable_params, cfg.optimization.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_opt_step += self._wall_time() - t0

                        global_step += 1
                        epoch_optimizer_steps += 1
                        ema.step(trainable_params, global_step)

                        # Snapshot metrics for logging
                        accumulated_metrics_snapshot = {k: v.item() for k, v in accumulated_metrics.items()}
                        accumulated_metrics = {}

                        # Eval/checkpoint at optimizer step boundaries
                        if rl_cfg.video_save_interval is not None and self._validation_prompts and global_step % rl_cfg.video_save_interval == 0:
                            self._save_validation_videos(step=global_step)
                            if rl_cfg.validation_grid_interval and global_step % rl_cfg.validation_grid_interval == 0:
                                self._generate_validation_grid(global_step)
                            self._transformer.train()
                            self._set_adapter("default")

                        if rl_cfg.checkpoint_save_interval and global_step % rl_cfg.checkpoint_save_interval == 0:
                            self._save_checkpoint(global_step, epoch, prompt_cycle_idx, optimizer)
                            self._set_adapter("default")

                        if global_step >= num_steps:
                            break

                del v_old_all, v_ref_all, mega_modality, all_modalities, all_input_lists

                if global_step >= num_steps:
                    break

            # Flush any remaining gradients (if backward_count not aligned)
            remaining = backward_count % effective_grad_accum
            if remaining > 0 and global_step < num_steps:
                t0 = self._wall_time()
                if cfg.optimization.max_grad_norm > 0:
                    self._accelerator.clip_grad_norm_(trainable_params, cfg.optimization.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                total_opt_step += self._wall_time() - t0

                global_step += 1
                epoch_optimizer_steps += 1
                ema.step(trainable_params, global_step)
                accumulated_metrics_snapshot = {k: v.item() for k, v in accumulated_metrics.items()}
                accumulated_metrics = {}

            # ============================================================
            # Phase 4: DECAY old adapter (once per epoch)
            # ============================================================
            t0 = self._wall_time()
            decay_value = min(global_step * rl_cfg.decay_rate, rl_cfg.max_decay)
            self._decay_old_adapter(decay_value)
            timings["decay"] = self._wall_time() - t0

            timings["adapter_switch"] = total_adapter_switch
            timings["fwd_new"] = total_fwd_new
            timings["fwd_old"] = total_fwd_old
            timings["fwd_ref"] = total_fwd_ref
            timings["nft_loss"] = total_nft_loss
            timings["backward"] = total_backward
            timings["opt_step"] = total_opt_step
            timings["phase3_total"] = self._cuda_time() - t_phase3

            # ============================================================
            # Phase 5: LOG (epoch-level summary)
            # ============================================================
            epoch_time = self._cuda_time() - epoch_start
            timings["epoch_total"] = epoch_time

            if IS_MAIN_PROCESS:
                all_rewards = all_flat_rewards
                mean_reward = all_rewards.mean().item()
                std_reward = all_rewards.std().item() if len(all_rewards) > 1 else 0.0
                max_reward = all_rewards.max().item()
                min_reward = all_rewards.min().item()

                # Mean advantage across all samples
                all_advs = [a for advs in per_prompt_advantages for a in advs]
                mean_advantage = sum(all_advs) / len(all_advs)

                log_metrics = {
                    "rl/mean_reward": mean_reward,
                    "rl/reward_std": std_reward,
                    "rl/max_reward": max_reward,
                    "rl/min_reward": min_reward,
                    "rl/mean_advantage": mean_advantage,
                    "rl/decay": decay_value,
                    "rl/epoch_time": epoch_time,
                    "rl/epoch": epoch,
                    "rl/global_step": global_step,
                    "rl/optimizer_steps_this_epoch": epoch_optimizer_steps,
                    "rl/num_timesteps_per_sample": num_timesteps,
                    "rl/ema_decay": ema.get_current_decay(global_step),
                }
                if accumulated_metrics_snapshot:
                    log_metrics["rl/loss"] = accumulated_metrics_snapshot.get("total_loss", 0.0)
                    log_metrics["rl/policy_loss"] = accumulated_metrics_snapshot.get("policy_loss", 0.0)
                    log_metrics["rl/kl_loss"] = accumulated_metrics_snapshot.get("kl_loss", 0.0)
                    log_metrics["rl/pos_loss"] = accumulated_metrics_snapshot.get("pos_loss", 0.0)
                    log_metrics["rl/neg_loss"] = accumulated_metrics_snapshot.get("neg_loss", 0.0)

                # Aggregate individual rewards across all prompts in epoch
                for _, name in self._reward_fns:
                    all_vals = torch.cat([gi[name] for gi in all_gathered_individual])
                    log_metrics[f"rl/reward/{name}"] = all_vals.mean().item()
                    log_metrics[f"rl/reward/{name}_std"] = all_vals.std().item() if len(all_vals) > 1 else 0.0

                for tname, tval in timings.items():
                    log_metrics[f"rl/time/{tname}"] = tval
                self._log_metrics(log_metrics)

                # Build per-reward detail string
                reward_parts = []
                for _, name in self._reward_fns:
                    all_vals = torch.cat([gi[name] for gi in all_gathered_individual])
                    r_mean = all_vals.mean().item()
                    r_std = all_vals.std().item() if len(all_vals) > 1 else 0.0
                    reward_parts.append(f"{name}: mu={r_mean:.2f} s={r_std:.2f}")
                reward_detail = " | ".join(reward_parts)

                loss_str = ""
                if accumulated_metrics_snapshot:
                    kl_raw = accumulated_metrics_snapshot.get("kl_loss", 0.0)
                    kl_weighted = rl_cfg.kl_beta * kl_raw
                    loss_str = (
                        f"loss={accumulated_metrics_snapshot.get('total_loss', 0.0):.3f} "
                        f"(pol={accumulated_metrics_snapshot.get('policy_loss', 0.0):.1f} "
                        f"kl={kl_raw:.2f}x{rl_cfg.kl_beta}={kl_weighted:.3f}) | "
                    )

                gen_time = timings.get("generation", 0.0)
                rew_time = timings.get("decode", 0.0) + timings.get("reward", 0.0)
                train_time = timings.get("phase3_total", 0.0)

                logger.info(
                    f"Step {global_step}/{num_steps} (epoch {epoch}, {num_prompts_per_epoch}p) | "
                    f"{loss_str}"
                    f"reward: mu={mean_reward:.2f} s={std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}] | "
                    f"{reward_detail} | "
                    f"adv={mean_advantage:.3f} | "
                    f"{epoch_time:.1f}s (gen={gen_time:.1f} rew={rew_time:.1f} train={train_time:.1f})"
                )

            # Check eval/checkpoint for steps from the flush path
            if remaining > 0 and global_step <= num_steps:
                if rl_cfg.video_save_interval is not None and self._validation_prompts and global_step % rl_cfg.video_save_interval == 0:
                    self._save_validation_videos(step=global_step)
                    if rl_cfg.validation_grid_interval and global_step % rl_cfg.validation_grid_interval == 0:
                        self._generate_validation_grid(global_step)
                if rl_cfg.checkpoint_save_interval and global_step % rl_cfg.checkpoint_save_interval == 0:
                    self._save_checkpoint(global_step, epoch, prompt_cycle_idx, optimizer)

            self._accelerator.wait_for_everyone()

            if global_step >= num_steps:
                break

        # Final validation videos + grid + checkpoint
        if rl_cfg.video_save_interval is not None and self._validation_prompts:
            self._save_validation_videos(step=global_step)
            if rl_cfg.validation_grid_interval:
                self._generate_validation_grid(global_step)
        self._save_checkpoint(global_step, epoch, prompt_cycle_idx, optimizer)

        if IS_MAIN_PROCESS and self._wandb_run is not None:
            self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()
        logger.info("RL training complete.")

    # ========================================================================
    # Model loading and setup
    # ========================================================================

    def _load_text_encoder_and_cache_embeddings(
        self,
    ) -> tuple[list[CachedPromptEmbeddings] | LazyEmbeddingStore, list[_ValidationPrompt]]:
        """Load text encoder, cache all prompt embeddings, unload heavy parts.

        If ``precomputed_embeddings_dir`` is set, training embeddings are loaded
        lazily from disk (one .pt file per prompt) instead of being encoded at
        startup.  The text encoder is still loaded briefly when validation
        prompts are configured.
        """
        rl_cfg = self._rl_config
        need_text_encoder = (
            rl_cfg.precomputed_embeddings_dir is None
            or rl_cfg.validation_prompts_file is not None
        )

        text_encoder = None
        if need_text_encoder:
            logger.debug("Loading text encoder...")
            text_encoder = load_text_encoder(
                checkpoint_path=self._config.model.model_path,
                gemma_model_path=self._config.model.text_encoder_path,
                device="cuda",
                dtype=torch.bfloat16,
                load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            )

        # --- Training embeddings ---
        if rl_cfg.precomputed_embeddings_dir is not None:
            cached: list[CachedPromptEmbeddings] | LazyEmbeddingStore = LazyEmbeddingStore(
                embeddings_dir=rl_cfg.precomputed_embeddings_dir,
                prompts_file=rl_cfg.prompts_file,
            )
            logger.info(
                f"Using lazy embedding store: {len(cached)} prompts from "
                f"{rl_cfg.precomputed_embeddings_dir}"
            )
        else:
            # Read training prompts from file
            prompts_path = Path(rl_cfg.prompts_file)
            prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

            # Cache embeddings for all training prompts
            cached = []
            with torch.inference_mode():
                for prompt in prompts:
                    v_ctx, a_ctx, _ = text_encoder(prompt)
                    cached.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx.cpu(),
                            audio_context_positive=a_ctx.cpu(),
                            prompt_text=prompt,
                        )
                    )

        # Cache validation prompt embeddings (if configured)
        validation_prompts: list[_ValidationPrompt] = []
        if rl_cfg.validation_prompts_file is not None:
            vp_path = Path(rl_cfg.validation_prompts_file)
            for line in vp_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                nickname, prompt_text = line.split("|", 1)
                nickname = nickname.strip()
                prompt_text = prompt_text.strip()
                with torch.inference_mode():
                    v_ctx, a_ctx, _ = text_encoder(prompt_text)
                validation_prompts.append(
                    _ValidationPrompt(
                        nickname=nickname,
                        embeddings=CachedPromptEmbeddings(
                            video_context_positive=v_ctx.cpu(),
                            audio_context_positive=a_ctx.cpu(),
                            prompt_text=prompt_text,
                        ),
                    )
                )
            logger.info(f"Cached embeddings for {len(validation_prompts)} validation prompts.")

        # Keep the embedding connectors, unload heavy Gemma model
        if text_encoder is not None:
            self._text_encoder = text_encoder
            self._text_encoder.model = None
            self._text_encoder.tokenizer = None
            self._text_encoder.feature_extractor_linear = None
        else:
            self._text_encoder = None

        free_gpu_memory()
        logger.info(f"Training embeddings ready: {len(cached)} prompts. Text encoder unloaded.")
        return cached, validation_prompts

    def _load_models(self) -> None:
        """Load transformer, VAE decoder, and scheduler."""
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=True,
            with_audio_vae_decoder=False,
            with_vocoder=False,
            with_text_encoder=False,
        )

        self._transformer = components.transformer.to(dtype=torch.bfloat16)
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler

        # Freeze everything initially
        self._transformer.requires_grad_(False)
        self._vae_decoder.requires_grad_(False)

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator."""
        # find_unused_parameters=True is required because the "old" LoRA adapter
        # parameters exist in the model but never participate in the gradient graph
        # (only "default" adapter params receive gradients).
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=1,
            kwargs_handlers=[ddp_kwargs],
        )
        if self._accelerator.num_processes > 1:
            logger.info(
                f"DDP enabled with {self._accelerator.num_processes} processes"
            )

    def _setup_dual_lora(self) -> None:
        """Setup dual LoRA adapters: 'default' (trainable) and 'old' (frozen reference)."""
        lora_cfg = self._config.lora
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights=True,
        )

        # Add default adapter
        self._transformer = get_peft_model(self._transformer, lora_config)

        # Add 'old' adapter as a copy of default
        self._transformer.add_adapter("old", lora_config)

        # Copy default weights to old adapter
        self._sync_old_adapter_from_default()

        # Freeze 'old' adapter parameters
        for name, param in self._transformer.named_parameters():
            if "old" in name and "lora" in name:
                param.requires_grad_(False)

        # Set back to default for initial state
        self._set_adapter("default")

        trainable_count = sum(p.numel() for p in self._transformer.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self._transformer.parameters())
        logger.info(f"Dual LoRA setup: {trainable_count:,} trainable / {total_count:,} total params")

    def _prepare_for_training(self) -> None:
        """Prepare models for distributed training."""
        # Enable gradient checkpointing if requested
        if self._config.optimization.enable_gradient_checkpointing:
            base_transformer = self._transformer.get_base_model()
            base_transformer.set_gradient_checkpointing(True)

        # Keep VAE on GPU -- ~250MB in bf16, plenty of headroom on 80GB H100s
        self._vae_decoder = self._vae_decoder.to(self._accelerator.device)

        # Prepare transformer with accelerator
        self._transformer = self._accelerator.prepare(self._transformer)

        vram_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory after model preparation: {vram_gb:.2f} GB")

    # ========================================================================
    # Adapter management
    # ========================================================================

    def _set_adapter(self, adapter_name: str) -> None:
        """Switch the active LoRA adapter."""
        unwrapped = self._accelerator.unwrap_model(self._transformer)
        unwrapped.set_adapter(adapter_name)

    def _sync_old_adapter_from_default(self) -> None:
        """Copy default adapter weights to old adapter."""
        default_params = {}
        old_params = {}
        for name, param in self._transformer.named_parameters():
            if "lora" not in name:
                continue
            if ".default." in name:
                key = name.replace(".default.", ".ADAPTER.")
                default_params[key] = param
            elif ".old." in name:
                key = name.replace(".old.", ".ADAPTER.")
                old_params[key] = param

        for key in default_params:
            if key in old_params:
                old_params[key].data.copy_(default_params[key].data)

    def _decay_old_adapter(self, decay: float) -> None:
        """Update old adapter: old = decay * old + (1 - decay) * default."""
        if decay <= 0:
            return

        unwrapped = self._accelerator.unwrap_model(self._transformer)
        default_params = {}
        old_params = {}
        for name, param in unwrapped.named_parameters():
            if "lora" not in name:
                continue
            if ".default." in name:
                key = name.replace(".default.", ".ADAPTER.")
                default_params[key] = param
            elif ".old." in name:
                key = name.replace(".old.", ".ADAPTER.")
                old_params[key] = param

        with torch.no_grad():
            for key in default_params:
                if key in old_params:
                    old_params[key].data.copy_(
                        decay * old_params[key].data + (1 - decay) * default_params[key].data
                    )

    # ========================================================================
    # Reward and advantage computation
    # ========================================================================

    def _compute_advantages(self, all_rewards: Tensor, adv_clip_max: float = 5.0) -> list[float]:
        """Compute batch-normalized advantages for all K rewards.

        Normalizes rewards within the current batch (z-score), clips to
        [-adv_clip_max, adv_clip_max], and maps to [0, 1] for use as NFT
        interpolation weights.

        Args:
            all_rewards: All rewards gathered across GPUs [K].
            adv_clip_max: Maximum absolute value for advantage clipping.

        Returns:
            List of advantage values r in [0, 1], one per reward in all_rewards.
        """
        rewards = all_rewards.tolist()
        mean = sum(rewards) / len(rewards)
        std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-6
        result = []
        for r in rewards:
            adv = (r - mean) / std
            adv = max(-adv_clip_max, min(adv_clip_max, adv))  # clip to [-adv_clip_max, adv_clip_max]
            result.append(adv / adv_clip_max / 2.0 + 0.5)  # map to [0, 1]
        return result

    # ========================================================================
    # VAE decode
    # ========================================================================

    def _decode_latent_to_pixels(self, latent: Tensor, device: torch.device) -> Tensor:
        """Decode patchified latent to pixel-space video.

        Args:
            latent: Patchified latent [1, seq_len, 128].
            device: Device to run decoding on.

        Returns:
            Pixel video [C, F, H, W] in [0, 1] range on CPU.
        """
        rl_cfg = self._rl_config

        latent_frames = rl_cfg.generation_num_frames // VIDEO_SCALE_FACTORS.time + 1
        latent_height = rl_cfg.generation_height // VIDEO_SCALE_FACTORS.height
        latent_width = rl_cfg.generation_width // VIDEO_SCALE_FACTORS.width

        # Unpatchify
        unpatchified = self._video_patchifier.unpatchify(
            latent,
            output_shape=VideoLatentShape(
                height=latent_height,
                width=latent_width,
                frames=latent_frames,
                batch=1,
                channels=128,
            ),
        )

        # Decode (VAE stays on GPU permanently)
        unpatchified = unpatchified.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            decoded = self._vae_decoder(unpatchified)

        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return decoded[0].float().cpu()  # [C, F, H, W]

    # ========================================================================
    # Saving
    # ========================================================================

    def _save_validation_videos(self, step: int) -> None:
        """Generate and save one validation video per GPU using EMA weights.

        Swaps EMA weights into the default adapter for generation, then restores.
        Each GPU generates its rank-th validation prompt with a fixed seed,
        producing files like `samples/step_00000_dog.mp4`.
        """
        device = self._accelerator.device
        rank = self._accelerator.process_index
        rl_cfg = self._rl_config

        vp = self._validation_prompts[rank]
        video_prompt_embeds = vp.embeddings.video_context_positive.to(device)

        self._transformer.eval()

        # Use EMA weights for validation if available
        if self._ema is not None and self._trainable_params is not None:
            self._ema.copy_ema_to(self._trainable_params)

        self._set_adapter("default")

        latent, _, _ = generate_video_latent(
            transformer=self._transformer,
            video_prompt_embeds=video_prompt_embeds,
            num_frames=rl_cfg.generation_num_frames,
            height=rl_cfg.generation_height,
            width=rl_cfg.generation_width,
            num_steps=rl_cfg.generation_steps,
            frame_rate=rl_cfg.frame_rate,
            seed=42,
            device=device,
        )
        pixel_video = self._decode_latent_to_pixels(latent, device)

        # Restore original weights
        if self._ema is not None and self._trainable_params is not None:
            self._ema.restore(self._trainable_params)

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"step_{step:05d}_{vp.nickname}.mp4"
        save_video(video_tensor=pixel_video, output_path=output_path, fps=rl_cfg.frame_rate)

        del pixel_video, latent
        free_gpu_memory()

        logger.info(f"Saved validation video: {output_path.name}")
        self._accelerator.wait_for_everyone()

    def _generate_validation_grid(self, step: int) -> None:
        """Run the validation_grid.py script to create a grid video."""
        if not IS_MAIN_PROCESS:
            return

        samples_dir = Path(self._config.output_dir) / "samples"
        if not samples_dir.is_dir():
            return

        script = Path(__file__).resolve().parents[5] / "scripts" / "validation_grid.py"
        if not script.exists():
            logger.warning(f"validation_grid.py not found at {script}")
            return

        cmd = [sys.executable, str(script), str(samples_dir), "--limit", "10"]
        if self._rl_config.validation_prompts_file:
            cmd.extend(["--prompts-file", str(self._rl_config.validation_prompts_file)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Validation grid generated at step {step}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to generate validation grid: {e.stderr[:200]}")

    def _save_checkpoint(
        self,
        step: int,
        epoch: int = 0,
        prompt_cycle_idx: int = 0,
        optimizer: AdamW | None = None,
    ) -> None:
        """Save LoRA checkpoint using EMA weights, plus full training state."""
        self._accelerator.wait_for_everyone()

        # Use EMA weights for checkpoint if available
        if self._ema is not None and self._trainable_params is not None:
            self._ema.copy_ema_to(self._trainable_params)

        self._set_adapter("default")
        # Collective op -- all processes must call this even if only main saves
        self._accelerator.get_state_dict(self._transformer)

        if IS_MAIN_PROCESS:
            save_dir = Path(self._config.output_dir) / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"rl_lora_weights_step_{step:05d}.safetensors"
            save_path = save_dir / filename

            save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=None)

            # Remove PEFT prefix, add ComfyUI prefix
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}
            state_dict = {k: v.to(save_dtype) for k, v in state_dict.items()}

            save_file(state_dict, save_path)
            logger.info(f"Checkpoint saved: {save_path.relative_to(self._config.output_dir)}")

        # Restore original weights
        if self._ema is not None and self._trainable_params is not None:
            self._ema.restore(self._trainable_params)

        # Save full training state for resumability
        if optimizer is not None:
            self._save_training_state(step, epoch, prompt_cycle_idx, optimizer)

    def _save_training_state(
        self,
        step: int,
        epoch: int,
        prompt_cycle_idx: int,
        optimizer: AdamW,
    ) -> None:
        """Save full training state (both adapters, optimizer, EMA, counters).

        Only runs on main process. Saved alongside the .safetensors checkpoint.
        """
        if not IS_MAIN_PROCESS:
            return

        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"rl_training_state_step_{step:05d}.pt"

        unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)

        # Extract default adapter weights
        unwrapped.set_adapter("default")
        default_state = get_peft_model_state_dict(unwrapped, adapter_name="default")
        default_state = {k: v.cpu().clone() for k, v in default_state.items()}

        # Extract old adapter weights
        unwrapped.set_adapter("old")
        old_state = get_peft_model_state_dict(unwrapped, adapter_name="old")
        old_state = {k: v.cpu().clone() for k, v in old_state.items()}

        # Restore active adapter
        unwrapped.set_adapter("default")

        training_state = {
            "global_step": step,
            "epoch": epoch,
            "prompt_cycle_idx": prompt_cycle_idx,
            "default_adapter": default_state,
            "old_adapter": old_state,
            "optimizer": optimizer.state_dict(),
            "ema": self._ema.state_dict() if self._ema is not None else None,
        }

        torch.save(training_state, save_path)
        logger.info(f"Training state saved: {save_path.relative_to(self._config.output_dir)}")

    def _load_training_state(
        self,
        checkpoint_path: str | Path,
        optimizer: AdamW,
    ) -> tuple[int, int, int]:
        """Load full training state from checkpoint.

        Must be called after dual LoRA setup, accelerator.prepare(optimizer),
        and EMA init.

        Args:
            checkpoint_path: Path to .pt file or checkpoint directory.
            optimizer: The prepared optimizer to restore state into.

        Returns:
            Tuple of (global_step, epoch, prompt_cycle_idx).
        """
        checkpoint_path = Path(checkpoint_path)

        # If directory, find the latest training state file
        if checkpoint_path.is_dir():
            state_files = sorted(checkpoint_path.glob("rl_training_state_step_*.pt"))
            if not state_files:
                raise FileNotFoundError(f"No training state files found in {checkpoint_path}")
            checkpoint_path = state_files[-1]

        logger.info(f"Loading training state from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)

        # Restore default adapter
        set_peft_model_state_dict(unwrapped, state["default_adapter"], adapter_name="default")

        # Restore old adapter
        set_peft_model_state_dict(unwrapped, state["old_adapter"], adapter_name="old")

        # Restore optimizer state
        optimizer.load_state_dict(state["optimizer"])

        # Restore EMA state
        if self._ema is not None and state.get("ema") is not None:
            self._ema.load_state_dict(state["ema"])

        global_step = state["global_step"]
        epoch = state["epoch"]
        prompt_cycle_idx = state["prompt_cycle_idx"]

        logger.info(
            f"Resumed from step {global_step} (epoch {epoch}, prompt_cycle_idx {prompt_cycle_idx})"
        )
        return global_step, epoch, prompt_cycle_idx

    def _save_config(self) -> None:
        """Save training config to output directory."""
        if not IS_MAIN_PROCESS:
            return
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

    # ========================================================================
    # Logging
    # ========================================================================

    def _init_wandb(self) -> None:
        """Initialize W&B if configured."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_cfg = self._config.wandb
        self._wandb_run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_cfg.tags + ["rl", "nft"],
            config=self._config.model_dump(),
        )

    def _setup_log_file(self) -> None:
        """Add a file handler so all logs go to {output_dir}/training.log."""
        if not IS_MAIN_PROCESS:
            return
        log_path = Path(self._config.output_dir) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        logger.info(f"Logging to {log_path}")

    @staticmethod
    def _cuda_time() -> float:
        """Return wall-clock time after synchronizing CUDA for accurate timing."""
        torch.cuda.synchronize()
        return time.time()

    @staticmethod
    def _wall_time() -> float:
        """Return wall-clock time without CUDA sync (for inner-loop timing)."""
        return time.time()

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to W&B."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
