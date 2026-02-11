"""RL Trainer for LTX-2 using the DiffusionNFT algorithm.

This trainer generates K videos per prompt using the current model, scores them
with a reward function, and uses the NFT loss formulation to update LoRA weights.

DDP strategy: With K=16 on 8 GPUs, each GPU generates K/num_gpus samples sequentially.
Rewards are all-gathered so each GPU sees all K rewards for advantage normalization.
Each GPU trains on its own generated samples.
"""

import logging
import os
import time
from pathlib import Path

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
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

from .generation import generate_video_latent
from .nft_loss import compute_nft_loss
from .rewards import get_reward_function

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class PerPromptStatTracker:
    """Track reward statistics per prompt for advantage normalization.

    Matches the reference DiffusionNFT stat_tracking.py:
    - Per-prompt mean subtraction adjusts for prompt-specific baselines
    - Global std is computed over the CURRENT BATCH only (not historical)
    - Stats are cleared each epoch to prevent stale baselines
    """

    def __init__(self) -> None:
        self._stats: dict[int, list[float]] = {}  # prompt_idx -> reward history

    def update(self, prompt_idx: int, rewards: list[float]) -> list[float]:
        """Add rewards for a prompt, return advantages (mean-subtracted, std-normalized).

        Args:
            prompt_idx: Index of the prompt these rewards came from.
            rewards: List of reward values for this batch.

        Returns:
            List of advantage values (same length as rewards).
        """
        if prompt_idx not in self._stats:
            self._stats[prompt_idx] = []
        self._stats[prompt_idx].extend(rewards)

        mean = sum(self._stats[prompt_idx]) / len(self._stats[prompt_idx])

        # Global std over CURRENT BATCH rewards only (matches reference)
        batch_mean = sum(rewards) / len(rewards)
        std = (sum((r - batch_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-4

        return [(r - mean) / std for r in rewards]

    def clear(self) -> None:
        """Clear all accumulated stats. Call at epoch boundaries."""
        self._stats.clear()


class RLTrainer:
    """RL trainer using DiffusionNFT for LTX-2.

    Generates videos, scores them with a reward function, and trains LoRA
    adapters using the NFT loss formulation.
    """

    def __init__(self, config: LtxTrainerConfig) -> None:
        self._config = config
        self._rl_config = config.rl

        # Load text encoder, cache prompt embeddings, then unload heavy parts
        self._cached_prompt_embeddings = self._load_text_encoder_and_cache_embeddings()

        # Load models
        self._load_models()

        # Setup accelerator
        self._setup_accelerator()

        # Setup dual LoRA adapters
        self._setup_dual_lora()

        # Prepare model with accelerator
        self._prepare_for_training()

        # Setup reward functions: list of (RewardFunction, weight, name) tuples
        self._reward_fns = [
            (get_reward_function(rc.type), rc.weight, rc.type)
            for rc in self._rl_config.rewards
        ]

        # Per-prompt stat tracker for advantage normalization
        self._stat_tracker = PerPromptStatTracker()

        # Patchifier for unpatchify during decode
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)

        # W&B
        self._wandb_run = None
        self._init_wandb()

        # File logging into output dir
        self._setup_log_file()

    def train(self) -> None:
        """Run the full RL training loop."""
        cfg = self._config
        rl_cfg = self._rl_config
        device = self._accelerator.device
        rank = self._accelerator.process_index
        num_processes = self._accelerator.num_processes
        accum_steps = cfg.optimization.gradient_accumulation_steps

        set_seed(cfg.seed + rank)

        # Compute samples per GPU and validate
        samples_per_gpu = rl_cfg.num_samples_per_prompt // num_processes
        assert rl_cfg.num_samples_per_prompt % num_processes == 0, (
            f"num_samples_per_prompt ({rl_cfg.num_samples_per_prompt}) must be "
            f"divisible by num_processes ({num_processes})"
        )

        # Setup optimizer with configurable AdamW params (reference uses weight_decay=1e-4)
        trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=cfg.optimization.learning_rate,
            betas=cfg.optimization.adam_betas,
            eps=cfg.optimization.adam_epsilon,
            weight_decay=cfg.optimization.adam_weight_decay,
        )
        optimizer = self._accelerator.prepare(optimizer)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        num_prompts = len(self._cached_prompt_embeddings)
        num_steps = cfg.optimization.steps

        logger.info(
            f"Starting RL training: {num_steps} iterations, "
            f"{num_steps // accum_steps} optimizer steps (accum={accum_steps}), "
            f"{num_prompts} prompts, K={rl_cfg.num_samples_per_prompt} "
            f"({samples_per_gpu} per GPU)"
        )

        # Convert optimizer-step intervals to raw-step intervals
        video_interval = (
            rl_cfg.video_save_interval * accum_steps
            if rl_cfg.video_save_interval is not None else None
        )
        ckpt_interval = (
            rl_cfg.checkpoint_save_interval * accum_steps
            if rl_cfg.checkpoint_save_interval is not None else None
        )

        # Save comparison videos at step 0 (before any training)
        if video_interval is not None:
            self._save_comparison_videos(step=0, comparison_prompt_idx=0, comparison_seed=42)

        self._accelerator.wait_for_everyone()
        opt_step = 0

        for step in range(num_steps):
            step_start = self._cuda_time()
            timings: dict[str, float] = {}

            # Pick prompt (cycle through). Clear stat tracker at epoch boundaries
            # (after cycling through all prompts) to prevent stale baselines.
            prompt_idx = step % num_prompts
            if prompt_idx == 0 and step > 0:
                self._stat_tracker.clear()

            cached = self._cached_prompt_embeddings[prompt_idx]
            video_prompt_embeds = cached.video_context_positive.to(device)

            # ============================================================
            # Phase 1: GENERATE all sub-samples for this GPU (batched)
            # ============================================================
            self._transformer.eval()
            self._set_adapter("old")

            local_latents = []
            local_positions = []
            local_rewards = []
            local_individual_rewards = []

            t0 = self._cuda_time()

            # Generate all samples in one batched forward pass
            gen_seeds = [
                step * num_processes * samples_per_gpu + rank * samples_per_gpu + k
                for k in range(samples_per_gpu)
            ]

            t_gen = self._cuda_time()
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
            total_gen_time = self._cuda_time() - t_gen

            # Decode and score each sample individually
            total_decode_time = 0.0
            total_reward_time = 0.0

            for k in range(samples_per_gpu):
                latent = all_latents[k : k + 1]      # [1, seq_len, 128]
                positions = all_positions[k : k + 1]  # [1, 3, seq_len, 2]

                t_dec = self._cuda_time()
                pixel_video = self._decode_latent_to_pixels(latent, device)
                total_decode_time += self._cuda_time() - t_dec

                t_rew = self._cuda_time()
                individual = {name: fn.compute(pixel_video) for fn, _, name in self._reward_fns}
                reward = sum(w * individual[name] for _, w, name in self._reward_fns)
                total_reward_time += self._cuda_time() - t_rew

                local_latents.append(latent)
                local_positions.append(positions)
                local_rewards.append(reward)
                local_individual_rewards.append(individual)

                # Save generated video for first sub-sample only
                if k == 0 and video_interval is not None and IS_MAIN_PROCESS and step % video_interval == 0:
                    self._save_labeled_video(pixel_video, step, "generated")

                del pixel_video

            del all_latents, all_positions

            timings["generation"] = total_gen_time
            timings["decode"] = total_decode_time
            timings["reward"] = total_reward_time
            timings["phase1_total"] = self._cuda_time() - t0

            # ============================================================
            # Phase 2: GATHER all K rewards across GPUs
            # ============================================================
            t0 = self._cuda_time()
            local_reward_tensor = torch.tensor(local_rewards, device=device, dtype=torch.float32)
            all_rewards = self._accelerator.gather(local_reward_tensor)  # [K]

            # Gather per-reward-function breakdowns across GPUs
            gathered_individual: dict[str, Tensor] = {}
            for _, _, name in self._reward_fns:
                local_vals = torch.tensor(
                    [d[name] for d in local_individual_rewards], device=device, dtype=torch.float32,
                )
                gathered_individual[name] = self._accelerator.gather(local_vals)  # [K]
            timings["gather"] = self._cuda_time() - t0

            # Log per-sample reward breakdown (all K samples) every accum cycle
            if IS_MAIN_PROCESS and step % accum_steps == 0:
                logger.info(f"  Per-sample rewards (K={len(all_rewards)}):")
                for i in range(len(all_rewards)):
                    parts = " ".join(
                        f"{name}={gathered_individual[name][i].item():+.4f}"
                        for _, _, name in self._reward_fns
                    )
                    logger.info(
                        f"    [{i:2d}] combined={all_rewards[i].item():+.4f} {parts}"
                    )

            # ============================================================
            # Phase 3: TRAIN on each sub-sample
            # Multi-timestep: for each sample, loop over N timesteps from the
            # generation sigma schedule, extracting more gradient signal per
            # expensive generation. Batched by adapter type per timestep.
            # ============================================================
            self._transformer.train()

            total_fwd_new = 0.0
            total_fwd_old = 0.0
            total_fwd_ref = 0.0
            total_nft_loss = 0.0
            total_backward = 0.0
            total_opt_step = 0.0
            total_adapter_switch = 0.0
            t_phase3 = self._cuda_time()

            # Compute per-prompt advantages for all K samples at once
            all_advantages = self._compute_advantages(prompt_idx, all_rewards)
            # This GPU's advantages are at offset rank*samples_per_gpu
            local_offset = rank * samples_per_gpu

            # Select training timesteps from the generation sigma schedule.
            # gen_sigmas has num_steps+1 entries (including 0). We use the
            # non-terminal sigmas as noise levels, matching the reference which
            # trains on the actual generation trajectory timesteps.
            num_train_ts = min(rl_cfg.num_train_timesteps, len(gen_sigmas) - 1)
            if num_train_ts > 1:
                # Uniformly sample num_train_ts indices from the generation schedule
                indices = torch.linspace(0, len(gen_sigmas) - 2, num_train_ts).long()
                train_sigmas = gen_sigmas[indices]  # [num_train_ts]
            else:
                # Single timestep: sample random from generation schedule
                idx = torch.randint(0, len(gen_sigmas) - 1, (1,))
                train_sigmas = gen_sigmas[idx]

            autocast_dtype = torch.bfloat16
            device_type = str(device).split(":")[0]

            # Loop over timesteps (outer) × samples (inner)
            for t_idx, sigma in enumerate(train_sigmas):
                t_val = sigma.float()

                # Pre-compute noisy inputs for all samples at this timestep
                sample_inputs: list[dict] = []
                for k in range(samples_per_gpu):
                    latent = local_latents[k]
                    positions = local_positions[k]
                    r = all_advantages[local_offset + k]
                    r_tensor = torch.tensor([r], device=device, dtype=torch.float32)

                    seq_len = latent.shape[1]
                    t_expanded = t_val.view(1, 1, 1)

                    noise = torch.randn_like(latent)
                    xt = (1 - t_expanded) * latent + t_expanded * noise

                    timesteps = t_val.expand(1, seq_len)
                    video_modality = Modality(
                        enabled=True,
                        latent=xt,
                        timesteps=timesteps,
                        positions=positions,
                        context=video_prompt_embeds,
                        context_mask=None,
                    )
                    sample_inputs.append({
                        "latent": latent, "xt": xt, "t_expanded": t_expanded,
                        "t_scalar": t_val, "r": r, "r_tensor": r_tensor,
                        "video_modality": video_modality,
                    })

                # --- All fwd_new passes (default adapter, with grad) ---
                t0 = self._cuda_time()
                self._set_adapter("default")
                total_adapter_switch += self._cuda_time() - t0

                v_new_list = []
                for k in range(samples_per_gpu):
                    t0 = self._cuda_time()
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        v_new, _ = self._transformer(
                            video=sample_inputs[k]["video_modality"], audio=None, perturbations=None,
                        )
                    v_new_list.append(v_new)
                    total_fwd_new += self._cuda_time() - t0

                # --- All fwd_old passes (old adapter, no grad) ---
                t0 = self._cuda_time()
                self._set_adapter("old")
                total_adapter_switch += self._cuda_time() - t0

                v_old_list = []
                for k in range(samples_per_gpu):
                    t0 = self._cuda_time()
                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        v_old, _ = self._transformer(
                            video=sample_inputs[k]["video_modality"], audio=None, perturbations=None,
                        )
                    v_old_list.append(v_old)
                    total_fwd_old += self._cuda_time() - t0

                # --- All fwd_ref passes (base model, no LoRA, no grad) ---
                unwrapped = self._accelerator.unwrap_model(self._transformer)
                t0 = self._cuda_time()
                unwrapped.base_model.disable_adapter_layers()
                unwrapped._adapters_disabled = True
                total_adapter_switch += self._cuda_time() - t0

                v_ref_list = []
                for k in range(samples_per_gpu):
                    t0 = self._cuda_time()
                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        v_ref, _ = self._transformer(
                            video=sample_inputs[k]["video_modality"], audio=None, perturbations=None,
                        )
                    v_ref_list.append(v_ref)
                    total_fwd_ref += self._cuda_time() - t0

                t0 = self._cuda_time()
                unwrapped.base_model.enable_adapter_layers()
                unwrapped._adapters_disabled = False
                self._set_adapter("default")
                total_adapter_switch += self._cuda_time() - t0

                # --- Loss + backward for each sample at this timestep ---
                for k in range(samples_per_gpu):
                    si = sample_inputs[k]

                    t0 = self._cuda_time()
                    loss, metrics = compute_nft_loss(
                        xt=si["xt"],
                        x0=si["latent"],
                        t=si["t_expanded"],
                        forward_pred=v_new_list[k],
                        old_pred=v_old_list[k].detach(),
                        ref_pred=v_ref_list[k].detach(),
                        r=si["r_tensor"],
                        beta=rl_cfg.nft_beta,
                        kl_beta=rl_cfg.kl_beta,
                        adv_clip_max=rl_cfg.adv_clip_max,
                    )
                    # Scale loss by number of training timesteps (accumulated gradients)
                    loss = loss / num_train_ts
                    total_nft_loss += self._cuda_time() - t0

                    with self._accelerator.accumulate(self._transformer):
                        t0 = self._cuda_time()
                        self._accelerator.backward(loss)
                        total_backward += self._cuda_time() - t0

                        t0 = self._cuda_time()
                        # Only step optimizer on the last timestep
                        if t_idx == num_train_ts - 1:
                            if self._config.optimization.max_grad_norm > 0:
                                self._accelerator.clip_grad_norm_(trainable_params, self._config.optimization.max_grad_norm)

                            optimizer.step()
                            optimizer.zero_grad()
                        total_opt_step += self._cuda_time() - t0

                del v_new_list, v_old_list, v_ref_list, sample_inputs

            timings["adapter_switch"] = total_adapter_switch
            timings["fwd_new"] = total_fwd_new
            timings["fwd_old"] = total_fwd_old
            timings["fwd_ref"] = total_fwd_ref
            timings["nft_loss"] = total_nft_loss
            timings["backward"] = total_backward
            timings["opt_step"] = total_opt_step
            timings["phase3_total"] = self._cuda_time() - t_phase3
            timings["num_train_timesteps"] = num_train_ts

            del local_latents, local_positions

            # ============================================================
            # 4. DECAY old adapter (only on actual optimizer steps)
            # ============================================================
            t0 = self._cuda_time()
            if self._accelerator.sync_gradients:
                opt_step += 1
                decay = min(opt_step * rl_cfg.decay_rate, rl_cfg.max_decay)
                self._decay_old_adapter(decay)
            timings["decay"] = self._cuda_time() - t0

            # ============================================================
            # 5. LOG
            # ============================================================
            step_time = self._cuda_time() - step_start
            timings["step_total"] = step_time

            if IS_MAIN_PROCESS:
                mean_reward = all_rewards.mean().item()
                max_reward = all_rewards.max().item()
                min_reward = all_rewards.min().item()

                mean_advantage = sum(all_advantages) / len(all_advantages)
                log_metrics = {
                    "rl/loss": metrics["total_loss"],
                    "rl/policy_loss": metrics["policy_loss"],
                    "rl/kl_loss": metrics["kl_loss"],
                    "rl/pos_loss": metrics["pos_loss"],
                    "rl/neg_loss": metrics["neg_loss"],
                    "rl/mean_reward": mean_reward,
                    "rl/max_reward": max_reward,
                    "rl/min_reward": min_reward,
                    "rl/mean_advantage": mean_advantage,
                    "rl/decay": decay if self._accelerator.sync_gradients else -1,
                    "rl/step_time": step_time,
                    "rl/num_train_timesteps": num_train_ts,
                    "rl/opt_step": opt_step,
                }
                # Log individual reward means (local GPU samples)
                for _, _, name in self._reward_fns:
                    vals = [d[name] for d in local_individual_rewards]
                    log_metrics[f"rl/reward/{name}"] = sum(vals) / len(vals)
                # Log all timings to W&B
                for tname, tval in timings.items():
                    log_metrics[f"rl/time/{tname}"] = tval
                self._log_metrics(log_metrics)

                # Build timing summary string
                timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in timings.items())

                if step % accum_steps == 0:
                    # Build individual reward string
                    reward_parts = []
                    for _, _, name in self._reward_fns:
                        vals = [d[name] for d in local_individual_rewards]
                        reward_parts.append(f"{name}={sum(vals)/len(vals):.4f}")
                    reward_detail = " ".join(reward_parts)

                    logger.info(
                        f"Step {step}/{num_steps} (opt {opt_step}) | "
                        f"loss={metrics['total_loss']:.4f} "
                        f"(policy={metrics['policy_loss']:.1f} kl={metrics['kl_loss']:.6f} "
                        f"kl_weighted={rl_cfg.kl_beta * metrics['kl_loss']:.4f}) | "
                        f"reward={mean_reward:.4f} [{min_reward:.4f}, {max_reward:.4f}] | "
                        f"{reward_detail} | "
                        f"adv={mean_advantage:.3f} | "
                        f"time={step_time:.1f}s"
                    )
                    logger.info(f"  Timings: {timing_str}")

            # Save comparison videos periodically
            if video_interval is not None and step > 0 and step % video_interval == 0:
                self._save_comparison_videos(
                    step=step, comparison_prompt_idx=0, comparison_seed=42,
                )

            # Save checkpoint
            if ckpt_interval and step > 0 and step % ckpt_interval == 0:
                self._save_checkpoint(step)

            self._accelerator.wait_for_everyone()

        # Final comparison videos + checkpoint
        if video_interval is not None:
            self._save_comparison_videos(step=num_steps, comparison_prompt_idx=0, comparison_seed=42)
        self._save_checkpoint(num_steps)

        if IS_MAIN_PROCESS and self._wandb_run is not None:
            self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()
        logger.info("RL training complete.")

    # ========================================================================
    # Model loading and setup
    # ========================================================================

    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings]:
        """Load text encoder, cache all prompt embeddings, unload heavy parts."""
        logger.debug("Loading text encoder...")
        text_encoder = load_text_encoder(
            checkpoint_path=self._config.model.model_path,
            gemma_model_path=self._config.model.text_encoder_path,
            device="cuda",
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        # Read prompts from file
        prompts_path = Path(self._rl_config.prompts_file)
        prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

        # Cache embeddings for all prompts
        cached = []
        with torch.inference_mode():
            for prompt in prompts:
                v_ctx, a_ctx, _ = text_encoder(prompt)
                cached.append(
                    CachedPromptEmbeddings(
                        video_context_positive=v_ctx.cpu(),
                        audio_context_positive=a_ctx.cpu(),
                    )
                )

        # Keep the embedding connectors, unload heavy Gemma model
        self._text_encoder = text_encoder
        self._text_encoder.model = None
        self._text_encoder.tokenizer = None
        self._text_encoder.feature_extractor_linear = None

        free_gpu_memory()
        logger.info(f"Cached embeddings for {len(cached)} prompts. Text encoder unloaded.")
        return cached

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
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
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

        # Keep VAE on GPU — ~250MB in bf16, plenty of headroom on 80GB H100s
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

    def _compute_advantages(self, prompt_idx: int, all_rewards: Tensor) -> list[float]:
        """Compute per-prompt normalized advantages for all K rewards.

        Uses PerPromptStatTracker for cross-batch normalization: per-prompt mean
        subtraction with global std normalization. This handles multi-reward
        scenarios where different prompts have different reward scales.

        Args:
            prompt_idx: Index of the current prompt.
            all_rewards: All rewards gathered across GPUs [K].

        Returns:
            List of advantage values r in [0, 1], one per reward in all_rewards.
        """
        reward_list = all_rewards.tolist()

        # Get per-prompt normalized advantages
        advantages = self._stat_tracker.update(prompt_idx, reward_list)

        adv_clip_max = self._rl_config.adv_clip_max
        result = []
        for adv in advantages:
            # Clip to [-adv_clip_max, adv_clip_max]
            adv = max(-adv_clip_max, min(adv_clip_max, adv))
            # Normalize to [0, 1]
            r = (adv + adv_clip_max) / (2 * adv_clip_max)
            result.append(r)
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

    def _save_labeled_video(self, pixel_video: Tensor, step: int, label: str) -> None:
        """Save a video to disk with a descriptive label."""
        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"step_{step:05d}_{label}.mp4"
        save_video(
            video_tensor=pixel_video,
            output_path=output_path,
            fps=self._rl_config.frame_rate,
        )

    def _save_comparison_videos(
        self, step: int, comparison_prompt_idx: int, comparison_seed: int,
    ) -> None:
        """Generate and save comparison videos: old adapter, new adapter, base model.

        Uses a fixed prompt and seed so videos are directly comparable across steps.
        All GPUs generate (required by DDP), but only rank 0 saves.
        """
        device = self._accelerator.device
        rl_cfg = self._rl_config
        cached = self._cached_prompt_embeddings[comparison_prompt_idx]
        video_prompt_embeds = cached.video_context_positive.to(device)

        self._transformer.eval()

        for adapter_name, label in [("old", "old"), ("default", "new")]:
            self._set_adapter(adapter_name)
            latent, _, _ = generate_video_latent(
                transformer=self._transformer,
                video_prompt_embeds=video_prompt_embeds,
                num_frames=rl_cfg.generation_num_frames,
                height=rl_cfg.generation_height,
                width=rl_cfg.generation_width,
                num_steps=rl_cfg.generation_steps,
                frame_rate=rl_cfg.frame_rate,
                seed=comparison_seed,
                device=device,
            )
            pixel_video = self._decode_latent_to_pixels(latent, device)
            if IS_MAIN_PROCESS:
                self._save_labeled_video(pixel_video, step, label)
            del pixel_video, latent
            free_gpu_memory()

        # Base model (no LoRA)
        unwrapped = self._accelerator.unwrap_model(self._transformer)
        with unwrapped.disable_adapter():
            latent, _, _ = generate_video_latent(
                transformer=self._transformer,
                video_prompt_embeds=video_prompt_embeds,
                num_frames=rl_cfg.generation_num_frames,
                height=rl_cfg.generation_height,
                width=rl_cfg.generation_width,
                num_steps=rl_cfg.generation_steps,
                frame_rate=rl_cfg.frame_rate,
                seed=comparison_seed,
                device=device,
            )
        pixel_video = self._decode_latent_to_pixels(latent, device)
        if IS_MAIN_PROCESS:
            self._save_labeled_video(pixel_video, step, "ref")
        del pixel_video, latent
        free_gpu_memory()

        if IS_MAIN_PROCESS:
            logger.info(f"Saved comparison videos at step {step}")

        self._accelerator.wait_for_everyone()

    def _save_checkpoint(self, step: int) -> None:
        """Save LoRA checkpoint (default adapter weights only)."""
        self._accelerator.wait_for_everyone()

        self._set_adapter("default")
        # Collective op — all processes must call this even if only main saves
        self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return

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

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to W&B."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
