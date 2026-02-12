"""RL Trainer for LTX-2 using the DiffusionNFT algorithm.

This trainer generates K videos per prompt using the current model, scores them
with a reward function, and uses the NFT loss formulation to update LoRA weights.

DDP strategy: With K=16 on 8 GPUs, each GPU generates K/num_gpus samples sequentially.
Rewards are all-gathered so each GPU sees all K rewards for advantage normalization.
Each GPU trains on its own generated samples.
"""

import logging
import os
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

    def train(self) -> None:
        """Run the full RL training loop."""
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

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        num_prompts = len(self._cached_prompt_embeddings)
        num_steps = cfg.optimization.steps

        assert rl_cfg.num_timesteps_per_sample <= rl_cfg.generation_steps, (
            f"num_timesteps_per_sample ({rl_cfg.num_timesteps_per_sample}) must be "
            f"<= generation_steps ({rl_cfg.generation_steps})"
        )

        logger.info(
            f"Starting RL training: {num_steps} optimizer steps, "
            f"{num_prompts} prompts, K={rl_cfg.num_samples_per_prompt} "
            f"({samples_per_gpu} per GPU), "
            f"T={rl_cfg.num_timesteps_per_sample} timesteps/sample"
        )

        # Save validation videos at step 0 (before any training)
        if rl_cfg.video_save_interval is not None and self._validation_prompts:
            self._save_validation_videos(step=0)

        self._accelerator.wait_for_everyone()

        for step in range(num_steps):
            step_start = self._cuda_time()
            timings: dict[str, float] = {}

            # Pick prompt (cycle through)
            prompt_idx = step % num_prompts
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

            t0 = self._wall_time()

            # Generate all samples in one batched forward pass
            gen_seeds = [
                step * num_processes * samples_per_gpu + rank * samples_per_gpu + k
                for k in range(samples_per_gpu)
            ]

            t_gen = self._wall_time()
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
            total_gen_time = self._wall_time() - t_gen

            # Decode and score each sample individually
            total_decode_time = 0.0
            total_reward_time = 0.0

            for k in range(samples_per_gpu):
                latent = all_latents[k : k + 1]      # [1, seq_len, 128]
                positions = all_positions[k : k + 1]  # [1, 3, seq_len, 2]

                t_dec = self._wall_time()
                pixel_video = self._decode_latent_to_pixels(latent, device) # [C, F, H, W]
                total_decode_time += self._wall_time() - t_dec

                t_rew = self._wall_time()
                prompt_text = cached.prompt_text
                individual = {name: fn.compute(pixel_video, prompt=prompt_text) for fn, name in self._reward_fns}
                reward = sum(individual.values())
                total_reward_time += self._wall_time() - t_rew

                local_latents.append(latent)
                local_positions.append(positions)
                local_rewards.append(reward)
                local_individual_rewards.append(individual)

                del pixel_video

            del all_latents, all_positions

            timings["generation"] = total_gen_time
            timings["decode"] = total_decode_time
            timings["reward"] = total_reward_time
            timings["phase1_total"] = self._cuda_time() - t0

            # ============================================================
            # Phase 2: GATHER all K rewards across GPUs
            # ============================================================
            t0 = self._wall_time()
            local_reward_tensor = torch.tensor(local_rewards, device=device, dtype=torch.float32)
            all_rewards = self._accelerator.gather(local_reward_tensor)  # [K]

            # Gather per-reward-function breakdowns across GPUs
            gathered_individual: dict[str, Tensor] = {}
            for _, name in self._reward_fns:
                local_vals = torch.tensor(
                    [d[name] for d in local_individual_rewards], device=device, dtype=torch.float32,
                )
                gathered_individual[name] = self._accelerator.gather(local_vals)  # [K]
            timings["gather"] = self._wall_time() - t0

            # ============================================================
            # Phase 3: TRAIN on each sub-sample (multi-timestep)
            # Restructured: group by adapter to minimize adapter switches
            # (3 switches instead of 4*num_timesteps).
            # ============================================================
            self._transformer.train()

            total_fwd_new = 0.0
            total_fwd_old = 0.0
            total_fwd_ref = 0.0
            total_nft_loss = 0.0
            total_backward = 0.0
            total_adapter_switch = 0.0
            t_phase3 = self._cuda_time()

            # Compute advantages for all K samples at once
            all_advantages = self._compute_advantages(all_rewards)
            # This GPU's advantages are at offset rank*samples_per_gpu
            local_offset = rank * samples_per_gpu

            # Select random timestep indices from the generation sigma schedule
            num_timesteps = min(rl_cfg.num_timesteps_per_sample, len(gen_sigmas) - 1)
            sigma_indices = torch.randperm(len(gen_sigmas) - 1)[:num_timesteps]

            autocast_dtype = torch.bfloat16
            device_type = str(device).split(":")[0]

            # Total gradient accumulation divisor: we do one backward per
            # timestep with a batched loss that already averages over
            # samples_per_gpu, so divide by num_timesteps only.
            total_grad_accum = num_timesteps
            backward_pass_count = 0

            accumulated_metrics: dict[str, Tensor] = {}

            # ----------------------------------------------------------
            # 1. Prepare ALL inputs for ALL timesteps upfront
            # ----------------------------------------------------------
            # all_inputs[t_idx] = list of dicts, one per sample
            all_inputs: list[list[dict]] = []
            # all_batched_modalities[t_idx] = batched Modality for this timestep
            all_batched_modalities: list[Modality] = []

            for t_idx in range(num_timesteps):
                t_val = gen_sigmas[sigma_indices[t_idx]].float()
                timestep_inputs: list[dict] = []
                xt_list = []
                ts_list = []
                pos_list = []

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

                    timestep_inputs.append({
                        "latent": latent, "xt": xt, "t_expanded": t_expanded,
                        "r_tensor": r_tensor,
                    })
                    xt_list.append(xt)
                    ts_list.append(timesteps)
                    pos_list.append(positions)

                all_inputs.append(timestep_inputs)

                # Build batched Modality for this timestep
                batched_modality = Modality(
                    enabled=True,
                    latent=torch.cat(xt_list, dim=0),
                    timesteps=torch.cat(ts_list, dim=0),
                    positions=torch.cat(pos_list, dim=0),
                    context=video_prompt_embeds.expand(samples_per_gpu, -1, -1),
                    context_mask=None,
                )
                all_batched_modalities.append(batched_modality)

            # ----------------------------------------------------------
            # 2. ALL fwd_old passes (1 adapter switch, no_grad, batched)
            # ----------------------------------------------------------
            t0 = self._wall_time()
            self._set_adapter("old")
            total_adapter_switch += self._wall_time() - t0

            v_old_all: list[Tensor] = []  # [num_timesteps] of [samples_per_gpu, seq, C]
            t0 = self._wall_time()
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                for t_idx in range(num_timesteps):
                    v_old_batched, _ = self._transformer(
                        video=all_batched_modalities[t_idx], audio=None, perturbations=None,
                    )
                    v_old_all.append(v_old_batched.detach())
            total_fwd_old = self._wall_time() - t0

            # ----------------------------------------------------------
            # 3. ALL fwd_ref passes (disable adapters, no_grad, batched)
            # ----------------------------------------------------------
            unwrapped = self._accelerator.unwrap_model(self._transformer)
            t0 = self._wall_time()

            v_ref_all: list[Tensor] = []  # [num_timesteps] of [samples_per_gpu, seq, C]
            with unwrapped.disable_adapter():
                total_adapter_switch += self._wall_time() - t0
                t0 = self._wall_time()
                with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    for t_idx in range(num_timesteps):
                        v_ref_batched, _ = self._transformer(
                            video=all_batched_modalities[t_idx], audio=None, perturbations=None,
                        )
                        v_ref_all.append(v_ref_batched.detach())
            total_fwd_ref = self._wall_time() - t0

            # ----------------------------------------------------------
            # 4. fwd_new + loss + backward (default adapter, with grad)
            #    One batched fwd + backward per timestep.
            # ----------------------------------------------------------
            t0 = self._wall_time()
            self._set_adapter("default")
            total_adapter_switch += self._wall_time() - t0

            for t_idx in range(num_timesteps):
                t0 = self._wall_time()
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    v_new_batched, _ = self._transformer(
                        video=all_batched_modalities[t_idx], audio=None, perturbations=None,
                    )
                total_fwd_new += self._wall_time() - t0

                # Build batched tensors for NFT loss
                batched_xt = torch.cat([si["xt"] for si in all_inputs[t_idx]], dim=0)
                batched_x0 = torch.cat([si["latent"] for si in all_inputs[t_idx]], dim=0)
                batched_t = torch.cat([si["t_expanded"] for si in all_inputs[t_idx]], dim=0)
                batched_r = torch.cat([si["r_tensor"] for si in all_inputs[t_idx]], dim=0)

                t0 = self._wall_time()
                loss, metrics = compute_nft_loss(
                    xt=batched_xt,
                    x0=batched_x0,
                    t=batched_t,
                    forward_pred=v_new_batched,
                    old_pred=v_old_all[t_idx].detach(),
                    ref_pred=v_ref_all[t_idx].detach(),
                    r=batched_r,
                    beta=rl_cfg.nft_beta,
                    kl_beta=rl_cfg.kl_beta,
                )
                loss = loss / total_grad_accum
                total_nft_loss += self._wall_time() - t0

                # Accumulate metrics as tensors (no .item() yet)
                for mkey, mval in metrics.items():
                    if mkey not in accumulated_metrics:
                        accumulated_metrics[mkey] = mval / total_grad_accum
                    else:
                        accumulated_metrics[mkey] = accumulated_metrics[mkey] + mval / total_grad_accum

                t0 = self._wall_time()
                backward_pass_count += 1
                if backward_pass_count < total_grad_accum:
                    with self._accelerator.no_sync(self._transformer):
                        self._accelerator.backward(loss)
                else:
                    # DDP sync on the very last backward pass
                    self._accelerator.backward(loss)
                total_backward += self._wall_time() - t0

                del v_new_batched

            del v_old_all, v_ref_all, all_batched_modalities, all_inputs

            # Convert accumulated tensor metrics to floats
            accumulated_metrics = {k: v.item() for k, v in accumulated_metrics.items()}

            # Optimizer step
            t0 = self._wall_time()
            if cfg.optimization.max_grad_norm > 0:
                self._accelerator.clip_grad_norm_(trainable_params, cfg.optimization.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_opt_step = self._wall_time() - t0

            timings["adapter_switch"] = total_adapter_switch
            timings["fwd_new"] = total_fwd_new
            timings["fwd_old"] = total_fwd_old
            timings["fwd_ref"] = total_fwd_ref
            timings["nft_loss"] = total_nft_loss
            timings["backward"] = total_backward
            timings["opt_step"] = total_opt_step
            timings["phase3_total"] = self._cuda_time() - t_phase3

            del local_latents, local_positions

            # ============================================================
            # 4. DECAY old adapter (every old_update_interval optimizer steps)
            # ============================================================
            t0 = self._wall_time()
            opt_step_num = step + 1
            decay_value = -1.0
            if opt_step_num % rl_cfg.old_update_interval == 0:
                decay_value = min(opt_step_num * rl_cfg.decay_rate, rl_cfg.max_decay)
                self._decay_old_adapter(decay_value)
            timings["decay"] = self._wall_time() - t0

            # ============================================================
            # 5. LOG
            # ============================================================
            step_time = self._cuda_time() - step_start
            timings["step_total"] = step_time

            if IS_MAIN_PROCESS:
                mean_reward = all_rewards.mean().item()
                std_reward = all_rewards.std().item() if len(all_rewards) > 1 else 0.0
                max_reward = all_rewards.max().item()
                min_reward = all_rewards.min().item()

                mean_advantage = sum(all_advantages) / len(all_advantages)
                log_metrics = {
                    "rl/loss": accumulated_metrics["total_loss"],
                    "rl/policy_loss": accumulated_metrics["policy_loss"],
                    "rl/kl_loss": accumulated_metrics["kl_loss"],
                    "rl/pos_loss": accumulated_metrics["pos_loss"],
                    "rl/neg_loss": accumulated_metrics["neg_loss"],
                    "rl/mean_reward": mean_reward,
                    "rl/reward_std": std_reward,
                    "rl/max_reward": max_reward,
                    "rl/min_reward": min_reward,
                    "rl/mean_advantage": mean_advantage,
                    "rl/decay": decay_value,
                    "rl/step_time": step_time,
                    "rl/num_timesteps_per_sample": num_timesteps,
                }
                # Log individual reward means and stds (all K samples)
                for _, name in self._reward_fns:
                    log_metrics[f"rl/reward/{name}"] = gathered_individual[name].mean().item()
                    log_metrics[f"rl/reward/{name}_std"] = gathered_individual[name].std().item() if len(gathered_individual[name]) > 1 else 0.0
                # Log all timings to W&B
                for tname, tval in timings.items():
                    log_metrics[f"rl/time/{tname}"] = tval
                self._log_metrics(log_metrics)

                # Build per-reward detail string
                reward_parts = []
                for _, name in self._reward_fns:
                    r_mean = gathered_individual[name].mean().item()
                    r_std = gathered_individual[name].std().item() if len(gathered_individual[name]) > 1 else 0.0
                    reward_parts.append(f"{name}: \u03bc={r_mean:.2f} \u03c3={r_std:.2f}")
                reward_detail = " | ".join(reward_parts)

                kl_raw = accumulated_metrics["kl_loss"]
                kl_weighted = rl_cfg.kl_beta * kl_raw

                gen_time = timings.get("generation", 0.0)
                rew_time = timings.get("decode", 0.0) + timings.get("reward", 0.0)
                train_time = timings.get("phase3_total", 0.0)

                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"loss={accumulated_metrics['total_loss']:.3f} "
                    f"(pol={accumulated_metrics['policy_loss']:.1f} "
                    f"kl={kl_raw:.2f}\u00d7{rl_cfg.kl_beta}={kl_weighted:.3f}) | "
                    f"reward: \u03bc={mean_reward:.2f} \u03c3={std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}] | "
                    f"{reward_detail} | "
                    f"adv={mean_advantage:.3f} | "
                    f"{step_time:.1f}s (gen={gen_time:.1f} rew={rew_time:.1f} train={train_time:.1f})"
                )

            # Save validation videos periodically
            if rl_cfg.video_save_interval is not None and self._validation_prompts and (step + 1) % rl_cfg.video_save_interval == 0:
                self._save_validation_videos(step=step + 1)
                if rl_cfg.validation_grid_interval and (step + 1) % rl_cfg.validation_grid_interval == 0:
                    self._generate_validation_grid(step + 1)

            # Save checkpoint
            if rl_cfg.checkpoint_save_interval and (step + 1) % rl_cfg.checkpoint_save_interval == 0:
                self._save_checkpoint(step + 1)

            self._accelerator.wait_for_everyone()

        # Final validation videos + grid + checkpoint
        if rl_cfg.video_save_interval is not None and self._validation_prompts:
            self._save_validation_videos(step=num_steps)
            if rl_cfg.validation_grid_interval:
                self._generate_validation_grid(num_steps)
        self._save_checkpoint(num_steps)

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

    def _compute_advantages(self, all_rewards: Tensor) -> list[float]:
        """Compute batch-normalized advantages for all K rewards.

        Normalizes rewards within the current batch (z-score), clips to [-1, 1],
        and maps to [0, 1] for use as NFT interpolation weights.

        Args:
            all_rewards: All rewards gathered across GPUs [K].

        Returns:
            List of advantage values r in [0, 1], one per reward in all_rewards.
        """
        rewards = all_rewards.tolist()
        mean = sum(rewards) / len(rewards)
        std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-6
        result = []
        for r in rewards:
            adv = (r - mean) / std
            adv = max(-1.0, min(1.0, adv))  # clip to [-1, 1]
            result.append(adv * 0.5 + 0.5)  # map to [0, 1]
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
        """Generate and save one validation video per GPU using the old adapter.

        Each GPU generates its rank-th validation prompt with a fixed seed,
        producing files like `samples/step_00000_dog.mp4`.
        """
        device = self._accelerator.device
        rank = self._accelerator.process_index
        rl_cfg = self._rl_config

        vp = self._validation_prompts[rank]
        video_prompt_embeds = vp.embeddings.video_context_positive.to(device)

        self._transformer.eval()
        self._set_adapter("old")

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

    def _save_checkpoint(self, step: int) -> None:
        """Save LoRA checkpoint (old adapter weights)."""
        self._accelerator.wait_for_everyone()

        self._set_adapter("old")
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

    @staticmethod
    def _wall_time() -> float:
        """Return wall-clock time without CUDA sync (for inner-loop timing)."""
        return time.time()

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to W&B."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
