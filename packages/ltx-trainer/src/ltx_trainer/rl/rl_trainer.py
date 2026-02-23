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
from .nft_loss import compute_nft_loss, compute_per_objective_nft_loss
from .preference_sampling import sample_preference_for_prompt, sample_preferences_with_subgroups
from .rewards import get_reward_functions
from .stat_tracker import compute_pareto_advantages

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

        # Enable preference conditioning before PEFT wraps the model (pareto mode only)
        if self._rl_config.preference_mode == "pareto":
            from .rewards import count_reward_dimensions
            num_rewards_pred = sum(count_reward_dimensions(rc.type) for rc in self._rl_config.rewards)
            self._transformer.enable_preference_conditioning(num_rewards_pred)

        # Setup accelerator
        self._setup_accelerator()

        # Setup dual LoRA adapters
        self._setup_dual_lora()

        # Prepare model with accelerator
        self._prepare_for_training()

        # Setup reward functions: list of (RewardFunction, name) tuples
        # video_score expands into one reward per dimension
        self._reward_fns = []
        self._reward_names = []
        for rc in self._rl_config.rewards:
            expanded = get_reward_functions(rc.type)
            self._reward_fns.extend(expanded)
            if rc.name and len(expanded) == 1:
                self._reward_names.append(rc.name)
            else:
                self._reward_names.extend([name for _, name in expanded])

        # Reward metadata for pareto mode
        reward_weights_list: list[float] = []
        for rc in self._rl_config.rewards:
            expanded = get_reward_functions(rc.type)
            reward_weights_list.extend([rc.weight] * len(expanded))
        self._reward_weights = torch.tensor(reward_weights_list, dtype=torch.float32)
        self._num_rewards = len(self._reward_fns)

        # Verify preference conditioning dimension matches actual rewards
        if self._rl_config.preference_mode == "pareto":
            from .rewards import count_reward_dimensions
            expected = sum(count_reward_dimensions(rc.type) for rc in self._rl_config.rewards)
            assert self._num_rewards == expected, (
                f"Reward dimension mismatch: count_reward_dimensions predicted {expected}, "
                f"but actual reward functions give {self._num_rewards}"
            )

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
        preference_mode = rl_cfg.preference_mode
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
            f"ema_decay={rl_cfg.ema_decay}, "
            f"num_pref_per_prompt={rl_cfg.num_pref_per_prompt}"
        )
        if rl_cfg.num_pref_per_prompt > 1:
            sub_group_size = rl_cfg.num_samples_per_prompt // rl_cfg.num_pref_per_prompt
            logger.info(
                f"Multi-preference: {rl_cfg.num_pref_per_prompt} preferences per prompt, "
                f"{sub_group_size} samples per sub-group for GDPO normalization"
            )

        # Save validation videos at step 0 (before any training)
        if rl_cfg.video_save_interval is not None and self._validation_prompts:
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

                # Sample preference vectors BEFORE generation (for model conditioning)
                pref = None  # (R,) single pref or None
                all_prefs = None  # (K, R) per-sample prefs or None
                pref_slots = None  # (K,) slot indices or None
                num_pref_per_prompt = rl_cfg.num_pref_per_prompt

                if preference_mode == "pareto":
                    if num_pref_per_prompt > 1:
                        # Multi-pref: K distinct preferences cycled across all samples
                        all_prefs_full, pref_slots_full = sample_preferences_with_subgroups(
                            cached.prompt_text, self._num_rewards, num_pref_per_prompt,
                            rl_cfg.num_samples_per_prompt, cfg.seed, epoch,
                        )  # (num_samples_per_prompt, R) and (num_samples_per_prompt,)
                        # This GPU's slice
                        local_start = rank * samples_per_gpu
                        local_end = local_start + samples_per_gpu
                        all_prefs = all_prefs_full[local_start:local_end]  # (samples_per_gpu, R)
                        pref_slots = pref_slots_full  # keep full for gathering later
                        # For generation: pass per-sample preferences (samples_per_gpu, R)
                        pref = all_prefs  # will be (B, R) in generate_video_latent
                    else:
                        # Single pref: all K repeats share the same preference
                        pref = sample_preference_for_prompt(
                            cached.prompt_text, self._num_rewards, cfg.seed, epoch, prompt_cycle_idx - 1,
                        )  # (R,) tensor

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
                    preference=pref,
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
                    individual = {rname: fn.compute(pixel_video, prompt=prompt_text) for (fn, _), rname in zip(self._reward_fns, self._reward_names)}
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
                    "prompt_text": cached.prompt_text,
                    "preferences": pref,  # (R,) or (B, R) tensor or None
                    "all_prefs": all_prefs,  # (samples_per_gpu, R) or None (multi-pref)
                    "pref_slots": pref_slots,  # (K,) full or None (multi-pref)
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
                for rname in self._reward_names:
                    local_vals = torch.tensor(
                        [d[rname] for d in ps["individual_rewards"]],
                        device=device, dtype=torch.float32,
                    )
                    gi[rname] = self._accelerator.gather(local_vals)  # [K]
                all_gathered_individual.append(gi)

            all_flat_rewards = torch.cat(all_prompt_rewards)  # [num_prompts_per_epoch * K]

            # Per-prompt advantages and (for pareto mode) per-objective advantages/preferences
            per_prompt_advantages: list[list[float]] = []
            per_prompt_advantages_per_obj: list[Tensor] = []  # pareto: list of (K, R)
            per_prompt_preferences: list[Tensor] = []  # pareto: list of (K, R)

            if preference_mode == "pareto":
                # Build reward_vectors (N_total, R) and group keys from gathered individual rewards
                # Matching ParetoControl: NO reward weight multiplication on raw rewards.
                # Weights are not used in per-objective mode (preferences drive weighting).
                K = rl_cfg.num_samples_per_prompt
                num_pref_pp = rl_cfg.num_pref_per_prompt
                reward_vectors_list = []
                group_keys_list: list = []

                for p_idx, gi in enumerate(all_gathered_individual):
                    # gi[name] is (K,) for each reward name
                    prompt_reward_vec = torch.stack([gi[name] for name in self._reward_names], dim=1)  # (K, R)
                    reward_vectors_list.append(prompt_reward_vec)

                    # Build group keys: composite (prompt__prefSlot) for multi-pref, else prompt index
                    if num_pref_pp > 1:
                        ps = epoch_samples[p_idx]
                        # Gather pref_slots across GPUs to get full (K,) slots
                        local_slots = ps["pref_slots"][rank * samples_per_gpu:(rank + 1) * samples_per_gpu]
                        local_slots_t = local_slots.to(device)
                        gathered_slots = self._accelerator.gather(local_slots_t)  # (K,)
                        for slot_val in gathered_slots.tolist():
                            group_keys_list.append(f"{p_idx}__pref{int(slot_val)}")
                    else:
                        group_keys_list.extend([p_idx] * K)

                reward_vectors = torch.cat(reward_vectors_list, dim=0)  # (N_total, R)
                all_advantages_per_obj = compute_pareto_advantages(
                    reward_vectors, group_keys_list,
                )  # (N_total, R)

                # Build per-prompt structures
                offset = 0
                for p_idx, ps in enumerate(epoch_samples):
                    prompt_advs_obj = all_advantages_per_obj[offset:offset + K]  # (K, R)
                    per_prompt_advantages_per_obj.append(prompt_advs_obj)

                    # Build per-sample preferences (K, R)
                    if num_pref_pp > 1:
                        # Multi-pref: gather per-sample preferences across GPUs
                        local_prefs = ps["all_prefs"].to(device)  # (samples_per_gpu, R)
                        pref_expanded = self._accelerator.gather(local_prefs)  # (K, R)
                    else:
                        # Single pref: expand (R,) to (K, R)
                        pref_expanded = ps["preferences"].unsqueeze(0).expand(K, -1).to(device)
                    per_prompt_preferences.append(pref_expanded)

                    # Legacy scalar advantages not used in pareto mode, but fill for logging
                    per_prompt_advantages.append([0.5] * K)
                    offset += K
            elif preference_mode == "per_reward_zscore":
                # Independent per-reward z-score: z-score each reward separately, average into one scalar
                reward_names = self._reward_names
                R = len(reward_names)
                K = rl_cfg.num_samples_per_prompt

                # Compute global std per reward across all prompts
                global_std_per_reward: dict[str, float] = {}
                for rname in reward_names:
                    all_vals = torch.cat([gi[rname] for gi in all_gathered_individual])
                    global_std_per_reward[rname] = all_vals.std().item() + 1e-6

                for p_idx, gi in enumerate(all_gathered_individual):
                    # Z-score each reward independently: (val - prompt_mean) / global_std
                    zscores = []
                    for rname in reward_names:
                        vals = gi[rname]  # (K,)
                        prompt_mean = vals.mean().item()
                        g_std = global_std_per_reward[rname]
                        z = (vals - prompt_mean) / g_std  # (K,)
                        zscores.append(z)

                    # Average z-scores across rewards → single scalar per sample
                    avg_z = torch.stack(zscores, dim=0).mean(dim=0)  # (K,)

                    # Clip and map to [0, 1] (same as legacy)
                    advs = []
                    for z_val in avg_z.tolist():
                        adv = max(-rl_cfg.adv_clip_max, min(rl_cfg.adv_clip_max, z_val))
                        advs.append(adv / rl_cfg.adv_clip_max / 2.0 + 0.5)
                    per_prompt_advantages.append(advs)
            else:
                # Legacy path: per-prompt z-score → clip → map to [0, 1]
                global_std = all_flat_rewards.std().item() + 1e-6
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
                    sample_dict = {
                        "latent": ps["latents"][k],
                        "positions": ps["positions"][k],
                        "advantage": per_prompt_advantages[p_idx][local_offset + k],
                        "prompt_embeds": ps["prompt_embeds"],
                        "gen_sigmas": ps["gen_sigmas"],
                    }
                    if preference_mode == "pareto":
                        global_k = local_offset + k
                        sample_dict["advantages_per_obj"] = per_prompt_advantages_per_obj[p_idx][global_k]  # (R,)
                        sample_dict["preferences"] = per_prompt_preferences[p_idx][global_k]  # (R,)
                    flat_samples.append(sample_dict)

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

                        input_item = {
                            "latent": latent, "xt": xt,
                            "t_expanded": t_expanded, "r_tensor": r_tensor,
                        }
                        if preference_mode == "pareto":
                            input_item["r_per_obj"] = sample["advantages_per_obj"]  # (R,)
                            input_item["preferences"] = sample["preferences"]  # (R,)
                        input_list.append(input_item)
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

                # --- Build preference tensor for this micro-batch (pareto mode) ---
                mb_pref = None
                if preference_mode == "pareto":
                    mb_pref = torch.stack([s["preferences"] for s in mb_samples]).to(device)  # (mb_size, R)

                # --- Build mega-modality (all timesteps concatenated along batch dim) ---
                mega_pref = mb_pref.repeat(num_timesteps, 1) if mb_pref is not None else None
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
                        preference=mega_pref,
                    )
                    v_old_all = list(v_old_mega.split(mb_size, dim=0))
                total_fwd_old += self._wall_time() - t0

                # --- ALL ref fwd passes (mega-batched, unconditioned baseline) ---
                unwrapped = self._accelerator.unwrap_model(self._transformer)
                t0 = self._wall_time()
                with unwrapped.disable_adapter():
                    total_adapter_switch += self._wall_time() - t0
                    t0 = self._wall_time()
                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        v_ref_mega, _ = self._transformer(
                            video=mega_modality, audio=None, perturbations=None,
                            preference=None,
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
                                preference=mb_pref,
                            )
                        v_new_chunks = [v_new_batched]
                    else:
                        # Multiple timesteps: cat modalities, single forward, split output
                        group_pref = mb_pref.repeat(group_size, 1) if mb_pref is not None else None
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
                                preference=group_pref,
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

                        if preference_mode == "pareto":
                            batched_adv_per_obj = torch.stack([si["r_per_obj"] for si in input_list])  # (B, R)
                            batched_prefs = torch.stack([si["preferences"] for si in input_list])  # (B, R)
                            loss, metrics = compute_per_objective_nft_loss(
                                xt=batched_xt, x0=batched_x0, t=batched_t,
                                forward_pred=v_new_chunks[k],
                                old_pred=v_old_all[j_idx].detach(),
                                ref_pred=v_ref_all[j_idx].detach(),
                                advantages_per_obj=batched_adv_per_obj,
                                preferences=batched_prefs,
                                beta=rl_cfg.nft_beta,
                                kl_beta=rl_cfg.kl_beta,
                                adv_clip_max=rl_cfg.adv_clip_max,
                            )
                        else:
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
                for rname in self._reward_names:
                    all_vals = torch.cat([gi[rname] for gi in all_gathered_individual])
                    log_metrics[f"rl/reward/{rname}"] = all_vals.mean().item()
                    log_metrics[f"rl/reward/{rname}_std"] = all_vals.std().item() if len(all_vals) > 1 else 0.0

                # Pareto-specific metrics
                if preference_mode == "pareto":
                    # Mean preference weight per objective
                    all_prefs = torch.cat(per_prompt_preferences, dim=0)  # (N_total, R)
                    for r_idx, name in enumerate(self._reward_names):
                        log_metrics[f"rl/pref_w/{name}"] = all_prefs[:, r_idx].mean().item()
                    # Per-objective policy losses from accumulated metrics
                    if accumulated_metrics_snapshot:
                        for r_idx, name in enumerate(self._reward_names):
                            key = f"policy_loss_obj_{r_idx}"
                            if key in accumulated_metrics_snapshot:
                                log_metrics[f"rl/policy_loss/{name}"] = accumulated_metrics_snapshot[key]
                    # Log preference gate scalar and adaln norm for monitoring conditioning strength
                    for pname, pparam in self._transformer.named_parameters():
                        if "pref_gate" in pname and "default" in pname and pparam.numel() == 1:
                            log_metrics["rl/pref_gate"] = pparam.item()
                        if "pref_adaln" in pname and "default" in pname and pname.endswith("weight"):
                            log_metrics["rl/pref_adaln_norm"] = pparam.norm().item()
                        if "rl/pref_gate" in log_metrics and "rl/pref_adaln_norm" in log_metrics:
                            break

                for tname, tval in timings.items():
                    log_metrics[f"rl/time/{tname}"] = tval
                self._log_metrics(log_metrics)

                # Build per-reward detail string
                reward_parts = []
                for rname in self._reward_names:
                    all_vals = torch.cat([gi[rname] for gi in all_gathered_individual])
                    r_mean = all_vals.mean().item()
                    r_std = all_vals.std().item() if len(all_vals) > 1 else 0.0
                    reward_parts.append(f"{rname}: mu={r_mean:.2f} s={r_std:.2f}")
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

        # Include preference conditioning modules in per-adapter copies if present
        has_pref_conditioning = hasattr(self._transformer, "pref_conditioner")
        modules_to_save = ["pref_conditioner", "pref_gate", "pref_adaln"] if has_pref_conditioning else None

        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights=True,
            modules_to_save=modules_to_save,
        )

        # Add default adapter
        self._transformer = get_peft_model(self._transformer, lora_config)

        # Add 'old' adapter as a copy of default
        self._transformer.add_adapter("old", lora_config)

        # Copy default weights to old adapter
        self._sync_old_adapter_from_default()

        # Freeze 'old' adapter parameters (LoRA weights + modules_to_save)
        for name, param in self._transformer.named_parameters():
            if "old" in name and ("lora" in name or "modules_to_save" in name):
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
            if "lora" not in name and "modules_to_save" not in name:
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
            if "lora" not in name and "modules_to_save" not in name:
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

    def _get_validation_preferences(self) -> list[tuple[str, Tensor | None]]:
        """Build list of (label, preference_tensor) for validation in pareto mode.

        Returns one-hot vectors for each reward dimension plus a uniform vector.
        Non-pareto modes return a single (empty-label, None) entry.
        """
        if self._rl_config.preference_mode != "pareto":
            return [("", None)]

        R = self._num_rewards
        prefs: list[tuple[str, Tensor]] = []
        # One-hot for each reward
        for i, name in enumerate(self._reward_names):
            vec = torch.zeros(R)
            vec[i] = 1.0
            prefs.append((f"pref_{name}", vec))
        # Uniform
        prefs.append(("pref_uniform", torch.ones(R) / R))
        return prefs

    def _save_validation_videos(self, step: int) -> None:
        """Generate and save validation videos per GPU using EMA weights.

        In pareto mode, generates one video per preference vector (one-hot per
        reward + uniform) for each GPU's validation prompt.  Non-pareto mode
        generates a single video per GPU.
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

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        val_prefs = self._get_validation_preferences()
        saved_names: list[str] = []

        for pref_label, pref_vec in val_prefs:
            pref = pref_vec.to(device) if pref_vec is not None else None

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
                preference=pref,
            )
            pixel_video = self._decode_latent_to_pixels(latent, device)

            suffix = f"_{pref_label}" if pref_label else ""
            output_path = output_dir / f"step_{step:05d}_{vp.nickname}{suffix}.mp4"
            save_video(video_tensor=pixel_video, output_path=output_path, fps=rl_cfg.frame_rate)
            saved_names.append(output_path.name)

            del pixel_video, latent

        # Restore original weights
        if self._ema is not None and self._trainable_params is not None:
            self._ema.restore(self._trainable_params)

        free_gpu_memory()

        logger.info(f"Saved validation videos: {', '.join(saved_names)}")
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

        Supports two checkpoint formats:
        - .pt file: full training state (adapters, optimizer, EMA, counters)
        - directory: finds latest .pt file in the directory

        When the checkpoint comes from a run with different trainable params
        (e.g. non-pareto → pareto, which adds pref_conditioner/pref_gate),
        adapter LoRA weights are loaded and new layers keep their init values.
        Optimizer and EMA state are skipped if param counts don't match.

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

        # Restore adapters. Pad checkpoint state dicts with current model values
        # for any new keys (e.g. pref_conditioner/pref_gate added by pareto mode)
        # so that PEFT doesn't error on missing keys.
        for adapter_name, state_key in [("default", "default_adapter"), ("old", "old_adapter")]:
            current = get_peft_model_state_dict(unwrapped, adapter_name=adapter_name)
            ckpt = state[state_key]
            merged = {k: v.cpu().clone() for k, v in current.items()}
            loaded_keys = []
            for k, v in ckpt.items():
                if k in merged:
                    merged[k] = v
                    loaded_keys.append(k)
            skipped = set(merged.keys()) - set(loaded_keys)
            if skipped:
                logger.info(
                    f"Adapter '{adapter_name}': loaded {len(loaded_keys)} keys, "
                    f"kept init for {len(skipped)} new keys: {sorted(skipped)[:5]}..."
                )
            set_peft_model_state_dict(unwrapped, merged, adapter_name=adapter_name)

        # Restore optimizer state (may fail if trainable param count changed)
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Could not restore optimizer state (param count likely changed): {e}. "
                f"Continuing with fresh optimizer."
            )

        # Restore EMA state (may fail if trainable param count changed)
        if self._ema is not None and state.get("ema") is not None:
            try:
                self._ema.load_state_dict(state["ema"])
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    f"Could not restore EMA state (param count likely changed): {e}. "
                    f"Continuing with fresh EMA."
                )

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
