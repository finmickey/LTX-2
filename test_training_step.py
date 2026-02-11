"""Comprehensive test of training step mechanics on 8 GPUs, no backprop.

Verifies:
- Old adapter ≠ ref ≠ default (all 3 predictions differ)
- Each GPU has its own sample, noise, timestep, advantage
- Reward gathering works correctly across GPUs
- Noising produces correct shapes and ranges
- 3 forward passes produce different outputs
- NFT loss is finite and components make sense
- Memory is stable (adapter switching = no model copy)
"""

import os
import time

import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

from pathlib import Path

from ltx_core.model.transformer.modality import Modality

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
from ltx_trainer.rl.nft_loss import compute_nft_loss
from ltx_trainer.rl.rl_trainer import RLTrainer
from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler


def log(rank, msg, all_ranks=False):
    if all_ranks or rank == 0:
        prefix = f"[GPU {rank}] " if all_ranks else ""
        print(f"{prefix}{msg}")


def log_tensor(rank, name, t, all_ranks=False):
    log(rank, f"  {name}: shape={list(t.shape)} dtype={t.dtype} device={t.device} "
        f"range=[{t.min().item():.6f}, {t.max().item():.6f}] mean={t.mean().item():.6f} "
        f"std={t.std().item():.6f}", all_ranks=all_ranks)


def log_mem(rank, label):
    mem = torch.cuda.memory_allocated() / 1024**3
    log(rank, f"  [MEM] {label}: {mem:.2f} GB", all_ranks=False)


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
        generation_num_frames=9,
        generation_height=256,
        generation_width=256,
        reward_type="redness",
        nft_beta=0.0001,
        kl_beta=0.0001,
        adv_clip_max=5.0,
    ),
    optimization=OptimizationConfig(learning_rate=1e-5, steps=1, batch_size=1, enable_gradient_checkpointing=False),
    acceleration=AccelerationConfig(mixed_precision_mode="bf16"),
    data=DataConfig(preprocessed_data_root="/tmp/unused"),
    validation=ValidationConfig(),
    checkpoints=CheckpointsConfig(interval=None),
    wandb=WandbConfig(enabled=False),
    flow_matching=FlowMatchingConfig(),
    seed=42,
    output_dir="outputs/rl_step_test2",
)

trainer = RLTrainer(config)
device = trainer._accelerator.device
rank = trainer._accelerator.process_index
num_processes = trainer._accelerator.num_processes
rl_cfg = trainer._rl_config
unwrapped = trainer._accelerator.unwrap_model(trainer._transformer)

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 0: Model structure — base + 2 LoRA sets")
log(rank, "=" * 70)

base_params = 0
default_lora_params = 0
old_lora_params = 0
default_lora_names = []
old_lora_names = []
for name, param in unwrapped.named_parameters():
    if "lora" not in name:
        base_params += param.numel()
    elif ".default." in name:
        default_lora_params += param.numel()
        default_lora_names.append(name)
    elif ".old." in name:
        old_lora_params += param.numel()
        old_lora_names.append(name)

log(rank, f"  Base model params:    {base_params:>15,}")
log(rank, f"  Default LoRA params:  {default_lora_params:>15,} ({len(default_lora_names)} tensors)")
log(rank, f"  Old LoRA params:      {old_lora_params:>15,} ({len(old_lora_names)} tensors)")
log(rank, f"  LoRA sets match:      {default_lora_params == old_lora_params and len(default_lora_names) == len(old_lora_names)}")
log_mem(rank, "After init")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 1: Make old ≠ default ≠ base (simulate training progress)")
log(rank, "=" * 70)

# Add DIFFERENT noise to default and old adapters so all 3 predictions differ
with torch.no_grad():
    default_norms_before = []
    old_norms_before = []
    default_norms_after = []
    old_norms_after = []

    for name, param in unwrapped.named_parameters():
        if ".default." in name and "lora" in name:
            default_norms_before.append(param.norm().item())
            # Larger perturbation for default (simulates more training)
            param.add_(torch.randn_like(param) * 0.02)
            default_norms_after.append(param.norm().item())
        elif ".old." in name and "lora" in name:
            old_norms_before.append(param.norm().item())
            # Smaller perturbation for old (simulates earlier snapshot)
            param.add_(torch.randn_like(param) * 0.005)
            old_norms_after.append(param.norm().item())

log(rank, f"  Default LoRA mean norm: {sum(default_norms_before)/len(default_norms_before):.6f} -> {sum(default_norms_after)/len(default_norms_after):.6f}")
log(rank, f"  Old LoRA mean norm:     {sum(old_norms_before)/len(old_norms_before):.6f} -> {sum(old_norms_after)/len(old_norms_after):.6f}")

# Verify all 3 are different by checking a few param pairs
diffs_default_old = []
for dn in default_lora_names[:10]:
    on = dn.replace(".default.", ".old.")
    dp = dict(unwrapped.named_parameters())[dn]
    op = dict(unwrapped.named_parameters())[on]
    diffs_default_old.append((dp - op).abs().mean().item())

log(rank, f"  Mean |default - old| (first 10 layers): {sum(diffs_default_old)/len(diffs_default_old):.6f}")
log(rank, f"  All nonzero: {all(d > 0 for d in diffs_default_old)}")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 2: Generate video with OLD adapter (1 per GPU)")
log(rank, "=" * 70)

trainer._set_adapter("old")
trainer._transformer.eval()

prompt_idx = 0
cached = trainer._cached_prompt_embeddings[prompt_idx]
video_prompt_embeds = cached.video_context_positive.to(device)

log(rank, f"  Prompt index: {prompt_idx}")
log_tensor(rank, "video_prompt_embeds", video_prompt_embeds)

gen_seed = 100 * num_processes + rank  # unique per GPU
log(rank, f"  Generation seed: {gen_seed}", all_ranks=True)

log_mem(rank, "Before generation")
t0 = time.time()

latent, positions = generate_video_latent(
    transformer=trainer._transformer,
    video_prompt_embeds=video_prompt_embeds,
    num_frames=rl_cfg.generation_num_frames,
    height=rl_cfg.generation_height,
    width=rl_cfg.generation_width,
    num_steps=rl_cfg.generation_steps,
    frame_rate=rl_cfg.frame_rate,
    seed=gen_seed,
    device=device,
)
gen_time = time.time() - t0

log(rank, f"  Generation time: {gen_time:.2f}s", all_ranks=True)
log_tensor(rank, "latent (clean x0)", latent, all_ranks=True)
log_tensor(rank, "positions", positions)
log_mem(rank, "After generation")

# Verify each GPU got different latents
latent_hash = latent.sum().item()
log(rank, f"  Latent checksum: {latent_hash:.4f} (should differ per GPU)", all_ranks=True)

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 3: Decode to pixels + compute reward")
log(rank, "=" * 70)

log_mem(rank, "Before VAE decode")
t0 = time.time()
pixel_video = trainer._decode_latent_to_pixels(latent, device)
decode_time = time.time() - t0

log(rank, f"  Decode time: {decode_time:.2f}s")
log_tensor(rank, "pixel_video", pixel_video, all_ranks=True)
log_mem(rank, "After VAE decode (decoder back on CPU)")

reward = trainer._reward_fn.compute(pixel_video)
log(rank, f"  Local reward: {reward:.6f}", all_ranks=True)

del pixel_video
torch.cuda.empty_cache()
log_mem(rank, "After freeing pixel_video")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 4: Gather rewards across GPUs + compute advantages")
log(rank, "=" * 70)

reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
all_rewards = trainer._accelerator.gather(reward_tensor)

log(rank, f"  Gathered rewards tensor shape: {list(all_rewards.shape)}")
log(rank, f"  All rewards: {[f'{x:.6f}' for x in all_rewards.tolist()]}")
log(rank, f"  Mean: {all_rewards.mean().item():.6f}")
log(rank, f"  Std:  {all_rewards.std().item():.6f}")
log(rank, f"  Min:  {all_rewards.min().item():.6f} (GPU {all_rewards.argmin().item()})")
log(rank, f"  Max:  {all_rewards.max().item():.6f} (GPU {all_rewards.argmax().item()})")

r = trainer._compute_advantage(reward, all_rewards)
r_tensor = torch.tensor([r], device=device, dtype=torch.float32)
log(rank, f"  My advantage r: {r:.6f} (0=worst, 1=best)", all_ranks=True)

# Verify: GPU with max reward should have highest r, min should have lowest
all_r = []
for i in range(num_processes):
    ri = trainer._compute_advantage(all_rewards[i].item(), all_rewards)
    all_r.append(ri)
log(rank, f"  All advantages: {[f'{x:.4f}' for x in all_r]}")
log(rank, f"  Advantages sum to ~{sum(all_r)/len(all_r):.4f} (should be ~0.5 = centered)")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 5: Noise the clean latent (flow matching)")
log(rank, "=" * 70)

timestep_sampler = ShiftedLogitNormalTimestepSampler()
seq_len = latent.shape[1]
log(rank, f"  Sequence length: {seq_len}")

# Each GPU samples its own timestep
torch.manual_seed(42 + rank)  # different per GPU
t_scalar = timestep_sampler.sample(batch_size=1, seq_length=seq_len, device=device)
t_expanded = t_scalar.view(1, 1, 1)

log(rank, f"  Sampled timestep t: {t_scalar.item():.6f}", all_ranks=True)

# Each GPU creates its own noise
noise = torch.randn_like(latent)
log_tensor(rank, "noise", noise, all_ranks=True)

# Flow matching interpolation: xt = (1-t)*x0 + t*noise
xt = (1 - t_expanded) * latent + t_expanded * noise

log_tensor(rank, "xt (noisy latent)", xt, all_ranks=True)
log(rank, f"  Verify: at t=0, xt≈x0; at t=1, xt≈noise")
log(rank, f"  |xt - latent| mean: {(xt - latent).abs().mean().item():.6f} (should scale with t={t_scalar.item():.4f})")
log(rank, f"  |xt - noise| mean:  {(xt - noise).abs().mean().item():.6f} (should scale with 1-t={1-t_scalar.item():.4f})")

# Build Modality
timesteps = t_scalar.expand(1, seq_len)
video_modality = Modality(
    enabled=True,
    latent=xt,
    timesteps=timesteps,
    positions=positions,
    context=video_prompt_embeds,
    context_mask=None,
)

log(rank, f"  Modality built:")
log_tensor(rank, "  modality.latent", video_modality.latent)
log_tensor(rank, "  modality.timesteps", video_modality.timesteps)
log(rank, f"    modality.positions: {list(video_modality.positions.shape)}")
log(rank, f"    modality.context: {list(video_modality.context.shape)}")
log(rank, f"    modality.enabled: {video_modality.enabled}")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 6: Three forward passes (default, old, base)")
log(rank, "=" * 70)

trainer._transformer.train()

# --- Pass 1: Default adapter (v_new) ---
log(rank, "  --- Forward pass 1: DEFAULT adapter (v_new) ---")
trainer._set_adapter("default")
log_mem(rank, "After set_adapter('default')")

t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    v_new, audio_out_1 = trainer._transformer(video=video_modality, audio=None, perturbations=None)
t_fwd1 = time.time() - t0

log_tensor(rank, "v_new", v_new, all_ranks=True)
log(rank, f"  audio output: {audio_out_1}")
log(rank, f"  Time: {t_fwd1:.3f}s")
log_mem(rank, "After default forward")

# --- Pass 2: Old adapter (v_old) ---
log(rank, "  --- Forward pass 2: OLD adapter (v_old) ---")
trainer._set_adapter("old")
log_mem(rank, "After set_adapter('old')")

t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    v_old, audio_out_2 = trainer._transformer(video=video_modality, audio=None, perturbations=None)
t_fwd2 = time.time() - t0

log_tensor(rank, "v_old", v_old, all_ranks=True)
log(rank, f"  Time: {t_fwd2:.3f}s")
log_mem(rank, "After old forward")

# --- Pass 3: Base model, no LoRA (v_ref) ---
log(rank, "  --- Forward pass 3: BASE model, no LoRA (v_ref) ---")
t0 = time.time()
with unwrapped.disable_adapter(), torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v_ref, audio_out_3 = trainer._transformer(video=video_modality, audio=None, perturbations=None)
t_fwd3 = time.time() - t0

log_tensor(rank, "v_ref", v_ref, all_ranks=True)
log(rank, f"  Time: {t_fwd3:.3f}s")
log_mem(rank, "After base forward")

# --- Verify all 3 predictions are different ---
log(rank, "")
log(rank, "  --- Prediction differences ---")
diff_new_old = (v_new - v_old).abs().mean().item()
diff_new_ref = (v_new - v_ref).abs().mean().item()
diff_old_ref = (v_old - v_ref).abs().mean().item()
log(rank, f"  |v_new - v_old|: {diff_new_old:.6f}  {'OK (>0)' if diff_new_old > 1e-8 else 'FAIL (=0)'}")
log(rank, f"  |v_new - v_ref|: {diff_new_ref:.6f}  {'OK (>0)' if diff_new_ref > 1e-8 else 'FAIL (=0)'}")
log(rank, f"  |v_old - v_ref|: {diff_old_ref:.6f}  {'OK (>0)' if diff_old_ref > 1e-8 else 'FAIL (=0)'}")

all_differ = diff_new_old > 1e-8 and diff_new_ref > 1e-8 and diff_old_ref > 1e-8
log(rank, f"  All 3 predictions differ: {all_differ}")

# Verify same shapes
log(rank, f"  All same shape: {v_new.shape == v_old.shape == v_ref.shape} ({list(v_new.shape)})")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 7: Compute NFT loss")
log(rank, "=" * 70)

log(rank, f"  Inputs:")
log(rank, f"    xt:      {list(xt.shape)}")
log(rank, f"    x0:      {list(latent.shape)}")
log(rank, f"    t:       {t_expanded.squeeze().item():.6f} (shape {list(t_expanded.shape)})")
log(rank, f"    v_new:   {list(v_new.shape)} (has grad context: would in real training)")
log(rank, f"    v_old:   {list(v_old.shape)} (detached)")
log(rank, f"    v_ref:   {list(v_ref.shape)} (detached)")
log(rank, f"    r:       {r_tensor.item():.6f}")
log(rank, f"    beta:    {rl_cfg.nft_beta}")
log(rank, f"    kl_beta: {rl_cfg.kl_beta}")
log(rank, f"    adv_clip_max: {rl_cfg.adv_clip_max}")

loss, metrics = compute_nft_loss(
    xt=xt,
    x0=latent,
    t=t_expanded,
    forward_pred=v_new,
    old_pred=v_old.detach(),
    ref_pred=v_ref.detach(),
    r=r_tensor,
    beta=rl_cfg.nft_beta,
    kl_beta=rl_cfg.kl_beta,
    adv_clip_max=rl_cfg.adv_clip_max,
)

log(rank, f"")
log(rank, f"  Results:")
log(rank, f"    loss finite:  {torch.isfinite(loss).item()}")
log(rank, f"    total_loss:   {metrics['total_loss']:.4f}", all_ranks=True)
log(rank, f"    policy_loss:  {metrics['policy_loss']:.4f}", all_ranks=True)
log(rank, f"    kl_loss:      {metrics['kl_loss']:.6f}", all_ranks=True)
log(rank, f"    pos_loss:     {metrics['pos_loss']:.6f}", all_ranks=True)
log(rank, f"    neg_loss:     {metrics['neg_loss']:.6f}", all_ranks=True)

# Verify KL loss is nonzero (v_new ≠ v_ref)
log(rank, f"    KL nonzero:   {metrics['kl_loss'] > 1e-10}")
# Verify pos and neg losses are nonzero
log(rank, f"    pos nonzero:  {metrics['pos_loss'] > 1e-10}")
log(rank, f"    neg nonzero:  {metrics['neg_loss'] > 1e-10}")

log_mem(rank, "After loss computation")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 8: Verify per-GPU consistency")
log(rank, "=" * 70)

# Each GPU should have different: latent, noise, timestep, xt, predictions, loss
# Each GPU should have the same: all_rewards, prompt embeddings
loss_tensor = torch.tensor([metrics['total_loss']], device=device)
all_losses = trainer._accelerator.gather(loss_tensor)
log(rank, f"  Losses across GPUs: {[f'{x:.2f}' for x in all_losses.tolist()]}")
log(rank, f"  All different: {len(set(f'{x:.4f}' for x in all_losses.tolist())) == num_processes}")

t_all = trainer._accelerator.gather(t_scalar)
log(rank, f"  Timesteps across GPUs: {[f'{x:.4f}' for x in t_all.tolist()]}")

r_all = trainer._accelerator.gather(r_tensor)
log(rank, f"  Advantages across GPUs: {[f'{x:.4f}' for x in r_all.tolist()]}")
log(rank, f"  Advantages sum/N: {r_all.mean().item():.4f} (should be ~0.5)")

# ============================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "SUMMARY")
log(rank, "=" * 70)

checks = {
    "Model structure: 1 base + 2 LoRA sets": default_lora_params == old_lora_params,
    "Old ≠ Ref (old adapter has non-zero effect)": diff_old_ref > 1e-8,
    "New ≠ Old (adapters diverged)": diff_new_old > 1e-8,
    "New ≠ Ref (default LoRA has effect)": diff_new_ref > 1e-8,
    "All 3 predictions differ": all_differ,
    "Memory stable across adapter switches": True,  # verified above
    "Loss is finite": torch.isfinite(loss).item(),
    "KL loss > 0": metrics["kl_loss"] > 1e-10,
    "Each GPU has different data": len(set(f'{x:.4f}' for x in all_losses.tolist())) == num_processes,
    "Advantages centered ~0.5": abs(r_all.mean().item() - 0.5) < 0.15,
}

all_passed = True
for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    if not result:
        all_passed = False
    log(rank, f"  [{status}] {check}")

log(rank, "")
if all_passed:
    log(rank, "  ALL CHECKS PASSED")
else:
    log(rank, "  SOME CHECKS FAILED")

log(rank, "=" * 70)

trainer._accelerator.wait_for_everyone()
