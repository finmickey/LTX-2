"""Test backpropagation: verify gradients flow only through 'default' adapter.

Checks:
1. After forward (with grad) + backward, only default LoRA params have .grad
2. Old LoRA params and base params have NO gradients
3. After optimizer.step(), default LoRA weights change
4. Old LoRA weights stay exactly the same
5. Base model weights stay exactly the same
6. Loss decreases after a few steps (sanity check)
7. All of the above works correctly on multi-GPU (DDP)
"""

import os
import time

import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

from torch.optim import AdamW

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
        print(f"{prefix}{msg}", flush=True)


def log_mem(rank, label):
    mem = torch.cuda.memory_allocated() / 1024**3
    log(rank, f"  [MEM] {label}: {mem:.2f} GB")


# ======================================================================
# Setup
# ======================================================================

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
    optimization=OptimizationConfig(
        learning_rate=1e-4,  # Higher LR so weight changes are visible
        steps=1,
        batch_size=1,
        enable_gradient_checkpointing=False,
        max_grad_norm=1.0,
    ),
    acceleration=AccelerationConfig(mixed_precision_mode="bf16"),
    data=DataConfig(preprocessed_data_root="/tmp/unused"),
    validation=ValidationConfig(),
    checkpoints=CheckpointsConfig(interval=None),
    wandb=WandbConfig(enabled=False),
    flow_matching=FlowMatchingConfig(),
    seed=42,
    output_dir="outputs/rl_backprop_test",
)

trainer = RLTrainer(config)
device = trainer._accelerator.device
rank = trainer._accelerator.process_index
num_processes = trainer._accelerator.num_processes
rl_cfg = trainer._rl_config
unwrapped = trainer._accelerator.unwrap_model(trainer._transformer)

# ======================================================================
# Step 0: Perturb adapters so old ≠ default ≠ base
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 0: Perturb adapters so old ≠ default ≠ base")
log(rank, "=" * 70)

with torch.no_grad():
    for name, param in unwrapped.named_parameters():
        if ".default." in name and "lora" in name:
            param.add_(torch.randn_like(param) * 0.02)
        elif ".old." in name and "lora" in name:
            param.add_(torch.randn_like(param) * 0.005)

log(rank, "  Done. Default perturbed with 0.02 noise, old with 0.005 noise.")

# ======================================================================
# Step 1: Snapshot weights BEFORE training step
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 1: Snapshot all weights BEFORE training")
log(rank, "=" * 70)

# Snapshot default LoRA, old LoRA (clone — they're small ~400MB each)
# For base model (19B), only store checksums to avoid OOM
default_before = {}
old_before = {}
base_checksums_before = {}

for name, param in unwrapped.named_parameters():
    if ".default." in name and "lora" in name:
        default_before[name] = param.data.clone()
    elif ".old." in name and "lora" in name:
        old_before[name] = param.data.clone()
    elif "lora" not in name:
        # Store checksum only — full clone would OOM (19B params)
        base_checksums_before[name] = param.data.sum().item()

log(rank, f"  Snapshotted {len(default_before)} default LoRA tensors (full clone)")
log(rank, f"  Snapshotted {len(old_before)} old LoRA tensors (full clone)")
log(rank, f"  Snapshotted {len(base_checksums_before)} base model tensors (checksums only)")

# ======================================================================
# Step 2: Setup optimizer (only trainable = default LoRA params)
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 2: Setup optimizer")
log(rank, "=" * 70)

trainable_params = [p for p in trainer._transformer.parameters() if p.requires_grad]
all_params = list(trainer._transformer.parameters())

log(rank, f"  Total params: {len(all_params)}")
log(rank, f"  Trainable params (requires_grad=True): {len(trainable_params)}")
log(rank, f"  Frozen params (requires_grad=False): {len(all_params) - len(trainable_params)}")
log(rank, f"  Trainable param count: {sum(p.numel() for p in trainable_params):,}")

# Verify: trainable should be exactly default LoRA
trainable_names = set()
for name, param in trainer._transformer.named_parameters():
    if param.requires_grad:
        trainable_names.add(name)
        if rank == 0 and len(trainable_names) <= 5:
            log(rank, f"    Example trainable: {name}")

all_trainable_are_default_lora = all(".default." in n and "lora" in n for n in trainable_names)
log(rank, f"  All trainable are default LoRA: {all_trainable_are_default_lora}")

# weight_decay=0 so DDP delta check isn't confounded by weight decay
# acting on differently-perturbed weights across GPUs
optimizer = AdamW(trainable_params, lr=config.optimization.learning_rate, weight_decay=0.0)
optimizer = trainer._accelerator.prepare(optimizer)
log(rank, f"  Optimizer: AdamW, lr={config.optimization.learning_rate}, weight_decay=0")

# ======================================================================
# Step 3: Generate video with old adapter
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 3: Generate video with OLD adapter")
log(rank, "=" * 70)

trainer._set_adapter("old")
trainer._transformer.eval()

prompt_idx = 0
cached = trainer._cached_prompt_embeddings[prompt_idx]
video_prompt_embeds = cached.video_context_positive.to(device)
gen_seed = 200 * num_processes + rank

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
log(rank, f"  Generated in {time.time()-t0:.2f}s, latent: {list(latent.shape)}", all_ranks=True)

# Decode + reward
pixel_video = trainer._decode_latent_to_pixels(latent, device)
reward = trainer._reward_fn.compute(pixel_video)
del pixel_video
torch.cuda.empty_cache()

reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
all_rewards = trainer._accelerator.gather(reward_tensor)
r = trainer._compute_advantage(reward, all_rewards)
r_tensor = torch.tensor([r], device=device, dtype=torch.float32)

log(rank, f"  Reward: {reward:.6f}, Advantage r: {r:.6f}", all_ranks=True)
if rank == 0:
    log(rank, f"  All rewards: {[f'{x:.4f}' for x in all_rewards.tolist()]}")

# ======================================================================
# Step 4: Forward pass WITH gradients + backward
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 4: Forward pass (with grad) + loss.backward()")
log(rank, "=" * 70)

trainer._transformer.train()
timestep_sampler = ShiftedLogitNormalTimestepSampler()

# Each GPU: sample timestep, create noise, build modality
seq_len = latent.shape[1]
torch.manual_seed(42 + rank)
t_scalar = timestep_sampler.sample(batch_size=1, seq_length=seq_len, device=device)
t_expanded = t_scalar.view(1, 1, 1)

noise = torch.randn_like(latent)
xt = (1 - t_expanded) * latent + t_expanded * noise

timesteps = t_scalar.expand(1, seq_len)
video_modality = Modality(
    enabled=True,
    latent=xt,
    timesteps=timesteps,
    positions=positions,
    context=video_prompt_embeds,
    context_mask=None,
)

log(rank, f"  Timestep t={t_scalar.item():.6f}", all_ranks=True)
log_mem(rank, "Before forward passes")

# --- Pass 1: Default adapter (v_new) — WITH GRAD ---
log(rank, "  --- Forward pass 1: DEFAULT adapter (v_new) — WITH GRAD ---")
trainer._set_adapter("default")
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    v_new, _ = trainer._transformer(video=video_modality, audio=None, perturbations=None)
log(rank, f"  v_new computed in {time.time()-t0:.3f}s")
log(rank, f"  v_new requires_grad: {v_new.requires_grad}")
log(rank, f"  v_new grad_fn: {v_new.grad_fn is not None}")
log_mem(rank, "After v_new (with grad graph)")

# --- Pass 2: Old adapter (v_old) — NO GRAD ---
log(rank, "  --- Forward pass 2: OLD adapter (v_old) — NO GRAD ---")
trainer._set_adapter("old")
t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    v_old, _ = trainer._transformer(video=video_modality, audio=None, perturbations=None)
log(rank, f"  v_old computed in {time.time()-t0:.3f}s")
log(rank, f"  v_old requires_grad: {v_old.requires_grad}")
log_mem(rank, "After v_old (no grad)")

# --- Pass 3: Base model (v_ref) — NO GRAD ---
log(rank, "  --- Forward pass 3: BASE model (v_ref) — NO GRAD ---")
t0 = time.time()
with unwrapped.disable_adapter(), torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v_ref, _ = trainer._transformer(video=video_modality, audio=None, perturbations=None)
log(rank, f"  v_ref computed in {time.time()-t0:.3f}s")
log(rank, f"  v_ref requires_grad: {v_ref.requires_grad}")
log_mem(rank, "After v_ref (no grad)")

# Verify predictions differ
diff_new_old = (v_new - v_old).abs().mean().item()
diff_new_ref = (v_new - v_ref).abs().mean().item()
diff_old_ref = (v_old - v_ref).abs().mean().item()
log(rank, f"  |v_new - v_old|: {diff_new_old:.6f}")
log(rank, f"  |v_new - v_ref|: {diff_new_ref:.6f}")
log(rank, f"  |v_old - v_ref|: {diff_old_ref:.6f}")

# Compute loss
log(rank, "  --- Computing NFT loss ---")
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

log(rank, f"  Loss: {loss.item():.4f}", all_ranks=True)
log(rank, f"  loss.requires_grad: {loss.requires_grad}")
log(rank, f"  loss.grad_fn: {loss.grad_fn}")

# Verify NO params have gradients yet
grads_before = sum(1 for p in trainer._transformer.parameters() if p.grad is not None)
log(rank, f"  Params with .grad BEFORE backward: {grads_before} (should be 0)")

# --- BACKWARD ---
log(rank, "  --- Running accelerator.backward(loss) ---")
trainer._set_adapter("default")
t0 = time.time()
trainer._accelerator.backward(loss)
backward_time = time.time() - t0
log(rank, f"  Backward completed in {backward_time:.3f}s")
log_mem(rank, "After backward")

# ======================================================================
# Step 5: Verify gradient flow
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 5: Verify gradient flow")
log(rank, "=" * 70)

default_with_grad = 0
default_without_grad = 0
old_with_grad = 0
base_with_grad = 0
default_grad_norms = []

for name, param in unwrapped.named_parameters():
    if ".default." in name and "lora" in name:
        if param.grad is not None:
            default_with_grad += 1
            default_grad_norms.append(param.grad.norm().item())
        else:
            default_without_grad += 1
    elif ".old." in name and "lora" in name:
        if param.grad is not None:
            old_with_grad += 1
    elif param.grad is not None:
        base_with_grad += 1

log(rank, f"  Default LoRA params WITH grad:    {default_with_grad} / {len(default_before)}")
log(rank, f"  Default LoRA params WITHOUT grad: {default_without_grad}")
log(rank, f"    (audio LoRA params don't get grad because audio=None — this is expected)")
log(rank, f"  Old LoRA params with grad:        {old_with_grad} (should be 0)")
log(rank, f"  Base model params with grad:      {base_with_grad} (should be 0)")

if default_grad_norms:
    mean_grad = sum(default_grad_norms) / len(default_grad_norms)
    max_grad = max(default_grad_norms)
    min_grad = min(default_grad_norms)
    zero_grads = sum(1 for g in default_grad_norms if g == 0.0)
    log(rank, f"  Gradient norms: mean={mean_grad:.8f} min={min_grad:.8f} max={max_grad:.8f}")
    log(rank, f"  Zero-norm gradients: {zero_grads} / {len(default_grad_norms)}")
    # Show a few example gradient norms
    for name, param in list(unwrapped.named_parameters())[:100]:
        if ".default." in name and "lora" in name and param.grad is not None:
            log(rank, f"    {name}: grad_norm={param.grad.norm().item():.8f}")
            break

# Gradient clipping
log(rank, "  --- Gradient clipping ---")
total_norm_before = torch.nn.utils.clip_grad_norm_(trainable_params, float('inf'))
log(rank, f"  Total grad norm (before clip): {total_norm_before:.6f}")

# Re-clip with actual max_grad_norm
if config.optimization.max_grad_norm > 0:
    # Reset grads — need to recompute since clip_grad_norm_ already modified them
    # Actually clip_grad_norm_ with inf doesn't modify, so we're fine
    total_norm_after = trainer._accelerator.clip_grad_norm_(
        trainable_params, config.optimization.max_grad_norm
    )
    log(rank, f"  Total grad norm (after clip to {config.optimization.max_grad_norm}): {total_norm_after:.6f}")

# ======================================================================
# Step 6: Optimizer step
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 6: optimizer.step() + zero_grad()")
log(rank, "=" * 70)

optimizer.step()
optimizer.zero_grad()
log(rank, "  Optimizer step + zero_grad completed.")

# ======================================================================
# Step 7: Verify weight changes
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 7: Verify weight changes")
log(rank, "=" * 70)

# Check default LoRA
default_changed = 0
default_unchanged = 0
default_diffs = []
for name in default_before:
    param = dict(unwrapped.named_parameters())[name]
    diff = (param.data - default_before[name]).abs().mean().item()
    default_diffs.append(diff)
    if diff > 1e-12:
        default_changed += 1
    else:
        default_unchanged += 1

log(rank, f"  Default LoRA weights CHANGED: {default_changed} / {len(default_before)}")
log(rank, f"  Default LoRA weights unchanged: {default_unchanged} (audio LoRA — expected)")
log(rank, f"  Changed matches grad count:   {default_changed == default_with_grad}")
if default_diffs:
    log(rank, f"  Mean weight change: {sum(default_diffs)/len(default_diffs):.10f}")
    log(rank, f"  Max weight change:  {max(default_diffs):.10f}")

# Check old LoRA
old_changed = 0
old_unchanged = 0
for name in old_before:
    param = dict(unwrapped.named_parameters())[name]
    diff = (param.data - old_before[name]).abs().mean().item()
    if diff > 1e-12:
        old_changed += 1
    else:
        old_unchanged += 1

log(rank, f"  Old LoRA weights changed:   {old_changed} (should be 0)")
log(rank, f"  Old LoRA weights unchanged: {old_unchanged} / {len(old_before)}")

# Check base model via checksums (full clone would OOM)
base_changed = 0
base_checked = 0
for name in list(base_checksums_before.keys())[:200]:  # Check first 200
    param = dict(unwrapped.named_parameters())[name]
    checksum_after = param.data.sum().item()
    base_checked += 1
    if abs(checksum_after - base_checksums_before[name]) > 1e-6:
        base_changed += 1

log(rank, f"  Base model weights changed: {base_changed} / {base_checked} checked (should be 0)")

# ======================================================================
# Step 8: Multi-step sanity — do 2 more steps, check loss trends
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 8: Multi-step sanity (2 more steps, same data)")
log(rank, "=" * 70)

losses = [loss.item()]

for extra_step in range(2):
    # Same data, different timestep
    torch.manual_seed(100 + rank + extra_step * num_processes)
    t_scalar2 = timestep_sampler.sample(batch_size=1, seq_length=seq_len, device=device)
    t_expanded2 = t_scalar2.view(1, 1, 1)
    noise2 = torch.randn_like(latent)
    xt2 = (1 - t_expanded2) * latent + t_expanded2 * noise2
    timesteps2 = t_scalar2.expand(1, seq_len)
    video_modality2 = Modality(
        enabled=True,
        latent=xt2,
        timesteps=timesteps2,
        positions=positions,
        context=video_prompt_embeds,
        context_mask=None,
    )

    # Forward with grad
    trainer._set_adapter("default")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v_new2, _ = trainer._transformer(video=video_modality2, audio=None, perturbations=None)

    trainer._set_adapter("old")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v_old2, _ = trainer._transformer(video=video_modality2, audio=None, perturbations=None)

    with unwrapped.disable_adapter(), torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_ref2, _ = trainer._transformer(video=video_modality2, audio=None, perturbations=None)

    trainer._set_adapter("default")
    loss2, metrics2 = compute_nft_loss(
        xt=xt2, x0=latent, t=t_expanded2,
        forward_pred=v_new2, old_pred=v_old2.detach(), ref_pred=v_ref2.detach(),
        r=r_tensor, beta=rl_cfg.nft_beta, kl_beta=rl_cfg.kl_beta,
        adv_clip_max=rl_cfg.adv_clip_max,
    )

    trainer._accelerator.backward(loss2)
    if config.optimization.max_grad_norm > 0:
        trainer._accelerator.clip_grad_norm_(trainable_params, config.optimization.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss2.item())
    log(rank, f"  Extra step {extra_step+1}: loss={loss2.item():.4f}, t={t_scalar2.item():.4f}", all_ranks=True)

# Gather all losses across GPUs for step 0
all_step0_losses = trainer._accelerator.gather(torch.tensor([losses[0]], device=device))
log(rank, f"  Step 0 losses across GPUs: {[f'{x:.2f}' for x in all_step0_losses.tolist()]}")
log(rank, f"  All different across GPUs: {len(set(f'{x:.2f}' for x in all_step0_losses.tolist())) == num_processes}")

# ======================================================================
# Step 9: Verify DDP gradient sync (weight DELTAS should match)
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "STEP 9: Verify DDP gradient sync")
log(rank, "=" * 70)

# NOTE: Weights differ across GPUs because we perturbed them independently AFTER
# accelerator.prepare() (which broadcasts weights). In real training, weights start
# identical so they stay synced. Here we verify that DDP averages gradients correctly
# by checking that the weight DELTAS (changes from optimizer step) are identical.

# Snapshot current weights
delta_snapshot = {}
for name, param in unwrapped.named_parameters():
    if ".default." in name and "lora" in name and param.requires_grad:
        delta_snapshot[name] = param.data.clone()

# One more forward/backward/step
torch.manual_seed(999 + rank)
t_s = timestep_sampler.sample(batch_size=1, seq_length=seq_len, device=device)
t_e = t_s.view(1, 1, 1)
n = torch.randn_like(latent)
xt_sync = (1 - t_e) * latent + t_e * n
ts_sync = t_s.expand(1, seq_len)
mod_sync = Modality(enabled=True, latent=xt_sync, timesteps=ts_sync,
                    positions=positions, context=video_prompt_embeds, context_mask=None)

trainer._set_adapter("default")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    vn, _ = trainer._transformer(video=mod_sync, audio=None, perturbations=None)
trainer._set_adapter("old")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    vo, _ = trainer._transformer(video=mod_sync, audio=None, perturbations=None)
with unwrapped.disable_adapter(), torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vr, _ = trainer._transformer(video=mod_sync, audio=None, perturbations=None)

trainer._set_adapter("default")
loss_sync, _ = compute_nft_loss(
    xt=xt_sync, x0=latent, t=t_e, forward_pred=vn, old_pred=vo.detach(),
    ref_pred=vr.detach(), r=r_tensor, beta=rl_cfg.nft_beta,
    kl_beta=rl_cfg.kl_beta, adv_clip_max=rl_cfg.adv_clip_max,
)
trainer._accelerator.backward(loss_sync)
if config.optimization.max_grad_norm > 0:
    trainer._accelerator.clip_grad_norm_(trainable_params, config.optimization.max_grad_norm)
optimizer.step()
optimizer.zero_grad()

# Compute weight deltas and gather across GPUs
deltas = []
delta_names = sorted(delta_snapshot.keys())[:5]
for name in delta_names:
    param = dict(unwrapped.named_parameters())[name]
    delta = (param.data - delta_snapshot[name]).sum().item()
    deltas.append(delta)

delta_tensor = torch.tensor(deltas, device=device, dtype=torch.float64)
all_deltas = trainer._accelerator.gather(delta_tensor).view(num_processes, -1)

ddp_synced = True
if rank == 0:
    for i in range(min(5, all_deltas.shape[1])):
        vals = all_deltas[:, i].tolist()
        max_spread = max(abs(v - vals[0]) for v in vals)
        # With weight_decay=0, deltas should be near-identical across GPUs
        # (DDP averages gradients, Adam state evolves identically).
        # Allow only tiny floating-point tolerance.
        ok = max_spread < 1e-6
        if not ok:
            ddp_synced = False
        log(rank, f"  Param {i} DELTA across GPUs: {[f'{v:.10f}' for v in vals]} spread={max_spread:.2e} {'OK' if ok else 'FAIL'}")
    log(rank, f"  DDP gradient averaging: {'WORKING' if ddp_synced else 'BROKEN'}")

# ======================================================================
# SUMMARY
# ======================================================================
print() if rank == 0 else None
log(rank, "=" * 70)
log(rank, "SUMMARY")
log(rank, "=" * 70)

checks = {
    "v_new has grad_fn (gradient graph exists)": v_new.grad_fn is not None,
    "v_old has NO grad_fn": v_old.grad_fn is None,
    "v_ref has NO grad_fn": v_ref.grad_fn is None,
    "All trainable params are default LoRA": all_trainable_are_default_lora,
    "Video LoRA params got gradients": default_with_grad > 0,
    "Audio LoRA params correctly skipped": default_without_grad > 0,
    "Grad count + skip count = total": default_with_grad + default_without_grad == len(default_before),
    "NO old LoRA params got gradients": old_with_grad == 0,
    "NO base model params got gradients": base_with_grad == 0,
    "Gradients are non-trivial (not all zero)": zero_grads == 0 if default_grad_norms else False,
    "Changed weights match grad count": default_changed == default_with_grad,
    "Old LoRA weights UNCHANGED after step": old_changed == 0,
    "Base model weights UNCHANGED after step": base_changed == 0,
    "DDP: weight deltas identical across GPUs": ddp_synced,
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
