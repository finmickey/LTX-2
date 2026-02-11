# DiffusionNFT RL Training for LTX-2 — Status

## What Was Implemented

A complete RL training pipeline for LTX-2 video generation using the DiffusionNFT algorithm. Instead of training on ground-truth data, the model generates K videos per prompt, scores them with a reward function, and uses the NFT loss to update LoRA weights.

### Files Created/Modified

All paths relative to `packages/ltx-trainer/` unless otherwise noted.

| File | Action | Description |
|------|--------|-------------|
| `src/ltx_trainer/config.py` | Modified | Added `RLConfig` class + `rl` field to `LtxTrainerConfig` |
| `src/ltx_trainer/rl/__init__.py` | Created | Package init |
| `src/ltx_trainer/rl/rl_trainer.py` | Created | `RLTrainer` class — full training loop |
| `src/ltx_trainer/rl/generation.py` | Created | Simplified generation (no CFG/STG, audio=None) |
| `src/ltx_trainer/rl/rewards.py` | Created | Reward functions (`RednessReward`) |
| `src/ltx_trainer/rl/nft_loss.py` | Created | NFT loss computation |
| `scripts/rl_train.py` | Created | Entry point (typer CLI) |
| `configs/ltx2_rl_nft.yaml` | Created | Template config |
| `configs/ltx2_rl_redness.yaml` | Created | Redness training config |
| `data/rl_prompts.txt` | Created | 15 red-themed test prompts |
| `data/rl_prompts_50.txt` | Created | 50 diverse prompts (no color words) |

### Architecture

- **RLTrainer** — standalone class (does NOT inherit from LtxvTrainer). Reuses model loading utilities and accelerator patterns.
- **1 base model + 2 LoRA adapter sets**: "default" (trainable) and "old" (frozen reference). No model duplication — adapter switching just changes which LoRA is active (~0 memory overhead).
- **DDP**: 8 GPUs, each generates 1 video per prompt, rewards all-gathered for advantage normalization.
- **Gradient accumulation**: `accelerator.accumulate()` wraps the training step. `gradient_accumulation_steps=4` means 4 prompts per optimizer step.
- **`find_unused_parameters=True`**: Required in DDP because "old" adapter params exist in the model but never participate in the gradient graph.

### Algorithm (DiffusionNFT)

For each prompt:
1. **Generate** 1 video per GPU using "old" adapter (no CFG/STG, 20 steps) → clean latent x₀
2. **Decode** latent → pixels via VAE, compute reward (redness), discard pixels
3. **Gather** rewards across 8 GPUs, compute advantage r ∈ [0,1]
4. **Noise** the clean latent: xₜ = (1-t)·x₀ + t·ε (flow matching, random t)
5. **3 forward passes** at timestep t (single pass each, NOT full denoising):
   - v_new = transformer(xₜ, t) with "default" adapter (WITH grad)
   - v_old = transformer(xₜ, t) with "old" adapter (no grad)
   - v_ref = transformer(xₜ, t) with no adapter/base model (no grad)
6. **NFT loss**: positive_pred = β·v_new + (1-β)·v_old, negative_pred = (1+β)·v_old - β·v_new, adaptive-weighted MSE + KL regularization (v_new vs v_ref)
7. **Backward** through v_new only → update "default" adapter
8. **Decay**: old = decay·old + (1-decay)·default (per optimizer step)

### Key Implementation Details

- **Audio disabled**: Pass `audio=None` to transformer. Do NOT pass `Modality(enabled=False, ...)` — causes shape mismatch errors.
- **Modality is frozen dataclass**: Use `dataclasses.replace()` to modify.
- **VAE decode**: VAE decoder shuttled GPU↔CPU per decode to save memory.
- **768/2304 LoRA params get gradients**: The other 1536 are audio attention LoRA (unused since audio=None). This is correct behavior, not a bug.
- **v_ref KL term**: NOT in the original paper (Equation 5), but IS in the NVlabs reference implementation. Needed for LoRA training to prevent collapse.
- **Comparison videos**: `_save_comparison_videos()` generates old/new/ref with fixed seed for direct visual comparison across steps.

## What Was Tested

### Stage 1: Generation (PASSED)
- Single GPU: 9 frames, 81 frames
- Compared RL generation (no CFG) with inference pipeline (CFG=1.0 and CFG=disabled) — matched
- Multi-GPU: 4 prompts × 8 GPUs = 32 videos with different seeds

### Stage 2: Reward + Advantages (PASSED)
- Redness rewards vary across GPUs (different seeds → different videos)
- `accelerator.gather()` correctly collects rewards
- Advantages correctly normalized, centered at 0.5

### Stage 3: Training Step Without Backprop (PASSED)
- `test_training_step.py`: All 3 predictions differ (old ≠ new ≠ ref with perturbed adapters)
- NFT loss is finite, KL loss nonzero
- Memory stable at 38.5 GB across adapter switches
- Each GPU has different data (loss, timestep, advantage)

### Stage 4: Backpropagation (PASSED)
- `test_backprop.py`: 14/14 checks pass
- v_new has grad_fn, v_old and v_ref do not
- 768 video LoRA params get gradients, 0 old LoRA, 0 base model
- After optimizer.step(): exactly those 768 params change, old/base unchanged
- DDP gradient sync verified: weight deltas identical across GPUs (within fp precision)

### Stage 5: Full Training Run (IN PROGRESS — PROBLEM FOUND)
- Ran 200 iterations with lr=1e-5, gradient_accumulation_steps=4
- Training loop runs correctly: rewards gathered, loss computed, optimizer steps happen
- Comparison videos saved every 20 steps

## Current Problem

**Comparison videos (old, new, ref) look identical across all steps.**

The `generated` videos (from reward computation) do change per step (different prompts/seeds), but the comparison videos — which use a fixed prompt and seed to show training progress — don't change.

**Root cause**: Learning rate 1e-5 is too low for visible changes in generation.

- After ~15 optimizer steps at lr=1e-5, LoRA B matrix elements change by ~7e-5
- LoRA contribution to model output: `(alpha/rank) * B @ A @ input ≈ 32 * 7e-5 * 0.018 ≈ 4e-5`
- Generation runs in **bf16** where precision at magnitude 1.0 is ~0.001
- LoRA contribution (0.00004) is **~25x below bf16 precision** → rounded to zero during generation
- Training loss IS computed in float32 (via autocast grad accumulation), so gradients flow correctly
- But the weight changes are invisible to the bf16 generation loop

**Fix**: Increase learning rate significantly (e.g., 1e-3) so weight changes are large enough to affect bf16 generation.

**Verification plan**: Restart with lr=1e-3, save checkpoint at step 20, load and inspect that weights actually changed meaningfully.

## How to Run

```bash
# Training
uv run --no-sync accelerate launch --num_processes=8 \
  packages/ltx-trainer/scripts/rl_train.py \
  packages/ltx-trainer/configs/ltx2_rl_redness.yaml

# Test scripts (in repo root)
uv run --no-sync accelerate launch --num_processes=8 test_training_step.py
uv run --no-sync accelerate launch --num_processes=8 test_backprop.py
```

## Model Paths

- Transformer: `models/ltx-2-19b-dev.safetensors`
- Text encoder: `models/gemma-3-12b-it-qat-q4_0-unquantized`

## Config Reference (RLConfig fields)

| Field | Default | Description |
|-------|---------|-------------|
| `prompts_file` | required | Text file, one prompt per line |
| `num_samples_per_prompt` | 8 | K samples (should = num_gpus) |
| `generation_steps` | 20 | Denoising steps |
| `generation_num_frames` | 9 | Frames (must satisfy frames % 8 == 1) |
| `generation_height` | 256 | Height (divisible by 32) |
| `generation_width` | 256 | Width (divisible by 32) |
| `reward_type` | "redness" | Reward function name |
| `nft_beta` | 0.0001 | NFT interpolation weight |
| `kl_beta` | 0.0001 | KL regularization weight |
| `adv_clip_max` | 5.0 | Advantage clipping range |
| `decay_rate` | 0.001 | Old adapter decay per optimizer step |
| `max_decay` | 0.5 | Maximum decay value |
| `frame_rate` | 25.0 | Video frame rate |

## Known Issues / Future Work

1. **Audio LoRA waste**: LoRA targets ALL attention layers including audio. Could save memory by only targeting video attention (`attn1` layers).
2. **No learning rate scheduler**: Currently constant LR. Could add warmup or cosine decay.
3. **Single reward function**: Only "redness" implemented. The `rewards.py` has a clean ABC for adding more.
4. **No gradient checkpointing tested**: `enable_gradient_checkpointing` exists but hasn't been verified with the RL loop.
5. **Checkpoint loading for resume**: Not implemented — training always starts from scratch.
