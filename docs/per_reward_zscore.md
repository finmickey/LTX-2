# Per-Reward Z-Score Advantage Mode

## Problem

When combining rewards with very different scales (e.g. sketch ~1.2 vs clip_score ~2.8), the legacy advantage computation z-scores their **sum**. The reward with the larger range dominates the sample ranking, making the smaller reward effectively invisible.

## Solution: `per_reward_zscore`

Z-score each reward independently, then average the z-scores into a single scalar advantage.

Given R reward functions and K samples per prompt:
1. For each reward r, compute per-prompt mean and global std
2. Z-score: `(reward_r - prompt_mean_r) / global_std_r`
3. Average the R z-scores per sample → single scalar
4. Clip to `[-adv_clip_max, adv_clip_max]` and map to `[0, 1]`

This ensures each reward contributes equally to sample ranking regardless of raw scale. Uses the standard single-objective NFT loss — no random preferences or per-objective loss needed.

## Usage

Set `preference_mode: per_reward_zscore` in the `rl:` section and list rewards separately:

```yaml
rl:
  preference_mode: per_reward_zscore
  rewards:
  - type: sketch
  - type: clip_score
```

## Current Run: `rl_sketchclip_perzscore_run1`

**Config:** `configs/rl_sketchclip_perzscore_run1.yaml`
**Output:** `outputs/rl_sketchclip_perzscore_run1/`

Settings (identical to `rl_nft_1k_clip_run1` except for preference_mode and rewards):

| Setting | Value |
|---|---|
| Resume from | `rl_nft_1k_clip_run1` step 1500 |
| preference_mode | `per_reward_zscore` |
| Rewards | `sketch` + `clip_score` (independent) |
| LR | 3e-5 |
| LoRA rank | 64, alpha 64 |
| Samples/prompt | 16 |
| Prompts/epoch | 4 |
| Timesteps/sample | 5 |
| Grad accum | 4 |
| NFT beta | 0.1 |
| KL beta | 0.004 |
| adv_clip_max | 5.0 |
| Generation | 256x256, 121 frames, 20 steps |
| Steps | 3000 total (continuing from 1500) |
| Prompts | 1000 (precomputed embeddings) |

## Comparison of Advantage Modes

| Mode | `preference_mode` | How it works |
|---|---|---|
| Legacy | `null` (default) | Sum rewards → z-score the sum. Largest-scale reward dominates. |
| Per-reward z-score | `per_reward_zscore` | Z-score each reward independently → average. Equal contribution. |
| Pareto | `pareto` | Per-objective NFT loss with random preference sampling. Most complex. |
