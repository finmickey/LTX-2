# Reward Calibration: SketchPlusClip

## Background

The `sketch_plus_clip` reward combines two signals:
- **Sketch score**: PACS classifier evidence + Sobel edge metrics (from `_SketchScorer`)
- **CLIP score**: Text-image cosine similarity (from `openai/clip-vit-large-patch14`)

### Previous Rescaling (Removed)

The original implementation applied arbitrary rescaling to both components:
- Sketch: `raw + 1.0` clamped to `[1, 4]` — assumed raw range ~[-1, 3]
- CLIP: `(sim - 0.15) / 0.25 * 3 + 1` — assumed raw range ~[0.15, 0.40]

The `[1, 4]` target range was borrowed from VideoScore (irrelevant for this reward).

### Why Rescaling Was Removed

The NFT algorithm z-score normalizes rewards within each batch. Only relative ordering matters, not absolute magnitudes. Artificial clamping destroys gradient signal at the boundaries (e.g., a sketch score of 3.5 and 5.0 both map to 4.0, losing information).

## Raw Value Ranges

Measured on 20 images from flwrlabs/pacs (5 per domain), resized to 256x256.

Run the calibration script to reproduce:
```bash
uv run --no-sync python packages/ltx-trainer/scripts/calibrate_rewards.py
```

### Overall Ranges

| Component | Min | Max | Mean | Std | Span |
|-----------|-----|-----|------|-----|------|
| sketch_raw | 0.076 | 1.953 | 0.543 | 0.424 | 1.878 |
| clip_raw | 0.139 | 0.202 | 0.172 | 0.017 | 0.064 |

### Per-Domain Sketch Means

| Domain | Sketch Mean | Sketch Span | CLIP Mean |
|--------|-------------|-------------|-----------|
| photo | 0.205 | 0.271 | 0.161 |
| art_painting | 0.289 | 0.353 | 0.167 |
| cartoon | 0.672 | 0.542 | 0.169 |
| sketch | 1.005 | 1.456 | 0.192 |

### Scale Imbalance

Sketch span (1.878) is ~30x wider than CLIP span (0.064). In a raw sum, sketch dominates ordering.

## Methodology

The calibration script uses the **PACS dataset** (`flwrlabs/pacs` on HuggingFace) which has 4 domains:
- **photo**: Real photographs
- **art_painting**: Artistic paintings
- **cartoon**: Cartoon-style images
- **sketch**: Hand-drawn sketches

This is ideal because our sketch scorer uses a PACS classifier (`prithivMLmods/PACS-DG-SigLIP2`), so we see exactly how it rates each domain.

For each image (resized to 256x256 to match generation resolution):
1. Compute raw `_SketchScorer.score()` → total + 5 component details
2. Compute raw CLIP cosine similarity (no rescaling)

## Scale Imbalance

The sketch score spans a much wider range than CLIP cosine similarity. In a naive sum, sketch dominates. This is acceptable because:

1. **NFT z-score normalization**: Rewards are normalized within batch, so absolute scale doesn't affect the algorithm
2. **Both signals contribute to ordering**: Even if sketch dominates the sum, CLIP still breaks ties among samples with similar sketch scores
3. **Monitoring**: Both raw values are logged separately via wandb for tracking

If training shows CLIP has no effect on sample ordering, a multiplier can be added later.

## Combined Reward

```
reward = sketch_raw + clip_raw
```

No clamping, no shifting, no rescaling. The NFT advantage computation handles normalization.
