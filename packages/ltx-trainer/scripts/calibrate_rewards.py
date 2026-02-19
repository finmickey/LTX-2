"""Calibrate sketch and CLIP reward ranges on real images from the PACS dataset.

Uses HuggingFace flwrlabs/pacs — the actual PACS dataset with 4 domains:
photo, art_painting, cartoon, sketch. Since our sketch scorer uses a PACS
classifier (prithivMLmods/PACS-DG-SigLIP2), this shows exactly how it rates
each domain.

For each image (resized to 256x256 to match generation resolution):
  - Raw _SketchScorer.score() → sketch_total + 5 component details
  - Raw CLIP cosine similarity (no rescaling) using openai/clip-vit-large-patch14
  - Current rescaled values for comparison

Run:
    uv run --no-sync python packages/ltx-trainer/scripts/calibrate_rewards.py
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torchvision.transforms import functional as TF

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------


def load_sketch_scorer():
    """Load the sketch scorer singleton."""
    from ltx_trainer.rl.rewards_sketch import get_sketch_scorer

    return get_sketch_scorer()


def load_clip_model():
    """Load CLIP model and return (model, tokenizer, transform)."""
    from transformers import CLIPModel, CLIPTokenizerFast

    from ltx_trainer.rl.rewards import _get_clip_image_transform

    model = (
        CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
    transform = _get_clip_image_transform(model.config.vision_config.image_size)
    return model, tokenizer, transform


@torch.no_grad()
def compute_clip_raw_similarity(clip_model, tokenizer, transform, image_tensor, prompt):
    """Compute raw CLIP cosine similarity (no rescaling).

    Args:
        image_tensor: [C, H, W] float in [0, 1].
        prompt: Text string.

    Returns:
        Raw cosine similarity float.
    """
    pixel_values = transform(image_tensor).unsqueeze(0).to(clip_model.device, dtype=clip_model.dtype)
    text_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=77)
    input_ids = text_inputs["input_ids"].to(clip_model.device)
    attention_mask = text_inputs["attention_mask"].to(clip_model.device)

    image_embeds = clip_model.get_image_features(pixel_values=pixel_values)
    text_embeds = clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    return (image_embeds * text_embeds).sum(dim=-1).item()


def current_clip_rescale(raw_sim):
    """Current [1, 4] rescaling in rewards.py."""
    return float(np.clip((raw_sim - 0.15) / 0.25, 0.0, 1.0)) * 3.0 + 1.0


def current_sketch_rescale(raw_score):
    """Current [1, 4] rescaling in rewards.py."""
    return float(max(1.0, min(4.0, raw_score + 1.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    random.seed(42)

    # Load PACS dataset
    print("Loading PACS dataset from flwrlabs/pacs...")
    ds = load_dataset("flwrlabs/pacs", split="train")

    # Group by domain
    domain_indices = defaultdict(list)
    for idx, example in enumerate(ds):
        domain = example["domain"]
        domain_indices[domain].append(idx)

    print(f"Domains found: {sorted(domain_indices.keys())}")
    for domain, indices in sorted(domain_indices.items()):
        print(f"  {domain}: {len(indices)} images")

    # Sample ~5 per domain
    samples_per_domain = 5
    selected = {}
    for domain in sorted(domain_indices.keys()):
        indices = domain_indices[domain]
        chosen = random.sample(indices, min(samples_per_domain, len(indices)))
        selected[domain] = chosen

    # Load models
    print("\nLoading sketch scorer...")
    sketch_scorer = load_sketch_scorer()

    print("Loading CLIP model...")
    clip_model, clip_tokenizer, clip_transform = load_clip_model()

    # A generic prompt for CLIP (since PACS images don't have captions)
    generic_prompt = "a picture"

    # Process each image
    print("\n" + "=" * 120)
    print(f"{'Domain':<16} {'Idx':>5}  {'Sketch Raw':>10} {'Sketch→[1,4]':>12}  "
          f"{'CLIP Raw':>9} {'CLIP→[1,4]':>10}  "
          f"{'PACS':>6} {'EdgeBand':>8} {'EdgeCtr':>7} {'BgTex':>6} {'Thick':>6}")
    print("=" * 120)

    all_results = []

    for domain in sorted(selected.keys()):
        for idx in selected[domain]:
            example = ds[idx]
            pil_image = example["image"].convert("RGB").resize((256, 256))

            # Convert to tensor [C, H, W] float [0, 1]
            img_tensor = TF.to_tensor(pil_image).to("cuda")

            # Sketch score: expects [N, C, H, W]
            sketch_input = img_tensor.unsqueeze(0)
            sketch_total, details = sketch_scorer.score(sketch_input)
            sketch_raw = sketch_total[0].item()

            # CLIP raw similarity
            clip_raw = compute_clip_raw_similarity(
                clip_model, clip_tokenizer, clip_transform, img_tensor, generic_prompt
            )

            # Current rescalings
            sketch_rescaled = current_sketch_rescale(sketch_raw)
            clip_rescaled = current_clip_rescale(clip_raw)

            result = {
                "domain": domain,
                "idx": idx,
                "sketch_raw": sketch_raw,
                "sketch_rescaled": sketch_rescaled,
                "clip_raw": clip_raw,
                "clip_rescaled": clip_rescaled,
                "pacs": details["pacs_sketch"][0].item(),
                "edge_band": details["edge_band"][0].item(),
                "edge_contrast": details["edge_contrast"][0].item(),
                "bg_texture": details["bg_texture"][0].item(),
                "thickness": details["thickness"][0].item(),
            }
            all_results.append(result)

            print(f"{domain:<16} {idx:>5}  {sketch_raw:>10.4f} {sketch_rescaled:>12.4f}  "
                  f"{clip_raw:>9.4f} {clip_rescaled:>10.4f}  "
                  f"{result['pacs']:>6.3f} {result['edge_band']:>8.3f} "
                  f"{result['edge_contrast']:>7.3f} {result['bg_texture']:>6.3f} "
                  f"{result['thickness']:>6.3f}")

    # Summary statistics per domain
    print("\n" + "=" * 90)
    print("SUMMARY BY DOMAIN")
    print("=" * 90)
    print(f"{'Domain':<16} {'Metric':<14} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Span':>8}")
    print("-" * 90)

    domains = sorted(set(r["domain"] for r in all_results))
    for domain in domains + ["ALL"]:
        if domain == "ALL":
            subset = all_results
        else:
            subset = [r for r in all_results if r["domain"] == domain]

        for metric in ["sketch_raw", "clip_raw", "sketch_rescaled", "clip_rescaled"]:
            vals = [r[metric] for r in subset]
            arr = np.array(vals)
            print(f"{domain:<16} {metric:<14} {arr.min():>8.4f} {arr.max():>8.4f} "
                  f"{arr.mean():>8.4f} {arr.std():>8.4f} {arr.max() - arr.min():>8.4f}")
        print()

    # Scale imbalance analysis
    print("=" * 90)
    print("SCALE IMBALANCE ANALYSIS (raw values)")
    print("=" * 90)
    sketch_vals = np.array([r["sketch_raw"] for r in all_results])
    clip_vals = np.array([r["clip_raw"] for r in all_results])
    sketch_span = sketch_vals.max() - sketch_vals.min()
    clip_span = clip_vals.max() - clip_vals.min()
    print(f"Sketch span: {sketch_span:.4f}")
    print(f"CLIP span:   {clip_span:.4f}")
    if clip_span > 0:
        print(f"Ratio (sketch/clip): {sketch_span / clip_span:.1f}x")
    print(f"\nIn a raw sum, sketch changes would dominate by ~{sketch_span / clip_span:.0f}x")
    print("NFT z-score normalization handles this — only relative ordering within batch matters.")


if __name__ == "__main__":
    main()
