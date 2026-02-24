"""UnifiedReward-based style reward functions using logits scoring (no CoT).

Same logits-extraction approach as rewards_vlm_style.py (softmax over digit
tokens 0-5, weighted sum → [0,1]) but using the UnifiedReward-Think model
(CodeGoat24/UnifiedReward-Think-qwen3vl-8b) which is a stronger reward model.

Can share the already-loaded model/processor from _UnifiedRewardThinkModel
(rewards_unifiedreward.py) to avoid loading the same 8B model twice on GPU.
"""

import logging

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from torch.nn import functional as F

from ltx_trainer.rl.rewards import RewardFunction, _video_content_hash

logger = logging.getLogger(__name__)


def _patch_conv3d_as_linear(model: object) -> None:
    """Monkey-patch Qwen3-VL's Conv3d patch embedding to use F.linear.

    Qwen3-VL's vision encoder uses Conv3d(3, 1152, kernel_size=(2,16,16),
    stride=(2,16,16)) as patch embedding. The kernel covers the entire spatial
    extent, making it functionally identical to a Linear(1536, 1152) layer.
    However, cuDNN's Conv3d has a pathological performance case with large
    batch + small spatial dims, taking ~13s per call vs ~0.1ms for F.linear.

    This replaces the patch_embed.forward method with a Linear equivalent.
    """
    visual = getattr(model, "visual", None)
    if visual is None:
        return
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        return
    proj = getattr(patch_embed, "proj", None)
    if proj is None or not isinstance(proj, torch.nn.Conv3d):
        return

    embed_dim = patch_embed.embed_dim
    weight = proj.weight  # [out_channels, in_channels, *kernel_size]
    bias = proj.bias
    weight_2d = weight.reshape(embed_dim, -1)  # [1152, 1536]

    orig_forward = patch_embed.forward

    def _fast_forward(hidden_states: Tensor) -> Tensor:
        target_dtype = weight.dtype
        x = hidden_states.to(dtype=target_dtype)  # [N, 1536]
        return F.linear(x, weight_2d, bias)  # [N, 1152]

    patch_embed.forward = _fast_forward
    logger.info(
        "URStyleModel: patched Conv3d patch_embed with F.linear "
        f"({weight_2d.shape[1]} -> {weight_2d.shape[0]})"
    )

NUM_FRAMES = 8  # Match UnifiedReward training expectations

# --- Prompt templates (copied from rewards_vlm_style.py for independent tuning) ---

PROMPT_PHOTOREALISM = """You are judging sampled frames from a generated video against a caption.
Caption: "A photorealistic video of {prompt}."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and photorealistic style.
Be strict. Do not hallucinate details.

Step 1) Style checks (pass/fail):
- Looks like real camera footage (not illustration, not cartoon, not 3D render).
- Realistic lighting and shadows consistent across frames.
- Realistic textures/materials (skin, fabric, metal, wood, etc. look natural if present).
- No painterly brush strokes, no heavy outlines, no flat shading.
- Temporal consistency: style is uniform across all frames.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present.
- Key attributes from {prompt} present (count, colors, distinctive parts).
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; frames are clear and detailed.
- 4: Style passes AND content mostly correct with only minor issues.
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but one style check fails.
- 2: Multiple content mistakes and/or multiple style failures, but some intent visible.
- 1: Very weak match; most requirements unmet.
- 0: Totally wrong or unusable frames.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""

PROMPT_WATERCOLOR = """You are judging sampled frames from a generated video against a caption.
Caption: "Watercolor painting animation of {prompt}. Soft, painterly brush strokes."

Task:
Output ONE integer score 0 to 5 for BOTH content alignment and demanded watercolor style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Soft, painterly brush strokes visible across frames.
- Watercolor blending and color washes (wet-on-wet or wet-on-dry look).
- No photorealistic textures (no sharp photo-like detail).
- Temporal consistency: painterly style is uniform across all frames.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable frames.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""

PROMPT_PIXAR = """You are judging sampled frames from a generated video against a caption.
Caption: "Pixar-style 3D animation of {prompt}. Stylized, expressive characters with smooth rendering."

Task:
Output ONE integer score 0 to 5 for BOTH content alignment and demanded Pixar animation style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- 3D-rendered appearance with smooth, clean surfaces (not photorealistic, not flat 2D).
- Stylized proportions: exaggerated or cartoon-like features (large eyes, rounded shapes).
- Soft, diffuse lighting typical of Pixar/animated films (no harsh real-world shadows).
- Rich, saturated colors with a polished, toy-like or plastic-like material quality.
- No photorealistic textures, no painterly brush strokes, no anime/cel-shading.
- Temporal consistency: 3D animated style is uniform across all frames.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable frames.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


# --- Shared utilities ---


def _sample_frames(video: Tensor, num_frames: int = NUM_FRAMES) -> list[Image.Image]:
    """Uniformly sample frames from video tensor and convert to PIL Images.

    Args:
        video: Video tensor [C, F, H, W] in [0, 1] range
        num_frames: Number of frames to sample

    Returns:
        List of PIL Images
    """
    total_frames = video.shape[1]
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        frame = video[:, idx]  # [C, H, W]
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_np))
    return frames


def _extract_score_from_logits(logits: Tensor, score_token_ids: list[int]) -> float:
    """Extract continuous score from logits over score tokens.

    Softmax over the 6 score tokens (0-5), weighted sum, normalized to [0, 1].
    """
    score_logits = logits[score_token_ids]  # [6]
    probs = torch.softmax(score_logits.float(), dim=-1)
    weighted_sum = (probs * torch.arange(len(score_token_ids), device=probs.device, dtype=probs.dtype)).sum()
    return (weighted_sum / 5.0).item()  # Normalize to [0, 1]


# --- UnifiedReward style model ---


class _URStyleModel:
    """UnifiedReward-Think model for style scoring via logits extraction (no CoT).

    Can either load its own model or share model/processor from an existing
    _UnifiedRewardThinkModel instance to avoid loading the same 8B model twice.
    """

    _TEMPLATES: dict[str, str] = {
        "realistic": PROMPT_PHOTOREALISM,
        "watercolor": PROMPT_WATERCOLOR,
        "pixar": PROMPT_PIXAR,
    }

    def __init__(
        self,
        model: object | None = None,
        processor: object | None = None,
        model_name: str = "CodeGoat24/UnifiedReward-Think-qwen3vl-8b",
    ) -> None:
        """Initialize URStyleModel.

        Args:
            model: Pre-loaded model to share (from _UnifiedRewardThinkModel._model).
            processor: Pre-loaded processor to share (from _UnifiedRewardThinkModel._processor).
            model_name: HuggingFace model identifier (used only if model/processor not provided).
        """
        if model is not None and processor is not None:
            logger.info("URStyleModel: sharing model/processor from existing UnifiedReward instance")
            self._model = model
            self._processor = processor
        else:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info(f"URStyleModel: loading model: {model_name}")
            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self._processor.tokenizer.padding_side = "left"
            self._model = (
                AutoModelForVision2Seq.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                .eval()
                .to("cuda")
            )

        # Fix cuDNN pathological Conv3d performance in Qwen3-VL vision encoder
        _patch_conv3d_as_linear(self._model)

        self._score_token_ids = self._resolve_score_tokens()
        logger.info(f"URStyleModel: score token IDs: {self._score_token_ids}")

        # Per-video cache (one video at a time, multiple styles)
        self._cache_key: bytes | None = None
        self._cache: dict[str, float] = {}

    def _resolve_score_tokens(self) -> list[int]:
        """Resolve token IDs for digits 0-5 via tokenizer, with fallback."""
        try:
            ids = []
            for digit in range(6):
                token_ids = self._processor.tokenizer.encode(str(digit), add_special_tokens=False)
                if len(token_ids) == 1:
                    ids.append(token_ids[0])
                else:
                    raise ValueError(f"Digit '{digit}' encoded to multiple tokens: {token_ids}")
            return ids
        except Exception as e:
            logger.warning(f"Failed to resolve score tokens via tokenizer ({e}), using fallback [15..20]")
            return [15, 16, 17, 18, 19, 20]

    def score_single(self, video: Tensor, prompt: str, template: str) -> float:
        """Score a video using a specific template via forward pass (no generation).

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video
            template: Prompt template with {prompt} placeholder

        Returns:
            Score in [0, 1]
        """
        frames = _sample_frames(video)
        eval_text = template.format(prompt=prompt)

        # Build chat messages with frames as images (Qwen3-VL API)
        content: list[dict] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": eval_text})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Extract logits at last position (predicts next token = the score digit)
        last_logits = outputs.logits[0, -1]  # [vocab_size]
        return _extract_score_from_logits(last_logits, self._score_token_ids)

    def get_score(self, video: Tensor, prompt: str, style: str) -> float:
        """Get style score with per-video caching.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video
            style: Style name ("realistic", "watercolor", "pixar")

        Returns:
            Score in [0, 1]
        """
        key = _video_content_hash(video)
        if key != self._cache_key:
            self._cache_key = key
            self._cache = {}
        if style not in self._cache:
            self._cache[style] = self.score_single(video, prompt, self._TEMPLATES[style])
        return self._cache[style]


# --- Reward classes ---


class URRealisticReward(RewardFunction):
    """UnifiedReward-based photorealistic style reward (logits scoring, no CoT)."""

    def __init__(self, style_model: _URStyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "realistic")


class URWatercolorReward(RewardFunction):
    """UnifiedReward-based watercolor style reward (logits scoring, no CoT)."""

    def __init__(self, style_model: _URStyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "watercolor")


class URPixarReward(RewardFunction):
    """UnifiedReward-based Pixar animation style reward (logits scoring, no CoT)."""

    def __init__(self, style_model: _URStyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "pixar")
