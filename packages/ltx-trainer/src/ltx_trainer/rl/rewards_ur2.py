"""UnifiedReward-2.0 style reward functions using logits scoring.

Uses CodeGoat24/UnifiedReward-2.0-qwen-7b (Qwen2.5-VL based, 7B params) for
style scoring via logits extraction over digit tokens 0-5.

Compared to rewards_ur_style.py (Qwen3-VL based):
- Different model architecture (Qwen2.5-VL vs Qwen3-VL)
- No Conv3d performance patch needed
- Uses qwen_vl_utils.process_vision_info for image preprocessing
- Supports 4 style dimensions: realistic, watercolor, pixar, bw

Performance (from ablations):
- 8 frames @ 960x544: ~1.3s per score
- 16 frames: ~2.6s per score
- Bottleneck: transformer forward pass (90% of time)
- Logits extraction = same speed as generation but gives continuous scores
"""

import logging

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ltx_trainer.rl.rewards import RewardFunction, _video_content_hash

logger = logging.getLogger(__name__)

NUM_FRAMES = 8  # Sweet spot: 1.3s/score with good discrimination (vs 2.6s for 16)

# --- Prompt templates ---
# These rubric-style prompts were validated to produce meaningful style
# discrimination on UR2 (tested on LTX-2 generated videos with 4 distinct styles).

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
- Vibrant but natural-looking colors with clean CG material quality (not oversaturated or blown-out).
- No photorealistic textures, no painterly brush strokes, no anime/cel-shading.
- Temporal consistency: 3D animated style is uniform across all frames.

Step 2) Quality checks (pass/fail):
- No visual glitches, color banding, oversaturation, or distorted geometry.
- Faces and bodies are well-formed (no melted or deformed features).

Step 3) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style, quality, and content checks pass; crisp and readable.
- 4: Style and quality pass AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style+quality pass but content has clear mistakes, or (B) content correct but 1 style/quality check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable frames.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""

PROMPT_BW = """You are judging sampled frames from a generated video against a caption.
Caption: "A black and white film noir video of {prompt}. High contrast monochrome, dramatic chiaroscuro lighting."

Task:
Output ONE integer score 0 to 5 for BOTH content alignment and demanded black-and-white film noir style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Frames are monochrome / grayscale (no color).
- High contrast with deep blacks and bright whites.
- Dramatic lighting with strong shadows (chiaroscuro).
- No color saturation, no vibrant hues.
- Temporal consistency: B&W style is uniform across all frames.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass.
- 4: Style passes AND content mostly correct.
- 3: Either style or content has issues but not both.
- 2: Multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable frames.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""

PROMPT_TEXT_QUALITY = """You are judging sampled frames from a generated video against a caption.
Caption: "{prompt}"

Task:
Output ONE integer score 0 to 5 based on BOTH text adherence and visual quality.
Be strict. Do not hallucinate details.

Step 1) Text adherence checks (pass/fail):
- Main subject(s) described in the caption are clearly present.
- Key attributes from the caption are correct (count, colors, shapes, sizes).
- Actions, poses, or relationships described in the caption are depicted correctly.
- Setting or background matches the caption (if specified).

Step 2) Visual quality checks (pass/fail):
- Frames are clear and sharp (no excessive blur or noise).
- Lighting is natural and consistent across frames.
- No visual artifacts, glitches, or color banding.
- Temporal consistency: objects maintain shape, size, and identity across frames.
- No deformed faces, hands, or body parts.

Scoring rule:
- 5: All text adherence checks pass AND all visual quality checks pass.
- 4: Text adherence passes AND quality mostly passes (one minor quality issue).
- 3: Either (A) text adherence passes but quality has clear issues, or (B) quality passes but one text adherence check fails.
- 2: Multiple text adherence or quality failures, but some intent visible.
- 1: Very weak match; most requirements unmet.
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


# --- UnifiedReward-2.0 style model ---


class _UR2StyleModel:
    """UnifiedReward-2.0 model for style scoring via logits extraction.

    Uses CodeGoat24/UnifiedReward-2.0-qwen-7b (Qwen2.5-VL based).
    Requires qwen_vl_utils for image preprocessing.
    """

    _TEMPLATES: dict[str, str] = {
        "realistic": PROMPT_PHOTOREALISM,
        "watercolor": PROMPT_WATERCOLOR,
        "pixar": PROMPT_PIXAR,
        "bw": PROMPT_BW,
        "text_quality": PROMPT_TEXT_QUALITY,
    }

    def __init__(
        self,
        model_name: str = "CodeGoat24/UnifiedReward-2.0-qwen-7b",
    ) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"UR2StyleModel: loading model: {model_name}")
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._processor.tokenizer.padding_side = "left"
        self._model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to("cuda")
        )

        self._score_token_ids = self._resolve_score_tokens()
        logger.info(f"UR2StyleModel: score token IDs: {self._score_token_ids}")

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
        from qwen_vl_utils import process_vision_info

        frames = _sample_frames(video)
        eval_text = template.format(prompt=prompt)

        # Build chat messages with frames as images (Qwen2.5-VL API)
        content: list[dict] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": eval_text})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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
            style: Style name ("realistic", "watercolor", "pixar", "bw")

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


class UR2RealisticReward(RewardFunction):
    """UnifiedReward-2.0 photorealistic style reward (logits scoring)."""

    def __init__(self, style_model: _UR2StyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "realistic")


class UR2WatercolorReward(RewardFunction):
    """UnifiedReward-2.0 watercolor style reward (logits scoring)."""

    def __init__(self, style_model: _UR2StyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "watercolor")


class UR2PixarReward(RewardFunction):
    """UnifiedReward-2.0 Pixar animation style reward (logits scoring)."""

    def __init__(self, style_model: _UR2StyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "pixar")


class UR2BWReward(RewardFunction):
    """UnifiedReward-2.0 black-and-white film noir style reward (logits scoring)."""

    def __init__(self, style_model: _UR2StyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "bw")


class UR2TextQualityReward(RewardFunction):
    """UnifiedReward-2.0 text adherence + visual quality reward (logits scoring)."""

    def __init__(self, style_model: _UR2StyleModel) -> None:
        self._style = style_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._style.get_score(video, prompt, "text_quality")
