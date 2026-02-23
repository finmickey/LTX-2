"""VLM-based style reward functions using Qwen2.5-VL logits scoring.

Adapted from ParetoControl's approach: instead of parsing generated text,
we extract scores directly from the model's logits over digit tokens (0-5).
This gives continuous, differentiable-friendly scores in [0, 1].

Two architectures:
1. Separate: one shared Qwen2.5-VL instance, called twice with different style prompts
   (vlm_realistic + vlm_watercolor reward types)
2. Combined: one Qwen2.5-VL instance, single prompt producing two scores via generation
   (vlm_style_dual reward type)
"""

import logging

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ltx_trainer.rl.rewards import RewardFunction

logger = logging.getLogger(__name__)

NUM_FRAMES = 5

# --- Prompt templates (adapted from ParetoControl for video) ---

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

PROMPT_COMBINED = """You are judging sampled frames from a generated video against a caption.
Caption: "{prompt}"

Task:
Give TWO integer scores from 0 to 5 separated by a comma.
Score A = photorealism (looks like real camera footage, realistic lighting/textures).
Score B = watercolor style (soft painterly brush strokes, watercolor blending/washes).

For each score, consider both the style match and content alignment with the caption.
Be strict.

Scoring rule (for each):
- 5: Style fully present AND content matches caption.
- 4: Style present AND content mostly correct.
- 3: Style partially present OR content has clear issues.
- 2: Multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong.

Response format: output ONLY two digits separated by a comma, like "3,2"."""


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

    Args:
        logits: Logits tensor [vocab_size]
        score_token_ids: Token IDs for digits 0-5

    Returns:
        Score in [0, 1]
    """
    score_logits = logits[score_token_ids]  # [6]
    probs = torch.softmax(score_logits.float(), dim=-1)
    weighted_sum = (probs * torch.arange(len(score_token_ids), device=probs.device, dtype=probs.dtype)).sum()
    return (weighted_sum / 5.0).item()  # Normalize to [0, 1]


# --- Single Qwen2.5-VL model ---


class _QwenVLStyleModel:
    """Single Qwen2.5-VL-7B model for style scoring via logits extraction."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Loading Qwen2.5-VL style model: {model_name}")
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = (
            AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            .eval()
            .to("cuda")
        )
        self._process_vision_info = process_vision_info

        # Resolve score token IDs via tokenizer (digits 0-5)
        self._score_token_ids = self._resolve_score_tokens()
        logger.info(f"Score token IDs: {self._score_token_ids}")

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

        # Build chat messages with frames as images
        content: list[dict] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": eval_text})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)

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

    def generate_dual_scores(self, video: Tensor, prompt: str, template: str) -> tuple[float, float]:
        """Generate two scores from a combined template via generation.

        Uses model.generate() with output_logits to extract scores from
        the "A,B" format output (logits at positions 0 and 2).

        Args:
            video: Video tensor [C, F, H, W]
            prompt: Text prompt
            template: Combined prompt template with {prompt} placeholder

        Returns:
            Tuple of (score_a, score_b), each in [0, 1]
        """
        frames = _sample_frames(video)
        eval_text = template.format(prompt=prompt)

        content: list[dict] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": eval_text})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )

        # output.logits is a tuple of tensors, one per generated token
        # Expected format: "A,B" -> 3 tokens: digit_A, comma, digit_B
        logits_a = output.logits[0][0]  # [vocab_size] - first generated token
        logits_b = output.logits[2][0]  # [vocab_size] - third generated token

        score_a = _extract_score_from_logits(logits_a, self._score_token_ids)
        score_b = _extract_score_from_logits(logits_b, self._score_token_ids)
        return score_a, score_b


# --- Version 1: Dual separate style scoring (shared Qwen model, two prompts) ---


class _VLMDualStyleModels:
    """Runs two style evaluations (photorealistic + watercolor) using one shared Qwen model.

    Caches results by video id so the second reward function reads from cache.
    """

    def __init__(self) -> None:
        self._model = _QwenVLStyleModel()
        self._cache_key: int | None = None
        self._cache_realistic: float | None = None
        self._cache_watercolor: float | None = None

    def _compute_both(self, video: Tensor, prompt: str) -> None:
        """Compute both style scores sequentially."""
        self._cache_realistic = self._model.score_single(video, prompt, PROMPT_PHOTOREALISM)
        self._cache_watercolor = self._model.score_single(video, prompt, PROMPT_WATERCOLOR)
        self._cache_key = id(video)

    def get_realistic_score(self, video: Tensor, prompt: str) -> float:
        vid_key = id(video)
        if self._cache_key != vid_key:
            self._compute_both(video, prompt)
        return self._cache_realistic

    def get_watercolor_score(self, video: Tensor, prompt: str) -> float:
        vid_key = id(video)
        if self._cache_key != vid_key:
            self._compute_both(video, prompt)
        return self._cache_watercolor


# --- Version 2: Combined single-inference model ---


class _VLMCombinedStyleModel:
    """Runs a single Qwen inference to produce both photorealistic and watercolor scores.

    Uses generation with "A,B" format output to extract two scores from one pass.
    """

    DIMENSIONS = ["vlm_realistic", "vlm_watercolor"]

    def __init__(self) -> None:
        self._model = _QwenVLStyleModel()
        self._cache_key: int | None = None
        self._cache_scores: dict[str, float] | None = None

    def _compute_both(self, video: Tensor, prompt: str) -> dict[str, float]:
        score_a, score_b = self._model.generate_dual_scores(video, prompt, PROMPT_COMBINED)
        return {"vlm_realistic": score_a, "vlm_watercolor": score_b}

    def get_dimension_score(self, video: Tensor, prompt: str, dimension: str) -> float:
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {self.DIMENSIONS}")

        vid_key = id(video)
        if self._cache_key != vid_key:
            self._cache_scores = self._compute_both(video, prompt)
            self._cache_key = vid_key

        return self._cache_scores[dimension]


# --- Reward classes ---


class VLMRealisticReward(RewardFunction):
    """VLM-based photorealistic style reward (separate model version)."""

    def __init__(self, dual_models: _VLMDualStyleModels) -> None:
        self._dual = dual_models

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._dual.get_realistic_score(video, prompt)


class VLMWatercolorReward(RewardFunction):
    """VLM-based watercolor style reward (separate model version)."""

    def __init__(self, dual_models: _VLMDualStyleModels) -> None:
        self._dual = dual_models

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._dual.get_watercolor_score(video, prompt)


class VLMCombinedRealisticReward(RewardFunction):
    """VLM-based photorealistic reward from combined model."""

    def __init__(self, model: _VLMCombinedStyleModel) -> None:
        self._model = model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._model.get_dimension_score(video, prompt, "vlm_realistic")


class VLMCombinedWatercolorReward(RewardFunction):
    """VLM-based watercolor reward from combined model."""

    def __init__(self, model: _VLMCombinedStyleModel) -> None:
        self._model = model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._model.get_dimension_score(video, prompt, "vlm_watercolor")
