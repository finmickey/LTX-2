"""UnifiedReward-Think reward function (multi-modal evaluation with reasoning).

UnifiedReward-Think uses Qwen3-VL to evaluate videos across multiple dimensions:
- alignment: How well the video matches the text prompt
- coherence: Internal consistency and flow of the video
- style: Quality of visual style and aesthetics

Uses step-by-step reasoning for more robust evaluation.
Model: CodeGoat24/UnifiedReward-Think-qwen3vl-8b
"""

import logging
import re
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.rl.rewards import RewardFunction
from ltx_trainer.video_utils import save_video

logger = logging.getLogger(__name__)


class _UnifiedRewardThinkModel:
    """Shared UnifiedReward-Think model (loaded once, used by all dimension rewards)."""

    DIMENSIONS = ["alignment", "coherence", "style"]

    def __init__(self, model_name: str = "CodeGoat24/UnifiedReward-Think-qwen3vl-8b", fps: float = 2.0) -> None:
        """Initialize UnifiedReward-Think model.

        Args:
            model_name: HuggingFace model identifier
            fps: Frames per second for temporary video files
        """
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Loading UnifiedReward-Think model: {model_name}")
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = (
            AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            .eval()
            .to("cuda")
        )
        self._process_vision_info = process_vision_info
        self._fps = fps

        # Cache for storing computed scores by video object id
        self._cache_key: int | None = None
        self._cache_scores: dict[str, float] | None = None

    def get_dimension_score(self, video: Tensor, prompt: str, dimension: str) -> float:
        """Get score for a specific dimension, computing all scores on first call per video.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video
            dimension: One of DIMENSIONS

        Returns:
            Score for the requested dimension (1.0 to 5.0 scale)
        """
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {self.DIMENSIONS}")

        vid_key = id(video)
        if self._cache_key != vid_key:
            self._cache_scores = self._compute_all(video, prompt)
            self._cache_key = vid_key

        return self._cache_scores[dimension]

    def _compute_all(self, video: Tensor, prompt: str) -> dict[str, float]:
        """Run UnifiedReward-Think model and return all dimension scores.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video

        Returns:
            Dictionary mapping dimension names to scores
        """
        # Save video to temporary file (UnifiedReward expects file paths)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_video(video, tmp_path, fps=self._fps)

            # Construct evaluation prompt for UnifiedReward
            eval_prompt = self._build_evaluation_prompt(prompt)

            # Prepare model inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(tmp_path)},
                        {"type": "text", "text": eval_prompt},
                    ],
                }
            ]

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

            # Generate model response
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
                # Remove input tokens from output
                output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                output_text = self._processor.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # Parse output to extract scores
            scores = self._parse_output(output_text)
            return scores

        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    def _build_evaluation_prompt(self, text_prompt: str) -> str:
        """Build the evaluation prompt for UnifiedReward.

        Args:
            text_prompt: Original text prompt used to generate the video

        Returns:
            Formatted evaluation prompt
        """
        return f"""You are an expert video quality evaluator. Please evaluate this video on the following criteria:

1. Alignment: How well does the video content match the text prompt?
2. Coherence: Is the video internally consistent with smooth transitions and logical flow?
3. Style: What is the quality of the visual style, aesthetics, and production value?

For each criterion, provide a score from 1.0 to 5.0 where:
- 1.0 = Very poor
- 2.0 = Poor
- 3.0 = Acceptable
- 4.0 = Good
- 5.0 = Excellent

Text prompt: "{text_prompt}"

Please provide your evaluation in exactly this format:
alignment: X.X
coherence: Y.Y
style: Z.Z

Provide only the three scores in the specified format."""

    def _parse_output(self, output_text: str) -> dict[str, float]:
        """Parse UnifiedReward model output to extract dimension scores.

        Expected format contains lines like:
        alignment: X.X
        coherence: Y.Y
        style: Z.Z

        Args:
            output_text: Raw model output text

        Returns:
            Dictionary mapping dimension names to float scores
        """
        # Try to extract scores using regex
        alignment_match = re.search(r"alignment:\s*([0-9.]+)", output_text, re.IGNORECASE)
        coherence_match = re.search(r"coherence:\s*([0-9.]+)", output_text, re.IGNORECASE)
        style_match = re.search(r"style:\s*([0-9.]+)", output_text, re.IGNORECASE)

        if not (alignment_match and coherence_match and style_match):
            logger.warning(f"Failed to parse UnifiedReward output. Got: {output_text}")
            # Return default mid-range scores if parsing fails
            return {
                "alignment": 3.0,
                "coherence": 3.0,
                "style": 3.0,
            }

        try:
            scores = {
                "alignment": float(alignment_match.group(1)),
                "coherence": float(coherence_match.group(1)),
                "style": float(style_match.group(1)),
            }

            # Clamp scores to valid range [1.0, 5.0]
            for key in scores:
                scores[key] = max(1.0, min(5.0, scores[key]))

            return scores

        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing UnifiedReward scores: {e}. Output: {output_text}")
            return {
                "alignment": 3.0,
                "coherence": 3.0,
                "style": 3.0,
            }


class UnifiedRewardThinkDimensionReward(RewardFunction):
    """Single dimension of UnifiedReward-Think evaluation."""

    def __init__(self, model: _UnifiedRewardThinkModel, dimension: str) -> None:
        """Initialize dimension reward.

        Args:
            model: Shared UnifiedReward-Think model instance
            dimension: Dimension name (one of _UnifiedRewardThinkModel.DIMENSIONS)
        """
        self._model = model
        self._dimension = dimension

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        """Compute reward for this dimension.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video

        Returns:
            Score for this dimension (1.0 to 5.0 scale)
        """
        return self._model.get_dimension_score(video, prompt, self._dimension)
