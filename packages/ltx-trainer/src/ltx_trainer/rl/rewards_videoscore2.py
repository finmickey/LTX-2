"""VideoScore2 reward function (3-dimensional evaluation with chain-of-thought).

VideoScore2 is the latest version from TIGER-Lab, evaluating videos on:
- visual_quality: Clearness, resolution, brightness, color
- text_to_video_alignment: Alignment between text prompt and video content
- physical_consistency: Physical/common-sense consistency across frames

Uses chain-of-thought reasoning to provide more robust evaluation.
Model: TIGER-Lab/VideoScore2 (Qwen-based vision-language model)
"""

import logging
import re
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.rl.rewards import RewardFunction, _video_content_hash
from ltx_trainer.video_utils import save_video

logger = logging.getLogger(__name__)


class _VideoScore2Model:
    """Shared VideoScore2 model (loaded once, used by all dimension rewards)."""

    DIMENSIONS = ["visual_quality", "text_to_video_alignment", "physical_consistency"]

    def __init__(self, model_name: str = "TIGER-Lab/VideoScore2", fps: float = 2.0) -> None:
        """Initialize VideoScore2 model.

        Args:
            model_name: HuggingFace model identifier
            fps: Frames per second for temporary video files
        """
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Loading VideoScore2 model: {model_name}")
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = (
            AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            .eval()
            .to("cuda")
        )
        self._process_vision_info = process_vision_info
        self._fps = fps
        self._cache_key: bytes | None = None
        self._cache_scores: dict[str, float] | None = None

    def get_dimension_score(self, video: Tensor, prompt: str, dimension: str) -> float:
        """Get score for a specific dimension, computing all on first call per video."""
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {self.DIMENSIONS}")

        key = _video_content_hash(video)
        if key != self._cache_key:
            self._cache_scores = self._compute_all(video, prompt)
            self._cache_key = key
        return self._cache_scores[dimension]

    def _compute_all(self, video: Tensor, prompt: str) -> dict[str, float]:
        """Run VideoScore2 model and return all dimension scores.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range
            prompt: Text prompt used to generate the video

        Returns:
            Dictionary mapping dimension names to scores
        """
        # Save video to temporary file (VideoScore2 expects file paths)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_video(video, tmp_path, fps=self._fps)

            # Construct evaluation prompt for VideoScore2
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
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
                )
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
        """Build the evaluation prompt for VideoScore2.

        Args:
            text_prompt: Original text prompt used to generate the video

        Returns:
            Formatted evaluation prompt
        """
        return f"""You are an expert for evaluating and thinking about the quality of AI videos from diverse dimensions.

We would like to evaluate its quality from three dimensions: 'visual quality', 'text-to-video alignment' and 'physical/common-sense consistency'. Below is the definition of each dimension:
(1) visual quality:
The dimension 'visual quality' cares about the video's visual and optical propertities, including resolution, overall clarity, local blurriness, smoothness, stability of brightness/contrast, distortion/misalignment, abrupt changes, and any other factors that affect the watching experience.
(2) text-to-video alignment:
The dimension 't2v_alignment' mainly assesses whether the generated video fully and accurately depicts the elements mentioned in the text prompt, such as characters, actions, animals, etc., as well as background, quantity, color, weather, and so on.
(3) physical/common-sense consistency:
The dimension 'physical/common-sense consistency' mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the video that appear strange or unnatural.

Here we provide an AI video generated by text-to-video models and its text prompt:
{text_prompt}.

Based on the video content and the dimension definitions, please evaluate the video and give the quality score.
The quality score must be integers in the range of 1 - 5.

Your output must be in the following format:
visual quality: <v_score>;
text-to-video alignment: <t_score>;
physical/common-sense consistency: <p_score>

DO NOT include any other things behind or after your output."""

    def _parse_output(self, output_text: str) -> dict[str, float]:
        """Parse VideoScore2 model output to extract dimension scores.

        Expected format: "visual quality: X; text-to-video alignment: Y; physical/common-sense consistency: Z"
        where X, Y, Z are integers from 1-5.

        Args:
            output_text: Raw model output text

        Returns:
            Dictionary mapping dimension names to float scores
        """
        # Try to extract scores using regex (integers 1-5)
        # Official pattern from VideoScore2 demo
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)

        if match:
            visual_score = float(match.group(1))
            alignment_score = float(match.group(2))
            consistency_score = float(match.group(3))

            scores = {
                "visual_quality": visual_score,
                "text_to_video_alignment": alignment_score,
                "physical_consistency": consistency_score,
            }

            # Clamp scores to valid range [1.0, 5.0]
            for key in scores:
                scores[key] = max(1.0, min(5.0, scores[key]))

            return scores

        # Parsing failed - log the output and return defaults
        logger.warning(f"Failed to parse VideoScore2 output. Got: {output_text}")
        return {
            "visual_quality": 3.0,
            "text_to_video_alignment": 3.0,
            "physical_consistency": 3.0,
        }


class VideoScore2DimensionReward(RewardFunction):
    """Single dimension of VideoScore2 evaluation."""

    def __init__(self, model: _VideoScore2Model, dimension: str) -> None:
        """Initialize dimension reward.

        Args:
            model: Shared VideoScore2 model instance
            dimension: Dimension name (one of _VideoScore2Model.DIMENSIONS)
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
