"""Reward functions for RL training.

Each reward function scores a video tensor and returns a scalar reward.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)
_logged_shapes = False


def _video_content_hash(video: Tensor) -> bytes:
    """Fast content-based hash key for a video tensor.

    Samples ~1024 evenly-spaced values from the flattened tensor and returns
    their raw bytes.  This is O(1) in video size and collision-free in practice
    (two videos that differ in even a single pixel will almost certainly differ
    in at least one of the 1024 sampled positions).
    """
    flat = video.flatten()
    stride = max(1, len(flat) // 1024)
    return flat[::stride].numpy().tobytes()


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        """Score a video.

        Args:
            video: Video tensor [C, F, H, W] in [0, 1] range.
            prompt: Text prompt used to generate the video.

        Returns:
            Scalar reward value.
        """


class RednessReward(RewardFunction):
    """Reward that measures how red the video is.

    Computes mean(R - max(G, B)) across all pixels and frames.
    Higher values indicate redder videos.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        r, g, b = video[0], video[1], video[2]
        return (r - torch.max(g, b)).mean().item()


class BluenessReward(RewardFunction):
    """Reward that measures how blue the video is.

    Computes mean(B - max(R, G)) across all pixels and frames.
    Higher values indicate bluer videos.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        r, g, b = video[0], video[1], video[2]
        return (b - torch.max(r, g)).mean().item()


class HorizontalEdgeReward(RewardFunction):
    """Reward that measures horizontal edge strength.

    Computes mean absolute difference between adjacent rows.
    Uniform frames score ~0, horizontally-striped frames score high.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return (video[:, :, 1:, :] - video[:, :, :-1, :]).abs().mean().item()


class HorizontalStripeReward(RewardFunction):
    """Reward that measures color contrast between horizontal bands.

    Groups rows into bands of 4, computes mean color per band, then rewards
    high contrast between adjacent bands. Penalizes adjacent bands that share
    the same color — encourages bold, visible stripes.
    """

    BAND_HEIGHT = 4

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        # video: [C, F, H, W]
        _C, _F, H, _W = video.shape
        n_bands = H // self.BAND_HEIGHT
        trimmed = video[:, :, : n_bands * self.BAND_HEIGHT, :]
        # [C, F, n_bands, band_height, W] -> mean over band_height & W -> [C, F, n_bands]
        bands = trimmed.reshape(_C, _F, n_bands, self.BAND_HEIGHT, _W).mean(dim=(3, 4))
        # Mean absolute diff between adjacent bands
        return (bands[:, :, 1:] - bands[:, :, :-1]).abs().mean().item()


class UniformFrameReward(RewardFunction):
    """Reward that measures how uniform each frame's color is.

    Each frame should be a single solid color. Computes negative mean absolute
    deviation of pixels from the per-frame mean color. Perfectly uniform
    frames score 0; noisy/textured frames score negative.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        global _logged_shapes
        # video: [C, F, H, W]
        if not _logged_shapes:
            logger.info(
                f"[UniformFrameReward] video: shape={list(video.shape)} "
                f"dtype={video.dtype} range=[{video.min().item():.4f}, {video.max().item():.4f}]"
            )
        frame_mean = video.mean(dim=(2, 3), keepdim=True)  # [C, F, 1, 1]
        deviation = (video - frame_mean).abs().mean()
        reward = -deviation.item()
        if not _logged_shapes:
            logger.info(
                f"[UniformFrameReward] frame_mean: shape={list(frame_mean.shape)} "
                f"range=[{frame_mean.min().item():.4f}, {frame_mean.max().item():.4f}] "
                f"deviation={deviation.item():.4f} reward={reward:.4f}"
            )
        return reward


class RedOrBlueReward(RewardFunction):
    """Per-frame max(redness, blueness), averaged across frames.

    Each frame is scored for being either red OR blue (whichever is stronger).
    Black scores 0, pure red/blue scores +1, alternating red/blue scores +1.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        r, g, b = video[0], video[1], video[2]  # [F, H, W]
        redness = (r - torch.max(g, b)).mean(dim=(1, 2))  # [F]
        blueness = (b - torch.max(r, g)).mean(dim=(1, 2))  # [F]
        per_frame = torch.max(redness, blueness)  # [F]
        return per_frame.mean().item()


class FrameContrastReward(RewardFunction):
    """Mean L1 color distance between consecutive frames.

    Measures how much color changes from one frame to the next.
    All-same-color scores 0, alternating red/blue scores ~2.0.
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        # video: [C, F, H, W] — mean color per frame: [C, F]
        frame_colors = video.mean(dim=(2, 3))
        if frame_colors.shape[1] < 2:
            return 0.0
        diffs = (frame_colors[:, 1:] - frame_colors[:, :-1]).abs().sum(dim=0)  # [F-1]
        return diffs.mean().item()


class ColorAlternationReward(RewardFunction):
    """Rewards having both red and blue frames in the video.

    Pure min(mean_red, mean_blue) — always points gradient toward the
    MINORITY color. No bootstrap needed: the per-prompt std normalization
    in the NFT advantage computation amplifies even tiny differences
    (variance 0.001 becomes advantage ±1 after division by std).

    All-red scores 0, alternating R/B scores 0.5. Use high weight (50)
    to ensure the min variance dominates the total reward variance.

    Scores: black=0, all-red=0, all-blue=0, alternating R/B=0.50
    """

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        r, g, b = video[0], video[1], video[2]  # [F, H, W]
        redness = (r - torch.max(g, b)).mean(dim=(1, 2))  # [F]
        blueness = (b - torch.max(r, g)).mean(dim=(1, 2))  # [F]

        mean_red = redness.clamp(min=0).mean()  # avg positive redness across frames
        mean_blue = blueness.clamp(min=0).mean()  # avg positive blueness across frames

        return torch.min(mean_red, mean_blue).item()

class ChangingColorsReward(RewardFunction):
    """Reward that measures how much each frame's color differs from recent frames.

    For each frame, computes the minimum L1 color distance to any of the
    previous `LOOKBACK` frames. This is a continuous reward — every bit of
    color change contributes, giving meaningful variance across samples.

    Penalizes degenerate luminance (too dark or too bright) to prevent
    collapse to black or white.

    Reward = mean_color_change - extremity_penalty
    - mean_color_change: mean over frames of min-L1-to-recent [0, ~3.0]
    - extremity_penalty: how far mean luminance is from 0.5 midpoint [0, 0.5]
      (black=0.4 penalty, white=0.4 penalty, mid-gray=0 penalty)
    """

    LOOKBACK = 5

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        global _logged_shapes
        # video: [C, F, H, W] — get mean color per frame: [C, F]
        frame_colors = video.mean(dim=(2, 3))
        num_frames = frame_colors.shape[1]
        if num_frames < 2:
            return 0.0

        if not _logged_shapes:
            logger.info(
                f"[ChangingColorsReward] frame_colors: shape={list(frame_colors.shape)} "
                f"range=[{frame_colors.min().item():.4f}, {frame_colors.max().item():.4f}] "
                f"num_frames={num_frames}"
            )
            for t in range(min(5, num_frames)):
                rgb = frame_colors[:, t].tolist()
                logger.info(f"  frame[{t}] RGB=[{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]")

        # Color change: min L1 distance to recent frames (continuous, no threshold)
        total_change = 0.0
        for t in range(1, num_frames):
            lb = min(t, self.LOOKBACK)
            prev = frame_colors[:, t - lb : t]  # [C, lb]
            curr = frame_colors[:, t : t + 1]  # [C, 1]
            diffs = (curr - prev).abs().sum(dim=0)  # [lb] L1 per prev frame
            total_change += diffs.min().item()
        change_reward = total_change / (num_frames - 1)

        # Extremity penalty: penalize per-frame luminance far from 0.5
        # Targets [0.1, 0.9] as "acceptable" — outside that range gets penalized
        luminance = frame_colors.mean(dim=0)  # [F] mean across RGB
        too_dark = (0.1 - luminance).clamp(min=0)   # penalty for < 0.1
        too_bright = (luminance - 0.9).clamp(min=0)  # penalty for > 0.9
        extremity_penalty = (too_dark + too_bright).mean().item()

        result = change_reward - extremity_penalty
        if not _logged_shapes:
            logger.info(
                f"[ChangingColorsReward] change={change_reward:.4f} "
                f"extremity_penalty={extremity_penalty:.4f} total={result:.4f}"
            )
            _logged_shapes = True

        return result


REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt that was used to generate the video,
then give scores from 5 perspectives:
(1) visual_quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal_consistency, the consistency of objects, characters and backgrounds across different frames of the video
(3) dynamic_degree, the degree of dynamic changes
(4) text_to_video_alignment, the alignment between the text prompt and the video content
(5) factual_consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, respectively.

The output format should be:
visual_quality: x.x, temporal_consistency: x.x, dynamic_degree: x.x, text_to_video_alignment: x.x, factual_consistency: x.x

Here is the prompt of the video: {text_prompt}

Here are the frames of the video:
""".strip()


class _VideoScoreModel:
    """Shared VideoScore-v1.1 model (loaded once, used by all dimension rewards)."""

    DIMENSIONS = [
        "visual_quality",
        "temporal_consistency",
        "dynamic_degree",
        "text_to_video_alignment",
        "factual_consistency",
    ]

    DIMENSION_WEIGHTS = {}

    def __init__(self, model_name: str = "TIGER-Lab/VideoScore-v1.1", max_num_frames: int = 48) -> None:
        # Patch DynamicCache for mantis compatibility with newer transformers
        from transformers import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = DynamicCache.get_seq_length

        from mantis.models.idefics2 import Idefics2ForSequenceClassification
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = (
            Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            .eval()
            .to("cuda")
        )
        self._max_num_frames = max_num_frames
        self._cache_key: bytes | None = None
        self._cache_scores: list[float] | None = None

    def get_dimension_score(self, video: Tensor, prompt: str, dim_idx: int) -> float:
        """Get score for a specific dimension, computing all on first call per video."""
        key = _video_content_hash(video)
        if key != self._cache_key:
            self._cache_scores = self._compute_all(video, prompt)
            self._cache_key = key
        return self._cache_scores[dim_idx]

    def _compute_all(self, video: Tensor, prompt: str) -> list[float]:
        """Run VideoScore model and return all dimension scores."""
        frames = video.permute(1, 0, 2, 3)  # [F, C, H, W]
        num_frames = frames.shape[0]
        if num_frames > self._max_num_frames:
            indices = np.linspace(0, num_frames - 1, self._max_num_frames, dtype=int)
            frames = frames[indices]

        pil_frames = [TF.to_pil_image(f) for f in frames]

        eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=prompt)
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(pil_frames):
            eval_prompt += "<image> " * (len(pil_frames) - num_image_token)

        inputs = self._processor(text=eval_prompt, images=pil_frames, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        logits = outputs.logits  # [1, 5]
        return logits[0].tolist()


def _get_clip_image_transform(size: int = 224) -> T.Compose:
    """Build a torchvision transform matching CLIPProcessor for tensor inputs."""
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


class _ClipScoreModel:
    """Shared CLIP model for text-image alignment scoring (loaded once, kept on GPU)."""

    MODEL_NAME = "openai/clip-vit-large-patch14"

    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPTokenizerFast

        self._model = (
            CLIPModel.from_pretrained(self.MODEL_NAME, dtype=torch.bfloat16)
            .eval()
            .to("cuda")
        )
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._transform = _get_clip_image_transform(self._model.config.vision_config.image_size)

    def _get_similarity(self, video: Tensor, prompt: str) -> float:
        """Compute raw CLIP text-image cosine similarity on the first frame."""
        # Extract first frame: video is [C, F, H, W] -> [C, H, W]
        frame = video[:, 0, :, :]

        # Apply CLIP image transform (expects [C, H, W] float in [0, 1])
        pixel_values = self._transform(frame).unsqueeze(0).to(self._model.device, dtype=self._model.dtype)

        # Tokenize text
        text_inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=77)
        input_ids = text_inputs["input_ids"].to(self._model.device)
        attention_mask = text_inputs["attention_mask"].to(self._model.device)

        with torch.inference_mode():
            image_embeds = self._model.get_image_features(pixel_values=pixel_values)
            text_embeds = self._model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # L2 normalize and compute cosine similarity
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)
        return (image_embeds * text_embeds).sum(dim=-1).item()

    def compute_raw_similarity(self, video: Tensor, prompt: str) -> float:
        """Compute raw CLIP cosine similarity (no rescaling)."""
        return self._get_similarity(video, prompt)

    def compute_score(self, video: Tensor, prompt: str) -> float:
        """Compute CLIP text-image cosine similarity on the first frame.

        Returns a score rescaled to [1, 4] to match VideoScore range.
        """
        sim = self._get_similarity(video, prompt)

        # Rescale from ~[0.15, 0.40] to [1, 4]
        score = float(np.clip((sim - 0.15) / 0.25, 0.0, 1.0)) * 3.0 + 1.0
        return score


class ClipScoreReward(RewardFunction):
    """CLIP text-image alignment reward on the first frame of the video."""

    def __init__(self, model: _ClipScoreModel) -> None:
        self._model = model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._model.compute_score(video, prompt)


class SketchPlusClipReward(RewardFunction):
    """Sum of raw sketch score and raw CLIP cosine similarity on the first frame.

    No artificial rescaling — NFT z-score normalization handles scale differences.
    """

    def __init__(self, clip_model: _ClipScoreModel) -> None:
        from ltx_trainer.rl.rewards_sketch import get_sketch_scorer

        self._sketch_scorer = get_sketch_scorer()
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        # Sketch score on first frame (raw, no rescaling)
        frame = video[:, 0, :, :].unsqueeze(0)
        sketch_total, _ = self._sketch_scorer.score(frame)
        sketch_raw = sketch_total[0].item()

        # CLIP cosine similarity (raw, no rescaling)
        clip_raw = self._clip_model.compute_raw_similarity(video, prompt)

        return sketch_raw + clip_raw


class RealisticClipReward(RewardFunction):
    """CLIP text-image alignment with a photorealistic prompt prefix.

    Prepends "A photorealistic, high quality, 4K, camera-captured snapshot of"
    to the prompt before computing CLIP similarity on the first frame.
    Score is in [1, 4] to match other rewards.
    """

    PREFIX = "A photorealistic, high quality, 4K, camera-captured snapshot of "

    def __init__(self, clip_model: _ClipScoreModel) -> None:
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        modified_prompt = self.PREFIX + prompt
        return self._clip_model.compute_score(video, modified_prompt)


class RealisticPlusClipReward(RewardFunction):
    """Sum of raw realism score and raw CLIP cosine similarity on the first frame.

    Symmetric counterpart to SketchPlusClipReward for balanced Pareto training.
    Uses PACS photo evidence + texture/color metrics + CLIP similarity.
    No artificial rescaling — NFT z-score normalization handles scale differences.
    """

    def __init__(self, clip_model: _ClipScoreModel) -> None:
        from ltx_trainer.rl.rewards_sketch import get_realistic_scorer

        self._realistic_scorer = get_realistic_scorer()
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        # Realism score on first frame (raw, no rescaling)
        frame = video[:, 0, :, :].unsqueeze(0)
        realistic_total, _ = self._realistic_scorer.score(frame)
        realistic_raw = realistic_total[0].item()

        # CLIP cosine similarity (raw, no rescaling)
        clip_raw = self._clip_model.compute_raw_similarity(video, prompt)

        return realistic_raw + clip_raw


class URRealisticPlusClipReward(RewardFunction):
    """Sum of UR realistic style score [0,1] and raw CLIP cosine similarity ~[0.15, 0.40]."""

    def __init__(self, style_model: object, clip_model: _ClipScoreModel) -> None:
        self._style_model = style_model
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        ur_score = self._style_model.get_score(video, prompt, "realistic")
        clip_raw = self._clip_model.compute_raw_similarity(video, prompt)
        return ur_score + clip_raw


class URPixarPlusClipReward(RewardFunction):
    """Sum of UR Pixar style score [0,1] and raw CLIP cosine similarity ~[0.15, 0.40]."""

    def __init__(self, style_model: object, clip_model: _ClipScoreModel) -> None:
        self._style_model = style_model
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        ur_score = self._style_model.get_score(video, prompt, "pixar")
        clip_raw = self._clip_model.compute_raw_similarity(video, prompt)
        return ur_score + clip_raw


class _PickScoreModel:
    """Shared PickScore model for text-image preference scoring (loaded once, kept on GPU)."""

    PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    MODEL_NAME = "yuvalkirstain/PickScore_v1"

    def __init__(self) -> None:
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.PROCESSOR_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME).eval().to("cuda", dtype=torch.bfloat16)

    def compute_raw_score(self, video: Tensor, prompt: str) -> float:
        """Compute PickScore on the first frame. Returns score in ~[0, 1] (logit_scale * cos_sim / 26)."""
        from PIL import Image as PILImage

        frame = video[:, 0, :, :]  # (C, H, W)
        frame_u8 = (frame * 255).round().clamp(0, 255).to(torch.uint8).cpu()
        pil_img = PILImage.fromarray(frame_u8.permute(1, 2, 0).numpy())

        image_inputs = self._processor(images=pil_img, return_tensors="pt")
        image_inputs = {k: v.to(self._model.device) for k, v in image_inputs.items()}

        text_inputs = self._processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {k: v.to(self._model.device) for k, v in text_inputs.items()}

        with torch.inference_mode():
            image_embs = self._model.get_image_features(**image_inputs)
            if not isinstance(image_embs, Tensor):
                image_embs = image_embs.pooler_output
            image_embs = nn.functional.normalize(image_embs, dim=-1)

            text_embs = self._model.get_text_features(**text_inputs)
            if not isinstance(text_embs, Tensor):
                text_embs = text_embs.pooler_output
            text_embs = nn.functional.normalize(text_embs, dim=-1)

        logit_scale = self._model.logit_scale.exp()
        score = (logit_scale * (text_embs @ image_embs.T)).item()
        return score / 26.0  # normalize to ~[0, 1] (same as ParetoNFT)


class SketchPlusPickScoreReward(RewardFunction):
    """Sum of raw sketch score and PickScore on the first frame."""

    def __init__(self, pickscore_model: _PickScoreModel) -> None:
        from ltx_trainer.rl.rewards_sketch import get_sketch_scorer

        self._sketch_scorer = get_sketch_scorer()
        self._pickscore_model = pickscore_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        frame = video[:, 0, :, :].unsqueeze(0)
        sketch_total, _ = self._sketch_scorer.score(frame)
        sketch_raw = sketch_total[0].item()
        pick_raw = self._pickscore_model.compute_raw_score(video, prompt)
        return sketch_raw + pick_raw


class RealisticPlusPickScoreReward(RewardFunction):
    """Sum of raw realism score and PickScore on the first frame."""

    def __init__(self, pickscore_model: _PickScoreModel) -> None:
        from ltx_trainer.rl.rewards_sketch import get_realistic_scorer

        self._realistic_scorer = get_realistic_scorer()
        self._pickscore_model = pickscore_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        frame = video[:, 0, :, :].unsqueeze(0)
        realistic_total, _ = self._realistic_scorer.score(frame)
        realistic_raw = realistic_total[0].item()
        pick_raw = self._pickscore_model.compute_raw_score(video, prompt)
        return realistic_raw + pick_raw


class PickScoreReward(RewardFunction):
    """Raw PickScore text-image preference score on the first frame."""

    def __init__(self, pickscore_model: _PickScoreModel) -> None:
        self._pickscore_model = pickscore_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._pickscore_model.compute_raw_score(video, prompt)


class UR2QualityPlusPickScoreReward(RewardFunction):
    """Sum of UR2 visual quality [0,1] and PickScore first-frame alignment ~[0,1]."""

    def __init__(self, ur2_model: object, pickscore_model: _PickScoreModel) -> None:
        self._ur2_model = ur2_model
        self._pickscore_model = pickscore_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        ur2_score = self._ur2_model.get_score(video, prompt, "visual_quality")
        pick_score = self._pickscore_model.compute_raw_score(video, prompt)
        return ur2_score + pick_score


class UR2APAPlusClipReward(RewardFunction):
    """Sum of UR2 Alignment + Physics + Aesthetics scores [0,1] each, plus raw CLIP cosine similarity.

    "APA" = Alignment + Physics + Aesthetics — the 3 default dimensions from the UR2 repo.
    Total reward ~ [0, 3.4] (three UR2 scores in [0,1] + CLIP ~[0.15, 0.40]).
    """

    def __init__(self, ur2_model: object, clip_model: _ClipScoreModel) -> None:
        self._ur2_model = ur2_model
        self._clip_model = clip_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        alignment = self._ur2_model.get_score(video, prompt, "alignment")
        physics = self._ur2_model.get_score(video, prompt, "physics")
        aesthetics = self._ur2_model.get_score(video, prompt, "aesthetics")
        clip_raw = self._clip_model.compute_raw_similarity(video, prompt)
        return alignment + physics + aesthetics + clip_raw


class RealisticPickScoreReward(RewardFunction):
    """PickScore with a photorealistic prompt prefix.

    Prepends "A photorealistic, high quality, 4K, camera-captured snapshot of"
    to the prompt before computing PickScore on the first frame.
    """

    PREFIX = "A photorealistic, high quality, 4K, camera-captured snapshot of "

    def __init__(self, pickscore_model: _PickScoreModel) -> None:
        self._pickscore_model = pickscore_model

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        return self._pickscore_model.compute_raw_score(video, self.PREFIX + prompt)


class VideoScoreDimensionReward(RewardFunction):
    """Single dimension of VideoScore-v1.1."""

    def __init__(self, model: _VideoScoreModel, dim_idx: int, weight: float = 1.0) -> None:
        self._model = model
        self._dim_idx = dim_idx
        self._weight = weight

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        score = self._model.get_dimension_score(video, prompt, self._dim_idx)
        return score * self._weight


_video_score_model: _VideoScoreModel | None = None
_clip_score_model: _ClipScoreModel | None = None
_pickscore_model: _PickScoreModel | None = None
_video_score2_model: object | None = None
_unifiedreward_think_model: object | None = None
_vlm_dual_style_models: object | None = None
_ur_style_model: object | None = None
_ur2_style_model: object | None = None


def count_reward_dimensions(reward_type: str) -> int:
    """Return the number of scalar reward dimensions for a given reward type.

    Lightweight lookup (no model loading) used to compute preference_dim
    before heavy reward models are loaded.
    """
    _REWARD_DIMENSIONS = {
        "video_score": 5,
        "video_score2": 3,
        "unifiedreward_think": 3,
    }
    return _REWARD_DIMENSIONS.get(reward_type, 1)


def get_reward_functions(name: str) -> list[tuple[RewardFunction, str]]:
    """Factory function to get reward function(s) by name.

    Most reward types return a single (function, name) pair.
    'video_score' expands into one reward per dimension (5 total),
    so each dimension is logged and tracked independently.

    Args:
        name: Name of the reward function.

    Returns:
        List of (RewardFunction, display_name) tuples.
    """
    global _video_score_model, _clip_score_model, _pickscore_model, _video_score2_model, _unifiedreward_think_model, _vlm_dual_style_models, _ur_style_model, _ur2_style_model

    if name == "video_score":
        if _video_score_model is None:
            _video_score_model = _VideoScoreModel()
        result = []
        for idx, dim_name in enumerate(_VideoScoreModel.DIMENSIONS):
            weight = _VideoScoreModel.DIMENSION_WEIGHTS.get(dim_name, 1.0)
            reward = VideoScoreDimensionReward(_video_score_model, idx, weight)
            result.append((reward, f"videoscore_{dim_name}"))
        return result

    if name == "clip_score":
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        return [(ClipScoreReward(_clip_score_model), "clip_score")]

    if name == "video_score2":
        from ltx_trainer.rl.rewards_videoscore2 import _VideoScore2Model, VideoScore2DimensionReward

        if _video_score2_model is None:
            _video_score2_model = _VideoScore2Model()
        result = []
        for dim_name in _VideoScore2Model.DIMENSIONS:
            reward = VideoScore2DimensionReward(_video_score2_model, dim_name)
            result.append((reward, f"videoscore2_{dim_name}"))
        return result

    if name == "unifiedreward_think":
        from ltx_trainer.rl.rewards_unifiedreward import _UnifiedRewardThinkModel, UnifiedRewardThinkDimensionReward

        if _unifiedreward_think_model is None:
            _unifiedreward_think_model = _UnifiedRewardThinkModel()
        result = []
        for dim_name in _UnifiedRewardThinkModel.DIMENSIONS:
            reward = UnifiedRewardThinkDimensionReward(_unifiedreward_think_model, dim_name)
            result.append((reward, f"unifiedreward_{dim_name}"))
        return result

    if name == "sketch":
        from ltx_trainer.rl.rewards_sketch import SketchReward, get_sketch_scorer

        scorer = get_sketch_scorer()
        return [(SketchReward(scorer), "sketch")]

    if name == "sketch_plus_clip":
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        return [(SketchPlusClipReward(_clip_score_model), "sketch_plus_clip")]

    if name == "realistic_clip":
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        return [(RealisticClipReward(_clip_score_model), "realistic_clip")]

    if name == "realistic_plus_clip":
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        return [(RealisticPlusClipReward(_clip_score_model), "realistic_plus_clip")]

    if name == "sketch_plus_pickscore":
        if _pickscore_model is None:
            _pickscore_model = _PickScoreModel()
        return [(SketchPlusPickScoreReward(_pickscore_model), "sketch_plus_pickscore")]

    if name == "realistic_plus_pickscore":
        if _pickscore_model is None:
            _pickscore_model = _PickScoreModel()
        return [(RealisticPlusPickScoreReward(_pickscore_model), "realistic_plus_pickscore")]

    if name == "pickscore":
        if _pickscore_model is None:
            _pickscore_model = _PickScoreModel()
        return [(PickScoreReward(_pickscore_model), "pickscore")]

    if name == "realistic_pickscore":
        if _pickscore_model is None:
            _pickscore_model = _PickScoreModel()
        return [(RealisticPickScoreReward(_pickscore_model), "realistic_pickscore")]

    if name == "vlm_realistic":
        from ltx_trainer.rl.rewards_vlm_style import VLMRealisticReward, _VLMDualStyleModels

        if _vlm_dual_style_models is None:
            _vlm_dual_style_models = _VLMDualStyleModels()
        return [(VLMRealisticReward(_vlm_dual_style_models), "vlm_realistic")]

    if name == "vlm_watercolor":
        from ltx_trainer.rl.rewards_vlm_style import VLMWatercolorReward, _VLMDualStyleModels

        if _vlm_dual_style_models is None:
            _vlm_dual_style_models = _VLMDualStyleModels()
        return [(VLMWatercolorReward(_vlm_dual_style_models), "vlm_watercolor")]

    if name == "vlm_pixar":
        from ltx_trainer.rl.rewards_vlm_style import VLMPixarReward, _VLMStyleModels

        if _vlm_dual_style_models is None:
            _vlm_dual_style_models = _VLMStyleModels()
        return [(VLMPixarReward(_vlm_dual_style_models), "vlm_pixar")]

    if name in ("ur_realistic", "ur_watercolor", "ur_pixar"):
        from ltx_trainer.rl.rewards_ur_style import URPixarReward, URRealisticReward, URWatercolorReward, _URStyleModel

        if _ur_style_model is None:
            if _unifiedreward_think_model is not None:
                # Share model/processor from already-loaded UnifiedReward-Think
                _ur_style_model = _URStyleModel(
                    model=_unifiedreward_think_model._model,
                    processor=_unifiedreward_think_model._processor,
                )
            else:
                _ur_style_model = _URStyleModel()
        _ur_reward_map = {
            "ur_realistic": (URRealisticReward, "ur_realistic"),
            "ur_watercolor": (URWatercolorReward, "ur_watercolor"),
            "ur_pixar": (URPixarReward, "ur_pixar"),
        }
        cls, display_name = _ur_reward_map[name]
        return [(cls(_ur_style_model), display_name)]

    if name in ("ur_realistic_plus_clip", "ur_pixar_plus_clip"):
        from ltx_trainer.rl.rewards_ur_style import _URStyleModel

        if _ur_style_model is None:
            if _unifiedreward_think_model is not None:
                _ur_style_model = _URStyleModel(
                    model=_unifiedreward_think_model._model,
                    processor=_unifiedreward_think_model._processor,
                )
            else:
                _ur_style_model = _URStyleModel()
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        _ur_clip_map = {
            "ur_realistic_plus_clip": (URRealisticPlusClipReward, "ur_realistic_plus_clip"),
            "ur_pixar_plus_clip": (URPixarPlusClipReward, "ur_pixar_plus_clip"),
        }
        cls, display_name = _ur_clip_map[name]
        return [(cls(_ur_style_model, _clip_score_model), display_name)]

    _ur2_single_styles = {
        "ur2_realistic", "ur2_watercolor", "ur2_pixar", "ur2_bw",
        "ur2_text_quality", "ur2_visual_quality",
        "ur2_alignment", "ur2_physics", "ur2_aesthetics",
    }
    if name in _ur2_single_styles:
        from ltx_trainer.rl.rewards_ur2 import (
            UR2AestheticsReward, UR2AlignmentReward, UR2BWReward,
            UR2PhysicsReward, UR2PixarReward, UR2RealisticReward,
            UR2TextQualityReward, UR2VisualQualityReward, UR2WatercolorReward,
            _UR2StyleModel,
        )

        if _ur2_style_model is None:
            _ur2_style_model = _UR2StyleModel()
        _ur2_reward_map: dict[str, tuple[type[RewardFunction], str]] = {
            "ur2_realistic": (UR2RealisticReward, "ur2_realistic"),
            "ur2_watercolor": (UR2WatercolorReward, "ur2_watercolor"),
            "ur2_pixar": (UR2PixarReward, "ur2_pixar"),
            "ur2_bw": (UR2BWReward, "ur2_bw"),
            "ur2_text_quality": (UR2TextQualityReward, "ur2_text_quality"),
            "ur2_visual_quality": (UR2VisualQualityReward, "ur2_visual_quality"),
            "ur2_alignment": (UR2AlignmentReward, "ur2_alignment"),
            "ur2_physics": (UR2PhysicsReward, "ur2_physics"),
            "ur2_aesthetics": (UR2AestheticsReward, "ur2_aesthetics"),
        }
        cls, display_name = _ur2_reward_map[name]
        return [(cls(_ur2_style_model), display_name)]

    if name == "ur2_quality_plus_pickscore":
        from ltx_trainer.rl.rewards_ur2 import _UR2StyleModel

        if _ur2_style_model is None:
            _ur2_style_model = _UR2StyleModel()
        if _pickscore_model is None:
            _pickscore_model = _PickScoreModel()
        return [(UR2QualityPlusPickScoreReward(_ur2_style_model, _pickscore_model), "ur2_quality_plus_pickscore")]

    if name == "ur2_apa_plus_clip":
        from ltx_trainer.rl.rewards_ur2 import _UR2StyleModel

        if _ur2_style_model is None:
            _ur2_style_model = _UR2StyleModel()
        if _clip_score_model is None:
            _clip_score_model = _ClipScoreModel()
        return [(UR2APAPlusClipReward(_ur2_style_model, _clip_score_model), "ur2_apa_plus_clip")]

    reward_classes: dict[str, type[RewardFunction]] = {
        "redness": RednessReward,
        "blueness": BluenessReward,
        "horizontal_edges": HorizontalEdgeReward,
        "horizontal_stripes": HorizontalStripeReward,
        "uniform_frame": UniformFrameReward,
        "changing_colors": ChangingColorsReward,
        "red_or_blue": RedOrBlueReward,
        "frame_contrast": FrameContrastReward,
        "color_alternation": ColorAlternationReward,
    }

    if name not in reward_classes:
        available = list(reward_classes.keys()) + [
            "video_score", "clip_score", "video_score2", "unifiedreward_think",
            "sketch", "sketch_plus_clip", "realistic_clip", "realistic_plus_clip",
            "sketch_plus_pickscore", "realistic_plus_pickscore",
            "pickscore", "realistic_pickscore",
            "vlm_realistic", "vlm_watercolor", "vlm_pixar",
            "ur_realistic", "ur_watercolor", "ur_pixar",
            "ur_realistic_plus_clip", "ur_pixar_plus_clip",
            "ur2_realistic", "ur2_watercolor", "ur2_pixar", "ur2_bw",
            "ur2_text_quality", "ur2_visual_quality",
            "ur2_alignment", "ur2_physics", "ur2_aesthetics",
            "ur2_quality_plus_pickscore", "ur2_apa_plus_clip",
        ]
        raise ValueError(f"Unknown reward function: {name}. Available: {available}")

    return [(reward_classes[name](), name)]
