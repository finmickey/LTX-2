"""Sketch reward function using PACS classification and Sobel edge metrics.

Self-contained implementation inspired by ParetoNFT's SketchScorerOptimized.
Combines PACS sketch evidence with GPU-based edge analysis for sketch-style scoring.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import functional as TF

from ltx_trainer.rl.rewards import RewardFunction

logger = logging.getLogger(__name__)


class _PACSScorer:
    """PACS domain classifier (photo/art_painting/cartoon/sketch).

    Uses prithivMLmods/PACS-DG-SigLIP2 to classify images.
    Returns softplus-scaled logits (not softmax) for smooth gradients.
    """

    MODEL_NAME = "prithivMLmods/PACS-DG-SigLIP2"

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._model = (
            AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
            .eval()
            .to("cuda")
        )
        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self._label2id = self._model.config.label2id

    @torch.no_grad()
    def score_sketch_evidence(self, pil_images: list) -> Tensor:
        """Return sketch evidence (softplus-scaled logit / 10) for each PIL image.

        Returns:
            Tensor [N] on CUDA.
        """
        inputs = self._processor(images=pil_images, return_tensors="pt").to("cuda")
        outputs = self._model(**inputs)
        scaled = F.softplus(outputs.logits) / 10.0
        sketch_idx = self._label2id["sketch"]
        return scaled[:, sketch_idx]

    @torch.no_grad()
    def score_photo_evidence(self, pil_images: list) -> Tensor:
        """Return photo evidence (softplus-scaled logit / 10) for each PIL image.

        Returns:
            Tensor [N] on CUDA.
        """
        inputs = self._processor(images=pil_images, return_tensors="pt").to("cuda")
        outputs = self._model(**inputs)
        scaled = F.softplus(outputs.logits) / 10.0
        photo_idx = self._label2id["photo"]
        return scaled[:, photo_idx]


class _SketchScorer:
    """GPU-optimized sketch scorer combining PACS evidence with Sobel edge metrics.

    Five scoring components:
    - PACS sketch evidence (classifier-based)
    - Edge density band-pass (target ~5% edge pixels)
    - Edge contrast (strong edges inside edge mask)
    - Background texture penalty (low texture outside edges)
    - Line thickness proxy (edge density ratio at two blur scales)
    """

    def __init__(
        self,
        pacs_scorer: _PACSScorer,
        edge_density_target: float = 0.05,
        edge_density_tol: float = 0.04,
        edge_thresh_quantile: float = 0.90,
        w_pacs: float = 1.0,
        w_edge_band: float = 1.2,
        w_edge_contrast: float = 0.4,
        w_bg_texture: float = 1.2,
        w_thickness: float = 0.4,
        thickness_target: float = 0.2,
        thickness_tol: float = 0.2,
    ) -> None:
        self._pacs = pacs_scorer
        self._edge_density_target = edge_density_target
        self._edge_density_tol = edge_density_tol
        self._edge_thresh_quantile = edge_thresh_quantile
        self._w_pacs = w_pacs
        self._w_edge_band = w_edge_band
        self._w_edge_contrast = w_edge_contrast
        self._w_bg_texture = w_bg_texture
        self._w_thickness = w_thickness
        self._thickness_target = thickness_target
        self._thickness_tol = thickness_tol

        # Sobel kernels
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device="cuda",
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device="cuda",
        ).view(1, 1, 3, 3)
        self._kx = kx
        self._ky = ky

        self._blur3 = self._make_gaussian(3, 1.0).to("cuda")
        self._blur7 = self._make_gaussian(7, 2.5).to("cuda")

    @staticmethod
    def _make_gaussian(k: int, s: float) -> Tensor:
        ax = torch.arange(k) - (k - 1) / 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        ker = torch.exp(-(xx**2 + yy**2) / (2 * s**2))
        ker = ker / ker.sum()
        return ker.view(1, 1, k, k).float()

    def _to_gray(self, x: Tensor) -> Tensor:
        """Convert NCHW float [0,1] to grayscale N1HW."""
        if x.shape[1] == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x[:, 0:1]

    def _sobel_mag(self, gray: Tensor) -> Tensor:
        gx = F.conv2d(gray, self._kx, padding=1)
        gy = F.conv2d(gray, self._ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def _edge_mask(self, mag: Tensor) -> Tensor:
        n = mag.shape[0]
        flat = mag.view(n, -1)
        thr = torch.quantile(flat, self._edge_thresh_quantile, dim=1).view(n, 1, 1, 1)
        return (mag >= thr).float()

    @torch.no_grad()
    def score(self, images_nchw: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute sketch score combining PACS evidence and edge metrics.

        Args:
            images_nchw: Tensor NCHW float in [0,1].

        Returns:
            (total_score [N], details dict of tensors [N])
        """
        x = images_nchw.to("cuda").float().clamp(0.0, 1.0)

        # 1) PACS sketch evidence
        x_u8 = (x * 255).round().to(torch.uint8).detach().cpu()
        from PIL import Image
        pil_imgs = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in x_u8]
        pacs_sketch = self._pacs.score_sketch_evidence(pil_imgs)  # [N]

        # 2) Edge metrics (GPU)
        gray = self._to_gray(x)
        mag = self._sobel_mag(gray)
        edge = self._edge_mask(mag)

        density = edge.mean(dim=(1, 2, 3))
        edge_band = 1.0 - torch.clamp(
            torch.abs(density - self._edge_density_target) / (self._edge_density_tol + 1e-8),
            0.0, 1.0,
        )

        edge_contrast = (mag * edge).sum(dim=(1, 2, 3)) / (edge.sum(dim=(1, 2, 3)) + 1e-6)

        non_edge = 1.0 - edge
        bg_texture = (mag * non_edge).sum(dim=(1, 2, 3)) / (non_edge.sum(dim=(1, 2, 3)) + 1e-6)

        # Thickness proxy: compare edge density at two blur scales
        g3 = F.conv2d(gray, self._blur3, padding=1)
        g7 = F.conv2d(gray, self._blur7, padding=3)
        e3 = self._edge_mask(self._sobel_mag(g3)).mean(dim=(1, 2, 3))
        e7 = self._edge_mask(self._sobel_mag(g7)).mean(dim=(1, 2, 3))
        thickness_ratio = e7 / (e3 + 1e-6)

        thickness = 1.0 - torch.clamp(
            torch.abs(thickness_ratio - self._thickness_target) / (self._thickness_tol + 1e-8),
            0.0, 1.0,
        )

        total = (
            self._w_pacs * pacs_sketch
            + self._w_edge_band * edge_band
            + self._w_edge_contrast * edge_contrast
            - self._w_bg_texture * bg_texture
            + self._w_thickness * thickness
        )

        details = {
            "pacs_sketch": pacs_sketch,
            "edge_density": density,
            "edge_band": edge_band,
            "edge_contrast": edge_contrast,
            "bg_texture": bg_texture,
            "thickness_ratio": thickness_ratio,
            "thickness": thickness,
        }
        return total, details


class _RealisticScorer:
    """GPU-optimized realism scorer: PACS photo evidence + texture/color metrics.

    Symmetric counterpart to _SketchScorer for balanced Pareto training.
    Four scoring components:
    - PACS photo evidence (classifier-based)
    - Low edge density bonus (inverse of sketch's edge band-pass)
    - Background texture richness (Sobel magnitude outside edges)
    - Natural color variance (penalizes flat/monochrome sketch look)
    """

    def __init__(
        self,
        pacs_scorer: _PACSScorer,
        edge_density_target: float = 0.05,
        edge_density_tol: float = 0.04,
        edge_thresh_quantile: float = 0.90,
        w_pacs: float = 1.0,
        w_low_edge: float = 1.2,
        w_texture: float = 1.2,
        w_color: float = 0.4,
    ) -> None:
        self._pacs = pacs_scorer
        self._edge_density_target = edge_density_target
        self._edge_density_tol = edge_density_tol
        self._edge_thresh_quantile = edge_thresh_quantile
        self._w_pacs = w_pacs
        self._w_low_edge = w_low_edge
        self._w_texture = w_texture
        self._w_color = w_color

        # Sobel kernels (same as _SketchScorer)
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device="cuda",
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device="cuda",
        ).view(1, 1, 3, 3)
        self._kx = kx
        self._ky = ky

    def _to_gray(self, x: Tensor) -> Tensor:
        """Convert NCHW float [0,1] to grayscale N1HW."""
        if x.shape[1] == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x[:, 0:1]

    def _sobel_mag(self, gray: Tensor) -> Tensor:
        gx = F.conv2d(gray, self._kx, padding=1)
        gy = F.conv2d(gray, self._ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def _edge_mask(self, mag: Tensor) -> Tensor:
        n = mag.shape[0]
        flat = mag.view(n, -1)
        thr = torch.quantile(flat, self._edge_thresh_quantile, dim=1).view(n, 1, 1, 1)
        return (mag >= thr).float()

    @torch.no_grad()
    def score(self, images_nchw: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute realism score combining PACS photo evidence and texture metrics.

        Args:
            images_nchw: Tensor NCHW float in [0,1].

        Returns:
            (total_score [N], details dict of tensors [N])
        """
        x = images_nchw.to("cuda").float().clamp(0.0, 1.0)

        # 1) PACS photo evidence
        x_u8 = (x * 255).round().to(torch.uint8).detach().cpu()
        from PIL import Image
        pil_imgs = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in x_u8]
        pacs_photo = self._pacs.score_photo_evidence(pil_imgs)  # [N]

        # 2) Edge metrics (GPU)
        gray = self._to_gray(x)
        mag = self._sobel_mag(gray)
        edge = self._edge_mask(mag)

        # Low edge density: inverse of sketch's edge_band
        # Sketch rewards ~5% density; realistic rewards being OUTSIDE that band
        density = edge.mean(dim=(1, 2, 3))
        edge_band = 1.0 - torch.clamp(
            torch.abs(density - self._edge_density_target) / (self._edge_density_tol + 1e-8),
            0.0, 1.0,
        )
        low_edge = 1.0 - edge_band

        # Background texture richness: Sobel mag outside edges
        # Sketch PENALIZES this (wants clean backgrounds); realistic REWARDS it
        non_edge = 1.0 - edge
        texture_richness = (mag * non_edge).sum(dim=(1, 2, 3)) / (non_edge.sum(dim=(1, 2, 3)) + 1e-6)

        # Natural color variance: penalize monochrome/flat (sketch look)
        # High RGB variance per pixel = colorful/realistic
        if x.shape[1] == 3:
            pixel_mean = x.mean(dim=1, keepdim=True)  # [N, 1, H, W]
            color_var = ((x - pixel_mean) ** 2).mean(dim=(1, 2, 3))  # [N]
        else:
            color_var = torch.zeros(x.shape[0], device=x.device)

        total = (
            self._w_pacs * pacs_photo
            + self._w_low_edge * low_edge
            + self._w_texture * texture_richness
            + self._w_color * color_var
        )

        details = {
            "pacs_photo": pacs_photo,
            "edge_density": density,
            "low_edge": low_edge,
            "texture_richness": texture_richness,
            "color_variance": color_var,
        }
        return total, details


# Lazy singletons
_pacs_scorer: _PACSScorer | None = None
_sketch_scorer: _SketchScorer | None = None
_realistic_scorer: _RealisticScorer | None = None


def get_sketch_scorer() -> _SketchScorer:
    """Get or create the shared sketch scorer singleton."""
    global _pacs_scorer, _sketch_scorer
    if _pacs_scorer is None:
        _pacs_scorer = _PACSScorer()
    if _sketch_scorer is None:
        _sketch_scorer = _SketchScorer(_pacs_scorer)
    return _sketch_scorer


def get_realistic_scorer() -> _RealisticScorer:
    """Get or create the shared realistic scorer singleton."""
    global _pacs_scorer, _realistic_scorer
    if _pacs_scorer is None:
        _pacs_scorer = _PACSScorer()
    if _realistic_scorer is None:
        _realistic_scorer = _RealisticScorer(_pacs_scorer)
    return _realistic_scorer


class SketchReward(RewardFunction):
    """Sketch-style reward on the first frame of the video.

    Combines PACS sketch classification with Sobel edge metrics.
    Score is rescaled to [1, 4] to match other reward ranges.
    """

    def __init__(self, scorer: _SketchScorer) -> None:
        self._scorer = scorer

    def compute(self, video: Tensor, prompt: str = "", **kwargs: object) -> float:
        # Extract first frame: video is [C, F, H, W] -> [1, C, H, W]
        frame = video[:, 0, :, :].unsqueeze(0)
        total, _details = self._scorer.score(frame)
        raw = total[0].item()
        # Rescale raw score (typically ~[-1, 3]) to [1, 4]
        score = float(max(1.0, min(4.0, raw + 1.0)))
        return score
