"""Lazy-loaded embedding store for RL training with large prompt sets.

Instead of holding all prompt embeddings in RAM, this loads each embedding
from a `.pt` file on demand. Each file contains a dict with keys
``video_context_positive`` and ``audio_context_positive``.

File index corresponds 1:1 with line number in the prompts file.
"""

from pathlib import Path

import torch

from ltx_trainer.validation_sampler import CachedPromptEmbeddings


class LazyEmbeddingStore:
    """Drop-in replacement for ``list[CachedPromptEmbeddings]`` that loads from disk.

    Supports ``__len__`` and ``__getitem__`` — the only two methods used by the
    RL training loop.
    """

    def __init__(self, embeddings_dir: str | Path, prompts_file: str | Path) -> None:
        embeddings_dir = Path(embeddings_dir)
        prompts_file = Path(prompts_file)

        # Discover .pt files sorted numerically (000000.pt, 000001.pt, ...)
        pt_files = sorted(embeddings_dir.glob("*.pt"), key=lambda p: int(p.stem))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {embeddings_dir}")
        self._files = pt_files

        # Load prompt strings (100k strings ~ 10-20 MB, negligible)
        self._prompts = [
            line.strip()
            for line in prompts_file.read_text().splitlines()
            if line.strip()
        ]

        if len(self._files) != len(self._prompts):
            raise ValueError(
                f"Mismatch: {len(self._files)} .pt files in {embeddings_dir} "
                f"but {len(self._prompts)} prompts in {prompts_file}"
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> CachedPromptEmbeddings:
        data = torch.load(self._files[idx], map_location="cpu", weights_only=True)
        return CachedPromptEmbeddings(
            video_context_positive=data["video_context_positive"],
            audio_context_positive=data["audio_context_positive"],
            prompt_text=self._prompts[idx],
        )
