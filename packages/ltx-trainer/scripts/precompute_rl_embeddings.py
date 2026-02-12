#!/usr/bin/env python

"""
Pre-compute text embeddings for RL training prompts.

Encodes each prompt with the Gemma text encoder and saves one .pt file per
prompt.  The resulting directory can be passed to the trainer via
``precomputed_embeddings_dir`` so that embeddings are lazily loaded instead of
encoded at startup (avoids OOM with 100k+ prompts).

Usage:
    python scripts/precompute_rl_embeddings.py prompts.txt \
        --output-dir embeddings/ \
        --model-path models/ltx-2-19b-dev.safetensors \
        --text-encoder-path models/gemma-3-12b-it-qat-q4_0-unquantized
"""

import os
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Pre-compute text embeddings for RL training prompts.",
)


@app.command()
def main(
    prompts_file: str = typer.Argument(
        ..., help="Text file with one prompt per line"
    ),
    output_dir: str = typer.Option(
        ..., help="Directory to save .pt embedding files"
    ),
    model_path: str = typer.Option(
        ..., help="Path to LTX-2 checkpoint (.safetensors)"
    ),
    text_encoder_path: str = typer.Option(
        ..., help="Path to Gemma text encoder directory"
    ),
    device: str = typer.Option(default="cuda", help="Device for encoding"),
    load_in_8bit: bool = typer.Option(
        default=False,
        help="Load Gemma in 8-bit precision (requires bitsandbytes)",
    ),
) -> None:
    """Encode each prompt and save as {idx:06d}.pt."""
    prompts_path = Path(prompts_file)
    if not prompts_path.is_file():
        raise typer.BadParameter(f"Prompts file not found: {prompts_file}")

    prompts = [
        line.strip()
        for line in prompts_path.read_text().splitlines()
        if line.strip()
    ]
    logger.info(f"Loaded {len(prompts):,} prompts from {prompts_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console = Console()
    with console.status("[bold]Loading Gemma text encoder...", spinner="dots"):
        text_encoder = load_text_encoder(
            checkpoint_path=model_path,
            gemma_model_path=text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )
    logger.info("Text encoder loaded.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding prompts", total=len(prompts))
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                v_ctx, a_ctx, _ = text_encoder(prompt)
                data = {
                    "video_context_positive": v_ctx.cpu(),
                    "audio_context_positive": a_ctx.cpu(),
                }
                torch.save(data, out / f"{idx:06d}.pt")
                progress.advance(task)

    logger.info(
        f"Saved {len(prompts):,} embedding files to {out}"
    )


if __name__ == "__main__":
    app()
