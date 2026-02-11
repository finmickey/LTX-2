#!/usr/bin/env python

"""
RL training for LTX-2 using the DiffusionNFT algorithm.

This script trains LoRA adapters using reward-guided optimization instead of
ground-truth data. It generates videos, scores them with a reward function,
and updates the model to produce higher-reward outputs.

Basic usage:
    python scripts/rl_train.py CONFIG_PATH

For multi-GPU training (recommended: 8 GPUs for K=8):
    accelerate launch --num_processes=8 scripts/rl_train.py CONFIG_PATH
"""

import os
from pathlib import Path

import torch
import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.rl.rl_trainer import RLTrainer

# Set CUDA device early for DDP
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="RL training for LTX-2 using DiffusionNFT.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
) -> None:
    """Run RL training using the provided configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    try:
        trainer_config = LtxTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration: {e}")
        raise typer.Exit(code=1) from e

    if trainer_config.rl is None:
        typer.echo("Error: RL configuration ('rl' section) is required for rl_train.py")
        raise typer.Exit(code=1)

    trainer = RLTrainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    app()
