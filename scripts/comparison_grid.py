#!/usr/bin/env python3
"""Create a 2x8 comparison grid video: base model vs LoRA checkpoint.

Y-axis: base, lora (step 500)
X-axis: 8 validation prompts (full prompt text as header)

Usage:
    python scripts/comparison_grid.py
"""

import subprocess
import sys
import textwrap
from pathlib import Path

BASE_DIR = Path("outputs/checkpoint_inference_step500_768x512x121_base")
BASE_CFG_STG_DIR = Path("outputs/checkpoint_inference_step500_768x512x121_base_cfg_stg")
LORA_DIR = Path("outputs/checkpoint_inference_step500_768x512x121")
OUTPUT_PATH = Path("outputs/comparison_grid_768x512x121.mp4")

ROWS = [
    ("Base", BASE_DIR),
    ("Base + CFG/STG", BASE_CFG_STG_DIR),
    ("Step 500", LORA_DIR),
]

PROMPTS = [
    ("drone", "A drone flying over a mountain landscape at golden hour"),
    ("cat", "A cat sitting on a windowsill watching birds outside"),
    ("waves", "Waves crashing against rocky cliffs during a storm"),
    ("guitar", "A street musician playing guitar in a busy city square"),
    ("snow", "Snow falling gently over a quiet village at night"),
    ("surfer", "A surfer riding a large wave in the ocean"),
    ("train", "A train passing through a tunnel in the mountains"),
    ("fireworks", "Fireworks exploding over a city skyline at night"),
]


def main():
    # Verify all files exist
    for label, d in ROWS:
        for nick, _ in PROMPTS:
            p = d / f"{nick}.mp4"
            if not p.exists():
                print(f"Error: missing {p}")
                sys.exit(1)

    # Probe tile dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0",
         str(ROWS[0][1] / f"{PROMPTS[0][0]}.mp4")],
        capture_output=True, text=True, check=True,
    )
    tile_w, tile_h = [int(x) for x in probe.stdout.strip().split(",")]
    print(f"Tile size: {tile_w}x{tile_h}")

    n_rows = len(ROWS)
    n_cols = len(PROMPTS)

    # Layout dimensions
    font_size = max(14, tile_h // 16)
    label_w = max(120, font_size * 8)  # row label width
    label_w = label_w if label_w % 2 == 0 else label_w + 1

    # Header height - need room for wrapped prompt text
    char_w = font_size * 0.6
    chars_per_line = max(8, int((tile_w - 16) / char_w))
    max_lines = max(len(textwrap.wrap(p, width=chars_per_line)) for _, p in PROMPTS)
    header_h = max_lines * (font_size + 4) + 16
    header_h = header_h if header_h % 2 == 0 else header_h + 1

    # Collect inputs in row-major order
    input_args = []
    for _, d in ROWS:
        for nick, _ in PROMPTS:
            input_args.extend(["-i", str(d / f"{nick}.mp4")])

    # xstack layout: offset by label_w horizontally and header_h vertically
    layout_parts = []
    for r in range(n_rows):
        for c in range(n_cols):
            layout_parts.append(f"{label_w + c * tile_w}_{header_h + r * tile_h}")

    n_inputs = n_rows * n_cols
    total_w = label_w + n_cols * tile_w
    total_h = header_h + n_rows * tile_h
    pad_w = total_w if total_w % 2 == 0 else total_w + 1
    pad_h = total_h if total_h % 2 == 0 else total_h + 1

    # Build filter: xstack -> pad -> labels
    filter_parts = [
        f"xstack=inputs={n_inputs}:layout={'|'.join(layout_parts)}:fill=black",
        f"pad={pad_w}:{pad_h}:0:0:black",
    ]

    # Column headers: full prompt text, centered and wrapped
    for c, (nick, prompt) in enumerate(PROMPTS):
        wrapped = textwrap.wrap(prompt, width=chars_per_line)
        line_h = font_size + 4
        total_text_h = len(wrapped) * line_h
        y_start = (header_h - total_text_h) // 2
        x_center = label_w + c * tile_w + tile_w // 2
        for li, line_text in enumerate(wrapped):
            safe = line_text.replace("'", "\u2019").replace(":", "\\:").replace("%", "%%")
            y = y_start + li * line_h
            filter_parts.append(
                f"drawtext=text='{safe}':"
                f"fontsize={font_size}:fontcolor=white:"
                f"x={x_center}-text_w/2:y={y}"
            )

    # Row labels: "Base" and "Step 500"
    for r, (label, _) in enumerate(ROWS):
        y_center = header_h + r * tile_h + tile_h // 2
        filter_parts.append(
            f"drawtext=text='{label}':"
            f"fontsize={font_size + 4}:fontcolor=white:"
            f"x=({label_w}-text_w)/2:y={y_center}-text_h/2"
        )

    filter_str = ",".join(filter_parts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(OUTPUT_PATH),
    ]

    print(f"Creating {n_rows}x{n_cols} grid ({pad_w}x{pad_h}) -> {OUTPUT_PATH}")
    subprocess.run(cmd, check=True)
    print(f"Done: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
