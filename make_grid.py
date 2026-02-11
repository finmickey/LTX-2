#!/usr/bin/env python3
"""Create a 2x5 grid video comparing RL training runs at different steps."""

import subprocess
import tempfile
import os

STEPS = [0, 20, 40, 60, 80]
RUNS = [
    ("outputs/rl_redness_run2/samples", "K=8, lr=1e-4, bs=4"),
    ("outputs/rl_redness_run3/samples", "K=16, lr=5e-5, bs=8"),
]
OUTPUT = "outputs/rl_grid_comparison.mp4"

# Video properties
VW, VH = 256, 256
HEADER_H = 40  # height for step labels
LABEL_W = 180  # width for row labels
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTSIZE_HEADER = 20
FONTSIZE_LABEL = 16
BG_COLOR = "black"
TEXT_COLOR = "white"

def build_ffmpeg_cmd():
    inputs = []
    input_idx = 0

    # Collect all input files (row-major: run0_step0, run0_step1, ..., run1_step0, ...)
    for run_dir, _ in RUNS:
        for step in STEPS:
            path = f"{run_dir}/step_{step:05d}_generated.mp4"
            inputs.extend(["-i", path])
            input_idx += 1

    n_inputs = input_idx  # 10 total

    # Build filter_complex
    # Strategy:
    # 1. For each cell, pad the video (no padding needed, just label)
    # 2. hstack each row
    # 3. Add header row with step labels
    # 4. Add left column with run labels
    # 5. vstack everything

    filters = []

    # Scale all inputs to ensure consistent size
    for i in range(n_inputs):
        filters.append(f"[{i}:v]scale={VW}:{VH},setsar=1[v{i}]")

    # hstack each row
    for row in range(len(RUNS)):
        cell_refs = "".join(f"[v{row * len(STEPS) + col}]" for col in range(len(STEPS)))
        filters.append(f"{cell_refs}hstack=inputs={len(STEPS)}[row{row}]")

    # vstack the rows
    row_refs = "".join(f"[row{row}]" for row in range(len(RUNS)))
    filters.append(f"{row_refs}vstack=inputs={len(RUNS)}[grid]")

    # Add left label column: create colored backgrounds with text for each row
    grid_w = VW * len(STEPS)
    grid_h = VH * len(RUNS)

    # Create label backgrounds for each row
    for row, (_, label) in enumerate(RUNS):
        filters.append(
            f"color=c={BG_COLOR}:s={LABEL_W}x{VH}:d=1,format=yuv420p,"
            f"drawtext=text='{label}':fontfile={FONT}:fontsize={FONTSIZE_LABEL}:"
            f"fontcolor={TEXT_COLOR}:x=(w-tw)/2:y=(h-th)/2[lbl{row}]"
        )

    # vstack labels
    lbl_refs = "".join(f"[lbl{row}]" for row in range(len(RUNS)))
    filters.append(f"{lbl_refs}vstack=inputs={len(RUNS)}[labels]")

    # hstack labels + grid
    filters.append(f"[labels][grid]hstack=inputs=2[body]")

    # Create header row: label spacer + step headers
    # First the spacer for the label column
    filters.append(
        f"color=c={BG_COLOR}:s={LABEL_W}x{HEADER_H}:d=1,format=yuv420p[hdr_spacer]"
    )

    # Step headers
    for col, step in enumerate(STEPS):
        filters.append(
            f"color=c={BG_COLOR}:s={VW}x{HEADER_H}:d=1,format=yuv420p,"
            f"drawtext=text='Step {step}':fontfile={FONT}:fontsize={FONTSIZE_HEADER}:"
            f"fontcolor={TEXT_COLOR}:x=(w-tw)/2:y=(h-th)/2[hdr{col}]"
        )

    hdr_refs = "".join(f"[hdr{col}]" for col in range(len(STEPS)))
    filters.append(f"[hdr_spacer]{hdr_refs}hstack=inputs={len(STEPS) + 1}[header]")

    # vstack header + body
    filters.append(f"[header][body]vstack=inputs=2[out]")

    filter_complex = ";\n".join(filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-shortest",
        OUTPUT,
    ]
    return cmd


if __name__ == "__main__":
    os.chdir("/home/user/LTX-2")
    cmd = build_ffmpeg_cmd()
    print("Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError("ffmpeg failed")
    print(f"Saved to {OUTPUT}")
