#!/usr/bin/env python3
"""Create a grid video from RL validation videos.

Y-axis: 8 validation prompts (nicknames)
X-axis: training steps (uniformly sampled)

Usage:
    python scripts/validation_grid.py outputs/rl_videoscore_run4/samples
    python scripts/validation_grid.py outputs/rl_videoscore_run4/samples --limit 6
"""

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create grid video from RL validation videos")
    parser.add_argument("folder", type=Path, help="Path to the samples/ folder")
    parser.add_argument("--limit", type=int, default=10, help="Max number of steps (columns) in the grid")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="Validation prompts file (nickname | prompt). If not given, labels use nicknames only.")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    # Parse all video files: step_00000_drone.mp4
    pattern = re.compile(r"^step_(\d+)_(.+)\.mp4$")
    entries: dict[int, dict[str, Path]] = {}  # step -> {nickname -> path}
    for f in sorted(folder.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        step = int(m.group(1))
        nickname = m.group(2)
        entries.setdefault(step, {})[nickname] = f

    if not entries:
        print(f"Error: no validation videos found in {folder}")
        sys.exit(1)

    all_steps = sorted(entries.keys())
    # Discover nicknames from the first step (preserves file-system order)
    nicknames = sorted(entries[all_steps[0]].keys())
    print(f"Found {len(all_steps)} steps, {len(nicknames)} prompts: {nicknames}")

    # Load full prompt texts from prompts file if provided
    nick_to_prompt: dict[str, str] = {}
    if args.prompts_file and args.prompts_file.is_file():
        for line in args.prompts_file.read_text().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            nick, prompt = line.split("|", 1)
            nick_to_prompt[nick.strip()] = prompt.strip()

    # Select steps: first, last, and limit-2 uniformly distributed in between
    limit = min(args.limit, len(all_steps))
    if limit <= 2:
        selected_steps = [all_steps[0], all_steps[-1]][:limit]
    else:
        selected_steps = [all_steps[0]]
        n_middle = limit - 2
        for i in range(n_middle):
            idx = int((i + 1) * (len(all_steps) - 1) / (n_middle + 1))
            selected_steps.append(all_steps[idx])
        selected_steps.append(all_steps[-1])

    print(f"Selected {len(selected_steps)} steps: {selected_steps}")

    # Verify all files exist
    for step in selected_steps:
        for nick in nicknames:
            if nick not in entries.get(step, {}):
                print(f"Warning: missing step_{step:05d}_{nick}.mp4, skipping step {step}")
                selected_steps = [s for s in selected_steps if s != step]
                break

    n_rows = len(nicknames)
    n_cols = len(selected_steps)

    # Probe tile dimensions from the first video
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "csv=p=0",
         str(entries[all_steps[0]][nicknames[0]])],
        capture_output=True, text=True, check=True,
    )
    parts = probe.stdout.strip().split(",")
    tile_w, tile_h = int(parts[0]), int(parts[1])
    print(f"Tile size: {tile_w}x{tile_h}")

    # Label column width — wide enough for wrapped prompt text
    label_w = tile_w
    # Ensure even
    label_w = label_w if label_w % 2 == 0 else label_w + 1
    font_size = max(14, tile_h // 16)
    # Approximate chars per line (~0.6 * font_size per char)
    char_w = font_size * 0.6
    chars_per_line = max(10, int((label_w - 16) / char_w))  # 8px padding each side

    # Build ffmpeg filter graph
    # Inputs: one per cell, row-major order (after label inputs)
    inputs = []
    for nick in nicknames:
        for step in selected_steps:
            inputs.append(entries[step][nick])

    input_args = []
    for p in inputs:
        input_args.extend(["-i", str(p)])

    # Header row height for step numbers
    header_h = font_size + 16
    # Ensure even
    header_h = header_h if header_h % 2 == 0 else header_h + 1

    # Build xstack layout with explicit pixel coordinates
    # Videos are offset by label_w horizontally and header_h vertically
    layout_parts = []
    for r in range(n_rows):
        for c in range(n_cols):
            layout_parts.append(f"{label_w + c * tile_w}_{header_h + r * tile_h}")

    n_inputs = n_rows * n_cols
    total_w = label_w + n_cols * tile_w
    total_h = header_h + n_rows * tile_h
    # Ensure even dimensions for h264
    pad_w = total_w if total_w % 2 == 0 else total_w + 1
    pad_h = total_h if total_h % 2 == 0 else total_h + 1

    # Build the filter: xstack -> pad -> header step labels -> row prompt labels
    filter_parts = [
        f"xstack=inputs={n_inputs}:layout={'|'.join(layout_parts)}:fill=black",
        f"pad={pad_w}:{pad_h}:0:0:black",
    ]

    # Draw step number headers centered over each column
    for c, step in enumerate(selected_steps):
        x_center = label_w + c * tile_w + tile_w // 2
        filter_parts.append(
            f"drawtext=text='Step {step}':"
            f"fontsize={font_size}:fontcolor=white:"
            f"x={x_center}-text_w/2:y=({header_h}-text_h)/2"
        )

    # Draw row prompt labels
    for r, nick in enumerate(nicknames):
        label = nick_to_prompt.get(nick, nick)
        # Word-wrap and render as multiple drawtext lines
        wrapped = textwrap.wrap(label, width=chars_per_line)
        line_h = font_size + 4
        total_text_h = len(wrapped) * line_h
        y_start = header_h + r * tile_h + (tile_h - total_text_h) // 2
        for li, line_text in enumerate(wrapped):
            safe = line_text.replace("'", "\u2019").replace(":", "\\:").replace("%", "%%")
            y = y_start + li * line_h
            filter_parts.append(
                f"drawtext=text='{safe}':"
                f"fontsize={font_size}:fontcolor=white:"
                f"x=8:y={y}"
            )

    filter_str = ",".join(filter_parts)

    output_path = folder.parent / f"validation_grid_step{selected_steps[-1]:05d}.mp4"

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(output_path),
    ]

    print(f"Creating {n_rows}x{n_cols} grid + labels ({pad_w}x{pad_h}) -> {output_path}")
    subprocess.run(cmd, check=True)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
