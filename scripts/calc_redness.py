"""Calculate redness reward for all videos in a samples folder."""

import sys
from pathlib import Path

import torch
import torchvision.io as tio


def calc_redness(video_path: Path) -> float:
    """Compute mean(R - max(G, B)) for a video file. Returns scalar."""
    video, _, _ = tio.read_video(str(video_path), pts_unit="sec")
    # video shape: [T, H, W, C] in uint8
    video = video.float() / 255.0  # normalize to [0, 1]
    # Rearrange to [C, T, H, W]
    video = video.permute(3, 0, 1, 2)
    r, g, b = video[0], video[1], video[2]
    return (r - torch.max(g, b)).mean().item()


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/rl_redness_run2/samples")

    # Collect all steps that have at least one video
    labels = ["generated", "old", "new", "ref"]
    steps = set()
    for f in folder.glob("step_*.mp4"):
        steps.add(int(f.stem.split("_")[1]))
    steps = sorted(steps)

    if not steps:
        print(f"No video files found in {folder}")
        return

    header = f"{'Step':>6}" + "".join(f"  {l:>10}" for l in labels)
    print(header)
    print("-" * len(header))

    for step in steps:
        row = f"{step:>6}"
        for label in labels:
            path = folder / f"step_{step:05d}_{label}.mp4"
            if path.exists():
                row += f"  {calc_redness(path):>10.4f}"
            else:
                row += f"  {'—':>10}"
        print(row)


if __name__ == "__main__":
    main()
