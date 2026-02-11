"""Tests for ColorAlternationReward scoring across different video scenarios."""

import sys
sys.path.insert(0, "packages/ltx-trainer/src")

import torch
from ltx_trainer.rl.rewards import ColorAlternationReward, UniformFrameReward

F = 81  # typical frame count
H, W = 8, 8  # small spatial dims for speed


def make_video(frame_fn):
    """Build a [3, F, H, W] video where frame_fn(f) returns (R, G, B) per frame."""
    video = torch.zeros(3, F, H, W)
    for f in range(F):
        r, g, b = frame_fn(f)
        video[0, f] = r
        video[1, f] = g
        video[2, f] = b
    return video


ca = ColorAlternationReward()
uf = UniformFrameReward()

CA_W = 50.0  # config weight (pure min, high weight)


def score(name, video):
    c = ca.compute(video)
    total = c * CA_W
    print(f"  {name:35s}  ca={c:+.4f} (x{CA_W}={c*CA_W:+.3f})  TOTAL={total:+.3f}")
    return c, 0, total


print("=" * 90)
print("ColorAlternationReward (new impl) — Scoring Table")
print("=" * 90)

print("\n--- Degenerate cases (should score ~0 or negative) ---")
c_black, _, _ = score("Black (0,0,0)", make_video(lambda f: (0, 0, 0)))
c_gray, _, _ = score("Gray (0.5,0.5,0.5)", make_video(lambda f: (0.5, 0.5, 0.5)))
c_noise, _, _ = score("Random noise", torch.rand(3, F, H, W))

print("\n--- Single-color (should score LOW — these are traps) ---")
c_red, _, t_red = score("All red (1,0,0)", make_video(lambda f: (1, 0, 0)))
c_blue, _, t_blue = score("All blue (0,0,1)", make_video(lambda f: (0, 0, 1)))
c_dimred, _, _ = score("Dim red (0.6,0.3,0.3)", make_video(lambda f: (0.6, 0.3, 0.3)))

print("\n--- Alternating (should score HIGH — these are the goal) ---")
c_alt1, _, t_alt1 = score("Alt R/B every frame",
    make_video(lambda f: (1, 0, 0) if f % 2 == 0 else (0, 0, 1)))
c_alt5, _, t_alt5 = score("Alt R/B every 5 frames",
    make_video(lambda f: (1, 0, 0) if (f // 5) % 2 == 0 else (0, 0, 1)))
c_alt10, _, t_alt10 = score("Alt R/B every 10 frames",
    make_video(lambda f: (1, 0, 0) if (f // 10) % 2 == 0 else (0, 0, 1)))

print("\n--- Partial / intermediate ---")
c_7030, _, _ = score("70/30 R/B split",
    make_video(lambda f: (1, 0, 0) if f < 57 else (0, 0, 1)))
c_half_sat, _, _ = score("Alt dim R/B (0.7,0.3,0.3)/(0.3,0.3,0.7)",
    make_video(lambda f: (0.7, 0.3, 0.3) if f % 2 == 0 else (0.3, 0.3, 0.7)))
c_rg, _, _ = score("Alt R/G (not blue!)",
    make_video(lambda f: (1, 0, 0) if f % 2 == 0 else (0, 1, 0)))

print("\n--- Gradient check: does escaping all-red have signal? ---")
# Simulate: 80 red frames + 1 blue frame (tiny perturbation from all-red)
c_almost_red, _, _ = score("80 red + 1 blue frame",
    make_video(lambda f: (1, 0, 0) if f < 80 else (0, 0, 1)))
# Compare to all-red
print(f"  -> Delta from all-red: {c_almost_red - c_red:+.6f}  "
      f"(should be > 0 for gradient signal)")

print("\n" + "=" * 90)
print("KEY CHECKS:")
print(f"  1. All-red total ({t_red:+.3f}) << Alt R/B total ({t_alt1:+.3f})?  "
      f"{'YES' if t_alt1 > t_red + 0.5 else 'NO — PROBLEM!'}")
print(f"  2. Black ca ({c_black:+.4f}) <= 0?  "
      f"{'YES' if c_black <= 0.001 else 'NO — PROBLEM!'}")
print(f"  3. All-red ca ({c_red:+.4f}) ~= 0 (no free reward)?  "
      f"{'YES' if abs(c_red) < 0.3 else 'NO — PROBLEM!'}")
print(f"  4. Alt R/B ca ({c_alt1:+.4f}) is the highest ca?  "
      f"{'YES' if c_alt1 >= max(c_red, c_blue, c_black, c_noise) else 'NO — PROBLEM!'}")
print(f"  5. 80red+1blue ({c_almost_red:+.4f}) > all-red ({c_red:+.4f})?  "
      f"{'YES' if c_almost_red > c_red + 1e-5 else 'NO — gradient vanishes!'}")
print(f"  6. Dim alternation ({c_half_sat:+.4f}) > 0?  "
      f"{'YES' if c_half_sat > 0.01 else 'NO — too weak for partial colors!'}")
print("=" * 90)
