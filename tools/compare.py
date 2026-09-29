"""Compares a render with its reference photo: prints PSNR and writes a side-by-side image
(photo on top, render below) for looking at.

    python tools/compare.py renders/render_00001.png data/tandt/train/images/00001.jpg renders/compare_00001.png

PSNR (peak signal-to-noise ratio) measures pixel difference in decibels; higher is closer.
Around 20 dB means clearly the same scene with visible differences, 25 dB close, 30+ dB
hard to tell apart.
"""
import sys

import numpy as np
from PIL import Image


def main(render_path, photo_path, out_path):
    render = Image.open(render_path).convert("RGB")
    photo = Image.open(photo_path).convert("RGB")
    if render.size != photo.size:
        sys.exit(f"sizes differ: render {render.size}, photo {photo.size}; render with --width {photo.size[0]}")
    a = np.asarray(render, dtype=np.float64) / 255.0
    b = np.asarray(photo, dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"PSNR {psnr:.2f} dB  ({render.size[0]}x{render.size[1]})")

    w, h = photo.size
    both = Image.new("RGB", (w, h * 2))
    both.paste(photo, (0, 0))
    both.paste(render, (0, h))
    both.save(out_path)
    print(f"side by side: {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
