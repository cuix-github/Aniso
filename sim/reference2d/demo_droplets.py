"""The escaped-particle demo, as video: the paper's free droplets and bubbles.

A tall column plunges from high into a pool, the same scene twice, side by side:
left WITHOUT the escape treatment (fine spray is dragged back into the surface by the
grid), right WITH it (escaped liquid flies as ballistic droplets, cyan; escaped air
rises as buoyant bubbles, white). Rendered per frame and assembled into an mp4.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe demo_droplets.py  (~15 min)
"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "validation")
NX, NY, FRAMES, DT_FRAME = 120, 160, 140, 0.05
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except Exception:
    FONT = ImageFont.load_default()


def mask(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def render(sim, scale=4):
    img = np.full((sim.nx * scale, sim.ny * scale, 3), 255, np.uint8)
    pix = np.clip((sim.pos * scale).astype(np.int64), 0, [sim.nx * scale - 1, sim.ny * scale - 1])
    esc = sim.escaped if len(sim.escaped) == len(sim.typ) else np.zeros(len(sim.typ), bool)
    layers = [(~esc & (sim.typ == 0), (225, 232, 240), 1),   # bulk air, pale
              (~esc & (sim.typ == 1), (25, 80, 190), 1),     # bulk liquid, blue
              (esc & (sim.typ == 0), (255, 255, 255), 2),    # bubbles, white, bigger
              (esc & (sim.typ == 1), (0, 190, 255), 2)]      # droplets, cyan, bigger
    for sel, colour, r in layers:
        for dx in range(r + 1):
            for dy in range(r + 1):
                img[np.clip(pix[sel, 0] + dx, 0, sim.nx * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, sim.ny * scale - 1)] = colour
    return Image.fromarray(np.rot90(img))


def main():
    os.makedirs(OUT, exist_ok=True)
    frames_dir = os.path.join(HERE, "results", "droplet_frames")
    os.makedirs(frames_dir, exist_ok=True)

    sims = []
    for esc in (False, True):
        s = Sim(NX, NY, escape=esc)
        s.seed(mask)
        s.calibrate()
        sims.append(s)

    head, pad = 34, 8
    labels = ("without escape treatment", "with droplets (cyan) + bubbles (white)")
    counts = []

    def save_frame(f):
        imgs = [render(s) for s in sims]
        w, h = imgs[0].size
        sheet = Image.new("RGB", ((w + pad) * 2 + pad, h + head + pad), "white")
        d = ImageDraw.Draw(sheet)
        for i, (im, lab) in enumerate(zip(imgs, labels)):
            d.text((pad + i * (w + pad), 6), lab, fill=(10, 10, 10), font=FONT)
            sheet.paste(im, (pad + i * (w + pad), head))
        sheet.save(os.path.join(frames_dir, f"fr{f:04d}.png"))

    save_frame(0)
    for f in range(1, FRAMES + 1):
        for s in sims:
            s.run(1, DT_FRAME)
        esc = sims[1].escaped
        counts.append((int((esc & (sims[1].typ == 1)).sum()),
                       int((esc & (sims[1].typ == 0)).sum())))
        save_frame(f)
        if f % 15 == 0:
            print(f"frame {f}/{FRAMES}: droplets {counts[-1][0]}, bubbles {counts[-1][1]}")

    peak = max(counts, key=lambda c: c[0])
    print(f"peak droplets in flight: {peak[0]}; bubbles at that moment: {peak[1]}")
    video = os.path.join(OUT, "droplets_2d.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                    "-i", os.path.join(frames_dir, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video],
                   check=True, capture_output=True)
    print("wrote", os.path.normpath(video))


if __name__ == "__main__":
    main()
