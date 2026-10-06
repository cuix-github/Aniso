"""The escaped-particle demo, as video: the paper's free droplets and bubbles.

A tall column plunges from high into a pool, the same scene twice, side by side:
left WITHOUT the escape treatment (fine spray is dragged back into the surface by the
grid), right WITH it. Escaped particles are drawn as outlined markers so they read at
a glance: droplets orange, bubbles white with a dark ring.

Per-frame particle states are cached as .npz, so rendering tweaks replay without
re-simulating:

  ..\\..\\.venv\\Scripts\\python.exe demo_droplets.py            # simulate + render
  ..\\..\\.venv\\Scripts\\python.exe demo_droplets.py rerender   # render from cache
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "validation")
CACHE = os.path.join(HERE, "results", "droplet_states")
FRAMES_DIR = os.path.join(HERE, "results", "droplet_frames")
NX, NY, FRAMES, DT_FRAME = 120, 160, 140, 0.05
try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()


def mask(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def render(pos, typ, escaped, scale=5):
    img = np.full((NX * scale, NY * scale, 3), 255, np.uint8)
    pix = np.clip((pos * scale).astype(np.int64), 0, [NX * scale - 1, NY * scale - 1])
    for sel, colour in ((~escaped & (typ == 0), (223, 230, 239)),
                        (~escaped & (typ == 1), (25, 80, 190))):
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, NX * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, NY * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    d = ImageDraw.Draw(im)
    H = im.height
    r = 5
    for sel, fill, ring in ((escaped & (typ == 1), (255, 120, 0), (120, 50, 0)),
                            (escaped & (typ == 0), (255, 255, 255), (10, 30, 70))):
        for x, y in pos[sel]:
            cx, cy = x * scale, H - y * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=ring, width=2)
    return im


def simulate():
    os.makedirs(CACHE, exist_ok=True)
    sims = []
    for esc in (False, True):
        s = Sim(NX, NY, escape=esc)
        s.seed(mask)
        s.calibrate()
        sims.append(s)
    for f in range(FRAMES + 1):
        if f > 0:
            for s in sims:
                s.run(1, DT_FRAME)
        e1 = sims[1].escaped if len(sims[1].escaped) == len(sims[1].typ) \
            else np.zeros(len(sims[1].typ), bool)
        np.savez_compressed(os.path.join(CACHE, f"st{f:04d}.npz"),
                            p0=sims[0].pos.astype(np.float32), t0=sims[0].typ,
                            p1=sims[1].pos.astype(np.float32), t1=sims[1].typ, e1=e1)
        if f % 20 == 0:
            print(f"frame {f}/{FRAMES}: droplets {int((e1 & (sims[1].typ == 1)).sum())}, "
                  f"bubbles {int((e1 & (sims[1].typ == 0)).sum())}")


def render_all():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    head, pad = 40, 8
    labels = ("without escape treatment", "with droplets (orange) + bubbles (white)")
    for f in range(FRAMES + 1):
        st = np.load(os.path.join(CACHE, f"st{f:04d}.npz"))
        none = np.zeros(len(st["t0"]), bool)
        imgs = [render(st["p0"], st["t0"], none),
                render(st["p1"], st["t1"], st["e1"].astype(bool))]
        w, h = imgs[0].size
        sheet = Image.new("RGB", ((w + pad) * 2 + pad, h + head + pad), "white")
        d = ImageDraw.Draw(sheet)
        for i, (im, lab) in enumerate(zip(imgs, labels)):
            d.text((pad + i * (w + pad), 8), lab, fill=(10, 10, 10), font=FONT)
            sheet.paste(im, (pad + i * (w + pad), head))
        sheet.save(os.path.join(FRAMES_DIR, f"fr{f:04d}.png"))
    video = os.path.join(OUT, "droplets_2d.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                    "-i", os.path.join(FRAMES_DIR, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video],
                   check=True, capture_output=True)
    print("wrote", os.path.normpath(video))


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "rerender"):
        simulate()
    render_all()
