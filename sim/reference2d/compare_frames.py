"""Side-by-side frame comparisons: plain single-phase FLIP (top row) vs PF-FLIP (bottom row),
same scene, same seeds, same instants. Two scenes:

  dam_break   air escapes freely, so the two methods should look broadly similar — an honest
              negative control
  air_cushion a slab falls on a sealed pocket of air, where only PF-FLIP has the physics

  python compare_frames.py        ->  results/validation/compare_{dam_break,air_cushion}.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim
from validation import SinglePhaseSim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "validation")
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
except Exception:
    FONT = ImageFont.load_default()


def render(sim, scale=5):
    img = np.full((sim.nx * scale, sim.ny * scale, 3), 255, np.uint8)
    pix = np.clip((sim.pos * scale).astype(np.int64), 0, [sim.nx * scale - 1, sim.ny * scale - 1])
    for t, colour in ((0, (225, 232, 240)), (1, (25, 80, 190))):   # air very light, liquid blue
        sel = sim.typ == t
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, sim.nx * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, sim.ny * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(120, 120, 120), width=2)
    return im


def run_pair(nx, ny, mask, times, dt_frame, label):
    sheets = {}
    for name, cls in (("plain FLIP (no air — free surface)", SinglePhaseSim),
                      ("PF-FLIP (air simulated as particles)", Sim)):
        sim = cls(nx, ny)
        sim.seed(mask)
        sim.calibrate()
        frames, t_now, snaps = int(round(max(times) / dt_frame)) + 1, 0.0, {}

        def cb(f, s, snaps=snaps):
            t = round((f + 1) * dt_frame, 6)
            for want in times:
                if abs(t - want) < dt_frame / 2 and want not in snaps:
                    snaps[want] = render(s)

        first = render(sim)
        sim.run(frames, dt_frame, on_frame=cb)
        row = [first if abs(t) < 1e-9 else snaps[min(snaps, key=lambda k: abs(k - t))] for t in times]
        sheets[name] = row

    w, h = sheets[next(iter(sheets))][0].size
    pad, head = 6, 34
    W = (w + pad) * len(times) + pad
    H = (h + head + pad) * 2 + pad + 30
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    y = pad
    for name, row in sheets.items():
        d.text((pad, y + 4), name, fill=(10, 10, 10), font=FONT)
        y += head
        for i, im in enumerate(row):
            out.paste(im, (pad + i * (w + pad), y))
        y += h + pad
    for i, t in enumerate(times):
        d.text((pad + i * (w + pad) + 6, H - 28), f"t = {t:.2f}", fill=(90, 90, 90), font=FONT)
    path = os.path.join(OUT, f"compare_{label}.png")
    out.save(path)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    run_pair(160, 80, lambda x, y: (x < 40) & (y < 56), [0.0, 0.9, 1.5, 2.4], 0.06, "dam_break")
    run_pair(96, 80, lambda x, y: (y < 20) | ((y > 30) & (y < 44)), [0.0, 0.4, 0.8, 1.2], 0.04,
             "air_cushion")
