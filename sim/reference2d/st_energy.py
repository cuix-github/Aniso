"""The one-glance accuracy picture: kinetic energy and surface roughness vs time for
the family-demo dam break, three ways. Accuracy = how closely a curve hugs the
trusted reference; aliasing shows as sustained agitation and roughness the
reference does not have.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe st_energy.py   (~3 min)
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim
from st_compare import surface_roughness

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "validation")
NX, NY, TOTAL_T, SAMPLE_T = 160, 96, 20.0, 0.1
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except Exception:
    FONT = ImageFont.load_default()


def mask(x, y):
    return (x < 36) & (y < 80)


def run(label, dt, **kw):
    sim = Sim(NX, NY, **kw)
    sim.seed(mask)
    sim.calibrate()
    t, next_s = 0.0, SAMPLE_T
    ts, ke, rg = [0.0], [0.0], [surface_roughness(sim, dt)]
    while t < TOTAL_T - 1e-9:
        sim.step(dt)
        t += dt
        if t >= next_s - 1e-9:
            m = sim.masses()
            ts.append(t)
            ke.append(float(0.5 * (m * (sim.vel ** 2).sum(1)).sum()))
            rg.append(surface_roughness(sim, dt))
            next_s += SAMPLE_T
    print(f"{label}: {len(ts)} samples, final KE {ke[-1]:.2e}, final roughness {rg[-1]:.2f}")
    return label, np.array(ts), np.array(ke), np.array(rg)


def panel(d, series, ys, title, x0, y0, w, h, log=False):
    colours = [(60, 60, 60), (200, 60, 40), (30, 120, 210)]
    lo = min(y.min() for y in ys)
    hi = max(y.max() for y in ys)
    if log:
        floor = max(hi * 1e-4, 1e-12)
        ys = [np.log10(np.maximum(y, floor)) for y in ys]
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
    hi = hi if hi > lo else lo + 1
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(120, 120, 120))
    d.text((x0, y0 - 28), title, fill=(10, 10, 10), font=FONT)
    for ci, ((label, ts, *_), y) in enumerate(zip(series, ys)):
        pts = [(x0 + w * t / TOTAL_T, y0 + h - h * (v - lo) / (hi - lo))
               for t, v in zip(ts, y)]
        d.line(pts, fill=colours[ci], width=3)
        d.text((x0 + 10, y0 + 8 + 22 * ci), label, fill=colours[ci], font=FONT)
    for frac in (0.25, 0.5, 0.75):
        d.text((x0 + w * frac - 10, y0 + h + 4), f"{TOTAL_T * frac:.0f}s",
               fill=(120, 120, 120), font=FONT)


def main():
    series = [run("PF-FLIP, trusted small dt", 0.012, sub_advect=True),
              run("PF-FLIP at 8x dt (strobe)", 0.096, sub_advect=True),
              run("ST-FLIP at 8x dt", 0.096, st=True)]
    W, H = 980, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    panel(d, series, [s[2] for s in series],
          "kinetic energy vs time (log scale) - agitation that will not settle is numerics",
          70, 60, W - 120, 280, log=True)
    panel(d, series, [s[3] for s in series],
          "surface roughness vs time - standing froth the reference does not have",
          70, 440, W - 120, 280)
    path = os.path.join(OUT, "st_energy_curves.png")
    im.save(path)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
