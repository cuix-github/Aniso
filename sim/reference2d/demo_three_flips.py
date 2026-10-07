"""The family portrait, as video: plain FLIP, PF-FLIP, and ST-FLIP on one violent
dam break, four panels.

  top-left     plain free-surface FLIP, small dt   - the classic: no air anywhere
  top-right    PF-FLIP, small dt (trusted)         - trapped pockets, cushioned crash
  bottom-left  PF-FLIP at 8x dt (strobe sampling)  - the same physics, strobed: blobs
  bottom-right ST-FLIP at 8x dt                    - the physics restored at 8x steps

All runs use substepped advection; the escape treatment is off everywhere so the
comparison isolates sampling and air. Per-panel wall-clock is stamped on the video.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe demo_three_flips.py  (~12 min)
"""
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim
from validation import SinglePhaseSim
from st_compare import sync_pos

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "validation")
FRAMES_DIR = os.path.join(HERE, "results", "three_flips_frames")
NX, NY = 160, 96
DT_SMALL, DT_BIG = 0.012, 0.096
TOTAL_T, FRAME_T = 3.5, 0.048
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
except Exception:
    FONT = ImageFont.load_default()


def mask(x, y):
    return (x < 36) & (y < 80)


def render(sim, dt, scale=4):
    pos = sync_pos(sim, dt)
    img = np.full((NX * scale, NY * scale, 3), 255, np.uint8)
    pix = np.clip((pos * scale).astype(np.int64), 0, [NX * scale - 1, NY * scale - 1])
    for t, colour in ((0, (223, 230, 239)), (1, (25, 80, 190))):
        sel = sim.typ == t
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, NX * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, NY * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1],
                                 outline=(150, 153, 160), width=2)
    return im


class Panel:
    def __init__(self, label, cls, dt, **kw):
        self.label, self.dt = label, dt
        self.sim = cls(NX, NY, **kw)
        self.sim.seed(mask)
        self.sim.calibrate()
        self.t = 0.0
        self.wall = 0.0

    def advance_to(self, t_target):
        t0 = time.time()
        while self.t < t_target - 1e-9:
            self.sim.step(self.dt)
            self.t += self.dt
        self.wall += time.time() - t0


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    panels = [
        Panel("plain FLIP - no air (small dt)", SinglePhaseSim, DT_SMALL, sub_advect=True),
        Panel("PF-FLIP - air as particles (small dt, trusted)", Sim, DT_SMALL,
              sub_advect=True),
        Panel("PF-FLIP at 8x dt - strobe sampling", Sim, DT_BIG, sub_advect=True),
        Panel("ST-FLIP at 8x dt - spacetime sampling", Sim, DT_BIG, st=True),
    ]
    n_frames = int(round(TOTAL_T / FRAME_T))
    pad, head = 8, 34
    for f in range(n_frames + 1):
        t_target = f * FRAME_T
        for p in panels:
            if f:
                p.advance_to(t_target)
        imgs = [render(p.sim, p.dt) for p in panels]
        w, h = imgs[0].size
        sheet = Image.new("RGB", ((w + pad) * 2 + pad, (h + head + pad) * 2 + pad), "white")
        d = ImageDraw.Draw(sheet)
        for i, (p, im) in enumerate(zip(panels, imgs)):
            x = pad + (i % 2) * (w + pad)
            y = pad + (i // 2) * (h + head + pad)
            d.text((x, y), f"{p.label}   [{p.wall:.0f}s compute]", fill=(10, 10, 10),
                   font=FONT)
            sheet.paste(im, (x, y + head))
        sheet.save(os.path.join(FRAMES_DIR, f"fr{f:04d}.png"))
        if f % 15 == 0:
            print(f"frame {f}/{n_frames}  wall-clocks:",
                  [f"{p.wall:.0f}s" for p in panels])
    print("final compute:", {p.label.split(" - ")[0]: f"{p.wall:.0f}s" for p in panels})
    video = os.path.join(OUT, "three_flips.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                    "-i", os.path.join(FRAMES_DIR, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video],
                   check=True, capture_output=True)
    print("wrote", os.path.normpath(video))


if __name__ == "__main__":
    main()
