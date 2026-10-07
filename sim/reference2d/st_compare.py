"""The ST-FLIP 2D gate: visual fidelity and cost at large time steps.

Each scene runs three ways, all with substepped advection (local CFL 1) so the
comparison is fair, exactly as the ST-FLIP paper does:

  baseline     PF-FLIP at the small, trusted dt (the validated regime)
  large-dt     PF-FLIP at a ~10x larger dt - expect strobe/aliasing surface artifacts
  ST-FLIP      the same large dt with spatiotemporal sampling - should track baseline

Outputs: a three-row frame sheet per scene, a cost table (steps, pressure
iterations, wall-clock per simulated second), and a surface-roughness metric
that quantifies the aliasing ripples.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe st_compare.py
"""
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfflip2d import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "validation")
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
except Exception:
    FONT = ImageFont.load_default()


def sync_pos(sim, dt):
    """Resynchronize jittered particle times to the slab centre for output, as the
    paper does: without this, rendering draws the time spread as motion blur."""
    if not getattr(sim, "st", False) or dt is None:
        return sim.pos
    return sim.pos - sim.tau[:, None] * dt * sim.vel


def render(sim, scale=5, dt=None):
    pos = sync_pos(sim, dt)
    img = np.full((sim.nx * scale, sim.ny * scale, 3), 255, np.uint8)
    pix = np.clip((pos * scale).astype(np.int64),
                  0, [sim.nx * scale - 1, sim.ny * scale - 1])
    for t, colour in ((0, (223, 230, 239)), (1, (25, 80, 190))):
        sel = sim.typ == t
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, sim.nx * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, sim.ny * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1],
                                 outline=(120, 120, 120), width=2)
    return im


def surface_roughness(sim, dt=None):
    """Std of the liquid surface height about its smoothed profile: the ripple metric."""
    pos = sync_pos(sim, dt)
    cols = np.clip(pos[sim.typ == 1, 0].astype(int), 0, sim.nx - 1)
    ys = pos[sim.typ == 1, 1]
    height = np.full(sim.nx, np.nan)
    for i in range(sim.nx):
        sel = cols == i
        if sel.any():
            height[i] = np.percentile(ys[sel], 99)
    ok = ~np.isnan(height)
    h = height[ok]
    k = np.ones(9) / 9.0
    smooth = np.convolve(np.pad(h, 4, mode="edge"), k, mode="valid")
    return float(np.std(h - smooth))


def run(name, nx, ny, mask, dt, total_t, sample_times, **kw):
    sim = Sim(nx, ny, **kw)
    sim.seed(mask)
    sim.calibrate()
    shots, rough = {}, []
    t_now, steps, iters = 0.0, 0, 0
    t0 = time.time()
    while t_now < total_t - 1e-9:
        sim.step(min(dt, total_t - t_now))
        t_now += dt
        steps += 1
        iters += sim.iters
        for want in sample_times:
            if want not in shots and t_now >= want - 1e-9:
                shots[want] = render(sim, dt=dt)
        if t_now > 0.55 * total_t:
            rough.append(surface_roughness(sim, dt=dt))
    wall = time.time() - t0
    return {"name": name, "shots": [shots[w] for w in sample_times],
            "rough": float(np.mean(rough)), "steps": steps, "iters": iters,
            "wall": wall, "per_sim_s": wall / total_t}


def scene(label, nx, ny, mask, dt_small, dt_big, total_t, sample_times):
    print(f"=== {label}")
    runs = [run("PF-FLIP, trusted small dt", nx, ny, mask, dt_small, total_t,
                sample_times, sub_advect=True),
            run(f"PF-FLIP at {dt_big / dt_small:.0f}x dt (strobe sampling)", nx, ny, mask,
                dt_big, total_t, sample_times, sub_advect=True),
            run(f"ST-FLIP at {dt_big / dt_small:.0f}x dt (spacetime sampling)", nx, ny, mask,
                dt_big, total_t, sample_times, st=True)]
    print(f"  {'config':44s} {'steps':>6s} {'CG iters':>9s} {'s/sim-s':>8s} {'roughness':>10s}")
    for r in runs:
        print(f"  {r['name']:44s} {r['steps']:6d} {r['iters']:9d} "
              f"{r['per_sim_s']:8.1f} {r['rough']:10.3f}")
    base, strobe, st = (r["rough"] for r in runs)
    print(f"  roughness vs baseline: strobe {strobe / base:.2f}x, ST {st / base:.2f}x; "
          f"cost ratio baseline/ST: {runs[0]['per_sim_s'] / runs[2]['per_sim_s']:.1f}x")

    w, h = runs[0]["shots"][0].size
    pad, head = 6, 34
    W = (w + pad) * len(sample_times) + pad
    H = (h + head + pad) * 3 + pad + 30
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    y = pad
    for r in runs:
        d.text((pad, y + 4), r["name"], fill=(10, 10, 10), font=FONT)
        y += head
        for i, im in enumerate(r["shots"]):
            out.paste(im, (pad + i * (w + pad), y))
        y += h + pad
    for i, t in enumerate(sample_times):
        d.text((pad + i * (w + pad) + 6, H - 28), f"t = {t:.2f}", fill=(90, 90, 90), font=FONT)
    path = os.path.join(OUT, f"stflip_{label}.png")
    out.save(path)
    print("  wrote", os.path.normpath(path))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    scene("dam_break", 160, 80, lambda x, y: (x < 40) & (y < 56),
          0.015, 0.15, 2.4, [0.6, 1.2, 1.8, 2.4])
    scene("plunge", 120, 160,
          lambda x, y: (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130)),
          0.012, 0.12, 5.4, [2.4, 3.3, 4.2, 5.4])
