"""Milestone-5 validation, layers 2 and 3: the full PF-FLIP loop running inside Bifrost
(sim_2d.json, one bifcmd run per scene) against the python reference and against physics.

Scenes:
  smoke          16x16 sanity run, a few substeps, no comparison
  dam_break      front-position curve vs the reference + side-by-side frame sheet
  hydrostatic    the column must stay at rest (physics anchor, no reference needed)
  air_cushion    sealed slab must float on trapped air; gap curve vs the reference
  rayleigh_taylor 19:1 fingers must grow (two-phase-only physics anchor) + sheet

The python runs use the same fixed dt as the graph, so frame times align exactly.
Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_loop.py [scene]
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402

RESULTS = os.path.join(HERE, "..", "reference2d", "results", "validation")
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
except Exception:
    FONT = ImageFont.load_default()


def seed_arrays(nx, ny, mask, rho_l=1000.0, rho_g=1.0):
    sim = Sim(nx, ny, rho_l=rho_l, rho_g=rho_g)
    sim.seed(mask)
    n = len(sim.pos)
    pos3 = np.concatenate([sim.pos, np.full((n, 1), 0.5)], 1).astype(np.float32)
    vel3 = np.zeros((n, 3), np.float32)
    typ = sim.typ.astype(np.float32)
    mass = np.where(sim.typ == 1, rho_l, rho_g).astype(np.float32)
    return sim, pos3, vel3, typ, mass


def probes(nx, ny):
    """Voxel-centre probe lattices of the two shifted splat volumes = MAC face centres."""
    def lattice(ni, nj):
        i, j = np.meshgrid(np.arange(ni) + 0.5, np.arange(nj) + 0.5, indexing="ij")
        return np.stack([i.ravel(), j.ravel(), np.full(i.size, 0.5)], 1).astype(np.float32)
    return lattice(nx + 1, ny), lattice(nx, ny + 1)


def run_bifrost(tag, nx, ny, mask, substeps, dt, rho_l=1000.0, rho_g=1.0,
                alpha_l=0.97, alpha_g=0.9):
    d = os.path.join(HERE, "out", "loop_" + tag)
    os.makedirs(d, exist_ok=True)
    sim, pos3, vel3, typ, mass = seed_arrays(nx, ny, mask, rho_l, rho_g)
    pu, pv = probes(nx, ny)
    arrs = {"positions": pos3, "velocities": vel3, "particle_phase": typ, "mass": mass,
            "momx": (mass * vel3[:, 0]).astype(np.float32),
            "momy": (mass * vel3[:, 1]).astype(np.float32),
            "probes_u": pu, "probes_v": pv}
    for k, v in arrs.items():
        np.save(os.path.join(d, k + ".npy"), v)
    args = [os.path.join(HERE, "run_sim_2d.bat"),
            "--set-port", "nx", str(nx), "--set-port", "ny", str(ny),
            "--set-port", "dt", str(dt), "--set-port", "substeps", str(substeps),
            "--set-port", "rho_liquid", str(rho_l), "--set-port", "rho_air", str(rho_g),
            "--set-port", "alpha_liquid", str(alpha_l), "--set-port", "alpha_air", str(alpha_g),
            "--set-port", "pos_pattern", (d + "/frame.####").replace("\\", "/"),
            "--set-port", "diag_pattern", (d + "/diag.####").replace("\\", "/"),
            "--set-port", "path_final_positions", (d + "/final.npy").replace("\\", "/")]
    for k in arrs:
        args += ["--set-port", "path_" + k, os.path.join(d, k + ".npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(os.path.join(d, f"frame.{substeps - 1:04d}.npy")):
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-2000:])
        raise SystemExit(f"bifcmd loop failed for {tag}")
    frames = [np.load(os.path.join(d, f"frame.{k:04d}.npy"))[:, :2] for k in range(substeps)]
    diags = np.stack([np.load(os.path.join(d, f"diag.{k:04d}.npy")) for k in range(substeps)])
    return sim.typ.copy(), frames, diags   # typ, per-substep positions, [iters,res,div,speed]


def run_python(nx, ny, mask, substeps, dt, rho_l=1000.0, rho_g=1.0,
               alpha_l=0.97, alpha_g=0.9):
    sim = Sim(nx, ny, rho_l=rho_l, rho_g=rho_g, alpha_l=alpha_l, alpha_g=alpha_g)
    sim.seed(mask)
    sim.calibrate()
    frames = []
    for _ in range(substeps):
        sim.step(dt)
        frames.append(sim.pos.copy())
    return sim.typ.copy(), frames


class Shim:
    def __init__(self, nx, ny, pos, typ):
        self.nx, self.ny, self.pos, self.typ = nx, ny, pos, typ


def render(nx, ny, pos, typ, scale=5):
    img = np.full((nx * scale, ny * scale, 3), 255, np.uint8)
    pix = np.clip((pos * scale).astype(np.int64), 0, [nx * scale - 1, ny * scale - 1])
    for t, colour in ((0, (225, 232, 240)), (1, (25, 80, 190))):
        sel = typ == t
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, nx * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, ny * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1],
                                 outline=(120, 120, 120), width=2)
    return im


def sheet(rows, times_idx, dt, label):
    """rows: list of (row_label, nx, ny, typ, frames)."""
    imgs = [[render(nx, ny, fr[k], typ) for k in times_idx] for _, nx, ny, typ, fr in rows]
    w, h = imgs[0][0].size
    pad, head = 6, 34
    W = (w + pad) * len(times_idx) + pad
    H = (h + head + pad) * len(rows) + pad + 30
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    y = pad
    for (name, *_), row in zip(rows, imgs):
        d.text((pad, y + 4), name, fill=(10, 10, 10), font=FONT)
        y += head
        for i, im in enumerate(row):
            out.paste(im, (pad + i * (w + pad), y))
        y += h + pad
    for i, k in enumerate(times_idx):
        d.text((pad + i * (w + pad) + 6, H - 28), f"t = {(k + 1) * dt:.2f}",
               fill=(90, 90, 90), font=FONT)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"bifrost_{label}.png")
    out.save(path)
    print("wrote", os.path.normpath(path))


def curve_plot(series, title, path):
    w, h, pad = 640, 360, 46
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    allv = np.concatenate([v for _, v in series])
    lo, hi = float(allv.min()), float(allv.max())
    hi = hi if hi > lo else lo + 1
    n = max(len(v) for _, v in series)
    colours = [(200, 60, 40), (30, 90, 200), (40, 160, 60)]
    for ci, (name, v) in enumerate(series):
        pts = [(pad + (w - 2 * pad) * k / (n - 1),
                h - pad - (h - 2 * pad) * (val - lo) / (hi - lo)) for k, val in enumerate(v)]
        d.line(pts, fill=colours[ci % 3], width=3)
        d.text((pad + 8, pad + 20 * ci), name, fill=colours[ci % 3], font=FONT)
    d.rectangle([pad, pad, w - pad, h - pad], outline=(120, 120, 120))
    d.text((pad, 8), title, fill=(10, 10, 10), font=FONT)
    im.save(path)
    print("wrote", os.path.normpath(path))


def dam_break():
    nx, ny, dt, sub = 160, 80, 0.015, 164
    mask = lambda x, y: (x < 40) & (y < 56)
    typ_b, fb, diag = run_bifrost("dam", nx, ny, mask, sub, dt)
    typ_p, fp = run_python(nx, ny, mask, sub, dt)
    front = lambda frames, typ: np.array([np.percentile(f[typ == 1, 0], 99.5) for f in frames])
    cb, cp = front(fb, typ_b), front(fp, typ_p)
    print(f"dam_break: front bifrost {cb[0]:.1f}->{cb.max():.1f}, python {cp[0]:.1f}->{cp.max():.1f}")
    print(f"  max |front difference|: {np.abs(cb - cp).max():.2f} cells of {nx}")
    print(f"  worst post-projection divergence: {diag[:, 2].max():.2e}; max CFL: {diag[:, 3].max() * dt:.2f}")
    curve_plot([("python reference", cp), ("Bifrost", cb)],
               "dam-break front position vs time",
               os.path.join(RESULTS, "bifrost_dam_front.png"))
    sheet([("python reference", nx, ny, typ_p, fp), ("Bifrost, full loop", nx, ny, typ_b, fb)],
          [0, 40, 80, 120, 163], dt, "dam_break")
    ok = np.abs(cb - cp).max() < 8 and cb.max() > 80 and diag[:, 2].max() < 1e-4
    print("dam_break:", "PASS" if ok else "FAIL")
    return ok


def hydrostatic():
    nx = ny = 64
    dt, sub = 0.025, 60
    typ_b, fb, diag = run_bifrost("hydro", nx, ny, lambda x, y: y < 32, sub, dt)
    tail = diag[-10:, 3].max()
    print(f"hydrostatic: max particle speed in final 10 substeps = {tail:.3f} (want small)")
    print(f"  CG iterations mean {diag[:, 0].mean():.0f}, worst divergence {diag[:, 2].max():.2e}")
    ok = tail < 1.0
    print("hydrostatic:", "PASS" if ok else "FAIL")
    return ok


def gap(frames, typ, lo_x=24, hi_x=72):
    out = []
    for f in frames:
        mid = (f[:, 0] > lo_x) & (f[:, 0] < hi_x)
        liq = typ == 1
        pool = f[mid & liq & (f[:, 1] < 28), 1]
        slab = f[mid & liq & (f[:, 1] >= 28), 1]
        if len(pool) == 0 or len(slab) == 0:
            out.append(np.nan)
            continue
        out.append(np.percentile(slab, 2) - np.percentile(pool, 98))
    return np.array(out)


def air_cushion():
    nx, ny, dt, sub = 96, 80, 0.01, 120
    mask = lambda x, y: (y < 20) | ((y > 30) & (y < 44))
    typ_b, fb, diag = run_bifrost("cushion", nx, ny, mask, sub, dt)
    typ_p, fp = run_python(nx, ny, mask, sub, dt)
    gb, gp = gap(fb, typ_b), gap(fp, typ_p)
    print(f"air_cushion: final gap bifrost {gb[-1]:.1f}, python {gp[-1]:.1f} (vacuum would collapse)")
    curve_plot([("python reference", gp), ("Bifrost", gb)],
               "sealed air pocket: pool-to-slab gap vs time",
               os.path.join(RESULTS, "bifrost_cushion_gap.png"))
    sheet([("python reference", nx, ny, typ_p, fp), ("Bifrost, full loop", nx, ny, typ_b, fb)],
          [0, 40, 80, 119], dt, "air_cushion")
    ok = gb[-1] > 4 and abs(gb[-1] - gp[-1]) < 3
    print("air_cushion:", "PASS" if ok else "FAIL")
    return ok


def rayleigh_taylor():
    nx, ny, dt, sub = 64, 192, 0.03, 184
    mask = lambda x, y: y > 96 + 3.0 * np.cos(2 * np.pi * x / 64)
    typ_b, fb, diag = run_bifrost("rt", nx, ny, mask, sub, dt,
                                  rho_l=19.0, rho_g=1.0, alpha_l=0.97, alpha_g=0.97)
    tip = np.array([np.percentile(f[typ_b == 1, 1], 0.5) for f in fb])
    print(f"rayleigh_taylor: heavy tip fell from y={tip[0]:.1f} to y={tip.min():.1f}")
    sheet([("Bifrost, full loop", nx, ny, typ_b, fb)], [0, 60, 120, 183], dt, "rayleigh_taylor")
    ok = tip.min() < tip[0] - 30
    print("rayleigh_taylor:", "PASS" if ok else "FAIL")
    return ok


def smoke():
    typ, fb, diag = run_bifrost("smoke", 16, 16, lambda x, y: y < 8, 4, 0.02)
    print("smoke: ran", len(fb), "substeps;", len(typ), "particles; diag[0] =", diag[0])
    return True


SCENES = {"smoke": smoke, "dam_break": dam_break, "hydrostatic": hydrostatic,
          "air_cushion": air_cushion, "rayleigh_taylor": rayleigh_taylor}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = [which] if which in SCENES else ["dam_break", "hydrostatic", "air_cushion",
                                             "rayleigh_taylor"]
    ok = True
    for n in names:
        print("===", n)
        ok = SCENES[n]() and ok
    print("OVERALL:", "PASS" if ok else "FAIL")
