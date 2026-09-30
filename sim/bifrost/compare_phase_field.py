"""Milestone 4: the phase field from Bifrost's STOCK splat node, compared against the
2D reference implementation on identical particles.

The same jittered two-block slab (liquid below, air above) is written to .npy files;
the Bifrost graph (phase_field_probe.json, all stock nodes) reads them, splats the
per-particle phase into a volume with splat_points_into_volume as a weighted average,
samples phi at every reference cell centre, and writes it back as .npy. This script
computes the reference phi on the same particles with two kernels (the tent kernel,
matching the stock kLinearKernel, and the paper's Eq. 6 kernel used by reference2d),
then renders a comparison figure and prints the agreement numbers.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe compare_phase_field.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RESULTS = os.path.join(HERE, "..", "reference2d", "results", "validation")

NX = NY = 64
INTERFACE_Y = 32.0
RADIUS = 1.5          # splat radius in voxels, matches the graph's default
EPS_DEN = 1e-6


def seed(rng):
    """Jittered 2x2-per-cell particles, liquid below INTERFACE_Y, air above —
    the same layout reference2d uses."""
    ii, jj = np.meshgrid(np.arange(NX * 2), np.arange(NY * 2), indexing="ij")
    base = np.stack([ii.ravel(), jj.ravel()], 1) * 0.5 + 0.25
    pos = base + (rng.random(base.shape) - 0.5) * 0.5
    phase = (pos[:, 1] < INTERFACE_Y).astype(np.float32)   # 1 = liquid, 0 = air
    return pos.astype(np.float32), phase


def cell_phi(pos, phase, kernel):
    """Reference weighted-average phase at every cell centre, 2D distances (particles
    and the sampled voxel layer share one z plane, so 3D and 2D distances agree)."""
    num = np.zeros((NX, NY))
    den = np.zeros((NX, NY))
    r = RADIUS
    span = int(np.ceil(r + 0.5))
    centers = np.arange(NX) + 0.5
    for p, ph in zip(pos, phase):
        i0, j0 = int(p[0]), int(p[1])
        for i in range(max(0, i0 - span), min(NX, i0 + span + 1)):
            for j in range(max(0, j0 - span), min(NY, j0 + span + 1)):
                d = np.hypot(centers[i] - p[0], centers[j] - p[1])
                if d < r:
                    w = (1.0 - d / r) if kernel == "tent" else (1.0 - (d / r) ** 2) ** 3
                    num[i, j] += w * ph
                    den[i, j] += w
    return num / (den + EPS_DEN)


def run_bifrost(pos, phase):
    os.makedirs(OUT, exist_ok=True)
    p3 = np.concatenate([pos, np.full((len(pos), 1), 0.5, np.float32)], 1)
    centers = np.stack(np.meshgrid(np.arange(NX) + 0.5, np.arange(NY) + 0.5,
                                   indexing="ij"), -1).reshape(-1, 2)
    probes = np.concatenate([centers, np.full((len(centers), 1), 0.5)], 1)
    np.save(os.path.join(OUT, "particles.npy"), p3.astype(np.float32))
    np.save(os.path.join(OUT, "phase.npy"), phase.astype(np.float32))
    np.save(os.path.join(OUT, "probes.npy"), probes.astype(np.float32))
    args = [os.path.join(HERE, "run_phase_field_probe.bat")]
    for name, fn in (("particles_npy", "particles.npy"), ("phase_npy", "phase.npy"),
                     ("probes_npy", "probes.npy"), ("phi_out_npy", "phi.npy")):
        args += ["--set-port", name, os.path.join(OUT, fn).replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        raise SystemExit(f"bifcmd failed with exit code {res.returncode}")
    phi = np.load(os.path.join(OUT, "phi.npy"))
    return phi.reshape(NX, NY)


def figure(ref_tent, ref_eq6, bif):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    scale = 6

    def panel(field, lo=0.0, hi=1.0):
        t = np.clip((field - lo) / (hi - lo), 0, 1)
        rgb = np.stack([225 - 200 * t, 232 - 152 * t, 240 - 50 * t], -1).astype(np.uint8)
        img = Image.fromarray(np.rot90(rgb)).resize((NX * scale, NY * scale), Image.NEAREST)
        ImageDraw.Draw(img).rectangle([0, 0, img.width - 1, img.height - 1],
                                      outline=(120, 120, 120), width=2)
        return img

    panels = [("2D reference (tent kernel)", panel(ref_tent)),
              ("Bifrost stock splat node", panel(bif)),
              ("abs difference x10", panel(np.abs(bif - ref_tent) * 10))]
    pad, head = 8, 30
    w = h = NX * scale
    sheet = Image.new("RGB", ((w + pad) * 3 + pad, h + head + 2 * pad), "white")
    d = ImageDraw.Draw(sheet)
    for k, (name, img) in enumerate(panels):
        d.text((pad + k * (w + pad), pad), name, fill=(10, 10, 10), font=font)
        sheet.paste(img, (pad + k * (w + pad), pad + head))
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "phase_field_bifrost.png")
    sheet.save(path)
    print("wrote", os.path.normpath(path))


def main():
    rng = np.random.default_rng(0)
    pos, phase = seed(rng)
    bif = run_bifrost(pos, phase)
    missed = int((bif < -0.5).sum())
    print(f"probes outside the splatted volume: {missed} of {bif.size}")
    ref_tent = cell_phi(pos, phase, "tent")
    ref_eq6 = cell_phi(pos, phase, "eq6")

    inner = np.s_[2:-2, 2:-2]
    diff = np.abs(bif - ref_tent)[inner]
    print(f"max |bifrost - reference(tent)| in the interior: {diff.max():.4f}")
    print(f"mean |bifrost - reference(tent)| in the interior: {diff.mean():.5f}")

    def midline(field):
        col = field[NX // 2]
        j = np.argmax(col < 0.5)
        a, b = col[j - 1], col[j]
        return (j - 1) + (a - 0.5) / (a - b)
    print(f"interface (phi=0.5) at x=centre: reference(tent) y={midline(ref_tent):.3f}, "
          f"bifrost y={midline(bif):.3f}, reference(Eq.6) y={midline(ref_eq6):.3f}")
    figure(ref_tent, ref_eq6, bif)


if __name__ == "__main__":
    main()
