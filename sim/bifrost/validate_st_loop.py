"""ST through the full Bifrost loop graph: the 2D plunge at 8x dt against the python
ST twin from the identical seed, plus a 3D big-step smoke. Peaks must land in the
same band; the step economics print alongside.

Run from sim/bifrost:  ..\..\.venv\Scripts\python.exe validate_st_loop.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from demo3d import run_bifrost3  # noqa: E402

NX, NY, DT, SUB = 120, 160, 0.096, 60


def mask2(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def seed_2d_slab(nx, ny, nz, mask):
    s = Sim(nx, ny)
    s.seed(lambda x, y: mask(x, y, None))
    pos = np.concatenate([s.pos, np.full((len(s.pos), 1), 0.5)], 1)
    return pos, s.typ


def main():
    st_ports = {"st": 1, "st_seed": 7, "escape": 1, "rho0_face": 0.0}
    # rho0 must match the reference calibration
    cal = Sim(NX, NY, st=True)
    cal.seed(mask2)
    cal.calibrate()
    st_ports["rho0_face"] = cal.rho0_face

    d, typ, diags = run_bifrost3(
        "st2d", NX, NY, 1, lambda x, y, z: mask2(x, y), SUB, DT,
        extra_ports=st_ports, seed_fn=lambda nx, ny, nz, m: seed_2d_slab(nx, ny, nz, m))
    peaks_b = [0, 0]
    for k in range(SUB):
        e = np.load(os.path.join(d, f"esc.{k:04d}.npy")) > 0.5
        peaks_b[0] = max(peaks_b[0], int((e & (typ == 1)).sum()))
        peaks_b[1] = max(peaks_b[1], int((e & (typ == 0)).sum()))

    sp = Sim(NX, NY, escape=True, st=True)
    sp.seed(mask2)
    sp.calibrate()
    peaks_p = [0, 0]
    for _ in range(SUB):
        sp.step(DT)
        e = sp.escaped
        peaks_p[0] = max(peaks_p[0], int((e & (sp.typ == 1)).sum()))
        peaks_p[1] = max(peaks_p[1], int((e & (sp.typ == 0)).sum()))

    print(f"2D plunge at 8x dt, {SUB} steps: peak droplets bifrost {peaks_b[0]} vs "
          f"python {peaks_p[0]}; bubbles {peaks_b[1]} vs {peaks_p[1]}")
    print(f"  CG iters mean {diags[:, 0].mean():.1f}; worst divergence {diags[:, 2].max():.2e}; "
          f"max CFL {diags[:, 3].max() * DT:.1f} (by design > 1 now)")
    ok = all(0.4 < (b + 1) / (p + 1) < 2.5 for b, p in zip(peaks_b, peaks_p))
    print("st 2D loop:", "PASS" if ok else "FAIL")

    d3, typ3, diag3 = run_bifrost3("st3d", 32, 24, 16, lambda x, y, z: (x < 10) & (y < 18),
                                   40, 0.12, extra_ports={"st": 1, "st_seed": 7})
    fin = np.load(os.path.join(d3, "final.npy"))
    ok3 = np.isfinite(fin).all() and diag3[:, 2].max() < 1e-3
    print(f"3D big-step smoke: worst div {diag3[:, 2].max():.2e}, iters mean "
          f"{diag3[:, 0].mean():.1f}, finite: {np.isfinite(fin).all()}")
    print("st 3D smoke:", "PASS" if ok3 else "FAIL")
    print("OVERALL:", "PASS" if ok and ok3 else "FAIL")


if __name__ == "__main__":
    main()
