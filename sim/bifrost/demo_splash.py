"""The splash demo: visual proof of what PF-FLIP buys, and that our Bifrost build
delivers it, on the violent scene where air is loudest.

A tall column plunges fast into a pool (the regime that matters: high impact speed,
trapped air, thin sheets). The same scene runs three ways onto one sheet:

  row 1  plain free-surface FLIP (air is vacuum) - big splash, but no air fingerprints
  row 2  PF-FLIP, python reference              - cushioned fall, air film, pockets
  row 3  PF-FLIP, the full Bifrost loop         - must look like row 2, not row 1

Rows 1 vs 2 show the algorithm. Rows 2 vs 3 visually validate the Bifrost build on a
violent scene, not just the tame dam break. Same seeds, same fixed dt, same instants.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe demo_splash.py   (~15 min)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from validate_loop import run_bifrost, run_python, sheet  # noqa: E402
from validation import SinglePhaseSim  # noqa: E402

NX, NY, DT, SUB = 100, 140, 0.01, 560
TIMES = [0, 300, 360, 440, 559]


def mask(x, y):
    return (y < 40) | ((np.abs(x - 50) < 9) & (y > 88) & (y < 130))


def run_single_phase():
    sim = SinglePhaseSim(NX, NY)
    sim.seed(mask)
    frames = []
    for _ in range(SUB):
        sim.step(DT)
        frames.append(sim.pos.copy())
    return sim.typ.copy(), frames


def main():
    print("1/3 plain free-surface FLIP (python)...")
    typ_s, fs = run_single_phase()
    print("2/3 PF-FLIP reference (python)...")
    typ_p, fp = run_python(NX, NY, mask, SUB, DT)
    print("3/3 PF-FLIP in Bifrost (one headless graph run)...")
    typ_b, fb, diag = run_bifrost("splash", NX, NY, mask, SUB, DT)
    print(f"   bifrost worst divergence {diag[:, 2].max():.2e}, max CFL {diag[:, 3].max() * DT:.2f}")
    sheet([("plain FLIP (no air - free surface)", NX, NY, typ_s, fs),
           ("PF-FLIP, python reference", NX, NY, typ_p, fp),
           ("PF-FLIP in Bifrost, full loop", NX, NY, typ_b, fb)],
          TIMES, DT, "splash_demo")


if __name__ == "__main__":
    main()
