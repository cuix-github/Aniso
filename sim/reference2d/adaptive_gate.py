"""Phase D1 gate: adaptive particles in the 2D reference.

  viscosity   all-coarse with the per-size blend correction must match all-fine
              shear decay (the Section 5.3 equalization, calibrated by measurement)
  cushion     the sealed-pocket anchor with adaptation on: the gap must hold
  dam_break   front curve vs the uniform reference, particle-count compression in
              deep air, and exact conservation of represented volume (sum scale^2)

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe adaptive_gate.py
"""
import numpy as np

from pfflip2d import Sim


def shear(all_coarse):
    sim = Sim(64, 64, alpha_l=0.9, alpha_g=0.9, g=0.0, adapt=True)
    sim.seed(lambda x, y: np.ones_like(x, dtype=bool))
    sim.calibrate()
    if all_coarse:
        keep = (sim.pos[:, 0] % 2 < 1) & (sim.pos[:, 1] % 2 < 1)
        sim.pos = sim.pos[keep] + 0.5
        sim.typ = sim.typ[keep]
        sim.vel = sim.vel[keep]
        sim.tau, sim.xi = sim.tau[keep], sim.xi[keep]
        sim.scale = np.full(len(sim.pos), 2.0)
    k = 2 * np.pi * 2 / 64
    sim.vel[:, 0] = 2.0 * np.sin(k * sim.pos[:, 1])
    amps, times, t = [], [], 0.0
    sim._adapt_off = True
    orig = sim._adapt_particles
    sim._adapt_particles = lambda: None     # freeze sizes for the measurement
    while t < 5.0:
        sim.step(0.05)
        t += 0.05
        amps.append(2.0 * (sim.vel[:, 0] * np.sin(k * sim.pos[:, 1])).mean())
        times.append(t)
    amps, times = np.array(amps), np.array(times)
    ok = amps > 0.1
    nu = -np.polyfit(times[ok], np.log(amps[ok]), 1)[0] / k ** 2
    return nu


def main():
    print("=== per-size blend: all-coarse vs all-fine shear decay")
    nf, nc = shear(False), shear(True)
    print(f"  nu fine {nf:.4f}, nu coarse(corrected) {nc:.4f}, ratio {nc / nf:.2f}")
    ok_v = 0.6 < nc / nf < 1.6
    print("viscosity:", "PASS" if ok_v else "FAIL")

    print("=== air cushion with adaptation on")
    sim = Sim(96, 80, adapt=True)
    sim.seed(lambda x, y: (y < 20) | ((y > 30) & (y < 44)))
    sim.calibrate()
    for _ in range(120):
        sim.step(0.01)
    liq = sim.typ == 1
    mid = (sim.pos[:, 0] > 24) & (sim.pos[:, 0] < 72)
    pool = sim.pos[mid & liq & (sim.pos[:, 1] < 28), 1]
    slab = sim.pos[mid & liq & (sim.pos[:, 1] >= 28), 1]
    gap = np.percentile(slab, 2) - np.percentile(pool, 98)
    print(f"  final gap {gap:.1f} (trusted 10.6; vacuum ~3)")
    ok_c = gap > 4
    print("cushion:", "PASS" if ok_c else "FAIL")

    print("=== dam break: adaptive vs uniform")
    runs = {}
    for adapt in (False, True):
        s = Sim(160, 80, adapt=adapt)
        s.seed(lambda x, y: (x < 40) & (y < 56))
        s.calibrate()
        n0, v0 = len(s.pos), float((s.scale ** 2).sum()) if adapt else len(s.pos)
        fronts, counts = [], []
        for _ in range(200):
            s.step(0.012)
            fronts.append(np.percentile(s.pos[s.typ == 1, 0], 99.5))
            counts.append(len(s.pos))
        runs[adapt] = (np.array(fronts), counts, n0,
                       float((s.scale ** 2).sum()) if adapt else len(s.pos), v0)
    fu, fa = runs[False][0], runs[True][0]
    dmax = np.abs(fu - fa).max()
    comp = 1 - min(runs[True][1]) / runs[True][2]
    vol_err = abs(runs[True][3] / runs[True][4] - 1)
    print(f"  max |front diff| {dmax:.1f} cells of 160; peak particle saving "
          f"{100 * comp:.0f} percent; represented-volume drift {vol_err:.2e}")
    ok_d = dmax < 10 and comp > 0.15 and vol_err < 1e-9
    print("dam_break:", "PASS" if ok_d else "FAIL")
    print("OVERALL:", "PASS" if ok_v and ok_c and ok_d else "FAIL")


if __name__ == "__main__":
    main()
