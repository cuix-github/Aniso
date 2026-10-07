"""Phase A of the milestone-6 push: the blend-viscosity law at production timesteps,
and the escape+ST combination check.

Eq. 13 says effective viscosity ~ (1-alpha)*dx^2/(C*dt): bigger steps mean less
damping at the same alpha. This measures the decay of a sinusoidal shear wave
(exact analytic decay exp(-nu k^2 t)) across (dt, alpha, st) combinations to
(a) confirm the 1/dt scaling survives ST sampling, and (b) verify the practical
knob: moving to 8x dt while scaling (1-alpha) by 8 restores the same damping.

Also runs the plunge with escape AND st together, which no gate has covered yet.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe st_viscosity.py
"""
import numpy as np

from pfflip2d import Sim

NX = NY = 64
U0, MODE = 2.0, 2


def shear_decay(dt, alpha, st, total_t=6.0):
    """Effective kinematic viscosity from the decay of u = U0 sin(k y)."""
    sim = Sim(NX, NY, alpha_l=alpha, alpha_g=alpha, g=0.0, st=st)
    sim.seed(lambda x, y: np.ones_like(x, dtype=bool))   # all liquid, no interface
    sim.calibrate()
    k = 2 * np.pi * MODE / NY
    sim.vel[:, 0] = U0 * np.sin(k * sim.pos[:, 1])
    amps, times = [], []
    t = 0.0
    while t < total_t - 1e-9:
        sim.step(dt)
        t += dt
        proj = sim.vel[:, 0] * np.sin(k * sim.pos[:, 1])
        amps.append(2.0 * proj.mean())
        times.append(t)
    amps, times = np.array(amps), np.array(times)
    ok = amps > 0.05 * U0
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(times[ok], np.log(amps[ok]), 1)[0]
    return -slope / k ** 2


def main():
    print("=== shear-decay viscosity: nu_eff and the Eq. 13 constant C = (1-a)dx^2/(nu dt)")
    rows = [("small dt, a=0.97, plain", 0.012, 0.97, False),
            ("small dt, a=0.97, ST", 0.012, 0.97, True),
            ("8x dt,    a=0.97, plain", 0.096, 0.97, False),
            ("8x dt,    a=0.97, ST", 0.096, 0.97, True),
            ("8x dt,    a=0.76, ST (adjusted: (1-a) x8)", 0.096, 0.76, True)]
    results = {}
    for label, dt, a, st in rows:
        nu = shear_decay(dt, a, st)
        C = (1 - a) / (nu * dt) if nu and not np.isnan(nu) and nu > 0 else np.nan
        results[label] = nu
        print(f"  {label:46s} nu_eff = {nu:8.4f}   C = {C:6.1f}")
    small = results["small dt, a=0.97, plain"]
    adj = results["8x dt,    a=0.76, ST (adjusted: (1-a) x8)"]
    big_same = results["8x dt,    a=0.97, ST"]
    print(f"  8x dt at same alpha keeps {big_same / small:.2f}x of the small-dt damping "
          f"(Eq. 13 predicts ~1/8 = 0.125)")
    print(f"  adjusted alpha restores {adj / small:.2f}x (want ~1.0)")
    ok_v = 0.05 < big_same / small < 0.4 and 0.5 < adj / small < 2.0
    print("viscosity:", "PASS" if ok_v else "FAIL")

    print("=== escape + ST together: the plunge, droplets and bubbles must still appear")
    s = Sim(120, 160, escape=True, st=True)
    s.seed(lambda x, y: (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130)))
    s.calibrate()
    peak_d = peak_b = 0
    t = 0.0
    while t < 6.0:
        s.step(0.096)
        t += 0.096
        e = s.escaped
        peak_d = max(peak_d, int((e & (s.typ == 1)).sum()))
        peak_b = max(peak_b, int((e & (s.typ == 0)).sum()))
    print(f"  peak droplets {peak_d}, peak bubbles {peak_b} "
          f"(small-dt escape-only run peaked at ~102 / ~140)")
    ok_e = peak_d > 20 and peak_b > 20
    print("escape+st:", "PASS" if ok_e else "FAIL")
    print("OVERALL:", "PASS" if ok_v and ok_e else "FAIL")


if __name__ == "__main__":
    main()
