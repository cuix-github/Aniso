"""The validation gate: does the paper's core story hold up in the simplest form that could
falsify it? Four experiments against the claims of PF-FLIP (see VALIDATION.md for verdicts).

  V1 air_pocket   Two-phase vs a plain single-phase free-surface FLIP on a slab of water
                  falling onto a pool: only the two-phase solver can cushion the trapped air.
  V2 transport    Particle-carried phase vs a grid-advected phase scalar in the reversing
                  vortex (Rider-Kothe): Lagrangian transport should not diffuse the interface.
  V3 viscosity    The FLIP blend as viscosity (paper Eq. 13): measure a shear wave's decay
                  and compare against the equation's prediction.
  V4 mixing       Fraction of particles on the wrong side of the interface over a dam break:
                  the paper claims no bounce-back heuristics are needed.

  python validation.py [air_pocket|transport|viscosity|mixing|all]

Outputs to results/validation/. Needs numpy, Pillow, matplotlib.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pfflip2d import Sim, draw, save_strip

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "validation")


# ---------------------------------------------------------------- single-phase baseline

class SinglePhaseSim(Sim):
    """Classic free-surface FLIP: liquid particles only, air is vacuum with p = 0."""

    def seed(self, liquid_mask, **kw):
        super().seed(liquid_mask, **kw)
        keep = self.typ == 1
        self.pos, self.vel, self.typ = self.pos[keep], self.vel[keep], self.typ[keep]

    def fluid_cells(self):
        counts = np.zeros((self.nx, self.ny))
        c = np.clip(self.pos.astype(np.int64), 0, [self.nx - 1, self.ny - 1])
        np.add.at(counts, (c[:, 0], c[:, 1]), 1)
        return counts > 0

    def project(self, dt, tol=1e-6, max_iter=4000):
        fluid = self.fluid_cells()
        beta = 1.0 / self.rho_l
        bu = np.full(self.u.shape, beta)
        bv = np.full(self.v.shape, beta)
        bu[0, :] = bu[-1, :] = 0.0
        bv[:, 0] = bv[:, -1] = 0.0
        # A face is active if at least one side is fluid; ghost pressure 0 in air.
        fu = np.zeros(self.u.shape, bool)
        fu[1:-1, :] = fluid[1:, :] | fluid[:-1, :]
        fv = np.zeros(self.v.shape, bool)
        fv[:, 1:-1] = fluid[:, 1:] | fluid[:, :-1]
        bu *= fu
        bv *= fv

        div = (self.u[1:, :] - self.u[:-1, :]) + (self.v[:, 1:] - self.v[:, :-1])
        b = np.where(fluid, -div / dt, 0.0)

        def A(p):
            P = np.where(fluid, p, 0.0)
            gpu = np.zeros_like(self.u)
            gpv = np.zeros_like(self.v)
            gpu[1:-1, :] = (P[1:, :] - P[:-1, :]) * bu[1:-1, :]
            gpv[:, 1:-1] = (P[:, 1:] - P[:, :-1]) * bv[:, 1:-1]
            out = -((gpu[1:, :] - gpu[:-1, :]) + (gpv[:, 1:] - gpv[:, :-1]))
            return np.where(fluid, out, 0.0)

        diag = bu[1:, :] + bu[:-1, :] + bv[:, 1:] + bv[:, :-1]
        inv_diag = np.where(fluid & (diag > 0), 1.0 / np.maximum(diag, 1e-300), 0.0)
        p = np.zeros((self.nx, self.ny))
        r = b - A(p)
        z = inv_diag * r
        d = z.copy()
        rz = float((r * z).sum())
        r0 = np.sqrt(float((r * r).sum())) or 1.0
        for it in range(max_iter):
            if np.sqrt(float((r * r).sum())) / r0 < tol:
                break
            q = A(d)
            dq = float((d * q).sum())
            if dq == 0:
                break
            a = rz / dq
            p += a * d
            r -= a * q
            z = inv_diag * r
            rz_new = float((r * z).sum())
            d = z + (rz_new / rz) * d
            rz = rz_new
        self.iters = it

        P = np.where(fluid, p, 0.0)
        self.u[1:-1, :] -= dt * bu[1:-1, :] * (P[1:, :] - P[:-1, :])
        self.v[:, 1:-1] -= dt * bv[:, 1:-1] * (P[:, 1:] - P[:, :-1])
        self._enforce_walls()
        self._extrapolate(fu, fv)

    def _extrapolate(self, fu, fv, passes=4):
        """Spread face velocities a few cells into the vacuum so surface particles advect."""
        for grid, known in ((self.u, fu.copy()), (self.v, fv.copy())):
            for _ in range(passes):
                s = np.zeros_like(grid)
                w = np.zeros_like(grid)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    src = np.roll(grid * known, (dx, dy), (0, 1))
                    kk = np.roll(known.astype(float), (dx, dy), (0, 1))
                    s += src
                    w += kk
                new = w > 0
                fill = ~known & new
                grid[fill] = (s / np.maximum(w, 1e-300))[fill]
                known |= new


# ---------------------------------------------------------------- V1: the trapped pocket

def air_pocket():
    def scene(sim):
        sim.seed(lambda x, y: (y < 20) | ((y > 30) & (y < 44)))
        sim.calibrate()

    def slab_mask(s):
        return (s.typ == 1) & (s.pos[:, 1] > 24)

    # The slab seals the box wall to wall, so the trapped air cannot escape sideways:
    # incompressible air must hold the slab up. The observable is the air gap between the
    # pool's surface and the slab's underside, in the central half of the box.
    results = {}
    for name, cls in (("two_phase", Sim), ("single_phase", SinglePhaseSim)):
        sim = cls(96, 80)
        scene(sim)
        liquid_y0 = sim.pos[:, 1].copy()
        slab_ids = np.where((sim.typ == 1) & (liquid_y0 > 24))[0]
        pool_ids = np.where((sim.typ == 1) & (liquid_y0 < 24))[0]
        gaps, shots = [], []

        def cb(f, s):
            mid_slab = s.pos[slab_ids]
            mid_pool = s.pos[pool_ids]
            in_mid_s = (mid_slab[:, 0] > 24) & (mid_slab[:, 0] < 72)
            in_mid_p = (mid_pool[:, 0] > 24) & (mid_pool[:, 0] < 72)
            bottom = float(np.percentile(mid_slab[in_mid_s, 1], 2.0))
            top = float(np.percentile(mid_pool[in_mid_p, 1], 98.0))
            gaps.append(bottom - top)
            if f % 5 == 0:
                shots.append(draw(s))

        sim.run(31, 0.04, on_frame=cb)
        save_strip(shots, os.path.join(OUT, f"air_pocket_{name}.png"), cols=7)
        results[name] = np.array(gaps)
        print(f"  {name}: air gap {gaps[0]:.1f} -> {gaps[-1]:.1f} cells")

    t = np.arange(31) * 0.04
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for name, gap in results.items():
        ax.plot(t, gap, label=name)
    ax.set(title="air gap between pool and falling slab (sealed box)", xlabel="t",
           ylabel="gap [cells]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "air_pocket_curves.png"), dpi=110)
    print("wrote", os.path.join(OUT, "air_pocket_curves.png"))

    tp, sp = results["two_phase"], results["single_phase"]
    print(f"  final gap: two-phase {tp[-1]:.1f}  single-phase {sp[-1]:.1f}")
    return tp[-1] > 4.0 and tp[-1] > 2.5 * max(sp[-1], 0.1)


# ---------------------------------------------------------------- V2: interface transport

def transport():
    L, T, steps = 64, 6.0, 720
    dt = T / steps

    def vel(p, t):
        mod = np.cos(np.pi * t / T)
        x, y = p[:, 0] / L, p[:, 1] / L
        u = -np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y) * mod
        v = np.sin(2 * np.pi * x) * np.sin(np.pi * y) ** 2 * mod
        return np.stack([u, v], -1) * L / 4.0

    def disc(x, y):
        return (x - 32) ** 2 + (y - 44) ** 2 < 12 ** 2

    # Particle-carried phase: advect markers through the analytic field (RK3).
    sim = Sim(L, L)
    sim.seed(disc)
    sim.calibrate()

    def phase_cells(s):
        counts = np.zeros((L, L))
        liq = np.zeros((L, L))
        c = np.clip(s.pos.astype(np.int64), 0, L - 1)
        np.add.at(counts, (c[:, 0], c[:, 1]), 1)
        np.add.at(liq, (c[:, 0], c[:, 1]), (s.typ == 1).astype(float))
        return np.where(counts > 0, liq / np.maximum(counts, 1), 0.0)

    phi0_p = phase_cells(sim)
    snaps_p = {}
    for s_i in range(steps):
        t = s_i * dt
        k1 = vel(sim.pos, t)
        k2 = vel(sim.pos + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vel(sim.pos - dt * k1 + 2 * dt * k2, t + dt)
        sim.pos = sim.pos + dt * (k1 + 4 * k2 + k3) / 6.0
        if s_i in (steps // 2 - 1, steps - 1):
            snaps_p[s_i] = phase_cells(sim)
    err_p = float(np.abs(snaps_p[steps - 1] - phi0_p).sum())

    # Grid-advected phase: the same field, semi-Lagrangian scalar advection.
    xs, ys = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5, indexing="ij")
    centres = np.stack([xs.ravel(), ys.ravel()], -1)
    phi_g = disc(centres[:, 0], centres[:, 1]).astype(float).reshape(L, L)
    phi0_g = phi_g.copy()

    def sample(phi, pts):
        q = np.clip(pts - 0.5, 0, L - 1.001)
        b = np.floor(q).astype(np.int64)
        f = q - b
        b1 = np.minimum(b + 1, L - 1)
        return (phi[b[:, 0], b[:, 1]] * (1 - f[:, 0]) * (1 - f[:, 1]) +
                phi[b1[:, 0], b[:, 1]] * f[:, 0] * (1 - f[:, 1]) +
                phi[b[:, 0], b1[:, 1]] * (1 - f[:, 0]) * f[:, 1] +
                phi[b1[:, 0], b1[:, 1]] * f[:, 0] * f[:, 1])

    snaps_g = {}
    for s_i in range(steps):
        t = s_i * dt
        back1 = centres - dt * vel(centres, t)                       # RK2 backtrace
        back = centres - dt * vel(0.5 * (centres + back1), t + 0.5 * dt)
        phi_g = sample(phi_g, back).reshape(L, L)
        if s_i in (steps // 2 - 1, steps - 1):
            snaps_g[s_i] = phi_g.copy()
    err_g = float(np.abs(snaps_g[steps - 1] - phi0_g).sum())

    def to_img(phi):
        a = np.clip(phi, 0, 1)
        img = (np.stack([1 - a, 1 - 0.6 * a, np.ones_like(a)], -1) * 255).astype(np.uint8)
        return Image.fromarray(np.rot90(img)).resize((L * 4, L * 4), Image.NEAREST)

    row = [to_img(phi0_p), to_img(snaps_p[steps // 2 - 1]), to_img(snaps_p[steps - 1]),
           to_img(phi0_g), to_img(snaps_g[steps // 2 - 1]), to_img(snaps_g[steps - 1])]
    save_strip(row, os.path.join(OUT, "transport.png"), cols=3)
    print(f"  L1 shape error after reversal: particles {err_p:.1f}  grid scalar {err_g:.1f} "
          f"(ratio {err_g / max(err_p, 1e-9):.1f}x)")
    return err_p < 0.5 * err_g


# ---------------------------------------------------------------- V3: Eq. 13 viscosity

def viscosity():
    H, W, dt, U = 32, 64, 0.05, 0.5
    k = 2 * np.pi / H
    alphas = [0.9, 0.7, 0.5]
    measured, predicted = [], []
    for a in alphas:
        sim = Sim(W, H, rho_l=1000.0, rho_g=1.0, alpha_l=a, alpha_g=a, g=0.0)
        sim.seed(lambda x, y: np.ones_like(x, dtype=bool))       # all liquid
        sim.calibrate()
        sim.vel[:, 0] = U * np.sin(k * sim.pos[:, 1])
        amps, times = [], []
        steps, t = 2000, 0.0
        for s_i in range(steps):
            sim.step(dt)                                          # fixed dt: Eq. 13 depends on it
            t += dt
            if s_i % 50 == 0:
                mid = (sim.pos[:, 0] > 24) & (sim.pos[:, 0] < 40)
                ny, sy = np.histogram(sim.pos[mid, 1], bins=H, range=(0, H),
                                      weights=sim.vel[mid, 0])
                cnt, _ = np.histogram(sim.pos[mid, 1], bins=H, range=(0, H))
                prof = ny / np.maximum(cnt, 1)
                amps.append(float(np.abs(prof).max()))
                times.append(t)
        times, amps = np.array(times), np.array(amps)
        sel = (times > 15) & (amps > 1e-4)
        slope = np.polyfit(times[sel], np.log(amps[sel]), 1)[0]
        nu_meas = -slope / k ** 2
        nu_pred = (1 - a) / (6 * dt)
        measured.append(nu_meas)
        predicted.append(nu_pred)
        print(f"  alpha={a}: nu measured {nu_meas:.3f}  predicted (Eq.13) {nu_pred:.3f}")

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(predicted, measured, "o-", label="measured")
    lim = [0, max(predicted) * 1.2]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    ax.set(xlabel="predicted nu (Eq. 13)", ylabel="measured nu", title="FLIP blend as viscosity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "viscosity.png"), dpi=110)
    print("wrote", os.path.join(OUT, "viscosity.png"))
    # The claim worth testing is the LAW's form: nu proportional to (1-alpha)/dt. The 1/6
    # constant in Eq. 13 assumes the reference implementation's transfer kernels; ours differ,
    # so we check linearity through the origin and report the calibrated constant.
    ratio = np.array(measured) / np.array(predicted)
    spread = float(ratio.max() - ratio.min()) / float(ratio.mean())
    print(f"  proportionality constant vs Eq. 13: {ratio.mean():.3f} (spread {spread * 100:.0f}%)")
    return spread < 0.35


# ---------------------------------------------------------------- V4: mixing fraction

def mixing():
    sim = Sim(160, 80)
    sim.seed(lambda x, y: (x < 40) & (y < 56))
    sim.calibrate()
    fracs = []

    def cell_phase(s):
        m = s.masses()
        base = np.floor(s.pos - 0.5 + 0.5).astype(np.int64)
        frac = (s.pos - 0.5) - base
        (raw,) = s._splat((s.nx, s.ny), (base, frac), [m])
        return s.phase(raw)

    def cb(f, s):
        phi = cell_phase(s)
        c = np.clip(s.pos.astype(np.int64), 0, [s.nx - 1, s.ny - 1])
        at = phi[c[:, 0], c[:, 1]]
        wrong = ((s.typ == 1) & (at < 0.3)) | ((s.typ == 0) & (at > 0.7))
        fracs.append(float(wrong.mean()))

    sim.run(41, 0.06, on_frame=cb)
    t = np.arange(len(fracs)) * 0.06
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(t, np.array(fracs) * 100)
    ax.set(xlabel="t", ylabel="% particles beyond the paper's escape threshold",
           title="wrong-side fraction, dam break (no bounce-back)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mixing.png"), dpi=110)
    print(f"  wrong-side fraction: start {fracs[0] * 100:.2f}%  end {fracs[-1] * 100:.2f}%  "
          f"max {max(fracs) * 100:.2f}%")
    print("wrote", os.path.join(OUT, "mixing.png"))
    return max(fracs) < 0.05


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    verdicts = {}
    for name, fn in (("air_pocket", air_pocket), ("transport", transport),
                     ("viscosity", viscosity), ("mixing", mixing)):
        if which in (name, "all"):
            print("===", name)
            verdicts[name] = fn()
    print("total %.1f s" % (time.time() - t0))
    for k, v in verdicts.items():
        print(f"  {k}: {'SUPPORTS THE CLAIM' if v else 'CHECK — claim not clearly supported'}")
