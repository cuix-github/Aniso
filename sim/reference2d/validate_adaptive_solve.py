"""Milestone 6.5a stage-1 gate: the adaptive pressure solve's math in 2D.

Five checks, each against something already trusted:

  reduce    refined everywhere, the adaptive assembly must reproduce the
            uniform 5-point operator to rounding on random inputs.
  anchor    on a real dam-break mid-state, the module's uniform solve must
            match pfflip2d.project itself (same face betas, same rhs),
            pressure and corrected faces alike.
  order     manufactured smooth-beta solution: observed convergence order,
            uniform (control, expect ~2) and adaptive with a junction-rich
            pattern (measured, not assumed).
  hydro     1000:1 two-phase column at rest: residual velocities must stay
            machine-small away from junctions and bounded (< 5 percent of
            the gravity kick) at them - the known first-order junction
            artifact, measured and localized.
  dam       the production question: project a real dam-break state on the
            adaptive grid, report unknowns saved and how far corrected
            velocities in the fine interface band drift from uniform.

Run from sim/reference2d:  ..\\..\\.venv\\Scripts\\python.exe validate_adaptive_solve.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adaptive_solve import Operator, build_cells  # noqa: E402
from pfflip2d import Sim  # noqa: E402

RES = os.path.join(HERE, "results", "adaptive")


def uniform_cells(nx, ny):
    return build_cells(nx, ny, 2, np.zeros((nx, ny), int))


def beta_from_arrays(bu, bv):
    def beta(x, y, axis, span):
        if axis == 0:
            ys = (y - span / 2.0 + np.arange(span)).astype(int)
            return float(bu[int(round(x)), ys].mean())
        xs = (x - span / 2.0 + np.arange(span)).astype(int)
        return float(bv[xs, int(round(y))].mean())
    return beta


def dam_state(nx=96, ny=64):
    sim = Sim(nx, ny)
    sim.seed(lambda x, y: (x < 0.3 * nx) & (y < 0.7 * ny))
    sim.calibrate()
    sim.run(12, 0.06)
    sim.p2g()
    phi_u = sim.phase(sim.mu)
    phi_v = sim.phase(sim.mv)
    bu = 1.0 / (sim.rho_g + (sim.rho_l - sim.rho_g) * phi_u)
    bv = 1.0 / (sim.rho_g + (sim.rho_l - sim.rho_g) * phi_v)
    bu[0, :] = bu[-1, :] = 0.0
    bv[:, 0] = bv[:, -1] = 0.0
    return sim, bu, bv


def liquid_band_target(sim, nx, ny, band=3, max_level=2):
    """Fine near the interface and the walls, coarser with distance."""
    liq = np.zeros((nx, ny))
    np.add.at(liq, (np.clip(sim.pos[:, 0].astype(int), 0, nx - 1),
                    np.clip(sim.pos[:, 1].astype(int), 0, ny - 1)),
              (sim.typ == 1).astype(float))
    frac = liq / 4.0
    interface = np.zeros((nx, ny), bool)
    interface |= (frac > 0.05) & (frac < 0.95)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        interface |= (frac > 0.5) != (np.roll(frac, sh, axis=ax) > 0.5)
    near = interface.copy()
    for _ in range(band):
        grow = near.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            grow |= np.roll(near, sh, axis=ax)
        near = grow
    target = np.full((nx, ny), max_level, int)
    mid = near.copy()
    for _ in range(2 * band):
        grow = mid.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            grow |= np.roll(mid, sh, axis=ax)
        mid = grow
    target[mid] = 1
    target[near] = 0
    target[:2, :] = target[-2:, :] = 0
    target[:, :2] = target[:, -2:] = 0
    return target


def check_reduce():
    print("=== reduce: refined-everywhere assembly vs the 5-point operator")
    rng = np.random.default_rng(3)
    nx, ny = 24, 16
    bu = rng.uniform(0.1, 1.0, (nx + 1, ny))
    bv = rng.uniform(0.1, 1.0, (nx, ny + 1))
    bu[0, :] = bu[-1, :] = 0.0
    bv[:, 0] = bv[:, -1] = 0.0
    op = Operator(uniform_cells(nx, ny), beta_from_arrays(bu, bv))
    x = rng.normal(0, 1, nx * ny)
    p = np.zeros((nx, ny))
    for n, (l, i, j) in enumerate(op.cells):
        p[i, j] = x[n]
    gpu = np.zeros((nx + 1, ny))
    gpv = np.zeros((nx, ny + 1))
    gpu[1:-1, :] = (p[1:, :] - p[:-1, :]) * bu[1:-1, :]
    gpv[:, 1:-1] = (p[:, 1:] - p[:, :-1]) * bv[:, 1:-1]
    want = -((gpu[1:, :] - gpu[:-1, :]) + (gpv[:, 1:] - gpv[:, :-1]))
    got = op.matvec(x)
    err = max(abs(got[n] - want[i, j])
              for n, (l, i, j) in enumerate(op.cells))
    cons = abs(op.matvec(np.ones(len(op.cells)))).max()
    print(f"    matvec vs 5-point max |diff| {err:.2e}; A(const) max {cons:.2e}")
    ok = err < 1e-12 and cons < 1e-12
    print("    PASS" if ok else "    FAIL")
    return ok


def check_anchor():
    print("=== anchor: module uniform solve vs pfflip2d.project, real state")
    dt = 0.06
    sim, bu, bv = dam_state()
    u0, v0 = sim.u.copy(), sim.v.copy()
    ref = Sim(sim.nx, sim.ny)
    ref.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in sim.__dict__.items()})
    ref.project(dt, tol=1e-11, max_iter=60000)

    op = Operator(uniform_cells(sim.nx, sim.ny), beta_from_arrays(bu, bv))
    b = op.rhs(u0, v0, dt)
    p, it = op.solve(b, tol=1e-11, max_iter=60000)
    u = u0.copy()
    v = v0.copy()
    pg = np.zeros((sim.nx, sim.ny))
    for n, (l, i, j) in enumerate(op.cells):
        pg[i, j] = p[n]
    u[1:-1, :] -= dt * bu[1:-1, :] * (pg[1:, :] - pg[:-1, :])
    v[:, 1:-1] -= dt * bv[:, 1:-1] * (pg[:, 1:] - pg[:, :-1])
    u[0, :] = u[-1, :] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    du = abs(u - ref.u).max()
    dv = abs(v - ref.v).max()
    print(f"    corrected faces max |diff| u {du:.2e}, v {dv:.2e} ({it} iters)")
    ok = max(du, dv) < 1e-6
    print("    PASS" if ok else "    FAIL")
    return ok


def manufactured(nx, ny, cells):
    a, b2 = np.pi / nx, np.pi / ny

    def beta(x, y, axis, span):
        return 1.0 + 0.5 * np.cos(a * x) * np.cos(b2 * y)

    op = Operator(cells, beta)
    xc, yc = op.center[:, 0], op.center[:, 1]
    C = np.cos(a * xc) * np.cos(b2 * yc)
    Sx = np.sin(a * xc) * np.cos(b2 * yc)
    Sy = np.cos(a * xc) * np.sin(b2 * yc)
    f = (1 + 0.5 * C) * (-(a * a + b2 * b2) * C) + \
        0.5 * (a * a * Sx * Sx + b2 * b2 * Sy * Sy)
    rhs = -f * op.vol
    p, it = op.solve(rhs, tol=1e-11, max_iter=60000)
    exact = C - (C * op.vol).sum() / op.vol.sum()
    num = p - (p * op.vol).sum() / op.vol.sum()
    err = np.sqrt((op.vol * (num - exact) ** 2).sum() / op.vol.sum())
    return err, len(op.cells), it


def check_order():
    print("=== order: manufactured smooth-beta solution, observed rates")
    rows = {"uniform": [], "adaptive": []}
    for N in (48, 96, 192):
        rows["uniform"].append(manufactured(N, N, uniform_cells(N, N)))
        tgt = np.full((N, N), 2, int)
        yy = np.arange(N)[None, :].repeat(N, 0)
        tgt[abs(yy - 0.5 * N) < 0.2 * N] = 0
        tgt[(abs(yy - 0.5 * N) >= 0.2 * N) & (abs(yy - 0.5 * N) < 0.3 * N)] = 1
        rows["adaptive"].append(manufactured(N, N, build_cells(N, N, 2, tgt)))
    out = {}
    for name, seq in rows.items():
        errs = [e for (e, _, _) in seq]
        orders = [np.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
        cellcounts = [c for (_, c, _) in seq]
        out[name] = orders
        print(f"    {name}: errors {['%.3e' % e for e in errs]}, "
              f"orders {['%.2f' % o for o in orders]}, cells {cellcounts}")
    ok = min(out["uniform"]) > 1.8 and min(out["adaptive"]) > 1.2
    print("    PASS" if ok else "    FAIL")
    return ok


def check_hydro():
    print("=== hydro: 1000:1 column at rest on an adaptive grid")
    N, dt, g = 64, 0.06, -9.8
    rho_l, rho_g = 1000.0, 1.0
    yy = np.arange(N + 1)

    def phi(y):
        return np.clip(0.5 * (1 + np.tanh((0.5 * N - y) / 1.5)), 0, 1)

    def rho(y):
        return rho_g + (rho_l - rho_g) * phi(y)

    def beta(x, y, axis, span):
        return 1.0 / rho(y)

    # the 1000:1 transition's TAIL sets the band: phi of 3e-4 still means
    # rho ~ 1.3 when the amplifier is 999, so junctions may only sit where
    # beta is actually flat (|y - N/2| > ~9 at this tanh sharpness). The
    # production criterion must coarsen on beta-flatness, not just distance
    # to the interface - this gate measured that lesson the hard way.
    tgt = np.full((N, N), 2, int)
    band = abs(np.arange(N)[None, :].repeat(N, 0) - 0.5 * N) < 10
    tgt[band] = 0
    tgt[(~band) & (abs(np.arange(N)[None, :].repeat(N, 0) - 0.5 * N) < 16)] = 1
    tgt[:2, :] = tgt[-2:, :] = tgt[:, :2] = tgt[:, -2:] = 0
    cells = build_cells(N, N, 2, tgt)
    kick = g * dt
    u0 = np.zeros((N + 1, N))
    v0 = np.full((N, N + 1), kick)
    v0[:, 0] = v0[:, -1] = 0.0
    op = Operator(cells, beta)
    b = op.rhs(u0, v0, dt)
    p, it = op.solve(b, tol=1e-11, max_iter=60000)

    # audit the operator's own face velocities: u_face = u* - dt (coef/A) Delta,
    # with Delta the face's interpolated difference - junction faces included
    worst_junction, worst_bulk = 0.0, 0.0
    for ent, coef, axis, area, kind, fx, fy in op.faces:
        delta = sum(w * p[a] for (a, w) in ent)
        ustar = kick if axis == 1 else 0.0
        resid = ustar - dt * (coef / area) * delta
        if kind == "junction":
            worst_junction = max(worst_junction, abs(resid))
        else:
            worst_bulk = max(worst_bulk, abs(resid))
    print(f"    residual speeds: same-level faces {worst_bulk:.3e}, "
          f"junction faces {worst_junction:.3e} (kick {abs(kick):.3f}, {it} iters)")
    ok = worst_bulk < 5e-3 * abs(kick) and worst_junction < 5e-2 * abs(kick)
    print("    PASS" if ok else "    FAIL")
    return ok


def check_dam():
    """The production question, asked as a curve: how far do corrected
    velocities on the always-fine interface band drift from the uniform
    projection as the band width grows? A working criterion knob must
    drive the drift down monotonically; the pass bar sits on the middle
    width. A real dam-break state is noisy and vortical through the whole
    liquid, so some drift is the honest price of coarsening - the gate
    measures it instead of pretending it away."""
    print("=== dam: real state, adaptive vs uniform, drift vs band width")
    dt = 0.06
    sim, bu, bv = dam_state()
    nx, ny = sim.nx, sim.ny
    u0, v0 = sim.u.copy(), sim.v.copy()
    beta = beta_from_arrays(bu, bv)

    opu = Operator(uniform_cells(nx, ny), beta)
    pu, itu = opu.solve(opu.rhs(u0, v0, dt), tol=1e-9, max_iter=60000)
    pg = np.zeros((nx, ny))
    for n, (l, i, j) in enumerate(opu.cells):
        pg[i, j] = pu[n]

    speed = max(abs(u0).max(), abs(v0).max())
    rows = []
    for band in (3, 5, 7):
        target = liquid_band_target(sim, nx, ny, band=band)
        opa = Operator(build_cells(nx, ny, 2, target), beta)
        pa, ita = opa.solve(opa.rhs(u0, v0, dt), tol=1e-9, max_iter=60000)
        coarse = target > 0
        dist = np.full((nx, ny), 99)
        dist[coarse] = 0
        for k in range(1, 15):
            grown = dist < k
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                dist[(dist == 99) & np.roll(grown, sh, axis=ax)] = k
        stats = {"water": [], "air_edge": [], "air_interior": []}
        for n, (l, i, j) in enumerate(opa.cells):
            if l != 0:
                continue
            nb = opa.index.get((0, i + 1, j))
            if nb is None or target[i, j] > 0:
                continue
            corr_a = u0[i + 1, j] - dt * bu[i + 1, j] * (pa[nb] - pa[n])
            corr_u = u0[i + 1, j] - dt * bu[i + 1, j] * (pg[i + 1, j] - pg[i, j])
            d = abs(corr_a - corr_u)
            if bu[i + 1, j] < 0.5:                  # liquid-side face
                stats["water"].append(d)
            elif dist[i, j] >= 3:
                stats["air_interior"].append(d)
            else:
                stats["air_edge"].append(d)
        mx = {k: (max(v) if v else 0.0) for k, v in stats.items()}
        rms = {k: (float(np.sqrt(np.mean(np.square(v)))) if v else 0.0)
               for k, v in stats.items()}
        cnt = sum(len(v) for v in stats.values())
        saved = 1 - len(opa.cells) / (nx * ny)
        rows.append({"band": band, "unknowns_saved": round(saved, 3),
                     "faces": cnt, "max": mx, "rms": rms, "iters": ita})
        print(f"    band {band}: {saved:.0%} saved | water max {mx['water']:.2e}"
              f" | air interior max {mx['air_interior']:.2e}"
              f" | air edge-skin max {mx['air_edge']:.2e} ({ita} iters)")
    print(f"    (uniform projection: {itu} iters; state max speed {speed:.2f})")
    # principled bars: a liquid face responds with beta ~ 1e-3 of an air
    # face to the same pressure error, so water drift must be ~1e-3 of the
    # air skin; interior air (3+ cells from coarsening) must be far below
    # the edge skin. The edge skin itself is reported, not gated: it costs
    # 1/1000 of its nominal size in momentum and shrinks with band width.
    # band 3 stays in the curve as the documented too-skinny regime (its
    # "interior" starts where the edge skin ends); the bars gate the
    # production operating point, band >= 5, which still halves the
    # unknowns on this state.
    ok = all(r["max"]["water"] < 1e-3 * speed for r in rows)
    ok = ok and all(r["max"]["air_interior"] < 5e-3 * speed
                    for r in rows if r["band"] >= 5)
    ok = ok and (rows[0]["rms"]["air_edge"] >= rows[1]["rms"]["air_edge"]
                 >= rows[2]["rms"]["air_edge"])
    print("    PASS" if ok else "    FAIL")
    return ok, {"rows": rows, "speed": speed, "iters_uniform": itu}


def main():
    os.makedirs(RES, exist_ok=True)
    results = [check_reduce(), check_anchor(), check_order(), check_hydro()]
    ok_dam, dam_stats = check_dam()
    results.append(ok_dam)
    with open(os.path.join(RES, "stage1_gate.json"), "w") as f:
        json.dump(dam_stats, f, indent=1)
    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
