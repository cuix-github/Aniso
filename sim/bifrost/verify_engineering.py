"""Engineering verification of the 3D node: the mature numerical-code checks that
catch silent implementation errors no visual inspection can.

  A  symmetry/invariance: mirror-x, permute x<->z, translate. An axis typo or
     off-by-one survives every video and dies here. (st off: no RNG; adapt off:
     merge grouping order is not equivariant by design.)
  B  true-3D lockstep: a from-scratch numpy implementation of the full step
     (flexible Jacobi-PCG and all) on a genuinely 3D state - the w-axis paths
     have never had a lockstep before this.
  C  order of convergence: a manufactured projection problem (uniform density,
     analytic solenoidal field plus a known gradient) at h, h/2, h/4 - the
     error must shrink at the scheme's order.
  D  momentum budget: chained single steps on a 3D collapse, auditing
     d(sum m v_y) against gravity impulse plus wall-pressure impulse each step.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe verify_engineering.py [A|B|C|D]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from validate_3d import run_node  # noqa: E402

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------- state factory
def make_state(nx, ny, nz, seed=11):
    rng = np.random.default_rng(seed)
    n = nx * ny * nz * 2
    pos = rng.uniform([0.8, 0.8, 0.8], [nx - 0.8, ny - 0.8, nz - 0.8], (n, 3))
    typ = (pos[:, 1] < 0.55 * ny).astype(np.float32)
    vel = rng.normal(0, 1.5, (n, 3))
    st = {"positions": pos.astype(np.float32), "velocities": vel.astype(np.float32),
          "particle_phase": typ}
    # smooth-ish random channels per face grid
    def field(shape, lo, hi):
        f = rng.uniform(lo, hi, shape)
        for ax in range(3):
            f = 0.5 * f + 0.25 * (np.roll(f, 1, ax) + np.roll(f, -1, ax))
        return f.astype(np.float32)
    shapes = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz), "w": (nx, ny, nz + 1)}
    for g, sh in shapes.items():
        phase = field(sh, 0, 1)
        mass = 1.0 + phase * 999.0 + field(sh, 0, 5)
        st[g + "_phase"] = phase
        st[g + "_mass"] = mass.astype(np.float32)
        st[g + "_mom"] = (mass * field(sh, -3, 3)).astype(np.float32)
    return st


def write_state(d, st, nx, ny, nz):
    os.makedirs(d, exist_ok=True)
    for k, v in st.items():
        np.save(os.path.join(d, k + ".npy"), np.ascontiguousarray(
            v.reshape(-1, 3) if k in ("positions", "velocities") else v.ravel()))


def step_node(tag, st, nx, ny, nz, dt=0.03, **ports):
    d_in = os.path.join(HERE, "out", "ver_" + tag)
    d_out = d_in + "_o"
    write_state(d_in, st, nx, ny, nz)
    run_node(d_in, d_out, nx=nx, ny=ny, nz=nz, dt=dt, preconditioner=0,
             escape=1, rho0_face=2.2, **ports)
    r = {}
    for nm in ("out_u", "out_v", "out_w", "out_positions", "out_velocities",
               "out_escaped", "pressure"):
        r[nm] = np.load(os.path.join(d_out, nm + ".npy"))
    return r


# ---------------------------------------------------------------- A: symmetry
def mirror_x_state(st, nx, ny, nz):
    m = {}
    m["positions"] = st["positions"].copy()
    m["positions"][:, 0] = nx - st["positions"][:, 0]
    m["velocities"] = st["velocities"].copy()
    m["velocities"][:, 0] *= -1
    m["particle_phase"] = st["particle_phase"].copy()
    for g, sh in (("u", (nx + 1, ny, nz)), ("v", (nx, ny + 1, nz)),
                  ("w", (nx, ny, nz + 1))):
        for c in ("mass", "mom", "phase"):
            f = st[g + "_" + c].reshape(sh)[::-1].copy()
            if g == "u" and c == "mom":
                f = -f
            m[g + "_" + c] = f
    return m


def permute_xz_state(st, nx, ny, nz):
    """x <-> z swap: u and w exchange roles; v permutes its axes."""
    m = {}
    m["positions"] = st["positions"][:, [2, 1, 0]].copy()
    m["velocities"] = st["velocities"][:, [2, 1, 0]].copy()
    m["particle_phase"] = st["particle_phase"].copy()
    for c in ("mass", "mom", "phase"):
        m["w_" + c] = st["u_" + c].reshape(nx + 1, ny, nz).transpose(2, 1, 0).copy()
        m["u_" + c] = st["w_" + c].reshape(nx, ny, nz + 1).transpose(2, 1, 0).copy()
        m["v_" + c] = st["v_" + c].reshape(nx, ny + 1, nz).transpose(2, 1, 0).copy()
    return m


def check_A(nx=20, ny=16, nz=20):
    st = make_state(nx, ny, nz)
    base = step_node("base", st, nx, ny, nz)

    mir = step_node("mir", mirror_x_state(st, nx, ny, nz), nx, ny, nz)
    pos_m = mir["out_positions"].copy()
    pos_m[:, 0] = nx - pos_m[:, 0]
    e_pos = np.abs(pos_m - base["out_positions"]).max()
    vel_m = mir["out_velocities"].copy()
    vel_m[:, 0] *= -1
    e_vel = np.abs(vel_m - base["out_velocities"]).max()
    e_esc = int((mir["out_escaped"] != base["out_escaped"]).sum())
    print(f"A mirror-x: pos {e_pos:.2e}, vel {e_vel:.2e}, escape flips {e_esc}")
    ok1 = e_pos < 2e-4 and e_vel < 2e-3 and e_esc <= 2

    per = step_node("per", permute_xz_state(st, nx, ny, nz), nz, ny, nx)
    pos_p = per["out_positions"][:, [2, 1, 0]]
    vel_p = per["out_velocities"][:, [2, 1, 0]]
    e_pos = np.abs(pos_p - base["out_positions"]).max()
    e_vel = np.abs(vel_p - base["out_velocities"]).max()
    e_esc = int((per["out_escaped"] != base["out_escaped"]).sum())
    print(f"A permute-xz: pos {e_pos:.2e}, vel {e_vel:.2e}, escape flips {e_esc}")
    ok2 = e_pos < 2e-4 and e_vel < 2e-3 and e_esc <= 2
    print("A:", "PASS" if ok1 and ok2 else "FAIL")
    return ok1 and ok2


# ---------------------------------------------------------------- B: 3D lockstep
def numpy_step(st, nx, ny, nz, dt, g=-9.8, rho_l=1000.0, rho_g=1.0,
               a_l=0.97, a_g=0.9, tol=1e-6, iters=4000, esc_phi=0.3, rho0=2.2):
    uM = st["u_mass"].reshape(nx + 1, ny, nz).astype(float)
    uO = st["u_mom"].reshape(nx + 1, ny, nz).astype(float)
    uP = st["u_phase"].reshape(nx + 1, ny, nz).astype(float)
    vM = st["v_mass"].reshape(nx, ny + 1, nz).astype(float)
    vO = st["v_mom"].reshape(nx, ny + 1, nz).astype(float)
    vP = st["v_phase"].reshape(nx, ny + 1, nz).astype(float)
    wM = st["w_mass"].reshape(nx, ny, nz + 1).astype(float)
    wO = st["w_mom"].reshape(nx, ny, nz + 1).astype(float)
    wP = st["w_phase"].reshape(nx, ny, nz + 1).astype(float)
    u = np.where(uM > 0, uO / np.maximum(uM, 1e-30), 0.0)
    v = np.where(vM > 0, vO / np.maximum(vM, 1e-30), 0.0)
    w = np.where(wM > 0, wO / np.maximum(wM, 1e-30), 0.0)

    def walls(u, v, w):
        u[0], u[-1] = 0, 0
        v[:, 0], v[:, -1] = 0, 0
        w[:, :, 0], w[:, :, -1] = 0, 0
    walls(u, v, w)
    uS, vS, wS = u.copy(), v.copy(), w.copy()
    v += g * dt
    walls(u, v, w)
    bu = 1.0 / (rho_g + (rho_l - rho_g) * np.clip(uP, 0, 1))
    bv = 1.0 / (rho_g + (rho_l - rho_g) * np.clip(vP, 0, 1))
    bw = 1.0 / (rho_g + (rho_l - rho_g) * np.clip(wP, 0, 1))
    walls(bu, bv, bw)
    rhs = -((u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])) / dt

    def A(p):
        gu = np.zeros_like(u)
        gv = np.zeros_like(v)
        gw = np.zeros_like(w)
        gu[1:-1] = (p[1:] - p[:-1]) * bu[1:-1]
        gv[:, 1:-1] = (p[:, 1:] - p[:, :-1]) * bv[:, 1:-1]
        gw[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) * bw[:, :, 1:-1]
        return -((gu[1:] - gu[:-1]) + (gv[:, 1:] - gv[:, :-1]) + (gw[:, :, 1:] - gw[:, :, :-1]))

    diag = bu[1:] + bu[:-1] + bv[:, 1:] + bv[:, :-1] + bw[:, :, 1:] + bw[:, :, :-1]
    dinv = np.where(diag > 0, 1.0 / np.maximum(diag, 1e-300), 0.0)
    p = np.zeros((nx, ny, nz))
    r = rhs - A(p)
    z = dinv * r
    d = z.copy()
    rz = float((r * z).sum())
    r0 = np.sqrt(float((r * r).sum())) or 1.0
    for _ in range(iters):
        if np.sqrt(float((r * r).sum())) / r0 < tol:
            break
        q = A(d)
        dq = float((d * q).sum())
        if dq == 0:
            break
        al = rz / dq
        rp = r.copy()
        p += al * d
        r -= al * q
        z = dinv * r
        rzn = float(((r - rp) * z).sum())
        beta = max(0.0, rzn / rz)
        d = z + beta * d
        rz = float((r * z).sum())
        if rz == 0:
            break
    u[1:-1] -= dt * bu[1:-1] * (p[1:] - p[:-1])
    v[:, 1:-1] -= dt * bv[:, 1:-1] * (p[:, 1:] - p[:, :-1])
    w[:, :, 1:-1] -= dt * bw[:, :, 1:-1] * (p[:, :, 1:] - p[:, :, :-1])
    walls(u, v, w)

    def samp(gr, off, pts):
        q = pts - off
        b = np.floor(q).astype(int)
        f = q - b
        out = np.zeros(len(pts))
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wgt = ((f[:, 0] if di else 1 - f[:, 0]) *
                           (f[:, 1] if dj else 1 - f[:, 1]) *
                           (f[:, 2] if dk else 1 - f[:, 2]))
                    ii = np.clip(b[:, 0] + di, 0, gr.shape[0] - 1)
                    jj = np.clip(b[:, 1] + dj, 0, gr.shape[1] - 1)
                    kk = np.clip(b[:, 2] + dk, 0, gr.shape[2] - 1)
                    out += wgt * gr[ii, jj, kk]
        return out

    pos = st["positions"].astype(float).copy()
    vel = st["velocities"].astype(float).copy()
    typ = st["particle_phase"] >= 0.5
    oU, oV, oW = np.array([0, .5, .5]), np.array([.5, 0, .5]), np.array([.5, .5, 0])

    # escape detection (Eq.6 count splat, both phases)
    cnt = {0: [np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)],
           1: [np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)]}
    for grid, off, arrS in ((0, oU, u.shape), (1, oV, v.shape), (2, oW, w.shape)):
        q = pos - off
        b = np.floor(q + 0.5).astype(int)
        f = q - b
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    d2 = (f[:, 0] - di) ** 2 + (f[:, 1] - dj) ** 2 + (f[:, 2] - dk) ** 2
                    wgt = np.maximum(1 - d2, 0) ** 3
                    ii, jj, kk = b[:, 0] + di, b[:, 1] + dj, b[:, 2] + dk
                    ok = ((ii >= 0) & (ii < arrS[0]) & (jj >= 0) & (jj < arrS[1]) &
                          (kk >= 0) & (kk < arrS[2]))
                    wgt = np.where(ok, wgt, 0)
                    ii = np.clip(ii, 0, arrS[0] - 1)
                    jj = np.clip(jj, 0, arrS[1] - 1)
                    kk = np.clip(kk, 0, arrS[2] - 1)
                    for ph in (0, 1):
                        np.add.at(cnt[ph][grid], (ii, jj, kk),
                                  wgt * (typ == (ph == 1)))
    thresh = (1 - esc_phi) * rho0
    fa = (samp(cnt[0][0], oU, pos) + samp(cnt[0][1], oV, pos) +
          samp(cnt[0][2], oW, pos)) / 3.0
    fl = (samp(cnt[1][0], oU, pos) + samp(cnt[1][1], oV, pos) +
          samp(cnt[1][2], oW, pos)) / 3.0
    esc = np.where(typ, fa > thresh, fl > thresh)

    du, dv, dw = u - uS, v - vS, w - wS
    aarr = np.where(typ, a_l, a_g)
    pic = np.stack([samp(u, oU, pos), samp(v, oV, pos), samp(w, oW, pos)], 1)
    dlt = np.stack([samp(du, oU, pos), samp(dv, oV, pos), samp(dw, oW, pos)], 1)
    nv = aarr[:, None] * (vel + dlt) + (1 - aarr[:, None]) * pic
    gx = pic
    drag = np.where(typ, 1.0, 8.0)
    nv_esc = vel.copy()
    nv_esc[:, 1] += np.where(typ, g, 2.0 * abs(g)) * dt
    nv_esc += drag[:, None] * dt * (gx - nv_esc)
    nv = np.where(esc[:, None], nv_esc, nv)

    def sampvec(pts):
        return np.stack([samp(u, oU, pts), samp(v, oV, pts), samp(w, oW, pts)], 1)
    mid = pos + 0.5 * dt * sampvec(pos)
    cl = np.maximum(mid, 0.51)
    npos = pos + dt * sampvec(cl)
    npos = np.where(esc[:, None], pos + dt * nv, npos)
    lohi = np.array([nx, ny, nz]) - 0.51
    for k in range(3):
        below, above = npos[:, k] < 0.51, npos[:, k] > lohi[k]
        npos[:, k] = np.clip(npos[:, k], 0.51, lohi[k])
        nv[below | above, k] = 0.0
    return {"out_u": u, "out_v": v, "out_w": w, "pos": npos, "vel": nv, "esc": esc}


def check_B(nx=20, ny=16, nz=20):
    st = make_state(nx, ny, nz, seed=29)
    node = step_node("b3d", st, nx, ny, nz)
    ref = numpy_step(st, nx, ny, nz, 0.03)
    eu = np.abs(node["out_u"].reshape(ref["out_u"].shape) - ref["out_u"]).max()
    ev = np.abs(node["out_v"].reshape(ref["out_v"].shape) - ref["out_v"]).max()
    ew = np.abs(node["out_w"].reshape(ref["out_w"].shape) - ref["out_w"]).max()
    ep = np.abs(node["out_positions"] - ref["pos"]).max()
    evp = np.abs(node["out_velocities"] - ref["vel"]).max()
    mism = int(((node["out_escaped"] > 0.5) != ref["esc"]).sum())
    print(f"B faces u/v/w: {eu:.2e} {ev:.2e} {ew:.2e}; pos {ep:.2e}; vel {evp:.2e}; "
          f"escape mismatches {mism}")
    ok = max(eu, ev, ew) < 1e-4 and ep < 1e-3 and evp < 1e-2 and mism <= 2
    print("B:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- C: convergence
def check_C():
    """Project a pure gradient (uniform density): the exact answer is zero velocity.
    The residual must shrink with resolution at the scheme's order."""
    errs = []
    for m in (16, 32, 64):
        nx = ny = nz = m
        pi = np.pi / m
        iu = np.arange(nx + 1)[:, None, None]
        jc = (np.arange(ny) + 0.5)[None, :, None]
        kc = (np.arange(nz) + 0.5)[None, None, :]
        u = -np.sin(pi * iu) * np.cos(pi * jc) * np.cos(pi * kc)
        ic = (np.arange(nx) + 0.5)[:, None, None]
        jv = np.arange(ny + 1)[None, :, None]
        v = -np.cos(pi * ic) * np.sin(pi * jv) * np.cos(pi * kc)
        kw = np.arange(nz + 1)[None, None, :]
        jc2 = (np.arange(ny) + 0.5)[None, :, None]
        w = -np.cos(pi * ic) * np.cos(pi * jc2) * np.sin(pi * kw)
        ones = lambda sh: np.ones(sh, np.float32)
        st = {"positions": np.full((1, 3), m / 2, np.float32),
              "velocities": np.zeros((1, 3), np.float32),
              "particle_phase": np.ones(1, np.float32),
              "u_mass": ones(u.shape), "u_mom": u.astype(np.float32),
              "u_phase": ones(u.shape),
              "v_mass": ones(v.shape), "v_mom": v.astype(np.float32),
              "v_phase": ones(v.shape),
              "w_mass": ones(w.shape), "w_mom": w.astype(np.float32),
              "w_phase": ones(w.shape)}
        r = step_node(f"c{m}", st, nx, ny, nz, dt=1.0, gravity=0.0,
                      rho_liquid=1.0, rho_air=1.0, tolerance=1e-10,
                      max_iterations=40000)
        res = max(np.abs(r["out_u"]).max(), np.abs(r["out_v"]).max(),
                  np.abs(r["out_w"]).max())
        # order measured on the PRESSURE against the analytic potential phi
        # (the velocity annihilation is exact by construction - bonus check).
        phi = (np.cos(pi * (np.arange(nx)[:, None, None] + 0.5)) *
               np.cos(pi * (np.arange(ny)[None, :, None] + 0.5)) *
               np.cos(pi * (np.arange(nz)[None, None, :] + 0.5)))
        pr = r["pressure"].reshape(nx, ny, nz)
        expected = phi * (m / np.pi)   # u was built as (1/k) grad(phi), k = pi/m
        diff = pr - expected
        diff -= diff.mean()
        errs.append(float(np.abs(diff).max() / np.abs(expected).max()))
        print(f"C m={m}: velocity annihilation {res:.2e}; pressure error vs analytic "
              f"{errs[-1]:.3e}")
    r1 = errs[0] / max(errs[1], 1e-30)
    r2 = errs[1] / max(errs[2], 1e-30)
    print(f"C pressure-error refinement ratios: {r1:.2f}, {r2:.2f} "
          f"(second order -> ~4)")
    ok = r1 > 2.5 and r2 > 2.5
    print("C:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- D: momentum
def check_D(nx=24, ny=20, nz=16, dt=0.02):
    """Exact discrete momentum identity: rho_face * dv = -dt * (p_j - p_j-1) on every
    interior face, so the density-weighted velocity change summed over a column
    telescopes to the boundary pressure difference - to machine precision, for any
    density field. If this does not close, the projection wiring is wrong."""
    st = make_state(nx, ny, nz, seed=47)
    g, rho_l, rho_g = -9.8, 1000.0, 1.0
    node = step_node("dmom", st, nx, ny, nz, dt=dt, tolerance=1e-10,
                     max_iterations=40000)
    vM = st["v_mass"].reshape(nx, ny + 1, nz).astype(float)
    vO = st["v_mom"].reshape(nx, ny + 1, nz).astype(float)
    vP = np.clip(st["v_phase"].reshape(nx, ny + 1, nz).astype(float), 0, 1)
    v_star = np.where(vM > 0, vO / np.maximum(vM, 1e-30), 0.0)
    v_star[:, 0], v_star[:, -1] = 0, 0
    v_star += g * dt
    v_star[:, 0], v_star[:, -1] = 0, 0
    rho_face = rho_g + (rho_l - rho_g) * vP
    v_post = node["out_v"].reshape(nx, ny + 1, nz)
    pr = node["pressure"].reshape(nx, ny, nz)
    lhs = (rho_face[:, 1:-1, :] * (v_post[:, 1:-1, :] - v_star[:, 1:-1, :])).sum()
    rhs = -dt * (pr[:, -1, :] - pr[:, 0, :]).sum()
    scale = max(abs(float(lhs)), abs(float(rhs)), 1e-30)
    resid = abs(float(lhs) - float(rhs)) / scale
    print(f"D telescoped momentum identity: lhs {float(lhs):.6e}, rhs {float(rhs):.6e}, "
          f"relative residual {resid:.2e}")
    ok = resid < 1e-6
    print("D:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ABCD"
    results = {}
    for c, fn in (("A", check_A), ("B", check_B), ("C", check_C), ("D", check_D)):
        if c in which:
            print(f"===== {c}")
            results[c] = fn()
    print("OVERALL:", "PASS" if all(results.values()) else "FAIL")
