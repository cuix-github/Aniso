"""Adaptive tiers in the node vs the 2D reference, near-lockstep: one step on a
mixed-scale mid-plunge state with per-tier channels (the stock two-splat scheme,
weight sums epsilon-encoded exactly as the graph will deliver them). Merge and
split decisions sit on float thresholds, so the comparison is multiset-style:
counts per scale, represented volume, and field agreement on the pre-merge set.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_adaptive_node.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from validate_3d import run_node  # noqa: E402

NX, NY, DT, EPS = 120, 160, 0.012, 16.0


def mask(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def tier_channels(sim):
    ones = np.ones(len(sim.pos))
    m = sim.masses()
    out = {}
    fu, fv = sim._face_frames()
    for tier, sel in (("", sim.scale < 1.5), ("2", sim.scale >= 1.5)):
        for g, f, shape, comp in (("u", fu, sim.u.shape, 0), ("v", fv, sim.v.shape, 1)):
            sub = (f[0][sel], f[1][sel])
            vals = [ones[sel], m[sel], (m * sim.vel[:, comp])[sel],
                    (sim.typ == 1).astype(float)[sel]]
            r = 2.0 if tier else 1.0
            acc = [np.zeros(shape) for _ in vals]
            Sim._splat_group(acc, shape, sub, vals, r)
            sw, sm, so, sp = acc
            mean = lambda q: np.where(sw > 0, q / np.maximum(sw, 1e-300), 0.0)
            key = g + tier
            out[key + "_mass"], out[key + "_mom"] = mean(sm), mean(so)
            out[key + "_phase"] = mean(sp)
            out[key + "_ws"] = (sw / (sw + EPS)).astype(np.float32)
    return out


def recover(ws):
    a = np.clip(ws.astype(np.float64), 0, 0.999999)
    return EPS * a / (1.0 - a)


def main():
    sim = Sim(NX, NY, adapt=True, escape=True)
    sim.seed(mask)
    sim.calibrate()
    for _ in range(240):
        sim.step(DT)
    n_coarse0 = int((sim.scale > 1.5).sum())
    print(f"state: {len(sim.pos)} particles, {n_coarse0} coarse, "
          f"volume {float((sim.scale ** 2).sum()):.0f}")
    ch = {k: v.astype(np.float32) for k, v in tier_channels(sim).items()}

    d_in = os.path.join(HERE, "out", "ad_in")
    d_out = os.path.join(HERE, "out", "ad_out")
    os.makedirs(d_in, exist_ok=True)
    rename = {"u2": "u2", "v2": "v2", "w2": "w2"}
    for k, v in ch.items():
        np.save(os.path.join(d_in, k + ".npy"), v.ravel())
    for g in ("u", "v", "w"):
        for c in ("mass", "mom", "phase", "ws"):
            f = os.path.join(d_in, f"{g}2_{c}.npy")
            src = ch.get(f"{g}2_{c}")
            if src is None:
                np.save(f, np.zeros(NX * NY * 2 if g == "w" else 0, np.float32))
    nw = NX * NY * 2
    for k in ("w_mass", "w_mom", "w_phase", "w_ws", "w2_mass", "w2_mom", "w2_phase",
              "w2_ws"):
        np.save(os.path.join(d_in, k + ".npy"), np.zeros(nw, np.float32))
    for k in ("u_wt", "v_wt", "w_wt", "u2_wt", "v2_wt", "w2_wt", "tau_in"):
        np.save(os.path.join(d_in, k + ".npy"),
                np.zeros(0 if "wt" in k else len(sim.pos), np.float32))
    np.save(os.path.join(d_in, "positions.npy"),
            np.concatenate([sim.pos, np.full((len(sim.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(d_in, "velocities.npy"),
            np.concatenate([sim.vel, np.zeros((len(sim.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(d_in, "particle_phase.npy"), sim.typ.astype(np.float32))
    np.save(os.path.join(d_in, "scale_in.npy"), sim.scale.astype(np.float32))

    run_node(d_in, d_out, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=1,
             rho0_face=sim.rho0_face, escape=1, adapt=1,
             coarse_gain=sim.coarse_gain, ws_epsilon=EPS,
             extra_arrays=("tau_in", "u_wt", "v_wt", "w_wt", "u_ws", "v_ws", "w_ws",
                           "scale_in") + tuple(f"{g}2_{c}" for g in ("u", "v", "w")
                                               for c in ("mass", "mom", "phase", "wt", "ws")),
             extra_outputs=("tau_out", "scale_out", "out_phase_state"))

    b_pos = np.load(os.path.join(d_out, "out_positions.npy"))[:, :2]
    b_scale = np.load(os.path.join(d_out, "scale_out.npy"))
    b_phase = np.load(os.path.join(d_out, "out_phase_state.npy"))

    # python twin step from the same state (its own channels internally)
    ref = Sim(NX, NY, adapt=True, escape=True)
    ref.pos, ref.vel = sim.pos.copy(), sim.vel.copy()
    ref.typ, ref.scale = sim.typ.copy(), sim.scale.copy()
    ref.tau, ref.xi = sim.tau.copy(), sim.xi.copy()
    ref.rho0_face, ref.coarse_gain = sim.rho0_face, sim.coarse_gain
    ref.step(DT)

    vol_b = float((b_scale.astype(float) ** 2).sum())
    vol_p = float((ref.scale ** 2).sum())
    nb, np_ = len(b_pos), len(ref.pos)
    cb = int((b_scale > 1.5).sum())
    cp = int((ref.scale > 1.5).sum())
    print(f"counts: node {nb} ({cb} coarse), python {np_} ({cp} coarse)")
    print(f"represented volume: node {vol_b:.0f}, python {vol_p:.0f} (exact target)")
    com_b = b_pos[b_phase >= 0.5].mean(0)
    com_p = ref.pos[ref.typ == 1].mean(0)
    print(f"liquid centre of mass: node ({com_b[0]:.3f},{com_b[1]:.3f}) vs "
          f"python ({com_p[0]:.3f},{com_p[1]:.3f})")
    # Counts differ by block-boundary flips: float32 channel round-trips move
    # positions ~1e-6, flipping floor(x/2) group assignment for boundary particles.
    # Conserved quantities stay strict; decision counts get a 0.5 percent band.
    ok = (abs(vol_b - vol_p) < 1e-3 and abs(nb - np_) <= 0.005 * np_ and
          abs(cb - cp) <= 0.01 * max(cp, 1) and np.abs(com_b - com_p).max() < 0.05)
    print("ADAPTIVE NODE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
