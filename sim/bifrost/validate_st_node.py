"""ST-in-the-node lockstep: the step node with st=1 against the 2D reference on an
identical mid-splash state, with the node's own exported tau draws replayed in
python, so the comparison is deterministic despite the randomness.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_st_node.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from validate_3d import run_node  # noqa: E402

NX, NY, DT = 120, 160, 0.096


def mask(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def st_channels(sim):
    """Stock-splat-equivalent channels of the wt-premultiplied properties."""
    wt = sim._wt(sim.tau) / np.maximum(sim._wt_norm(sim.xi), 1e-9)
    m = sim.masses() * wt
    ones = np.ones(len(sim.pos))
    fu, fv = sim._face_frames()
    out = {}
    for g, f, shape, comp in (("u", fu, sim.u.shape, 0), ("v", fv, sim.v.shape, 1)):
        sw, swt, sm, smom, sph = sim._splat(
            shape, f, [ones, wt, m, m * sim.vel[:, comp], wt * (sim.typ == 1)])
        mean = lambda q: np.where(sw > 0, q / np.maximum(sw, 1e-300), 0.0)
        out[g + "_wt"], out[g + "_mass"] = mean(swt), mean(sm)
        out[g + "_mom"], out[g + "_phase"] = mean(smom), mean(sph)
    return out, wt


def main():
    sim = Sim(NX, NY, escape=True, st=True)
    sim.seed(mask)
    sim.calibrate()
    for _ in range(36):            # t = 3.46 at the big step: mid-splash, jittered taus
        sim.step(DT)
    ch, wt_now = st_channels(sim)
    ch = {k: v.astype(np.float32) for k, v in ch.items()}

    d_in = os.path.join(HERE, "out", "stn_in")
    d_out = os.path.join(HERE, "out", "stn_out")
    os.makedirs(d_in, exist_ok=True)
    for k, v in ch.items():
        np.save(os.path.join(d_in, k + ".npy"), v.ravel())
    for k in ("w_mass", "w_mom", "w_phase", "w_wt"):
        np.save(os.path.join(d_in, k + ".npy"), np.zeros(NX * NY * 2, np.float32))
    np.save(os.path.join(d_in, "positions.npy"),
            np.concatenate([sim.pos, np.full((len(sim.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(d_in, "velocities.npy"),
            np.concatenate([sim.vel, np.zeros((len(sim.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(d_in, "particle_phase.npy"), sim.typ.astype(np.float32))
    np.save(os.path.join(d_in, "tau_in.npy"), sim.tau.astype(np.float32))

    run_node(d_in, d_out, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=1,
             rho0_face=sim.rho0_face, escape=1, st=1, st_seed=7, step_index=36,
             extra_arrays=("tau_in", "u_wt", "v_wt", "w_wt"),
             extra_outputs=("tau_out", "out_wt", "out_pos_synced"))

    tau_new = np.load(os.path.join(d_out, "tau_out.npy")).astype(float)
    b_esc = np.load(os.path.join(d_out, "out_escaped.npy")) > 0.5
    b_pos = np.load(os.path.join(d_out, "out_positions.npy"))[:, :2]
    b_vel = np.load(os.path.join(d_out, "out_velocities.npy"))[:, :2]

    # ---- python replay with identical channels and the node's tau draws.
    ref = Sim(NX, NY, escape=True, st=True)
    ref.pos = np.load(os.path.join(d_in, "positions.npy"))[:, :2].astype(float)
    ref.vel = np.load(os.path.join(d_in, "velocities.npy"))[:, :2].astype(float)
    ref.typ = sim.typ.copy()
    ref.tau = np.load(os.path.join(d_in, "tau_in.npy")).astype(float)
    ref.rho0_face = sim.rho0_face
    um, uo = ch["u_mass"].astype(float), ch["u_mom"].astype(float)
    vm, vo = ch["v_mass"].astype(float), ch["v_mom"].astype(float)
    ref.u = np.where(um > 0, uo / np.maximum(um, 1e-30), 0.0)
    ref.v = np.where(vm > 0, vo / np.maximum(vm, 1e-30), 0.0)
    ref._enforce_walls()
    u_star, v_star = ref.u.copy(), ref.v.copy()
    ref.v += ref.g * DT
    ref._enforce_walls()
    phi_u = np.clip(np.where(ch["u_wt"] > 1e-12, ch["u_phase"] / np.maximum(ch["u_wt"], 1e-12), 0), 0, 1)
    phi_v = np.clip(np.where(ch["v_wt"] > 1e-12, ch["v_phase"] / np.maximum(ch["v_wt"], 1e-12), 0), 0, 1)
    ref.mu, ref.mv = np.empty_like(ref.u), np.empty_like(ref.v)
    ref.phase = lambda raw: phi_u if raw is ref.mu else phi_v
    ref.project(DT)
    # escape detection, as Sim.step does it
    ones_air = (ref.typ == 0).astype(float)
    fu, fv = ref._face_frames()
    au, lu = ref._splat(ref.u.shape, fu, [ones_air, 1 - ones_air])
    av, lv = ref._splat(ref.v.shape, fv, [ones_air, 1 - ones_air])
    thresh = (1.0 - ref.esc_phi) * ref.rho0_face
    fa = 0.5 * (ref._sample_faces(au, np.array([0.0, 0.5]), ref.pos) +
                ref._sample_faces(av, np.array([0.5, 0.0]), ref.pos))
    fl = 0.5 * (ref._sample_faces(lu, np.array([0.0, 0.5]), ref.pos) +
                ref._sample_faces(lv, np.array([0.5, 0.0]), ref.pos))
    drop = (ref.typ == 1) & (fa > thresh)
    bub = (ref.typ == 0) & (fl > thresh)
    esc = drop | bub
    ref.g2p(u_star, v_star, skip=esc)
    if esc.any():
        vg = ref.sample_velocity(ref.pos)
        ref.vel[drop, 1] += ref.g * DT
        ref.vel[drop] += ref.drag_droplet * DT * (vg[drop] - ref.vel[drop])
        ref.vel[bub, 1] += ref.buoyancy * abs(ref.g) * DT
        ref.vel[bub] += ref.drag_bubble * DT * (vg[bub] - ref.vel[bub])
    # advection replay with the node's taus (clamp matches the node; inactive at const dt)
    dt_act = np.clip(DT * (1.0 + tau_new - ref.tau), 0.0, 2.0 * DT)
    remaining = dt_act.copy()
    ref.pos = ref.pos.copy()
    ballistic = esc
    ref.pos[ballistic] += remaining[ballistic, None] * ref.vel[ballistic]
    remaining = np.where(ballistic, 0.0, remaining)
    for _ in range(64):
        active = remaining > 1e-12
        if not active.any():
            break
        v = ref.sample_velocity(ref.pos)
        speed = np.maximum(np.abs(v).max(1), 1e-9)
        dts = np.where(active, np.minimum(remaining, 1.0 / speed), 0.0)
        mid = ref.pos + 0.5 * dts[:, None] * v
        v2 = ref.sample_velocity(np.clip(mid, 0.51, None))
        ref.pos = ref.pos + dts[:, None] * v2
        remaining = remaining - dts
    lo = 0.51
    for k, hi in ((0, NX - 0.51), (1, NY - 0.51)):
        below, above = ref.pos[:, k] < lo, ref.pos[:, k] > hi
        ref.pos[:, k] = np.clip(ref.pos[:, k], lo, hi)
        ref.vel[below | above, k] = 0.0

    mism = int((b_esc != esc).sum())
    dp = np.abs(b_pos - ref.pos).max()
    dv = np.abs(b_vel - ref.vel).max()
    print(f"escaped: python {int(esc.sum())}, node {int(b_esc.sum())}, mismatches {mism}")
    print(f"max |diff| positions {dp:.3e}, velocities {dv:.3e}")
    ok = mism <= 2 and dp < 2e-3 and dv < 5e-2
    print("ST NODE LOCKSTEP:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
