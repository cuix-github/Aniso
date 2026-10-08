"""Solid-obstacle lockstep: the node with AABB obstacles against the 2D reference on
an identical bore-with-block state, identical channels, one step. Penetration must
be zero on both sides and fields must agree at float rounding.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_solids.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from validate_step import channels  # noqa: E402
from validate_3d import run_node  # noqa: E402

NX, NY, DT = 200, 60, 0.02
BOX = (120.0, 0.0, 128.0, 14.0)


def mask(x, y):
    return (y < 6) | ((x < 40) & (y < 40))


def main():
    sim = Sim(NX, NY, escape=True, obstacles=[BOX])
    sim.seed(mask)
    sim.calibrate()
    for _ in range(75):     # t = 1.5: bore at the block, splash in progress
        sim.step(DT)
    ch = {k: v.astype(np.float32) for k, v in channels(sim).items()}

    d_in = os.path.join(HERE, "out", "sol_in")
    d_out = os.path.join(HERE, "out", "sol_out")
    os.makedirs(d_in, exist_ok=True)
    for k, v in ch.items():
        np.save(os.path.join(d_in, k + ".npy"), v.ravel())
    for k in ("w_mass", "w_mom", "w_phase"):
        np.save(os.path.join(d_in, k + ".npy"), np.zeros(NX * NY * 2, np.float32))
    np.save(os.path.join(d_in, "positions.npy"),
            np.concatenate([sim.pos, np.full((len(sim.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(d_in, "velocities.npy"),
            np.concatenate([sim.vel, np.zeros((len(sim.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(d_in, "particle_phase.npy"), sim.typ.astype(np.float32))
    np.save(os.path.join(d_in, "obstacles_min.npy"),
            np.array([[BOX[0], BOX[1], -1.0]], np.float32))
    np.save(os.path.join(d_in, "obstacles_max.npy"),
            np.array([[BOX[2], BOX[3], 2.0]], np.float32))

    run_node(d_in, d_out, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=1,
             rho0_face=sim.rho0_face, escape=1,
             extra_arrays=("obstacles_min", "obstacles_max"))

    # python twin with identical channels
    ref = Sim(NX, NY, escape=True, obstacles=[BOX])
    ref.pos = np.load(os.path.join(d_in, "positions.npy"))[:, :2].astype(float)
    ref.vel = np.load(os.path.join(d_in, "velocities.npy"))[:, :2].astype(float)
    ref.typ = sim.typ.copy()
    ref.rho0_face = sim.rho0_face
    um, uo = ch["u_mass"].astype(float), ch["u_mom"].astype(float)
    vm, vo = ch["v_mass"].astype(float), ch["v_mom"].astype(float)
    ref.u = np.where(um > 0, uo / np.maximum(um, 1e-30), 0.0)
    ref.v = np.where(vm > 0, vo / np.maximum(vm, 1e-30), 0.0)
    ref._enforce_walls()
    u_star, v_star = ref.u.copy(), ref.v.copy()
    ref.v += ref.g * DT
    ref._enforce_walls()
    phi_u = np.clip(ch["u_phase"].astype(float), 0, 1)
    phi_v = np.clip(ch["v_phase"].astype(float), 0, 1)
    ref.mu, ref.mv = np.empty_like(ref.u), np.empty_like(ref.v)
    ref.phase = lambda raw: phi_u if raw is ref.mu else phi_v
    ref.project(DT)
    w_rep = np.ones(len(ref.typ))
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
    ref.advect(DT, ballistic=esc)
    ref._push_out()

    b_pos = np.load(os.path.join(d_out, "out_positions.npy"))[:, :2]
    b_vel = np.load(os.path.join(d_out, "out_velocities.npy"))[:, :2]
    b_esc = np.load(os.path.join(d_out, "out_escaped.npy")) > 0.5
    pen_b = int(((b_pos[:, 0] > BOX[0]) & (b_pos[:, 0] < BOX[2]) &
                 (b_pos[:, 1] > BOX[1]) & (b_pos[:, 1] < BOX[3])).sum())
    pen_p = int(ref._inside_solid(ref.pos).sum())
    mism = int((b_esc != esc).sum())
    dp = np.abs(b_pos - ref.pos).max()
    dv = np.abs(b_vel - ref.vel).max()
    print(f"penetrations: node {pen_b}, python {pen_p}")
    print(f"escape mismatches: {mism}; max |diff| pos {dp:.3e}, vel {dv:.3e}")
    ok = pen_b == 0 and pen_p == 0 and mism <= 2 and dp < 2e-3 and dv < 5e-2
    print("SOLIDS LOCKSTEP:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
