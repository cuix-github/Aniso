"""Milestone-5 validation, layer 1: the C++ step node against the python reference in
lockstep on identical inputs.

A reference dam break is advanced a few frames; its state is splatted into exactly the
channels the Bifrost graph feeds the node (per-face mean mass, mean momentum, mean
phase, all float32). The same channel files then drive (a) the step node through bifcmd
and (b) the python project/g2p/advect with the same injected channels. Every output is
compared: face velocities, particle positions and velocities, iteration counts.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_step.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402

IN = os.path.join(HERE, "out", "step_in")
OUT = os.path.join(HERE, "out", "step_out")
NX, NY, DT = 160, 80, 0.02


def channels(sim):
    """The node's input channels from a reference state: per-face weighted means."""
    m = sim.masses()
    ones = np.ones(len(sim.pos))
    fu, fv = sim._face_frames()
    mu, pu, su, qu = sim._splat(sim.u.shape, fu, [m, m * sim.vel[:, 0], ones, sim.typ.astype(float)])
    mv, pv, sv, qv = sim._splat(sim.v.shape, fv, [m, m * sim.vel[:, 1], ones, sim.typ.astype(float)])

    def mean(num, den):
        return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    return {"u_mass": mean(mu, su), "u_mom": mean(pu, su), "u_phase": mean(qu, su),
            "v_mass": mean(mv, sv), "v_mom": mean(pv, sv), "v_phase": mean(qv, sv)}


def main():
    os.makedirs(IN, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    sim = Sim(NX, NY)
    sim.seed(lambda x, y: (x < 40) & (y < 56))
    sim.calibrate()
    sim.run(12, 0.06)   # a mid-collapse state with real velocities

    ch = {k: v.astype(np.float32) for k, v in channels(sim).items()}
    for k, v in ch.items():
        np.save(os.path.join(IN, f"{k}.npy"), v.ravel())
    np.save(os.path.join(IN, "positions.npy"),
            np.concatenate([sim.pos, np.full((len(sim.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(IN, "velocities.npy"),
            np.concatenate([sim.vel, np.zeros((len(sim.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(IN, "particle_phase.npy"), sim.typ.astype(np.float32))

    # ---- (a) the node, through bifcmd
    args = [os.path.join(HERE, "run_solve_step.bat"), "--set-port", "dt", str(DT)]
    for n in ['u_mass', 'u_mom', 'u_phase', 'v_mass', 'v_mom', 'v_phase', 'particle_phase', 'positions', 'velocities']:
        args += ["--set-port", f"path_{n}", os.path.join(IN, f"{n}.npy").replace("\\", "/")]
    for n in ['out_u', 'out_v', 'pressure', 'out_positions', 'out_velocities']:
        args += ["--set-port", f"path_{n}", os.path.join(OUT, f"{n}.npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "Output port" in line and "ok_" not in line:
            print(line.strip())
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"bifcmd failed: {res.returncode}")

    # ---- (b) python, same float32 channels injected
    ref = Sim(NX, NY)
    ref.pos, ref.vel, ref.typ = sim.pos.copy(), sim.vel.copy(), sim.typ.copy()
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
    ref.mu, ref.mv = np.empty_like(ref.u), np.empty_like(ref.v)   # identity markers
    ref.phase = lambda raw: phi_u if raw is ref.mu else phi_v
    ref.project(DT)
    ref.g2p(u_star, v_star)
    ref.advect(DT)

    # ---- compare
    bu = np.load(os.path.join(OUT, "out_u.npy")).reshape(NX + 1, NY)
    bv = np.load(os.path.join(OUT, "out_v.npy")).reshape(NX, NY + 1)
    bp = np.load(os.path.join(OUT, "out_positions.npy"))[:, :2]
    bw = np.load(os.path.join(OUT, "out_velocities.npy"))[:, :2]
    checks = [("face u", np.abs(bu - ref.u).max()),
              ("face v", np.abs(bv - ref.v).max()),
              ("particle position", np.abs(bp - ref.pos).max()),
              ("particle velocity", np.abs(bw - ref.vel).max())]
    print(f"python CG iterations: {ref.iters}")
    ok = True
    for name, err in checks:
        verdict = "OK" if err < 1e-3 else "FAIL"
        ok &= err < 1e-3
        print(f"  max |bifrost - python| {name}: {err:.3e}  {verdict}")
    print("LOCKSTEP:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
