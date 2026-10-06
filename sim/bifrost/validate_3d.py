"""Milestone-6 phase 1 validation: the 3D step node and its multigrid preconditioner.

Three tests, each against something already trusted:

  thin_slab   step_3d at nz=1 must reproduce the 2D method exactly: same dam-break
              channel inputs as validate_step.py, w channels all zero, compared field
              by field against the python reference (the lockstep bridge into 3D).
  multigrid   same inputs and tolerance, preconditioner 1 vs 0: identical solution,
              far fewer iterations. The 2D baseline to beat is ~533.
  hydro3d     a real 3D two-phase hydrostatic column (32x32x8) built analytically:
              after the gravity kick the node must return it to rest and reproduce
              the analytic pressure jump, multigrid on.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_3d.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from validate_step import channels  # noqa: E402

IN = os.path.join(HERE, "out", "step3_in")
OUT = os.path.join(HERE, "out", "step3_out")
NX, NY, DT = 160, 80, 0.02


def run_node(indir, outdir, **ports):
    os.makedirs(outdir, exist_ok=True)
    args = [os.path.join(HERE, "run_solve_step3.bat")]
    for k, v in ports.items():
        args += ["--set-port", k, str(v)]
    for n in ("u_mass", "u_mom", "u_phase", "v_mass", "v_mom", "v_phase",
              "w_mass", "w_mom", "w_phase", "particle_phase", "positions", "velocities"):
        args += ["--set-port", "path_" + n, os.path.join(indir, n + ".npy").replace("\\", "/")]
    for n in ("out_u", "out_v", "out_w", "pressure", "out_positions", "out_velocities",
              "out_escaped"):
        args += ["--set-port", "path_" + n, os.path.join(outdir, n + ".npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    stats = {}
    for line in res.stdout.splitlines():
        if "Output port '" in line and "ok_" not in line:
            name = line.split("'")[1]
            stats[name] = float(line.rsplit(")", 1)[-1].strip())
    if res.returncode != 0 or "iterations_used" not in stats:
        sys.stderr.write(res.stdout[-2500:] + res.stderr[-1500:])
        raise SystemExit("bifcmd step_3d failed")
    return stats


def make_slab_inputs():
    """The 2D dam-break mid-state, packaged for the 3D node at nz=1."""
    os.makedirs(IN, exist_ok=True)
    sim = Sim(NX, NY)
    sim.seed(lambda x, y: (x < 40) & (y < 56))
    sim.calibrate()
    sim.run(12, 0.06)
    ch = {k: v.astype(np.float32) for k, v in channels(sim).items()}
    for k, v in ch.items():
        np.save(os.path.join(IN, k + ".npy"), v.ravel())
    nw = NX * NY * 2
    for k in ("w_mass", "w_mom", "w_phase"):
        np.save(os.path.join(IN, k + ".npy"), np.zeros(nw, np.float32))
    np.save(os.path.join(IN, "positions.npy"),
            np.concatenate([sim.pos, np.full((len(sim.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(IN, "velocities.npy"),
            np.concatenate([sim.vel, np.zeros((len(sim.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(IN, "particle_phase.npy"), sim.typ.astype(np.float32))
    return sim, ch


def python_step(sim, ch):
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
    ref.mu, ref.mv = np.empty_like(ref.u), np.empty_like(ref.v)
    ref.phase = lambda raw: phi_u if raw is ref.mu else phi_v
    ref.project(DT)
    ref.g2p(u_star, v_star)
    ref.advect(DT)
    return ref


def main():
    sim, ch = make_slab_inputs()
    ref = python_step(sim, ch)

    print("=== thin_slab (nz=1 lockstep against the 2D reference)")
    s0 = run_node(IN, OUT, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=0)
    bu = np.load(os.path.join(OUT, "out_u.npy")).reshape(NX + 1, NY)
    bv = np.load(os.path.join(OUT, "out_v.npy")).reshape(NX, NY + 1)
    bp = np.load(os.path.join(OUT, "out_positions.npy"))[:, :2]
    bw = np.load(os.path.join(OUT, "out_velocities.npy"))[:, :2]
    checks = [("face u", np.abs(bu - ref.u).max()), ("face v", np.abs(bv - ref.v).max()),
              ("positions", np.abs(bp - ref.pos).max()), ("velocities", np.abs(bw - ref.vel).max())]
    ok = True
    print(f"  iterations: node {s0['iterations_used']:.0f}, python {ref.iters}")
    for name, err in checks:
        good = err < 1e-3
        ok &= good
        print(f"  max |diff| {name}: {err:.3e}  {'OK' if good else 'FAIL'}")
    print("thin_slab:", "PASS" if ok else "FAIL")

    print("=== multigrid (same system, preconditioner 1 vs 0)")
    out_mg = OUT + "_mg"
    s1 = run_node(IN, out_mg, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=1)
    mu = np.load(os.path.join(out_mg, "out_u.npy")).reshape(NX + 1, NY)
    mv = np.load(os.path.join(out_mg, "out_v.npy")).reshape(NX, NY + 1)
    du = max(np.abs(mu - bu).max(), np.abs(mv - bv).max())
    print(f"  iterations: jacobi {s0['iterations_used']:.0f} -> multigrid {s1['iterations_used']:.0f}")
    print(f"  residuals: {s0['final_residual']:.2e} vs {s1['final_residual']:.2e}; "
          f"solution difference {du:.3e}")
    ok_mg = s1["iterations_used"] < 100 and du < 1e-3
    ok &= ok_mg
    print("multigrid:", "PASS" if ok_mg else "FAIL")

    print("=== hydro3d (32x32x8 two-phase column, analytic channels, multigrid)")
    nx, ny, nz = 32, 32, 8
    d3 = os.path.join(HERE, "out", "step3_hydro")
    os.makedirs(d3, exist_ok=True)
    rho_l, rho_g, g = 1000.0, 1.0, -9.8
    half = ny // 2

    # y positions of face centres decide phi: u,w faces sit at cell-centre y (j+0.5);
    # v faces sit at integer y = j.
    phi_u = np.zeros((nx + 1, ny, nz))
    phi_u[:, :half, :] = 1.0
    phi_w = np.zeros((nx, ny, nz + 1))
    phi_w[:, :half, :] = 1.0
    phi_v = np.zeros((nx, ny + 1, nz))
    phi_v[:, :half, :] = 1.0
    phi_v[:, half, :] = 0.5
    for name, phi in (("u", phi_u), ("v", phi_v), ("w", phi_w)):
        rho = rho_g + (rho_l - rho_g) * phi
        np.save(os.path.join(d3, f"{name}_mass.npy"), rho.ravel().astype(np.float32))
        np.save(os.path.join(d3, f"{name}_mom.npy"), np.zeros(phi.size, np.float32))
        np.save(os.path.join(d3, f"{name}_phase.npy"), phi.ravel().astype(np.float32))
    probes = np.array([[nx / 2, 1.0, nz / 2], [nx / 2, ny - 1.0, nz / 2]], np.float32)
    np.save(os.path.join(d3, "positions.npy"), probes)
    np.save(os.path.join(d3, "velocities.npy"), np.zeros((2, 3), np.float32))
    np.save(os.path.join(d3, "particle_phase.npy"), np.array([1, 0], np.float32))
    s3 = run_node(d3, d3 + "_out", nx=nx, ny=ny, nz=nz, dt=1.0, preconditioner=1,
                  alpha_liquid=0.0, alpha_air=0.0)
    vel = np.load(os.path.join(d3 + "_out", "out_velocities.npy"))
    pr = np.load(os.path.join(d3 + "_out", "pressure.npy")).reshape(nx, ny, nz)
    jump = pr[nx // 2, 0, nz // 2] - pr[nx // 2, ny - 1, nz // 2]
    analytic = -g * (rho_l * half + rho_g * (ny - half - 1))
    still = np.abs(vel).max()
    print(f"  max |velocity| after projection: {still:.3e} (gravity kick was {abs(g):.1f})")
    print(f"  pressure jump: {jump:.0f} vs analytic ~{analytic:.0f} "
          f"({100 * jump / analytic:.1f} percent); iterations {s3['iterations_used']:.0f}")
    ok3 = still < 1e-3 and abs(jump / analytic - 1) < 0.05
    ok &= ok3
    print("hydro3d:", "PASS" if ok3 else "FAIL")

    print("OVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
