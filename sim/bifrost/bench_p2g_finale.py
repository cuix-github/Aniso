"""Finale-scale speed receipts for the fused P2G, directly comparable to the
spike-1 profile (Aniso PR #37): same 256x64x128 production config, resumed
from the same mid-bore overnight state (6.8M particles after adaptive
merging), walls differenced across two run lengths so startup cancels.

Baseline on record from spike 1 (stock graph, same state, same machine):
4 substeps 393.3 s, 12 substeps 1164.4 s -> 96.4 s per substep, of which
the step node is 5.6 s and the pressure solve 0.9 s. This script re-runs
the stock 4-substep point as an environment sanity check, then measures the
fused graph at 4 and 12 substeps.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe bench_p2g_finale.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from run_bore_suite import scene  # noqa: E402
from demo3d import lattice  # noqa: E402

NX, NY, NZ, DT = 256, 64, 128, 0.12
SRC = os.path.join(HERE, "out", "loop3_D_finale_c1")
STOCK_BASELINE = {"wall4_s": 393.3, "wall12_s": 1164.4, "per_substep_s": 96.4}


def init_arrays():
    pos = np.load(os.path.join(SRC, "final.npy")).astype(np.float32)
    ph = np.load(os.path.join(SRC, "final_phase.npy")).astype(np.float32)
    _, pmin, pmax, *_ = scene(NX, NY, NZ)
    a = {"positions": pos, "particle_phase": ph, "phase_st": ph,
         "obs_min": pmin.astype(np.float32), "obs_max": pmax.astype(np.float32),
         "probes_u": lattice(NX + 1, NY, NZ), "probes_v": lattice(NX, NY + 1, NZ),
         "probes_w": lattice(NX, NY, NZ + 1)}
    for k in ("velocities", "mass", "momx", "momy", "momz", "tau", "wt", "wtph",
              "scale"):
        a[k] = np.load(os.path.join(SRC, f"final_{k}.npy")).astype(np.float32)
    return a


def run(tag, bat, substeps, arrs, ports):
    d = os.path.join(HERE, "out", "loop3_" + tag)
    os.makedirs(d, exist_ok=True)
    for k, v in arrs.items():
        np.save(os.path.join(d, k + ".npy"), v)
    args = [os.path.join(HERE, bat),
            "--set-port", "nx", str(NX), "--set-port", "ny", str(NY),
            "--set-port", "nz", str(NZ), "--set-port", "dt", str(DT),
            "--set-port", "substeps", str(substeps),
            "--set-port", "pos_pattern", (d + "/frame.####").replace("\\", "/"),
            "--set-port", "diag_pattern", (d + "/diag.####").replace("\\", "/"),
            "--set-port", "path_final_positions", (d + "/final.npy").replace("\\", "/"),
            "--set-port", "esc_pattern", (d + "/esc.####").replace("\\", "/"),
            "--set-port", "scl_pattern", (d + "/scl.####").replace("\\", "/"),
            "--set-port", "ph_pattern", (d + "/ph.####").replace("\\", "/")]
    for k, v in ports.items():
        args += ["--set-port", k, str(v)]
    for k in arrs:
        args += ["--set-port", "path_" + k,
                 os.path.join(d, k + ".npy").replace("\\", "/")]
    t0 = time.time()
    res = subprocess.run(args, capture_output=True, text=True)
    wall = time.time() - t0
    ok = res.returncode == 0 and os.path.exists(
        os.path.join(d, f"frame.{substeps - 1:04d}.npy"))
    if not ok:
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-2000:])
        raise SystemExit(f"run {tag} failed")
    shutil.rmtree(d, ignore_errors=True)
    return wall


def main():
    cal = Sim(64, 64, adapt=True, st=True)
    cal.seed(lambda x, y: np.ones_like(x, dtype=bool))
    cal.calibrate()
    ports = {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1,
             "rho0_face": cal.rho0_face, "coarse_gain": cal.coarse_gain,
             "ws_epsilon": 16}
    arrs = init_arrays()
    print(f"init particles: {len(arrs['positions'])}", flush=True)

    stock4 = run("bench_stock4", "run_sim_3d.bat", 4, arrs, ports)
    print(f"stock 4 substeps: {stock4:.1f}s "
          f"(spike-1 recorded {STOCK_BASELINE['wall4_s']}s)", flush=True)
    fused4 = run("bench_fused4", "run_sim_3d_fused.bat", 4, arrs, ports)
    print(f"fused 4 substeps: {fused4:.1f}s", flush=True)
    fused12 = run("bench_fused12", "run_sim_3d_fused.bat", 12, arrs, ports)
    print(f"fused 12 substeps: {fused12:.1f}s", flush=True)

    per_fused = (fused12 - fused4) / 8.0
    out = {"grid": f"{NX}x{NY}x{NZ}", "particles": len(arrs["positions"]),
           "stock_per_substep_s": STOCK_BASELINE["per_substep_s"],
           "stock4_sanity_s": round(stock4, 1),
           "fused4_s": round(fused4, 1), "fused12_s": round(fused12, 1),
           "fused_per_substep_s": round(per_fused, 2),
           "speedup": round(STOCK_BASELINE["per_substep_s"] / per_fused, 1)}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(HERE, "out", "bench_p2g_finale.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
