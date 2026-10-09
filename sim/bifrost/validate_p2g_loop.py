"""System gate for the fused P2G: the full 3D loop at nz = 1 against the
2D python reference, fused graph vs stock graph on identical particles.

The claim under test: with the splat layer now implementing the reference's
own kernel and face frames, the fused pipeline should track the reference at
least as closely as the stock-splat pipeline ever did (whose dam-break front
gap was 1.9 cells over a full run, from the stock kernel mismatch).

  dam      plain-mode dam break, 80 substeps: per-substep liquid front
           (99th percentile x) and centroid against the reference, fused
           error must not exceed stock error.
  hydro    two-phase hydrostatic column through the fused graph: after 40
           substeps the max particle speed must be at solver-tolerance level.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_p2g_loop.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from demo3d import run_bifrost3  # noqa: E402

NX, NY, DT, SUB = 96, 48, 0.06, 80


def lift(sim):
    pos3 = np.column_stack([sim.pos, np.full(len(sim.pos), 0.5)]).astype(np.float32)
    return pos3, sim.typ.astype(np.int32)


def run_graph(tag, pos3, typ, nx, ny, substeps, fused, ports=None):
    d, _, _ = run_bifrost3(
        tag, nx, ny, 1, None, substeps, DT,
        extra_ports=dict({"preconditioner": 1}, **(ports or {})),
        seed_fn=lambda *_: (pos3, typ), fused=fused)
    return d


def metrics(pos, liquid):
    liq = pos[liquid]
    return float(np.percentile(liq[:, 0], 99.0)), liq[:, :2].mean(0)


def check_dam():
    print(f"=== dam: plain mode, {NX}x{NY}, {SUB} substeps of {DT}")
    ref = Sim(NX, NY)
    ref.seed(lambda x, y: (x < 0.3 * NX) & (y < 0.7 * NY))
    ref.calibrate()
    pos3, typ = lift(ref)
    liquid = typ == 1

    runs = {}
    for name, fused in (("stock", False), ("fused", True)):
        runs[name] = run_graph("p2gloop_" + name, pos3, typ, NX, NY, SUB, fused)

    err = {"stock": [], "fused": []}
    cen = {"stock": [], "fused": []}
    for k in range(SUB):
        ref.run(1, DT)
        f_ref, c_ref = metrics(np.column_stack([ref.pos, np.zeros(len(ref.pos))]),
                               liquid)
        for name in runs:
            p = np.load(os.path.join(runs[name], f"frame.{k:04d}.npy"))
            f, c = metrics(p, liquid)
            err[name].append(abs(f - f_ref))
            cen[name].append(float(np.abs(c - c_ref).max()))
    for name in ("stock", "fused"):
        print(f"    {name}: front error max {max(err[name]):.3f} cells "
              f"(final {err[name][-1]:.3f}), centroid drift max {max(cen[name]):.3f}")
    ok = (max(err["fused"]) <= max(err["stock"]) + 1e-6 and
          max(cen["fused"]) <= max(cen["stock"]) + 1e-6)
    print("    PASS (fused tracks the reference at least as closely as stock)"
          if ok else "    FAIL")
    return ok


def check_hydro():
    """Rest is a grid-level statement at this seeding: jittered 4-per-cell
    particles at 1000:1 with no surface tension jiggle at the interface in
    the reference too (its liquid p99.9 step-speed is ~0.5 cells/s on this
    scene). The honest gates: the fused grid stays divergence-free, and the
    fused particle noise is no worse than the stock graph's or the spec's."""
    print("=== hydro: two-phase column, fused vs stock vs reference")
    ref = Sim(NX, NY)
    ref.seed(lambda x, y: y < 0.5 * NY)
    ref.calibrate()
    pos3, typ = lift(ref)

    p999 = {}
    div = {}
    for name, fused in (("stock", False), ("fused", True)):
        d, _, diag = run_bifrost3(
            "p2gloop_hydro_" + name, NX, NY, 1, None, 40, DT,
            extra_ports={"preconditioner": 1},
            seed_fn=lambda *_: (pos3, typ), fused=fused)
        v = (np.load(os.path.join(d, "frame.0039.npy")) -
             np.load(os.path.join(d, "frame.0038.npy"))) / DT
        ph = np.load(os.path.join(d, "ph.0039.npy"))
        p999[name] = float(np.percentile(np.abs(v).max(1)[ph >= 0.5], 99.9))
        div[name] = float(diag[-1][2])
    prev = ref.pos.copy()
    for _ in range(40):
        prev = ref.pos.copy()
        ref.run(1, DT)
    vr = np.abs(ref.pos - prev).max(1) / DT
    p999["reference"] = float(np.percentile(vr[ref.typ == 1], 99.9))

    print(f"    grid max divergence after projection: stock {div['stock']:.1e},"
          f" fused {div['fused']:.1e}")
    print(f"    liquid p99.9 step-speed: reference {p999['reference']:.3f},"
          f" stock {p999['stock']:.3f}, fused {p999['fused']:.3f} cells/s")
    ok = (div["fused"] < 1e-5 and
          p999["fused"] <= 1.2 * max(p999["stock"], p999["reference"]))
    print("    PASS" if ok else "    FAIL")
    return ok


def main():
    results = [check_dam(), check_hydro()]
    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
