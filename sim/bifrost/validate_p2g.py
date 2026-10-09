"""Unit gate for the fused P2G node (PFFlip::Solve::p2g_3d).

The executable spec is the reference's splat (pfflip2d.py _face_frames +
_splat_group: Eq. 6 kernel, base = floor(p' + 0.5) face frames). Three checks:

  flat2d   nz=1 against the reference operator itself, all modes: plain,
           st (wt channel), adapt (two tiers, separate radii, ws channel).
           The reference accumulates raw sums; the node emits the stock-path
           means, so expectations are sums/(K + eps) computed from the
           reference's own output.
  rand3d   a from-scratch 3D numpy implementation of the same spec on a
           randomized state, all 30 channels, adapt + st on.
  permute  x<->z permutation of the 3D state must transpose u<->w channels
           exactly (the axis-bias lesson from verify_engineering check A).

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_p2g.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402

D = os.path.join(HERE, "out", "p2g_gate")
CHANS = [g + s for g in ("u", "v", "w", "u2", "v2", "w2")
         for s in ("_mass", "_mom", "_phase", "_wt", "_ws")]
EM, EW = 1e-6, 16.0


def run_node(tag, nx, ny, nz, adapt, st, arrs):
    d_in = os.path.join(D, tag + "_in")
    d_out = os.path.join(D, tag + "_out")
    os.makedirs(d_in, exist_ok=True)
    os.makedirs(d_out, exist_ok=True)
    for k, v in arrs.items():
        np.save(os.path.join(d_in, k + ".npy"), v)
    args = [os.path.join(HERE, "run_p2g_test.bat"),
            "--set-port", "nx", str(nx), "--set-port", "ny", str(ny),
            "--set-port", "nz", str(nz), "--set-port", "adapt", str(adapt),
            "--set-port", "st", str(st)]
    for k in arrs:
        args += ["--set-port", "path_" + k,
                 os.path.join(d_in, k + ".npy").replace("\\", "/")]
    for c in CHANS:
        args += ["--set-port", "path_" + c,
                 os.path.join(d_out, c + ".npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2500:] + res.stderr[-1500:])
        raise SystemExit("bifcmd p2g_test failed for " + tag)
    out = {}
    for c in CHANS:
        p = os.path.join(d_out, c + ".npy")
        if os.path.exists(p):
            out[c] = np.load(p)
    return out


def make_arrays(pos3, vel3, mass, phase, wt=None, scale=None):
    n = len(pos3)
    z = np.zeros(n, np.float32)
    return {"positions": pos3.astype(np.float32),
            "scale": (scale if scale is not None else np.ones(n)).astype(np.float32),
            "mass": mass.astype(np.float32),
            "momx": (mass * vel3[:, 0]).astype(np.float32),
            "momy": (mass * vel3[:, 1]).astype(np.float32),
            "momz": (mass * vel3[:, 2]).astype(np.float32),
            "wt": (wt if wt is not None else z).astype(np.float32),
            "phase": phase.astype(np.float32)}


def ref_means(sim, face_of, shape, values, sel=None, r=1.0):
    """Reference raw sums -> the node's mean semantics, optionally one tier."""
    ones = np.ones(len(face_of[0]))
    vals = [ones] + values
    if sel is not None:
        face_of = (face_of[0][sel], face_of[1][sel])
        vals = [np.asarray(v)[sel] for v in vals]
    out = [np.zeros(shape) for _ in vals]
    Sim._splat_group(out, shape, face_of, vals, r)
    K = out[0]
    return K, [a / (K + EM) for a in out[1:]]


def numpy_p2g3(nx, ny, nz, arrs, adapt, st):
    """Independent 3D implementation of the spec, dense loops over a stencil."""
    dims = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz), "w": (nx, ny, nz + 1)}
    offs = {"u": (0.0, 0.5, 0.5), "v": (0.5, 0.0, 0.5), "w": (0.5, 0.5, 0.0)}
    pos = arrs["positions"].astype(float)
    vals = {k: arrs[k].astype(float)
            for k in ("mass", "momx", "momy", "momz", "wt", "phase")}
    scale = arrs["scale"].astype(float)
    out = {}
    for g in ("u", "v", "w"):
        mom = {"u": "momx", "v": "momy", "w": "momz"}[g]
        for tier, r in ((1, 1.0), (2, 2.0)):
            sel = (scale >= 1.5) if tier == 2 else (scale < 1.5)
            if not adapt:
                sel = np.ones(len(pos), bool) if tier == 1 else np.zeros(len(pos), bool)
            p = pos[sel] - np.array(offs[g])
            b = np.floor(p + 0.5).astype(int)
            f = p - b
            span = int(np.ceil(r))
            acc = {k: np.zeros(dims[g]) for k in ("K", "m", "mo", "ph", "wtv")}
            src = {"m": vals["mass"][sel], "mo": vals[mom][sel],
                   "ph": vals["phase"][sel], "wtv": vals["wt"][sel]}
            for di in range(-span, span + 1):
                for dj in range(-span, span + 1):
                    for dk in range(-span, span + 1):
                        ii, jj, kk = b[:, 0] + di, b[:, 1] + dj, b[:, 2] + dk
                        ok = ((ii >= 0) & (ii < dims[g][0]) & (jj >= 0) &
                              (jj < dims[g][1]) & (kk >= 0) & (kk < dims[g][2]))
                        d2 = ((f[:, 0] - di) ** 2 + (f[:, 1] - dj) ** 2 +
                              (f[:, 2] - dk) ** 2) / r ** 2
                        w3 = np.where(ok, np.maximum(1.0 - d2, 0.0) ** 3, 0.0)
                        idx = (np.clip(ii, 0, dims[g][0] - 1),
                               np.clip(jj, 0, dims[g][1] - 1),
                               np.clip(kk, 0, dims[g][2] - 1))
                        np.add.at(acc["K"], idx, w3)
                        for k2 in src:
                            np.add.at(acc[k2[0:2] if k2 != "wtv" else "wtv"],
                                      idx, w3 * src[k2])
            tag = g if tier == 1 else g + "2"
            K = acc["K"]
            out[tag + "_mass"] = (acc["m"] / (K + EM)).ravel()
            out[tag + "_mom"] = (acc["mo"] / (K + EM)).ravel()
            out[tag + "_phase"] = (acc["ph"] / (K + EM)).ravel()
            if st:
                out[tag + "_wt"] = (acc["wtv"] / (K + EM)).ravel()
            if adapt:
                out[tag + "_ws"] = (K / (K + EW)).ravel()
    return out


def check_flat2d():
    print("=== flat2d: nz=1 against the reference operator, all modes")
    NX, NY = 48, 32
    rng = np.random.default_rng(7)
    n = 6000
    pos3 = np.column_stack([rng.uniform(0.6, NX - 0.6, n),
                            rng.uniform(0.6, NY - 0.6, n),
                            np.full(n, 0.5)]).astype(np.float32)
    # the node's port is float32; the reference must see the same rounding or
    # particles within an ulp of a floor(p + 0.5) boundary shift their stencil
    pos2 = pos3[:, :2].astype(float)
    vel3 = rng.normal(0, 3, (n, 3))
    mass = rng.uniform(0.5, 1200.0, n)
    phase = rng.integers(0, 2, n).astype(float)
    wt = rng.uniform(0.1, 2.1875, n)
    scale = np.where(rng.random(n) < 0.3, 2.0, 1.0)

    sim = Sim(NX, NY)
    sim.pos = pos2
    fu, fv = sim._face_frames()
    worst = 0.0
    for mode, adapt, st in (("plain", 0, 0), ("st", 0, 1), ("adapt", 1, 0),
                            ("full", 1, 1)):
        arrs = make_arrays(pos3, vel3, mass, phase,
                           wt=wt if st else None,
                           scale=scale if adapt else None)
        node = run_node("flat_" + mode, NX, NY, 1, adapt, st, arrs)
        for g, fo, shape, mom in (("u", fu, (NX + 1, NY), mass * vel3[:, 0]),
                                  ("v", fv, (NX, NY + 1), mass * vel3[:, 1])):
            tiers = [(g, scale < 1.5 if adapt else None, 1.0)]
            if adapt:
                tiers.append((g + "2", scale >= 1.5, 2.0))
            for tag, sel, r in tiers:
                vals = [mass, mom, phase]
                if st:
                    vals.append(wt)
                K, means = ref_means(sim, fo, shape, vals, sel, r)
                names = ["_mass", "_mom", "_phase"] + (["_wt"] if st else [])
                for nm, exp in zip(names, means):
                    got = node[tag + nm].reshape(shape)
                    worst = max(worst, float(np.abs(got - exp).max()))
                if adapt:
                    ws = node[tag + "_ws"].reshape(shape)
                    worst = max(worst, float(np.abs(ws - K / (K + EW)).max()))
    print(f"    worst |node - reference| over all modes/channels: {worst:.2e}")
    ok = worst < 5e-4
    print("    PASS" if ok else "    FAIL")
    return ok


def check_rand3d():
    print("=== rand3d: independent numpy 3D splat, adapt + st on")
    NX, NY, NZ = 24, 16, 12
    rng = np.random.default_rng(11)
    n = 40000
    pos3 = np.column_stack([rng.uniform(0.6, NX - 0.6, n),
                            rng.uniform(0.6, NY - 0.6, n),
                            rng.uniform(0.6, NZ - 0.6, n)])
    vel3 = rng.normal(0, 3, (n, 3))
    mass = rng.uniform(0.5, 1200.0, n)
    phase = rng.integers(0, 2, n).astype(float)
    wt = rng.uniform(0.1, 2.1875, n)
    scale = np.where(rng.random(n) < 0.3, 2.0, 1.0)
    arrs = make_arrays(pos3, vel3, mass, phase, wt=wt, scale=scale)
    node = run_node("rand3d", NX, NY, NZ, 1, 1, arrs)
    ref = numpy_p2g3(NX, NY, NZ, arrs, 1, 1)
    worst = max(float(np.abs(node[c] - ref[c]).max()) for c in ref)
    print(f"    worst |node - numpy| over all 30 channels: {worst:.2e}")
    ok = worst < 5e-4
    print("    PASS" if ok else "    FAIL")
    return ok


def check_permute():
    print("=== permute: x<->z state permutation must transpose u<->w channels")
    NX, NY, NZ = 20, 14, 20
    rng = np.random.default_rng(13)
    n = 30000
    pos3 = np.column_stack([rng.uniform(0.6, NX - 0.6, n),
                            rng.uniform(0.6, NY - 0.6, n),
                            rng.uniform(0.6, NZ - 0.6, n)])
    vel3 = rng.normal(0, 3, (n, 3))
    mass = rng.uniform(0.5, 1200.0, n)
    phase = rng.integers(0, 2, n).astype(float)
    wt = rng.uniform(0.1, 2.1875, n)
    scale = np.where(rng.random(n) < 0.3, 2.0, 1.0)
    a = make_arrays(pos3, vel3, mass, phase, wt=wt, scale=scale)
    b = dict(a)
    b["positions"] = a["positions"][:, [2, 1, 0]]
    b["momx"], b["momz"] = a["momz"], a["momx"]
    na = run_node("perm_a", NX, NY, NZ, 1, 1, a)
    nb = run_node("perm_b", NZ, NY, NX, 1, 1, b)
    worst = 0.0
    for t in ("", "2"):
        for s in ("_mass", "_mom", "_phase", "_wt", "_ws"):
            ua = na["u" + t + s].reshape(NX + 1, NY, NZ)
            wb = nb["w" + t + s].reshape(NZ, NY, NX + 1)
            worst = max(worst, float(np.abs(ua - wb.transpose(2, 1, 0)).max()))
    print(f"    worst |u(state) - w(permuted state)^T|: {worst:.2e}")
    ok = worst < 2e-6
    print("    PASS (float rounding)" if ok else "    FAIL")
    return ok


def main():
    results = [check_flat2d(), check_rand3d(), check_permute()]
    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
