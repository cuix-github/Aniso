"""Compare two built operator packs through the real Bifrost graphs.

Run packs sequentially: --mode gate checks every saved array bit-for-bit,
--mode bench resumes the canonical finale at 4/12 steps, and --mode demo3d
runs the 20-second 3D bore at 20/167 steps (demo is its 2D counterpart).
Each invocation gets a new output
directory; old results cannot accidentally satisfy a failed run. A copied
main-build pack is the independent pre-change baseline, not a serial switch
inside the new implementation. Nothing in this driver deletes results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference2d"))
from pfflip2d import Sim
from demo3d import lattice
from generate_sim_graph import P
from validate_p2g import CHANS, make_arrays


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def full_state_graph(root, raw_dump=False):
    graph = json.loads((HERE / "sim_3d_fused.json").read_text())
    top = graph["compounds"][0]
    states = top["compounds"][0]["iterateCompound"]["ports"]
    if raw_dump:
        # Diagnostic output only: distinguish the actual solver positions
        # from ST's time-resynchronized display positions. Keep both intact.
        body = top["compounds"][0]
        top["ports"].append(P("raw_pos_pattern", "input", "string", ""))
        body["ports"].append(P("raw_pos_pattern", "input", "string", ""))
        top["connections"].append({"source": ".raw_pos_pattern", "target": "loop.raw_pos_pattern"})
        body["compoundNodes"] += [
            {"nodeName": "w_raw", "nodeType": "File::NumPy::write_NumPy"},
            {"nodeName": "ok_raw", "nodeType": "Core::Type_Conversion::to_float"}]
        next(n for n in body["compoundNodes"] if n["nodeName"] == "ok_acc")["multiInPortNames"].append("raw")
        body["connections"] += [
            {"source": "step.out_positions", "target": "w_raw.data"},
            {"source": ".raw_pos_pattern", "target": "w_raw.file_path"},
            {"source": ".current_index", "target": "w_raw.frame"},
            {"source": "w_raw.success", "target": "ok_raw.from"},
            {"source": "ok_raw.float", "target": "ok_acc.first.raw"}]
        body["values"] += [
            {"valueName": "w_raw.overwrite", "valueType": "bool", "value": "true"},
            {"valueName": "w_raw.create_directories", "valueType": "bool", "value": "true"}]
    names = []
    for state in states:
        if state["portKind"] != "state" or state["inputPortName"] == "ok_in":
            continue
        name = state["outputPortName"]
        node = "save_" + name
        names.append(name)
        top["ports"] += [P("save_" + name, "input", "string", ""),
                         P("saved_" + name, "output", "bool")]
        top["compoundNodes"].append({"nodeName": node, "nodeType": "File::NumPy::write_NumPy"})
        top["connections"] += [
            {"source": "loop." + name, "target": node + ".data"},
            {"source": ".save_" + name, "target": node + ".file_path"},
            {"source": node + ".success", "target": ".saved_" + name}]
        top["values"] += [{"valueName": node + ".overwrite", "valueType": "bool", "value": "true"},
                          {"valueName": node + ".create_directories", "valueType": "bool", "value": "true"}]
    path = root / "full_state_graph.json"
    path.write_text(json.dumps(graph))
    return path, names


class Experiment:
    def __init__(self, args):
        self.root = HERE / "out" / "threading" / args.label
        self.root.mkdir(parents=True, exist_ok=False)
        self.pack = Path(args.pack).resolve()
        self.bifrost = Path(args.bifrost)
        self.env = os.environ.copy()
        self.env.update(BIFROST_LIB_CONFIG_FILES=str(self.pack), OMP_NUM_THREADS=str(args.threads),
                        OMP_DYNAMIC="FALSE", PFFLIP_PROFILE="1")
        self.env["PATH"] = str(self.bifrost / "bin") + os.pathsep + str(self.bifrost / "thirdparty/bin") + os.pathsep + self.env["PATH"]
        self.raw_dump = args.mode == "demo3d"
        self.graph, self.state_names = full_state_graph(self.root, self.raw_dump)
        self.result = {"mode": args.mode, "threads": args.threads, "pack": str(self.pack),
                       "dll_sha256": digest(self.pack.parent / "lib/PFFlipSpikeOps.dll"),
                       "cases": {}}

    def inputs(self, name, arrays):
        d = self.root / name
        d.mkdir()
        ports = {}
        for k, v in arrays.items():
            p = d / (k + ".npy")
            np.save(p, v)
            ports["path_" + k] = p.as_posix()
        return ports

    def invoke(self, tag, graph, inputs, ports, outputs):
        d = self.root / tag
        d.mkdir()
        all_ports = dict(inputs, **ports)
        all_ports.update({k: (d / v).as_posix() for k, v in outputs.items()})
        cmd = [str(self.bifrost / "bin/bifcmd.exe"), str(graph)]
        for k, v in all_ports.items():
            cmd += ["--set-port", k, str(v)]
        start = time.perf_counter()
        # Timestamp flushed profiling lines in this process. Consecutive step
        # completions give a steady interval from ONE run, so a phase pie need
        # not pretend that startup/load cancels perfectly between two runs.
        run = subprocess.Popen(cmd, env=self.env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        lines, profiles = [], []
        for line in run.stdout:
            lines.append(line)
            if line.startswith(("PFPROF ", "P2GPROF ")):
                profiles.append({"kind": line.split()[0], "received_s": time.perf_counter() - start,
                                 **{k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)(?:\s|$)", line)}})
        run.wait()
        wall = time.perf_counter() - start
        output = "".join(lines)
        (d / "bifcmd.log").write_text(output)
        if run.returncode:
            raise RuntimeError(f"{tag}: bifcmd failed; see {d}")
        arrays = sorted(d.glob("*.npy"))
        if not arrays:
            raise RuntimeError(f"{tag}: no outputs")
        hashes = {}
        for p in arrays:
            a = np.load(p, mmap_mode="r")
            if not np.isfinite(a).all():
                raise AssertionError(f"non-finite output: {p}")
            hashes[p.name] = digest(p)
        row = {"wall_s": wall, "arrays": hashes, "profiles": profiles}
        self.result["cases"][tag] = row
        (self.root / "receipt.json").write_text(json.dumps(self.result, indent=2))
        print(f"{tag}: {wall:.3f}s, {len(arrays)} finite arrays", flush=True)
        return row

    def loop(self, tag, inputs, shape, steps, ports, dump_every=1, full_state=True):
        outputs = {"pos_pattern": "frame.####", "diag_pattern": "diag.####",
                   "esc_pattern": "esc.####", "scl_pattern": "scl.####", "ph_pattern": "ph.####",
                   "path_final_positions": "final.npy"}
        if full_state:
            outputs.update({"save_" + n: "state_" + n + ".npy" for n in self.state_names})
        if self.raw_dump:
            outputs["raw_pos_pattern"] = "raw.####"
        row = self.invoke(tag, self.graph if full_state else HERE / "sim_3d_fused.json", inputs,
                          dict(nx=shape[0], ny=shape[1], nz=shape[2], substeps=steps,
                               preconditioner=1, dump_every=dump_every, **ports), outputs)
        expected = {"final.npy"} | {f"diag.{i:04d}.npy" for i in range(steps)}
        expected |= {f"{kind}.{i:04d}.npy" for kind in ("frame", "esc", "scl", "ph")
                     for i in range(0, steps, dump_every)}
        if full_state:
            expected |= {"state_" + n + ".npy" for n in self.state_names}
        if self.raw_dump:
            expected |= {f"raw.{i:04d}.npy" for i in range(steps)}
        assert expected == set(row["arrays"]), (tag, expected ^ set(row["arrays"]))
        return row


def initial_state(shape, pos, typ, obstacles=None):
    n = len(pos)
    zero = np.zeros(n, np.float32)
    far = np.array([[-100, -100, -100]], np.float32)
    return {"positions": pos.astype(np.float32), "velocities": np.zeros_like(pos, dtype=np.float32),
            "particle_phase": typ.astype(np.float32), "phase_st": typ.astype(np.float32),
            "mass": np.where(typ == 1, 1000, 1).astype(np.float32),
            "momx": zero, "momy": zero, "momz": zero, "tau": zero,
            "wt": np.ones(n, np.float32), "wtph": typ.astype(np.float32),
            "scale": np.ones(n, np.float32),
            "obs_min": obstacles[0] if obstacles else far,
            "obs_max": obstacles[1] if obstacles else far + 1,
            "probes_u": lattice(shape[0] + 1, shape[1], shape[2]),
            "probes_v": lattice(shape[0], shape[1] + 1, shape[2]),
            "probes_w": lattice(shape[0], shape[1], shape[2] + 1)}


def calibration():
    sim = Sim(64, 64, adapt=True, st=True)
    sim.seed(lambda x, y: np.ones_like(x, dtype=bool))
    sim.calibrate()
    return {"rho0_face": sim.rho0_face, "coarse_gain": sim.coarse_gain, "ws_epsilon": 16}


def gate(exp):
    rng = np.random.default_rng(711)
    for nz in (1, 12):
        shape = (24, 16, nz)
        n = 20000
        pos = rng.uniform([-2, -2, -2], np.array(shape) + 2, (n, 3)).astype(np.float32)
        if nz == 1:
            pos[:, 2] = 0.5
        # Exact and adjacent float32 values at ownership boundaries, plus
        # out-of-domain particles. Shuffle so spatial order != input order.
        b = np.tile(np.arange(0, 25, 4, dtype=np.float32), 30)
        pos[:len(b), 0] = b
        pos[len(b):2*len(b), 0] = np.nextafter(b, np.float32(-np.inf))
        pos[2*len(b):3*len(b), 0] = np.nextafter(b, np.float32(np.inf))
        rng.shuffle(pos)
        arr = make_arrays(pos, rng.normal(0, 5, (n, 3)), rng.uniform(0.1, 1200, n),
                          rng.integers(0, 2, n), rng.uniform(0.1, 2.2, n),
                          np.where(rng.random(n) < 0.4, 2, 1))
        inp = exp.inputs(f"p2g_input_{nz}", arr)
        for adapt, st in ((0, 0), (0, 1), (1, 0), (1, 1)):
            row = exp.invoke(f"p2g_{nz}_{adapt}_{st}", HERE / "p2g_test.json", inp,
                             dict(nx=24, ny=16, nz=nz, adapt=adapt, st=st),
                             {"path_" + c: c + ".npy" for c in CHANS})
            # Bifrost's NumPy writer omits empty disabled-channel arrays.
            suffixes = ["_mass", "_mom", "_phase"] + (["_wt"] if st else []) + (["_ws"] if adapt else [])
            grids = ["u", "v", "w"] + (["u2", "v2", "w2"] if adapt else [])
            assert set(row["arrays"]) == {g + s + ".npy" for g in grids for s in suffixes}
    for n in (0, 31):
        arr = make_arrays(np.full((n, 3), 0.5), np.zeros((n, 3)), np.ones(n), np.zeros(n))
        exp.invoke(f"p2g_small_{n}", HERE / "p2g_test.json", exp.inputs(f"small_input_{n}", arr),
                   dict(nx=2, ny=3, nz=1, adapt=1, st=1), {"path_" + c: c + ".npy" for c in CHANS})
    cal = calibration()
    from run_bore_suite import scene, seed_no_solids
    for nz in (1, 12):
        shape = (24, 16, nz)
        if nz == 1:
            sim = Sim(24, 16)
            sim.seed(lambda x, y: (y < 2) | ((x < 7) & (y < 12)))
            pos = np.column_stack([sim.pos, np.full(len(sim.pos), 0.5)])
            typ = sim.typ
            obs = (np.array([[13, 0, -1]], np.float32), np.array([[15, 7, 2]], np.float32))
            keep = ~((pos[:, 0] > 13) & (pos[:, 0] < 15) & (pos[:, 1] < 7))
            pos, typ = pos[keep], typ[keep]
        else:
            mask, pmin, pmax, *_ = scene(*shape)
            pos, typ = seed_no_solids(*shape, mask, pmin, pmax)
            obs = (pmin, pmax)
        inp = exp.inputs(f"loop_input_{nz}", initial_state(shape, pos, typ, obs))
        for adapt, st in ((0, 0), (0, 1), (1, 0), (1, 1)):
            exp.loop(f"loop_{nz}_{adapt}_{st}", inp, shape, 10,
                     dict(dt=0.12, adapt=adapt, st=st, escape=1, **cal))
        exp.loop(f"loop_{nz}_dump4", inp, shape, 10,
                 dict(dt=0.12, adapt=1, st=1, escape=1, **cal), dump_every=4)
        a = exp.result["cases"][f"loop_{nz}_1_1"]["arrays"]
        b = exp.result["cases"][f"loop_{nz}_dump4"]["arrays"]
        assert all(a[k] == v for k, v in b.items()), "dump cadence changes state"


def bench(exp):
    from bench_p2g_finale import init_arrays
    arr = init_arrays()
    inp = exp.inputs("finale_input", arr)
    exp.result["input_sha256"] = {p.name: digest(p) for p in (exp.root / "finale_input").glob("*.npy")}
    ports = dict(dt=0.12, st=1, adapt=1, escape=1, **calibration())
    rows = [exp.loop(f"finale_{n}", inp, (256, 64, 128), n, ports,
                     dump_every=1000, full_state=False) for n in (4, 12)]
    exp.result["per_substep_s"] = (rows[1]["wall_s"] - rows[0]["wall_s"]) / 8
    # Substeps 4..11 appear only in the long run, matching the wall difference.
    tail = rows[1]["profiles"][-16:]
    exp.result["phase_s"] = {key: sum(p.get(key, 0) for p in tail if p["kind"] == kind) / 8000
                             for key, kind in (("scatter", "P2GPROF"), ("pack", "P2GPROF"),
                                               ("asm", "PFPROF"), ("solve", "PFPROF"),
                                               ("corr", "PFPROF"), ("part", "PFPROF"))}
    steps = [p for p in rows[1]["profiles"] if p["kind"] == "PFPROF"]
    assert len(steps) == 12 and len(tail) == 16
    exp.result["steady_substep_s"] = (steps[-1]["received_s"] - steps[3]["received_s"]) / 8
    exp.result["phase_s"]["graph_and_dispatch"] = exp.result["steady_substep_s"] - sum(exp.result["phase_s"].values())
    print(json.dumps({k: exp.result[k] for k in ("per_substep_s", "steady_substep_s", "phase_s")}, indent=2), flush=True)


def flat(exp):
    # The tiny flat loop in gate deliberately exercises the small-input path.
    # This second flat case stays above the scatter parallelism threshold,
    # including after air coarsening, to test nz=1 in the actual threaded path.
    shape = (96, 48, 1)
    sim = Sim(*shape[:2])
    sim.seed(lambda x, y: (y < 6) | ((x < 28) & (y < 36)))
    pos = np.column_stack([sim.pos, np.full(len(sim.pos), 0.5)])
    inp = exp.inputs("flat_input", initial_state(shape, pos, sim.typ))
    cal = calibration()
    for adapt, st in ((0, 0), (0, 1), (1, 0), (1, 1)):
        exp.loop(f"flat_{adapt}_{st}", inp, shape, 10,
                 dict(dt=0.12, adapt=adapt, st=st, escape=1, **cal))
        d = exp.root / f"flat_{adapt}_{st}"
        assert min(len(np.load(d / f"frame.{k:04d}.npy", mmap_mode="r")) for k in range(10)) >= 4096


def demo(exp):
    from demo_p2g_speed import seed, PILLARS
    pos, typ = seed()
    inp = exp.inputs("demo_input", initial_state((160, 80, 1), pos, typ, (PILLARS[:, 0], PILLARS[:, 1])))
    ports = dict(dt=0.12, st=1, adapt=1, escape=1, **calibration())
    rows = [exp.loop(f"demo_{n}", inp, (160, 80, 1), n, ports) for n in (20, 167)]
    exp.result["per_substep_s"] = (rows[1]["wall_s"] - rows[0]["wall_s"]) / 147


def demo_scene(nx, ny, nz):
    """The demo's own obstacle layout (Xue, 2026-10-10): pillars mid-flume in
    both x and z so the camera looks straight at the collisions, one taller
    and one shorter, both thicker than the suite's. The validated bore-suite
    scene is untouched - this variant exists for observation, not for gates."""
    h1, h0, rx = 0.75 * ny, 0.125 * ny, 0.22 * nx
    pmin = np.array([[0.46 * nx, 0.0, 0.38 * nz],
                     [0.60 * nx, 0.0, 0.52 * nz]], np.float32)
    pmax = np.array([[0.52 * nx, 0.55 * ny, 0.50 * nz],
                     [0.66 * nx, 0.42 * ny, 0.64 * nz]], np.float32)

    def mask(x, y, z):
        return (y < h0) | ((x < rx) & (y < h1))
    return mask, pmin, pmax


def demo3d(exp):
    from run_bore_suite import seed_no_solids
    shape = (96, 24, 48)
    mask, pmin, pmax = demo_scene(*shape)
    pos, typ = seed_no_solids(*shape, mask, pmin, pmax)
    inp = exp.inputs("demo_input", initial_state(shape, pos, typ, (pmin, pmax)))
    exp.result["demo_shape"] = shape
    exp.result["demo_obstacles"] = [pmin.tolist(), pmax.tolist()]
    ports = dict(dt=0.12, st=1, adapt=1, escape=1, **calibration())
    rows = [exp.loop(f"demo_{n}", inp, shape, n, ports) for n in (20, 167)]
    exp.result["per_substep_s"] = (rows[1]["wall_s"] - rows[0]["wall_s"]) / 147
    initial_liquid = int((typ == 1).sum())
    peaks = [0, 0]
    raw_pen, display_pen, display_frames = 0, 0, []
    for k in range(167):
        d = exp.root / "demo_167"
        p, ph, scl, esc = [np.load(d / f"{key}.{k:04d}.npy") for key in ("frame", "ph", "scl", "esc")]
        raw = np.load(d / f"raw.{k:04d}.npy")
        vol = scl.astype(float) ** 3
        assert vol.sum() == len(pos), f"represented-volume drift at step {k}"
        assert vol[ph >= 0.5].sum() == initial_liquid, f"liquid-volume drift at step {k}"
        per_frame = 0
        for a, b in zip(pmin, pmax):
            raw_pen += int(((raw > a).all(1) & (raw < b).all(1)).sum())
            per_frame += int(((p > a).all(1) & (p < b).all(1)).sum())
        display_pen += per_frame
        if per_frame:
            display_frames.append({"step": k, "penetrations": per_frame})
        peaks = [max(peaks[0], int(((esc > 0.5) & (ph >= 0.5)).sum())),
                 max(peaks[1], int(((esc > 0.5) & (ph < 0.5)).sum()))]
    exp.result["invariants"] = {"frames_checked": 167, "represented_volume_drift": 0,
                                "liquid_volume_drift": 0, "solver_state_penetrations": raw_pen,
                                "display_position_penetrations": display_pen,
                                "display_audit_passed": display_pen == 0,
                                "display_penetrating_frames": display_frames,
                                "peak_droplets": peaks[0], "peak_bubbles": peaks[1]}
    print(json.dumps(exp.result["invariants"]), flush=True)
    assert raw_pen == 0, "actual solver-state obstacle penetration"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "flat", "bench", "demo", "demo3d"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--pack", default=str(HERE / "PFFlipSpike/build/PFFlipSpike-1.0.0/PFFlipSpikePackConfig.json"))
    parser.add_argument("--bifrost", default=r"C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    exp = Experiment(args)
    try:
        globals()[args.mode](exp)
    except Exception as error:
        exp.result["failure"] = f"{type(error).__name__}: {error}"
        (exp.root / "receipt.json").write_text(json.dumps(exp.result, indent=2))
        raise
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        if "input_sha256" in baseline:
            assert baseline["input_sha256"] == exp.result["input_sha256"], "benchmark inputs differ"
        assert set(baseline["cases"]) == set(exp.result["cases"])
        for name, row in exp.result["cases"].items():
            old = baseline["cases"][name]["arrays"]
            assert old == row["arrays"], f"bitwise mismatch: {name}: {[k for k in old if old[k] != row['arrays'].get(k)]}"
        exp.result["bitwise_equal_to"] = str(args.compare.resolve())
        print("ALL OUTPUT ARRAYS BIT-IDENTICAL TO BASELINE", flush=True)
    (exp.root / "receipt.json").write_text(json.dumps(exp.result, indent=2))
    if exp.result.get("invariants", {}).get("display_audit_passed") is False:
        raise SystemExit("DISPLAY AUDIT FAIL: penetrations recorded; solver-state and identity checks retained")


if __name__ == "__main__":
    main()
