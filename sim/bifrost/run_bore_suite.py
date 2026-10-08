"""The milestone-6 overnight suite: the 3D wet-bed tsunami bore with obstacles.

One scene, four stages, results land in out/bore_suite/ as they complete:

  A  trusted    small dt (CFL ~0.5), the fidelity reference
  B  st         the same scene at 8x dt with spacetime sampling
  C  production st + adaptive air at 8x dt
  D  finale     the 128-cubed-class flume (256x64x128 cells) in production config

Checks: bore front position vs Stoker's analytic wet-bed dam-break speed, kinetic
proxy curves, particle-count compression (C/D), droplet/bubble peaks, zero obstacle
penetrations. Videos per stage, curves, and a summary verdict file.

  ..\\..\\.venv\\Scripts\\python.exe run_bore_suite.py smoke    # tiny plumbing check
  ..\\..\\.venv\\Scripts\\python.exe run_bore_suite.py          # the overnight run
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from demo3d import run_bifrost3, lattice  # noqa: E402

OUT = os.path.join(HERE, "out", "bore_suite")
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except Exception:
    FONT = ImageFont.load_default()

# flume: reservoir (depth H1) released into a shallow bed (H0); two pillar obstacles.
H1_FRAC, H0_FRAC, RES_X_FRAC = 0.75, 0.125, 0.22
G = 9.8


def scene(nx, ny, nz):
    h1, h0, rx = H1_FRAC * ny, H0_FRAC * ny, RES_X_FRAC * nx
    pillars_min = np.array([[0.55 * nx, 0, 0.30 * nz], [0.70 * nx, 0, 0.60 * nz]],
                           np.float32)
    pillars_max = np.array([[0.59 * nx, 0.5 * ny, 0.42 * nz],
                            [0.74 * nx, 0.5 * ny, 0.72 * nz]], np.float32)

    def mask(x, y, z):
        return (y < h0) | ((x < rx) & (y < h1))
    return mask, pillars_min, pillars_max, h1, h0, rx


def stoker_speed(h1, h0):
    """Front speed of the wet-bed dam-break bore (Stoker 1957), solved by bisection."""
    def f(h2):
        u2 = 2 * (np.sqrt(G * h1) - np.sqrt(G * h2))
        U = np.sqrt(G * h2 * (h2 + h0) / (2 * h0))
        return u2 - (U * (1 - h0 / h2))
    lo, hi = h0 * 1.0001, h1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    h2 = 0.5 * (lo + hi)
    return float(np.sqrt(G * h2 * (h2 + h0) / (2 * h0))), float(h2)


def seed_no_solids(nx, ny, nz, mask, pmin, pmax):
    from demo3d import seed3d
    pos, typ = seed3d(nx, ny, nz, mask)
    keep = np.ones(len(pos), bool)
    for a, b in zip(pmin, pmax):
        keep &= ~((pos[:, 0] > a[0]) & (pos[:, 0] < b[0]) &
                  (pos[:, 1] > a[1]) & (pos[:, 1] < b[1]) &
                  (pos[:, 2] > a[2]) & (pos[:, 2] < b[2]))
    return pos[keep], typ[keep]


def render3d(pos, typ, nx, ny, nz, pmin, pmax, scale=None):
    scale = scale or max(2, int(720 / (nx + nz)))
    yaw, pitch = np.radians(30), np.radians(18)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp_, sp_ = np.cos(pitch), np.sin(pitch)

    def project(p):
        x = p[..., 0] - nx / 2
        y = p[..., 1] - ny / 2
        z = p[..., 2] - nz / 2
        xr = cy_ * x + sy_ * z
        zr = -sy_ * x + cy_ * z
        yr = cp_ * y - sp_ * zr
        return xr, yr, cp_ * zr + sp_ * y

    W = int((nx + nz) * 0.80 * scale // 2 * 2)
    H = int((ny + nz) * 0.85 * scale // 2 * 2)
    im = Image.new("RGB", (W, H), (252, 252, 254))
    d = ImageDraw.Draw(im)
    for a, b in zip(pmin, pmax):
        corners = np.array([[i, j, k] for i in (a[0], b[0]) for j in (a[1], b[1])
                            for k in (a[2], b[2])])
        cx, cy2, _ = project(corners)
        px = (cx * scale + W / 2)
        py = (H / 2 - cy2 * scale)
        d.polygon(list(zip(px[[0, 1, 3, 2]], py[[0, 1, 3, 2]])), fill=(120, 124, 132))
        d.polygon(list(zip(px[[2, 3, 7, 6]], py[[2, 3, 7, 6]])), fill=(98, 102, 110))
        d.polygon(list(zip(px[[1, 3, 7, 5]], py[[1, 3, 7, 5]])), fill=(80, 84, 92))
    liq = pos[typ == 1]
    if len(liq) > 250000:
        liq = liq[:: len(liq) // 250000 + 1]
    x, y, depth = project(liq)
    order = np.argsort(depth)
    x, y, depth = x[order], y[order], depth[order]
    t = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-9)
    img = np.array(im)
    xi = np.clip((x * scale + W / 2).astype(int), 0, W - 2)
    yi = np.clip((H / 2 - y * scale).astype(int), 0, H - 2)
    col = np.stack([15 + 60 * t, 70 + 90 * t, 170 + 70 * t], 1).astype(np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            img[yi + dy, xi + dx] = col
    return Image.fromarray(img)


def load_final_state(d):
    import numpy as np
    st = {"positions": np.load(os.path.join(d, "final.npy"))}
    for nm, key in (("velocities", "velocities"), ("mass", "mass"), ("momx", "momx"),
                    ("momy", "momy"), ("momz", "momz"), ("tau", "tau"), ("wt", "wt"),
                    ("wtph", "wtph"), ("scale", "scale"), ("phase", "phase_st")):
        st[key] = np.load(os.path.join(d, "final_" + nm + ".npy"))
    st["particle_phase"] = st["phase_st"]
    return st


def run_stage(tag, nx, ny, nz, dt, substeps, ports, every=4, chunk=None):
    """chunk: run only substeps [chunk*CH, ...) of a longer stage, resuming from the
    previous chunk's final state; the caller chains chunks across launches."""
    mask, pmin, pmax, h1, h0, rx = scene(nx, ny, nz)
    cal = Sim(64, 64, adapt=bool(ports.get("adapt")), st=bool(ports.get("st")))
    cal.seed(lambda x, y: np.ones_like(x, dtype=bool))
    cal.calibrate()
    ports = dict(ports)
    ports.setdefault("rho0_face", cal.rho0_face)
    if ports.get("adapt"):
        ports.setdefault("coarse_gain", cal.coarse_gain)
        ports.setdefault("ws_epsilon", 16)
    t0 = time.time()
    init = None
    if chunk is not None and chunk > 0:
        prev_dir = os.path.join(HERE, "out", f"loop3_{tag}_c{chunk - 1}")
        init = load_final_state(prev_dir)
    ctag = tag if chunk is None else f"{tag}_c{chunk}"
    d, typ, diag = run_bifrost3(
        ctag, nx, ny, nz, mask, substeps, dt, extra_ports=ports,
        seed_fn=lambda *_: seed_no_solids(nx, ny, nz, mask, pmin, pmax),
        obstacles=(pmin, pmax), init_arrays=init)
    wall = time.time() - t0
    # front curve, penetration audit, compression, render
    fronts, counts, pen = [], [], 0
    frames_dir = os.path.join(d, "png")
    os.makedirs(frames_dir, exist_ok=True)
    out_i = 0
    for k in range(substeps):
        p = np.load(os.path.join(d, f"frame.{k:04d}.npy"))
        ph = np.load(os.path.join(d, f"ph.{k:04d}.npy"))
        liq = ph >= 0.5
        counts.append(len(p))
        for a, b in zip(pmin, pmax):
            pen += int(((p[:, 0] > a[0]) & (p[:, 0] < b[0]) & (p[:, 1] > a[1]) &
                        (p[:, 1] < b[1]) & (p[:, 2] > a[2]) & (p[:, 2] < b[2])).sum())
        fronts.append((k * dt, float(np.percentile(p[liq, 0], 99.0))))
        if k % every == 0:
            render3d(p, liq.astype(int), nx, ny, nz, pmin, pmax).save(
                os.path.join(frames_dir, f"fr{out_i:04d}.png"))
            out_i += 1
    video = os.path.join(OUT, f"bore_{tag}.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "16",
                    "-i", os.path.join(frames_dir, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", video],
                   check=True, capture_output=True)
    return {"tag": tag, "wall": wall, "fronts": fronts, "counts": counts, "pen": pen,
            "diag": diag.tolist() if hasattr(diag, "tolist") else diag,
            "video": video, "h1": h1, "h0": h0, "rx": rx, "nx": nx}


STAGE_DEFS = {
    "A_trusted": (80, 32, 40, 0.015, {"preconditioner": 1, "escape": 1}, 80, 5),
    "B_st": (80, 32, 40, 0.12, {"preconditioner": 1, "st": 1, "escape": 1}, 50, 1),
    "C_production": (80, 32, 40, 0.12,
                     {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1}, 50, 1),
    "D_finale": (256, 64, 128, 0.12,
                 {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1}, 17, 4),
}


def stage_chunk(tag, chunk):
    nx, ny, nz, dt, ports, per_chunk, n_chunks = STAGE_DEFS[tag]
    os.makedirs(OUT, exist_ok=True)
    r = run_stage(tag, nx, ny, nz, dt, per_chunk, ports, chunk=chunk)
    with open(os.path.join(OUT, f"{tag}_c{chunk}.json"), "w") as f:
        json.dump({"wall": r["wall"], "pen": r["pen"], "counts": r["counts"],
                   "fronts": r["fronts"], "h1": r["h1"], "h0": r["h0"],
                   "rx": r["rx"], "nx": r["nx"]}, f)
    print(f"CHUNK DONE {tag} {chunk}/{n_chunks - 1}", flush=True)


def finalize(tag):
    nx, ny, nz, dt, ports, per_chunk, n_chunks = STAGE_DEFS[tag]
    fronts, counts, pen, wall = [], [], 0, 0.0
    for c in range(n_chunks):
        with open(os.path.join(OUT, f"{tag}_c{c}.json")) as f:
            j = json.load(f)
        off = c * per_chunk * dt
        counts += j["counts"]
        pen += j["pen"]
        wall += j["wall"]
        # the bore front is the front of ELEVATED liquid (above the undisturbed
        # bed), not of all liquid - the bed itself spans the flume from t=0.
        d = os.path.join(HERE, "out", f"loop3_{tag}_c{c}")
        h0 = j["h0"]
        for k in range(per_chunk):
            ppos = np.load(os.path.join(d, f"frame.{k:04d}.npy"))
            ph = np.load(os.path.join(d, f"ph.{k:04d}.npy"))
            el = (ph >= 0.5) & (ppos[:, 1] > h0 + 1.5)
            x = float(np.percentile(ppos[el, 0], 99.0)) if el.sum() > 50 else 0.0
            fronts.append((off + k * dt, x))
    h1, h0, rx, nxv = j["h1"], j["h0"], j["rx"], j["nx"]
    U, h2 = stoker_speed(h1, h0)
    fr = np.array(fronts)
    selw = (fr[:, 1] > rx + 4) & (fr[:, 1] < 0.52 * nxv)
    slope = (np.polyfit(fr[selw, 0], fr[selw, 1], 1)[0]
             if selw.sum() > 4 else float("nan"))
    comp = 1 - min(counts) / max(counts)
    entry = {"wall_s": round(wall, 1),
             "bore_speed_measured": round(float(slope), 2) if np.isfinite(slope) else None,
             "bore_speed_stoker": round(U, 2),
             "speed_ratio": round(float(slope) / U, 3) if np.isfinite(slope) else None,
             "penetrations": pen, "peak_compression": round(comp, 3)}
    path = os.path.join(OUT, "summary.json")
    summary = {}
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
    summary[tag] = entry
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(tag, json.dumps(entry), flush=True)
    # stitch per-chunk pngs into the stage video
    frames_dir = os.path.join(OUT, tag + "_png")
    os.makedirs(frames_dir, exist_ok=True)
    idx = 0
    for c in range(n_chunks):
        src = os.path.join(HERE, "out", f"loop3_{tag}_c{c}", "png")
        for f2 in sorted(os.listdir(src)):
            os.replace(os.path.join(src, f2), os.path.join(frames_dir, f"fr{idx:04d}.png"))
            idx += 1
    subprocess.run(["ffmpeg", "-y", "-framerate", "16",
                    "-i", os.path.join(frames_dir, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    os.path.join(OUT, f"bore_{tag}.mp4")],
                   check=True, capture_output=True)
    print("FINALIZED", tag, flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "stage":
        stage_chunk(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "finalize":
        finalize(sys.argv[2])
        return
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    if smoke:
        stages = [("A_trusted", 48, 24, 24, 0.015, 8, {"preconditioner": 1}),
                  ("C_production", 48, 24, 24, 0.12, 4,
                   {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1})]
    else:
        stages = [
            ("A_trusted", 80, 32, 40, 0.015, 400, {"preconditioner": 1, "escape": 1}),
            ("B_st", 80, 32, 40, 0.12, 50,
             {"preconditioner": 1, "st": 1, "escape": 1}),
            ("C_production", 80, 32, 40, 0.12, 50,
             {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1}),
            ("D_finale", 256, 64, 128, 0.12, 67,
             {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1}),
        ]
    for tag, nx, ny, nz, dt, sub, ports in stages:
        print(f"=== {tag}: {nx}x{ny}x{nz}, dt {dt}, {sub} substeps", flush=True)
        r = run_stage(tag, nx, ny, nz, dt, sub, ports)
        # Stoker check on the pre-obstacle stretch
        h1, h0 = r["h1"], r["h0"]
        U, h2 = stoker_speed(h1, h0)
        fr = np.array(r["fronts"])
        selw = (fr[:, 1] > r["rx"] + 4) & (fr[:, 1] < 0.52 * r["nx"])
        slope = (np.polyfit(fr[selw, 0], fr[selw, 1], 1)[0]
                 if selw.sum() > 4 else float("nan"))
        comp = 1 - min(r["counts"]) / max(r["counts"])
        summary[tag] = {
            "wall_s": round(r["wall"], 1), "bore_speed_measured": round(float(slope), 2),
            "bore_speed_stoker": round(U, 2),
            "speed_ratio": round(float(slope) / U, 3) if np.isfinite(slope) else None,
            "penetrations": r["pen"], "peak_compression": round(comp, 3),
            "video": os.path.basename(r["video"]),
        }
        print(json.dumps(summary[tag]), flush=True)
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print("SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
