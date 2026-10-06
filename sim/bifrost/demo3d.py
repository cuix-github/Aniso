"""Milestone 6: the first 3D run, rendered to video.

A 3D two-phase dam break (liquid column under air in a closed box) runs entirely
inside one headless Bifrost graph (sim_3d.json: three staggered stock splat chains
feeding the multigrid step node inside an iterate). Per-substep particle dumps are
rendered as a shaded point cloud from a three-quarter view and assembled into an mp4
with ffmpeg.

Run from sim/bifrost:
  ..\\..\\.venv\\Scripts\\python.exe demo3d.py smoke   # 16x16x8 sanity run
  ..\\..\\.venv\\Scripts\\python.exe demo3d.py         # the real scene + video
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "reference2d", "results", "validation")


def seed3d(nx, ny, nz, mask, jitter=0.35, seed=1):
    rng = np.random.default_rng(seed)
    cx, cy, cz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ps, ts = [], []
    for ox in (0.25, 0.75):
        for oy in (0.25, 0.75):
            for oz in (0.25, 0.75):
                p = np.stack([cx + ox, cy + oy, cz + oz], -1).reshape(-1, 3).astype(np.float64)
                p += rng.uniform(-jitter, jitter, p.shape) * 0.5
                ps.append(p)
                ts.append(mask(p[:, 0], p[:, 1], p[:, 2]).astype(np.int32))
    return np.concatenate(ps), np.concatenate(ts)


def lattice(ni, nj, nk):
    i, j, k = np.meshgrid(np.arange(ni) + 0.5, np.arange(nj) + 0.5, np.arange(nk) + 0.5,
                          indexing="ij")
    return np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(np.float32)


def run_bifrost3(tag, nx, ny, nz, mask, substeps, dt, rho_l=1000.0, rho_g=1.0):
    d = os.path.join(HERE, "out", "loop3_" + tag)
    os.makedirs(d, exist_ok=True)
    pos, typ = seed3d(nx, ny, nz, mask)
    mass = np.where(typ == 1, rho_l, rho_g).astype(np.float32)
    zeros = np.zeros(len(pos), np.float32)
    arrs = {"positions": pos.astype(np.float32), "velocities": np.zeros_like(pos, dtype=np.float32),
            "particle_phase": typ.astype(np.float32), "mass": mass,
            "momx": zeros, "momy": zeros, "momz": zeros,
            "probes_u": lattice(nx + 1, ny, nz), "probes_v": lattice(nx, ny + 1, nz),
            "probes_w": lattice(nx, ny, nz + 1)}
    for k, v in arrs.items():
        np.save(os.path.join(d, k + ".npy"), v)
    args = [os.path.join(HERE, "run_sim_3d.bat"),
            "--set-port", "nx", str(nx), "--set-port", "ny", str(ny),
            "--set-port", "nz", str(nz), "--set-port", "dt", str(dt),
            "--set-port", "substeps", str(substeps), "--set-port", "preconditioner", "1",
            "--set-port", "rho_liquid", str(rho_l), "--set-port", "rho_air", str(rho_g),
            "--set-port", "pos_pattern", (d + "/frame.####").replace("\\", "/"),
            "--set-port", "diag_pattern", (d + "/diag.####").replace("\\", "/"),
            "--set-port", "path_final_positions", (d + "/final.npy").replace("\\", "/")]
    for k in arrs:
        args += ["--set-port", "path_" + k, os.path.join(d, k + ".npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(os.path.join(d, f"frame.{substeps - 1:04d}.npy")):
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-2000:])
        raise SystemExit(f"bifcmd 3D loop failed for {tag}")
    diags = np.stack([np.load(os.path.join(d, f"diag.{k:04d}.npy")) for k in range(substeps)])
    return d, typ, diags


def render_frame(pos, typ, nx, ny, nz, scale=11):
    """Shaded liquid point cloud from a three-quarter view, box edges drawn."""
    yaw, pitch = np.radians(32), np.radians(20)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp_, sp_ = np.cos(pitch), np.sin(pitch)

    def project(p):
        x = p[:, 0] - nx / 2
        y = p[:, 1] - ny / 2
        z = p[:, 2] - nz / 2
        xr = cy_ * x + sy_ * z
        zr = -sy_ * x + cy_ * z
        yr = cp_ * y - sp_ * zr
        depth = cp_ * zr + sp_ * y
        return xr, yr, depth

    W = int((nx + nz) * 0.82 * scale // 2 * 2)
    H = int((ny + nz) * 0.95 * scale // 2 * 2)
    im = Image.new("RGB", (W, H), (252, 252, 254))
    d = ImageDraw.Draw(im)
    corners = np.array([[i * nx, j * ny, k * nz] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                       float)
    cx, cy2, _ = project(corners)
    px = (cx * scale + W / 2).astype(int)
    py = (H / 2 - cy2 * scale).astype(int)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        d.line([(px[a], py[a]), (px[b], py[b])], fill=(205, 208, 214), width=2)

    liq = pos[typ == 1]
    if len(liq) > 220000:
        liq = liq[:: len(liq) // 220000 + 1]
    x, y, depth = project(liq)
    order = np.argsort(depth)   # far first, near painted last
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


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    if smoke:
        d, typ, diags = run_bifrost3("smoke", 16, 16, 8,
                                     lambda x, y, z: y < 8, 4, 0.02)
        print("smoke ok:", diags[:, 0].tolist(), "max div", diags[:, 2].max())
        return

    nx, ny, nz, dt, sub = 64, 48, 32, 0.015, 160
    d, typ, diags = run_bifrost3("dam", nx, ny, nz,
                                 lambda x, y, z: (x < 20) & (y < 36), sub, dt)
    print(f"run done: CG iters mean {diags[:, 0].mean():.1f} max {diags[:, 0].max():.0f}; "
          f"worst divergence {diags[:, 2].max():.2e}; max CFL {diags[:, 3].max() * dt:.2f}")
    frames_dir = os.path.join(d, "png")
    os.makedirs(frames_dir, exist_ok=True)
    out_idx = 0
    for k in range(0, sub, 2):
        pos = np.load(os.path.join(d, f"frame.{k:04d}.npy"))
        render_frame(pos, typ, nx, ny, nz).save(
            os.path.join(frames_dir, f"fr{out_idx:04d}.png"))
        out_idx += 1
    video = os.path.join(RESULTS, "bifrost_dam3d.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "24",
                    "-i", os.path.join(frames_dir, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video],
                   check=True, capture_output=True)
    print("wrote", os.path.normpath(video))


if __name__ == "__main__":
    main()
