"""Renders a smooth fly-through of a splat scene to an MP4, with Aniso's renderer (the GPU one
when available) feeding frames straight into ffmpeg.

    python tools/flythrough.py <scene.ply> <colmap-sparse-dir> <out.mp4> --mode orbit
    python tools/flythrough.py <scene.ply> <colmap-sparse-dir> <out.mp4> --mode tour

The path comes from the scene's own photo cameras, so it stays where the scene was seen:
  orbit  follows the photographers' circle around the subject, smoothed, always looking at the
         point the photos look at most (for object-centred captures such as the train)
  tour   walks through the photo cameras themselves, chained by nearness in position and view
         direction and smoothed (for captures taken inside a room, such as the kitchen, where
         every photo camera is known to see real content)

Options: --size 1920x1080, --fps 60, --seconds 12, --fov 70 (horizontal, degrees), --cpu,
--radius R (orbit: fraction of the photos' radius, default 1.0), --stride N (tour: use every
Nth camera of the chain as a keyframe, default 6). Needs numpy, ffmpeg on the PATH, and build-cuda/aniso.exe (or set ANISO).
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def read_poses(sparse):
    """Camera centres and forward and down axes (world space) from COLMAP's images.bin."""
    centres, forwards, downs = [], [], []
    with open(os.path.join(sparse, "images.bin"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<I4d3dI", f.read(64))
            while f.read(1) != b"\0":
                pass
            (npts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * npts)
            w, x, y, z = qw, qx, qy, qz
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            centres.append(-R.T @ np.array([tx, ty, tz]))
            forwards.append(R[2])
            downs.append(R[1])
    return np.array(centres), np.array(forwards), np.array(downs)


def focus_point(centres, forwards):
    """The point closest, in least squares, to every camera's line of sight."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, forwards):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def smooth_loop(points, count, passes=3):
    """Resample a closed loop of points to `count` samples and smooth it with a circular box
    filter, so the camera glides instead of stepping between photo positions."""
    t = np.linspace(0, len(points), count, endpoint=False)
    i0 = np.floor(t).astype(int) % len(points)
    i1 = (i0 + 1) % len(points)
    u = (t - np.floor(t))[:, None]
    out = points[i0] * (1 - u) + points[i1] * u
    k = max(3, count // 30) | 1
    for _ in range(passes):
        pad = np.concatenate([out[-k:], out, out[:k]])
        kernel = np.ones(k) / k
        out = np.stack([np.convolve(pad[:, j], kernel, mode="same")[k:-k] for j in range(3)], 1)
    return out


def orbit_path(centres, forwards, up, frames, radius):
    focus = focus_point(centres, forwards)
    rel = centres - focus
    # A horizontal frame of reference around the focus point.
    e1 = rel.mean(0) - up * (rel.mean(0) @ up)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ang = np.arctan2(rel @ e2, rel @ e1)
    order = np.argsort(ang)
    # Keyframes: the median camera in each of 24 angular bins that has any.
    keys = []
    bins = np.linspace(-np.pi, np.pi, 25)
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = order[(ang[order] >= lo) & (ang[order] < hi)]
        if len(sel):
            keys.append(np.median(rel[sel], 0))
    keys = np.array(keys) * radius + focus
    pos = smooth_loop(keys, frames)
    fwd = focus - pos
    return pos, fwd


def tour_path(centres, forwards, frames, stride):
    """A walk through the photo cameras themselves, each of which is known to see the scene:
    chain them greedily by nearness in both position and viewing direction, keep every
    `stride`-th camera of the chain as a keyframe, and smooth positions and directions so the
    camera glides from view to view."""
    spread = np.percentile(np.linalg.norm(centres - centres.mean(0), axis=1), 90)
    feat = np.concatenate([centres / spread, 0.8 * forwards], 1)
    left = list(range(len(centres)))
    cur = int(np.argmin(np.linalg.norm(centres - centres.mean(0), axis=1)))
    chain = [cur]
    left.remove(cur)
    while left:
        d = np.linalg.norm(feat[left] - feat[cur], axis=1)
        cur = left[int(np.argmin(d))]
        if d.min() > 1.2:  # the rest are far from anything visited: stop rather than jump
            break
        chain.append(cur)
        left.remove(cur)
    keys = chain[::stride]
    kp, kf = centres[keys], forwards[keys]
    t = np.linspace(0, len(keys) - 1, frames)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(keys) - 1)
    u = (t - i0)[:, None]
    u = u * u * (3 - 2 * u)  # ease between keyframes
    pos = kp[i0] * (1 - u) + kp[i1] * u
    fwd = kf[i0] * (1 - u) + kf[i1] * u
    k = max(3, frames // 40) | 1
    for arr in (pos, fwd):
        for _ in range(2):
            pad = np.concatenate([np.repeat(arr[:1], k, 0), arr, np.repeat(arr[-1:], k, 0)])
            arr[:] = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="same")[k:-k] for j in range(3)], 1)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return pos, fwd, len(chain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("sparse")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["orbit", "tour"], default="orbit")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--fov", type=float, default=70)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    centres, forwards, downs = read_poses(a.sparse)
    up = -downs.mean(0)
    up /= np.linalg.norm(up)
    frames = int(round(a.fps * a.seconds))
    if a.mode == "orbit":
        pos, fwd = orbit_path(centres, forwards, up, frames, a.radius or 1.0)
    else:
        pos, fwd, used = tour_path(centres, forwards, frames, a.stride)
        print("tour through %d of %d photo cameras" % (used, len(centres)), flush=True)

    exe = os.environ.get("ANISO", os.path.join("build-cuda", "aniso.exe"))
    w, h = (int(v) for v in a.size.split("x"))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p, d in zip(pos, fwd):
            f.write("%f %f %f  %f %f %f  %f %f %f\n" % (*p, *d, *up))
        path_file = f.name
    render = [exe, "path", a.scene, path_file, a.size, str(a.fov)] + ([] if a.cpu else ["--gpu"])
    encode = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", a.size,
              "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    print("%d frames at %s, %d fps, %s path" % (frames, a.size, a.fps, a.mode), flush=True)
    r = subprocess.Popen(render, stdout=subprocess.PIPE)
    e = subprocess.Popen(encode, stdin=r.stdout)
    r.stdout.close()
    e.wait()
    r.wait()
    os.remove(path_file)
    if r.returncode or e.returncode:
        sys.exit("render exit %s, encode exit %s" % (r.returncode, e.returncode))
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
