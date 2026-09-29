"""Renders a directed video of a splat scene from a shot list: slow, steady camera moves with a
fixed point of interest per shot, hard cuts between shots, and an optional "reveal" that
shrinks every Gaussian so the scene visibly breaks apart into the ellipsoids it is made of.

    python tools/shots.py <scene.ply> <shots.json> <out.mp4> [--size 1920x1080] [--fps 60]

A shot list is JSON: {"up": [0, 1, 0], "fov": 60, "shots": [ ... ]}. Each shot has
"seconds" and one of two moves, both eased in and out so the camera never jerks:

  {"move": "dolly", "from": [x, y, z], "to": [x, y, z], "look": [x, y, z]}
      a straight move; the camera always looks at "look"
  {"move": "arc", "centre": [x, y, z], "radius": r, "height": y, "from_deg": a, "to_deg": b,
   "look": [x, y, z]}
      a horizontal arc around "centre" at height y; angles are measured in the horizontal
      plane from +x toward +z

Optional per shot: "reveal": [start, end, smallest] shrinks the Gaussians to `smallest` of
their size between the fractions `start` and `end` of the shot and grows them back, easing
both ways. Needs ffmpeg on the PATH and build-cuda/aniso.exe (or set ANISO).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np


def ease(u):
    """Smooth start and stop: zero speed at both ends."""
    return u * u * (3 - 2 * u)


def reveal_scale(u, spec):
    if not spec:
        return 1.0
    start, end, smallest = spec
    if u <= start or u >= end:
        return 1.0
    x = (u - start) / (end - start)  # 0 -> 1 across the reveal window
    # Shrink over the first third, hold, grow back over the last third.
    if x < 1 / 3:
        k = ease(x * 3)
    elif x > 2 / 3:
        k = ease((1 - x) * 3)
    else:
        k = 1.0
    return 1.0 + (smallest - 1.0) * k


def shot_frames(shot, fps):
    n = max(1, int(round(shot["seconds"] * fps)))
    look = np.array(shot["look"], float)
    for f in range(n):
        u = f / max(1, n - 1)
        e = ease(u)
        if shot["move"] == "dolly":
            p = (1 - e) * np.array(shot["from"], float) + e * np.array(shot["to"], float)
        elif shot["move"] == "arc":
            a = math.radians(shot["from_deg"] + e * (shot["to_deg"] - shot["from_deg"]))
            c = np.array(shot["centre"], float)
            p = np.array([c[0] + shot["radius"] * math.cos(a), shot["height"], c[2] + shot["radius"] * math.sin(a)])
        else:
            sys.exit("unknown move " + shot["move"])
        yield p, look - p, reveal_scale(u, shot.get("reveal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("shots")
    ap.add_argument("out")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.shots))
    up = spec.get("up", [0, 1, 0])

    lines = []
    for shot in spec["shots"]:
        for p, d, s in shot_frames(shot, a.fps):
            lines.append("%f %f %f  %f %f %f  %g %g %g  %f\n" % (*p, *d, *up, s))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
        path_file = f.name

    exe = os.environ.get("ANISO", os.path.join("build-cuda", "aniso.exe"))
    render = [exe, "path", a.scene, path_file, a.size, str(spec.get("fov", 60))] + ([] if a.cpu else ["--gpu"])
    encode = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", a.size,
              "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    print("%d shots, %d frames, %.0f seconds at %s" % (len(spec["shots"]), len(lines), len(lines) / a.fps, a.size),
          flush=True)
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
