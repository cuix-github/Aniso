"""Renders a run of consecutive photo cameras and assembles them into an animated GIF, so the
scene can be seen from a moving viewpoint.

    python tools/orbit.py data/train/point_cloud_7000.ply data/tandt/train/sparse/0 renders/orbit.gif --first 1 --count 60

Options: --first N (first photo number), --count N (how many cameras), --width N (frame
width, default 490), --sh-degree D (default 3), --fps N (default 12). Needs Pillow, and the
aniso executable built in build/Release (or set ANISO to its path).
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("sparse")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--width", type=int, default=490)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    exe = os.environ.get("ANISO", str(Path("build") / "Release" / "aniso.exe"))
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(a.first, a.first + a.count):
            name = f"{k:05d}.jpg"
            png = Path(tmp) / f"{k:05d}.png"
            subprocess.run([exe, "render", a.scene, a.sparse, name, str(png), "--width", str(a.width),
                            "--sh-degree", str(a.sh_degree)], check=True, stdout=subprocess.DEVNULL)
            frames.append(Image.open(png).convert("RGB"))
            print(f"\rrendered {len(frames)}/{a.count}", end="", flush=True)
    print()
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
