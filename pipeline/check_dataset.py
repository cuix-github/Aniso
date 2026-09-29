"""Checks a COLMAP-format dataset against its images: projects the sparse points through each
camera and overlays them on the image, and builds a contact sheet of all views.

    python pipeline/check_dataset.py <dataset-dir> <out-dir> [--max 16]

If the camera export is right, the overlaid points sit on the surfaces they came from.
Needs numpy and Pillow.
"""
import argparse
import os
import struct

import numpy as np
from PIL import Image, ImageDraw


def read_cameras(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        cams = {}
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            count = {0: 3, 1: 4}[model]
            p = struct.unpack("<%dd" % count, f.read(8 * count))
            fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model == 0 else p
            cams[cid] = (w, h, fx, fy, cx, cy)
    return cams


def read_images(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        out = []
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<I4d3dI", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * npts)
            out.append((name.decode(), (qw, qx, qy, qz), np.array([tx, ty, tz]), cid))
    return sorted(out)


def read_points(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        xyz = np.zeros((n, 3))
        rgb = np.zeros((n, 3), np.uint8)
        for i in range(n):
            _, x, y, z, r, g, b, _ = struct.unpack("<Q3d3Bd", f.read(43))
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(8 * tl)
            xyz[i] = (x, y, z)
            rgb[i] = (r, g, b)
    return xyz, rgb


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = os.path.join(a.dataset, "sparse", "0")
    cams = read_cameras(os.path.join(sp, "cameras.bin"))
    images = read_images(os.path.join(sp, "images.bin"))
    xyz, _ = read_points(os.path.join(sp, "points3D.bin"))
    sub = xyz[np.random.default_rng(0).choice(len(xyz), min(20000, len(xyz)), replace=False)]

    thumbs = []
    for name, q, t, cid in images[: a.max]:
        w, h, fx, fy, cx, cy = cams[cid]
        img = Image.open(os.path.join(a.dataset, "images", name)).convert("RGB")
        p = sub @ qmat(q).T + t
        p = p[p[:, 2] > 0.05]
        u = fx * p[:, 0] / p[:, 2] + cx
        v = fy * p[:, 1] / p[:, 2] + cy
        ov = img.copy()
        d = ImageDraw.Draw(ov)
        for x, y in zip(u, v):
            if 0 <= x < w and 0 <= y < h:
                d.point((x, y), fill=(255, 0, 255))
        both = Image.new("RGB", (w, h * 2))
        both.paste(img, (0, 0))
        both.paste(ov, (0, h))
        both.save(os.path.join(a.out, "overlay_" + name))
        thumbs.append(img.resize((w // 4, h // 4)))

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    tw, th = thumbs[0].size
    sheet = Image.new("RGB", (tw * cols, th * rows))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(a.out, "contact_sheet.png"))
    print("wrote %d overlays and contact_sheet.png to %s" % (len(thumbs), a.out))


if __name__ == "__main__":
    main()
