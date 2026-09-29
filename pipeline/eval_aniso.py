"""Scores a trained splat scene with Aniso's own renderer on the views training never saw.

    python pipeline/eval_aniso.py <dataset-dir> <train-dir> [--sheet 8]

For every view listed in <train-dir>/held_out.txt: renders it with build/Release/aniso.exe,
scores it against the true image (PSNR), and, where the trainer saved its own render, scores
the agreement between Aniso and gsplat on the same splats. Writes eval_aniso.json and a sheet
of the first N views, each row: true image | Aniso render | amplified difference.
Needs numpy and Pillow.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

ANISO = os.environ.get("ANISO", os.path.join("build", "Release", "aniso.exe"))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("train")
    ap.add_argument("--sheet", type=int, default=8)
    a = ap.parse_args()
    ply = os.path.join(a.train, "point_cloud.ply")
    sparse = os.path.join(a.dataset, "sparse", "0")
    out_dir = os.path.join(a.train, "aniso")
    os.makedirs(out_dir, exist_ok=True)
    names = [n.strip() for n in open(os.path.join(a.train, "held_out.txt")) if n.strip()]

    rows, results = [], []
    for name in names:
        out = os.path.join(out_dir, name)
        subprocess.run([ANISO, "render", ply, sparse, name, out], check=True, stdout=subprocess.DEVNULL)
        truth = np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"))
        ours = np.asarray(Image.open(out).convert("RGB"))
        r = dict(name=name, psnr_vs_truth=psnr(ours, truth))
        g = os.path.join(a.train, "gsplat_" + name)
        if os.path.exists(g):
            r["psnr_aniso_vs_gsplat"] = psnr(ours, np.asarray(Image.open(g).convert("RGB")))
        results.append(r)
        if len(rows) < a.sheet:
            diff = np.clip(np.abs(ours.astype(int) - truth.astype(int)).sum(2) * 3, 0, 255).astype(np.uint8)
            rows.append(np.concatenate([truth, ours, np.repeat(diff[..., None], 3, 2)], 1))

    mean = lambda k: float(np.mean([r[k] for r in results if k in r]))  # noqa: E731
    summary = dict(views=len(results), mean_psnr_vs_truth=mean("psnr_vs_truth"),
                   min_psnr_vs_truth=min(r["psnr_vs_truth"] for r in results),
                   max_psnr_vs_truth=max(r["psnr_vs_truth"] for r in results))
    if any("psnr_aniso_vs_gsplat" in r for r in results):
        summary["mean_psnr_aniso_vs_gsplat"] = mean("psnr_aniso_vs_gsplat")
    with open(os.path.join(a.train, "eval_aniso.json"), "w") as f:
        json.dump(dict(summary=summary, views=results), f, indent=1)
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet.thumbnail((1920, 4000))
    sheet.save(os.path.join(a.train, "eval_sheet.png"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
