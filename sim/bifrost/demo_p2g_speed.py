"""The fused-P2G speed demonstration: one 20-second 2D simulation, run twice.

Scene: a wet-bed bore (reservoir released onto a shallow layer) striking two
pillars, 160x80 cells, production config (ST spacetime sampling, adaptive
air, escaped-particle droplets, multigrid), dt 0.12, 167 substeps = 20.04
simulated seconds. The same particles go through the stock-splat graph and
the fused-P2G graph; the output video plays them side by side in real time
(video time = simulated time) with each pipeline's compute clock running at
its measured per-substep rate, startup excluded. A companion chart puts the
2D rates next to the finale-scale receipts from bench_p2g_finale.py.

Per-substep rates are measured honestly: each pipeline runs twice (20 and
167 substeps) and the rate is the difference over 147, so graph compile and
process startup cancel.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe demo_p2g_speed.py
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
from demo3d import run_bifrost3  # noqa: E402

NX, NY, DT, SUB, SHORT = 160, 80, 0.12, 167, 20
RES = os.path.join(HERE, "results", "p2g")
H1, H0, RX = 0.75 * NY, 0.125 * NY, 0.22 * NX
PILLARS = np.array([[[0.52 * NX, 0.0, -1.0], [0.56 * NX, 0.45 * NY, 2.0]],
                    [[0.68 * NX, 0.0, -1.0], [0.72 * NX, 0.35 * NY, 2.0]]],
                   np.float32)
try:
    FONT = ImageFont.truetype("arial.ttf", 18)
    FONT_BIG = ImageFont.truetype("arialbd.ttf", 22)
except Exception:
    FONT = FONT_BIG = ImageFont.load_default()


def seed():
    sim = Sim(NX, NY)
    sim.seed(lambda x, y: (y < H0) | ((x < RX) & (y < H1)))
    keep = np.ones(len(sim.pos), bool)
    for a, b in PILLARS:
        keep &= ~((sim.pos[:, 0] > a[0]) & (sim.pos[:, 0] < b[0]) &
                  (sim.pos[:, 1] > a[1]) & (sim.pos[:, 1] < b[1]))
    pos3 = np.column_stack([sim.pos[keep],
                            np.full(keep.sum(), 0.5)]).astype(np.float32)
    return pos3, sim.typ[keep].astype(np.int32)


def run_once(tag, fused, substeps, pos3, typ, ports):
    t0 = time.time()
    d, _, _ = run_bifrost3(tag, NX, NY, 1, None, substeps, DT,
                           extra_ports=dict(ports),
                           seed_fn=lambda *_: (pos3, typ),
                           obstacles=(PILLARS[:, 0], PILLARS[:, 1]),
                           fused=fused)
    return time.time() - t0, d


SCALE = 4
W, H = NX * SCALE, NY * SCALE
HDR = 56


def render_panel(pos, ph, esc, label, t_sim, t_cpu, color):
    im = Image.new("RGB", (W, H + HDR), (14, 16, 22))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, HDR], fill=(26, 29, 38))
    dr.text((12, 8), label, font=FONT_BIG, fill=color)
    dr.text((12, 33), f"sim {t_sim:5.1f} s    compute {t_cpu:7.1f} s",
            font=FONT, fill=(200, 203, 210))
    img = np.array(im)
    liq = ph >= 0.5
    p = pos[liq]
    e = esc[liq] > 0.5
    x = np.clip((p[:, 0] * SCALE).astype(int), 0, W - 2)
    y = np.clip(H - 2 - (p[:, 1] * SCALE).astype(int), 0, H - 2) + HDR
    t = np.clip(p[:, 1] / NY, 0, 1)
    col = np.stack([18 + 50 * t, 80 + 90 * t, 165 + 75 * t], 1).astype(np.uint8)
    col[e] = (235, 242, 250)
    for dx in (0, 1):
        for dy in (0, 1):
            img[y + dy, x + dx] = col
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for a, b in PILLARS:
        dr.rectangle([a[0] * SCALE, HDR + H - b[1] * SCALE,
                      b[0] * SCALE, HDR + H - a[1] * SCALE], fill=(104, 108, 118))
    return im


def main():
    os.makedirs(RES, exist_ok=True)
    cal = Sim(64, 64, adapt=True, st=True)
    cal.seed(lambda x, y: np.ones_like(x, dtype=bool))
    cal.calibrate()
    ports = {"preconditioner": 1, "st": 1, "escape": 1, "adapt": 1,
             "rho0_face": cal.rho0_face, "coarse_gain": cal.coarse_gain,
             "ws_epsilon": 16}
    pos3, typ = seed()
    print(f"{len(pos3)} particles, {SUB} substeps = {SUB * DT:.1f} sim-seconds",
          flush=True)

    walls, dirs = {}, {}
    for name, fused in (("stock", False), ("fused", True)):
        w_short, _ = run_once(f"p2gdemo_{name}_s", fused, SHORT, pos3, typ, ports)
        w_full, d = run_once(f"p2gdemo_{name}", fused, SUB, pos3, typ, ports)
        rate = (w_full - w_short) / (SUB - SHORT)
        walls[name] = {"short_s": round(w_short, 1), "full_s": round(w_full, 1),
                       "per_substep_s": round(rate, 3)}
        dirs[name] = d
        print(f"{name}: full {w_full:.1f}s, per-substep {rate:.3f}s", flush=True)

    frames_dir = os.path.join(RES, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    drift = 0.0
    for k in range(SUB):
        panels = []
        for name, color in (("stock", (255, 170, 90)), ("fused", (110, 220, 140))):
            d = dirs[name]
            pos = np.load(os.path.join(d, f"frame.{k:04d}.npy"))
            ph = np.load(os.path.join(d, f"ph.{k:04d}.npy"))
            esc = np.load(os.path.join(d, f"esc.{k:04d}.npy"))
            label = ("stock splats (30 nodes + 30 samples)" if name == "stock"
                     else "fused P2G (one custom node)")
            panels.append(render_panel(
                pos, ph, esc, label, (k + 1) * DT,
                (k + 1) * walls[name]["per_substep_s"], color))
            if name == "stock":
                c0 = pos[ph >= 0.5][:, :2].mean(0)
            else:
                drift = max(drift, float(np.abs(
                    pos[ph >= 0.5][:, :2].mean(0) - c0).max()))
        canvas = Image.new("RGB", (W * 2 + 4, H + HDR), (60, 60, 70))
        canvas.paste(panels[0], (0, 0))
        canvas.paste(panels[1], (W + 4, 0))
        canvas.save(os.path.join(frames_dir, f"fr{k:04d}.png"))
    print(f"max liquid-centroid drift stock vs fused: {drift:.3f} cells",
          flush=True)

    video = os.path.join(RES, "p2g_speed_sidebyside.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", f"{SUB / (SUB * DT):.4f}",
                    "-i", os.path.join(frames_dir, "fr%04d.png"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "21", video],
                   check=True, capture_output=True)

    summary = {"scene": f"{NX}x{NY} wet-bed bore + 2 pillars, production config",
               "substeps": SUB, "sim_seconds": round(SUB * DT, 2),
               "walls": walls, "centroid_drift_cells": round(drift, 3),
               "video": os.path.basename(video)}
    bench = os.path.join(HERE, "out", "bench_p2g_finale.json")
    if os.path.exists(bench):
        summary["finale"] = json.load(open(bench))
    with open(os.path.join(RES, "p2g_speed_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    chart(summary)
    print(json.dumps(summary, indent=1), flush=True)


def chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("2D demo\n160x80, 51k particles",
               summary["walls"]["stock"]["per_substep_s"],
               summary["walls"]["fused"]["per_substep_s"])]
    fin = summary.get("finale")
    if fin:
        groups.append((f"3D finale\n{fin['grid']}, {fin['particles']/1e6:.1f}M",
                       fin["stock_per_substep_s"], fin["fused_per_substep_s"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.4))
    axes = [axes] if len(groups) == 1 else list(axes)
    for ax, (label, s, f) in zip(axes, groups):
        bars = ax.bar(["stock splats", "fused P2G"], [s, f],
                      color=["#e8995a", "#5fc07a"], width=0.55)
        ax.bar_label(bars, fmt="%.2f s")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("wall seconds per substep (startup excluded)")
        ax.text(0.5, 0.92, f"{s / f:.1f}x", transform=ax.transAxes,
                ha="center", fontsize=16, fontweight="bold", color="#2d6e45")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("P2G: thirty stock splat/sample passes vs one fused node",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "p2g_speed_chart.png"), dpi=150)


if __name__ == "__main__":
    main()
