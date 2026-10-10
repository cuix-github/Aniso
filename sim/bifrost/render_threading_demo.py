"""Render the verified 20-second bore with measured, startup-excluded clocks.

Pass two --mode demo3d (or demo) directories from threading_experiment.py.
Frames stay in ignored out/; only the final video is a repository artifact.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from demo_p2g_speed import render_panel, HDR, DT, SUB, FONT, FONT_BIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("threaded", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    roots = (args.baseline, args.threaded)
    receipts = [json.loads((root / "receipt.json").read_text()) for root in roots]
    a, b = [r["cases"]["demo_167"]["arrays"] for r in receipts]
    assert a == b, "Do not render a claim of equivalence when any saved state differs"
    frames = args.threaded / "video_frames"
    frames.mkdir(exist_ok=False)
    labels = [f"before: serial transfer ({receipts[0]['threads']} workers)",
              f"after: threaded transfer ({receipts[1]['threads']} workers)"]
    colors = [(255, 170, 90), (110, 220, 140)]
    display_issue = any(r.get("invariants", {}).get("display_audit_passed") is False
                        for r in receipts)
    shape = receipts[0].get("demo_shape")
    if shape:
        from run_bore_suite import scene, render3d
        obs = receipts[0].get("demo_obstacles")
        if obs:
            pmin, pmax = np.array(obs[0], np.float32), np.array(obs[1], np.float32)
        else:
            _, pmin, pmax, *_ = scene(*shape)
    for k in range(SUB):
        panels = []
        for j, root in enumerate(roots):
            d = root / "demo_167"
            pos, ph, esc = [np.load(d / f"{n}.{k:04d}.npy") for n in ("frame", "ph", "esc")]
            clock = (k + 1) * receipts[j]["per_substep_s"]
            if shape:
                scene_image = render3d(pos, (ph >= 0.5).astype(int), *shape, pmin, pmax)
                panel = Image.new("RGB", (scene_image.width, scene_image.height + HDR), (26, 29, 38))
                panel.paste(scene_image, (0, HDR))
                draw = ImageDraw.Draw(panel)
                draw.text((12, 8), labels[j], font=FONT_BIG, fill=colors[j])
                draw.text((12, 33), f"sim {(k+1)*DT:5.1f} s    compute {clock:7.1f} s",
                          font=FONT, fill=(200, 203, 210))
            else:
                panel = render_panel(pos, ph, esc, labels[j], (k + 1) * DT, clock, colors[j])
            panels.append(panel)
        w, h = panels[0].size
        canvas = Image.new("RGB", (w * 2 + 4, h + (56 if display_issue else 28)), (26, 29, 38))
        for j, panel in enumerate(panels):
            canvas.paste(panel, (j * (w + 4), 0))
        shape_label = " x ".join(map(str, shape or (160, 80, 1)))
        ImageDraw.Draw(canvas).text((12, h + 4),
            f"{shape_label} | Identical states | clocks use average step cost; startup excluded",
            font=FONT, fill=(210, 214, 222))
        if display_issue:
            ImageDraw.Draw(canvas).text((12, h + 30),
                "Known issue in BOTH builds: some display positions enter pillars; solver positions do not.",
                font=FONT, fill=(255, 183, 105))
        canvas.save(frames / f"fr{k:04d}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(1 / DT), "-i", str(frames / "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", str(args.output)],
                   check=True, capture_output=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
