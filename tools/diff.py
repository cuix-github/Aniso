"""Makes an amplified difference image of two renders of the same view, stacked under both
inputs: bright where they differ, black where they match.

    python tools/diff.py renders/r_00001_sh0.png renders/r_00001_sh3.png renders/diff_sh.png

Used to see where view-dependent colour matters: render once with --sh-degree 0 and once with
the default, then diff. Needs Pillow and numpy.
"""
import sys

import numpy as np
from PIL import Image


def main(a_path, b_path, out_path, gain=3.0):
    a_img = Image.open(a_path).convert("RGB")
    b_img = Image.open(b_path).convert("RGB")
    a = np.asarray(a_img, dtype=np.int16)
    b = np.asarray(b_img, dtype=np.int16)
    diff = np.clip(np.abs(b - a).sum(axis=2) * gain, 0, 255).astype(np.uint8)
    print(f"mean absolute difference {np.abs(b - a).mean():.2f} of 255 per channel")
    w, h = a_img.size
    out = Image.new("RGB", (w, h * 3))
    out.paste(a_img, (0, 0))
    out.paste(b_img, (0, h))
    out.paste(Image.fromarray(diff).convert("RGB"), (0, 2 * h))
    out.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
