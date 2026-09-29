"""Independent cross-check for `aniso info`: reads the same PLY with numpy and prints the
same statistics, so the C++ loader can be compared number for number.

    python tools/reference_stats.py data/train/point_cloud_7000.ply
"""
import sys

import numpy as np

C0 = 0.28209479177387814


def read(path):
    with open(path, "rb") as f:
        props, count = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property"):
                props.append(line.split()[2])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype="<f4", count=count * len(props))
    return data.reshape(count, len(props)), {p: i for i, p in enumerate(props)}


def row(label, v):
    print(f"  {label:<14} min {v.min():12.6f}   max {v.max():12.6f}   mean {v.astype(np.float64).mean():12.6f}")


def main(path):
    d, c = read(path)
    print(path)
    print(f"  gaussians      {d.shape[0]}")
    for axis in "xyz":
        row(f"position.{axis}", d[:, c[axis]])
    scales = np.exp(d[:, [c["scale_0"], c["scale_1"], c["scale_2"]]])
    row("largest scale", scales.max(axis=1))
    row("opacity", 1.0 / (1.0 + np.exp(-d[:, c["opacity"]])))
    for ch, name in enumerate("rgb"):
        row(f"dc colour {name}", 0.5 + C0 * d[:, c[f"f_dc_{ch}"]])


if __name__ == "__main__":
    main(sys.argv[1])
