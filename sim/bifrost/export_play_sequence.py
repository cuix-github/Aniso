"""Exports a PF-FLIP 2D dam-break run as a numbered .npy sequence for the Maya play
graph (phase_field_play.json): per frame, particle positions (N,3, z=0.5) and phase
(N,), named particles.####.npy / phase.####.npy under out/seq/, Bifrost's frame-token
convention. The simulation is the same two-phase reference used by the validation
gate, so what plays in Maya's viewport is literally the python experiment.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe export_play_sequence.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402

SEQ = os.path.join(HERE, "out", "seq")
NX, NY, FRAMES, DT_FRAME = 160, 80, 60, 0.06


def dump(frame, sim):
    p3 = np.concatenate([sim.pos.astype(np.float32),
                         np.full((len(sim.pos), 1), 0.5, np.float32)], 1)
    np.save(os.path.join(SEQ, f"particles.{frame:04d}.npy"), p3)
    np.save(os.path.join(SEQ, f"phase.{frame:04d}.npy"),
            sim.typ.astype(np.float32))


def main():
    os.makedirs(SEQ, exist_ok=True)
    sim = Sim(NX, NY)
    sim.seed(lambda x, y: (x < 40) & (y < 56))
    sim.calibrate()
    dump(1, sim)  # Maya frame 1 = initial state
    sim.run(FRAMES - 1, DT_FRAME,
            on_frame=lambda f, s: dump(f + 2, s))
    print(f"wrote {FRAMES} frames ({NX}x{NY} dam break, "
          f"{len(sim.pos)} particles) to {os.path.normpath(SEQ)}")


if __name__ == "__main__":
    main()
