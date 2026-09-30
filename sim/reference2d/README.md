# The 2D reference implementation

`pfflip2d.py` is the executable spec for the Bifrost build: the core PF-FLIP loop from Section 3 of the paper, in about 300 lines of numpy, with none of the scale machinery. Both phases are particles; the phase field is read off the splatted face masses; the pressure solve is variable-coefficient with Jacobi-preconditioned conjugate gradient; the per-phase FLIP blend doubles as viscosity, as in the paper.

Run it (needs numpy and Pillow):

```
python pfflip2d.py            # all three scenes, ~35 s total
python pfflip2d.py dam_break  # one scene
```

## Scenes and what they prove

- **hydrostatic**: liquid under air at 1000:1 in a closed box. Proves the two-phase projection holds the column: velocities stay at FLIP-jitter level with no blow-up.
- **dam_break** (`results/dam_break.png`): a collapsing column running along the floor under air. Proves the coupled transport works; the front advances monotonically. The collapse is mildly damped by the FLIP blend, a knob to tune when matching the Bifrost build.
- **rayleigh_taylor** (`results/rayleigh_taylor.png`): heavy fluid over light at Atwood 0.9, the paper's own validation setup. Fingers grow from a seeded ripple into mushrooms. This phenomenon is two-phase-only: a single-phase solver with a free surface cannot produce it at all, which is what makes it the signature test.

## Deliberate omissions, for later milestones

Adaptive particles and the per-size blend correction, escaped-particle droplets, the divergence-drift correction, and liquid-velocity extrapolation. Each is listed in `../README.md`'s milestones.
