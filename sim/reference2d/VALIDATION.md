# The validation gate: are the paper's core claims true?

Before any Bifrost engineering, we tested PF-FLIP's central claims in the smallest setups that could falsify them, comparing against the obvious alternatives. Run everything with `python validation.py` (about 80 seconds); images and plots land in `results/validation/`. Verdicts below are from the run of 2026-09-30.

## 1. Two-phase captures what single-phase FLIP structurally cannot — SUPPORTED, decisively

A slab of water falls toward a pool in a sealed box, trapping a layer of air wall to wall. `SinglePhaseSim` in `validation.py` is a classic free-surface FLIP, the same solver with air treated as vacuum at zero pressure, so the only difference under test is whether air exists. Since the trapped air has nowhere to escape and cannot compress, the physical answer is that the slab must stop.

Result (`air_pocket_curves.png`): the two-phase slab settles onto the air cushion and the gap stays at 10.6 cells for the whole run, while the single-phase slab falls freely through its vacuum, 10.6 down to 3.3 and still accelerating. The frame strips make it visible at a glance. A first version of this experiment measured the pocket with a fixed spatial band, which conflated the slab entering the band with the pocket shrinking and diluted the contrast; the metric was redesigned to track the actual gap between pool surface and slab underside. Noted here because the gate should record its own mistakes.

## 2. Particle-carried phase does not diffuse — SUPPORTED, with an honest caveat

The reversing-vortex benchmark: a disc is sheared into a long filament and the flow is exactly reversed; a perfect method returns the original disc. We advect the same interface two ways through the same velocity field: as particle markers, PF-FLIP's way, and as a grid scalar moved by semi-Lagrangian advection, the classical way.

Result (`transport.png`): the grid scalar loses the thin filament to numerical diffusion and comes back visibly eroded, L1 shape error 293 in cell units. The particles come back essentially exact. Caveat recorded: with an analytic velocity field the particle path is reversible almost by construction, so the fair reading is not "particles are perfect" but "Eulerian scalar transport destroys thin interface features at this resolution and Lagrangian transport has no such term." That is precisely the property PF-FLIP buys by carrying phase on particles.

## 3. The FLIP blend acts as viscosity — the LAW holds; the paper's constant is kernel-dependent

Paper Eq. 13 says effective kinematic viscosity is (1−α)·Δx²/(6Δt). We measured the decay rate of a sinusoidal shear wave, which has an exact analytic decay, across α = 0.9, 0.7, 0.5 at fixed Δt.

Result (`viscosity.png`): measured viscosity is cleanly proportional to (1−α)/Δt, ratios to the predicted value of 0.18, 0.18, 0.22, but the constant is about 1/30 rather than 1/6. The proportionality is the physics; the constant depends on the transfer kernels, and ours differ from the reference implementation's. Practical consequence for the Bifrost build: treat Eq. 13 as a scaling law and calibrate the constant once per implementation with exactly this shear-decay test.

## 4. No bounce-back heuristics needed — SUPPORTED

The paper claims phases self-separate without the particle-bounce heuristics earlier two-phase FLIP needed. We ran the dam break under air at 1000:1 and measured the fraction of particles beyond the paper's own escape threshold on the wrong side of the interface.

Result (`mixing.png`): the wrong-side fraction peaks at 0.14 percent and does not grow. At the paper's threshold those few particles are exactly its droplet and bubble candidates, not a failure mode.

## Overall

The core story survives contact: air-as-particles produces qualitatively different, physically correct behavior that free-surface FLIP cannot; the particle-carried phase is diffusion-free where the classical alternative erodes; mixing is self-limiting; and the viscosity law holds in form with an implementation-specific constant. One claim was sharpened rather than confirmed: Eq. 13's factor 1/6 should be read as belonging to their kernels, not to the method.

Gate passed. Bifrost milestone 3 is unblocked, with one carried instruction: calibrate the viscosity constant in the Bifrost build using the shear-decay test before trusting any α-to-viscosity mapping.
