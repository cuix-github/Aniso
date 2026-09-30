# The simulation workstream: PF-FLIP in Bifrost

This folder implements the core of *Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation* (Braun, Bender, Thuerey, SIGGRAPH 2025) inside Maya's Bifrost, using only the public Bifrost SDK and shipped nodes, and couples the result to this repository's Gaussian-splat pipeline: simulate a two-phase air-and-water shot in Bifrost, render the particles as Gaussians with Aniso inside a captured splat scene. The authors released only their grid data structure (MSBG, Apache-2.0), not the solver, so this is an independent implementation from the paper's equations.

## Why this paper, in three sentences

PF-FLIP fills air with FLIP particles too, so wind, splashes, spray, and air entrainment come from physics instead of "white-water" heuristics. Its phase field costs nothing: it is the splatted particle mass, normalized, so there is no level set and no surface reconstruction anywhere. The only genuinely new machinery over textbook FLIP is a variable-coefficient pressure solve, where each cell face is weighted by the mass that landed on it.

## Architecture decisions (settled)

- **Uniform-grid pressure solve, adaptive particles.** The paper's grid adaptivity (its Sections 4 and 6) exists to reach ocean scale; a room-scale shot does not need it, and the particle side (Section 5, including the per-size FLIP-blend correction of 5.3) is where the cost explosion actually is, because air fills the whole domain with particles. Escalation paths if footage ever demands more: higher uniform resolution, then static nested refinement, then a standalone build on the authors' Apache-2.0 MSBG. Consequence accepted: domain size is capped by the uniform solve; a snug box around the action.
- **One custom C++ node, everything else stock.** The two-phase pressure projection does not exist in the product (the shipped liquid and gas projections each assume a single density) and is our one SDK node. Verified stock wheels for the rest: `splat_points_into_volume` (arbitrary properties, linear or quadratic kernel, exposes accumulated weights, which *is* the phase field's raw material), `sample_property` for grid-to-particle, `transport_particles`, standard field math, and the public iteration machinery the shipped solvers themselves are built from.
- **Public surface only.** Shipped files, public SDK, public docs. The custom node reads and writes plain arrays attached as custom properties, deliberately independent of Bifrost's internal volume layout.
- **Reference first.** A small pure-Python 2D implementation of the core loop (`reference2d/`) is the executable spec; every Bifrost stage is validated against it and against the paper's own test cases.

## What is in this folder

- `bifrost/PFFlipSpike/`: the feasibility spike, a Bifrost SDK operator pack containing the variable-coefficient projection with a built-in 1D two-phase hydrostatic test. Build with `PFFlipSpike\build.bat` (needs Visual Studio 2022 and Bifrost 3.1 installed with Maya 2027); run with `run_hydro_test.bat`.
- `bifrost/hydro_test.json`: the bifcmd wrapper compound for that test.
- `bifrost/phase_field_probe.json`: the milestone-4 graph, stock nodes only: particles in from .npy, phase splatted into a volume with `splat_points_into_volume`, phi sampled at probe positions and written back as .npy. Its `view_volume` output carries the volume with fog density set to phi, so Maya's viewport shows the phase field directly. `run_phase_field_probe.bat` runs it headlessly; `compare_phase_field.py` drives the whole comparison against the 2D reference and writes the figure; `pfflip_graphs_config.json` exposes the compound to Maya (point `BIFROST_LIB_CONFIG_FILES` at it).
- `reference2d/`: the 2D reference implementation and its validation scenes.

Spike results (2026-09-30): the projection returns a 1000:1 density-contrast column to rest at the 1e-6 level and reproduces the analytic hydrostatic pressure jump; unpreconditioned conjugate gradient needs 211 iterations at 64 cells and 1843 at 256, which is the measured argument for adding a preconditioner in 3D.

## Milestones, each with its "done when"

1. **Spike landed.** This folder builds and the hydrostatic test passes on a clean checkout. Done when `run_hydro_test.bat` reports max velocity after projection below 1e-4.
2. **2D reference validates the core loop.** Hydrostatic rest, a two-phase dam break, and a Rayleigh-Taylor instability, the last being a two-phase-only phenomenon no single-phase solver can produce. Done when RT fingers develop from a seeded perturbation and the dam break front advances plausibly, with images in the repository.
3. **Validation gate: the paper's claims tested comparatively.** Done 2026-09-30, passed: see `reference2d/VALIDATION.md` (single-phase FLIP baseline vs two-phase on a sealed trapped-air pocket; interface transport vs a grid scalar; the viscosity law calibrated; mixing measured). One carried instruction: recalibrate the blend-to-viscosity constant per implementation.
4. **Bifrost graph produces the phase field with stock nodes.** Done 2026-09-30: `phase_field_probe.json` reads the same jittered two-block slab the 2D reference seeds (identical particles via .npy), splats phase with the stock node, and the fields agree, mean |difference| 0.002, maximum 0.08 confined to the one-voxel interface band, interface midpoint within 0.03 voxels. Figure: `reference2d/results/validation/phase_field_bifrost.png`. Authoring facts that cost debugging time: a new point property needs `set_geo_property` (the `_data` variant only overwrites existing ones); `splat_points_into_volume` writes only into existing tiles, so build the volume from the points first with `points_to_volume`, and it needs fog density on or the volume has no voxels at all; give the splat one property name per node, a two-name filter string misbehaved.
5. **The custom node does the projection inside the graph loop**, on channels assembled by stock nodes; 2D dam break inside Bifrost matches the reference qualitatively. Done when a side-by-side strip of frames looks the same.
6. **3D at small scale** with adaptive particles (coarsen far-field air, per-size blend correction) and a preconditioned solve. Done when a 3D dam break in a box runs at around 128³ overnight or better and conserves volume reasonably.
7. **The shot.** A captured real scene, box colliders aligned by hand, one violent release, particles exported per frame and rendered as Gaussians by Aniso inside the splat scene, whitewater coloured from the phase field. Scene choice is still open.

## Open questions

- Hero scene: to be chosen (current candidates: flash flood down a captured staircase; dam break through a doorway; violently filling a dry fountain).
- Preconditioner for the 3D solve: incomplete Cholesky versus a simple multigrid V-cycle; decide from measured iteration counts at milestone 5.
- ~~Whether the stock splat's kernel is smooth enough for a quiet phase field~~ — answered at milestone 4: the stock linear kernel's field matches the reference, and swapping the paper's Eq. 6 kernel for a tent kernel in the reference moves the interface midpoint by under 0.01 voxels, so kernel choice barely matters for the field; the custom node does not need the paper's exact kernel.
- Volume-drift correction (the paper's divergence correction term): add at milestone 4 if the dam break visibly loses volume.
- How escaped-particle droplets and bubbles are handled for rendering; decide when the shot exists.
