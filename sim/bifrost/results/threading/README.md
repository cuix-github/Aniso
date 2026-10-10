# Deterministic transfer threading

On 2026-10-10, Xue authorized threading after asking the successor agent to
learn the experiment discipline from the earlier work. The starting point was
main at `9eb7bc6`, after the fused P2G and profiling changes. PR #31's withdrawn
4x claim mattered here: a faster inner loop is not evidence of a faster whole
simulation, and the old 10.3-second timing is not a substitute for a new
before/after pair.

The implementation assigns disjoint slabs of the destination grid to workers.
Each slab receives a stable list of particles in their original input order;
particles whose support crosses a boundary appear in both lists, but each
worker writes only its own faces. This preserves the old sequence of additions
at every destination, rather than merely making a new reduction reproducible.
There are no atomics and no full grid allocated per worker. The extra storage
is a bounded duplication of particle indices. Small inputs, one worker, and
builds without OpenMP take the serial path.

The same ownership scheme handles the two count scatters inside the particle
stage, which drive escape detection and adaptation. Independent sampling and
obstacle sweeps run in parallel. Spacetime draws still come from the original
serial random-number stream; only their velocity attenuation and normalization
run in parallel. Particle merge/split ordering, pressure reductions, and the
simulation graph's ports and equations are unchanged.

The independent baseline is a copied operator pack built from the starting
main, before editing the C++ files. `threading_experiment.py` runs either pack
through the actual Bifrost graphs in separate processes, with an explicit
worker count. Its unit cases include both particle sizes, all ST/adapt modes,
empty inputs, out-of-domain particles, shuffled ordering, and float32 values
at and immediately beside ownership boundaries. Its loop cases save all
eleven state arrays as well as every frame and diagnostic, in 2D and 3D, with
both dump cadences. Comparison uses file hashes, not a visual or tolerance
threshold. The older independent NumPy and physics tests remain separate gates.

For speed, the canonical 256x64x128 mid-bore state is resumed at four and
twelve substeps. Timing runs are serialized. The difference of their walls
removes startup approximately; timestamps on flushed per-step profiling lines
also measure substeps 4 through 11 inside the long run itself. The latter
interval provides the denominator for the phase breakdown, so a fluctuating
startup or CPU load cannot manufacture a negative graph slice. Both numbers
are retained in the receipts. Dumping every thousandth step limits disk use;
the same cadence and input hashes apply to every compared build.

The machine is a Ryzen 9 9950X3D: 16 physical cores, 32 logical workers. The
same-day baseline measured 8.44 and 8.26 seconds per substep; two threaded runs
at 32 workers measured 4.62 and 4.44 seconds. That supports about a 1.8x whole
pipeline improvement. The 16-worker run was 4.71 seconds and the eight-worker
run 4.85 seconds. These are measured cases, not a claim of linear scaling.
Every saved array in those large-state runs matched the original build exactly.

The phase breakdown below uses the last eight substeps of the final long run
for each build, all at 32 workers. Times are seconds per substep. Its own
observed interval is used as the total; the separate four/twelve wall
difference above varies slightly with startup and machine load.

| Work | Original build | Threaded build |
| --- | ---: | ---: |
| P2G scatter, including allocation and binning | 2.096 | 0.281 |
| Particle stage, including both count scatters | 3.662 | 1.447 |
| Pressure solve | 2.319 | 2.418 |
| Packing, assembly, correction, graph and dispatch | 0.313 | 0.336 |
| Total observed interval | 8.391 | 4.483 |

The pressure code is unchanged. Its small timing variation is not an algorithm
change, and its new roughly 54-percent share comes from this measured threaded
configuration. This is the next useful baseline for milestone 6.5a. Random
draw generation, stable merge/split ordering, and reductions still have serial
costs. Partitioning only x also limits available parallelism in very narrow
domains; this experiment establishes the improvement on the measured scenes,
not every possible grid shape.

The completed identity matrix compares 774 saved arrays per worker count at
1, 8, and 32 workers against the independent main build. Every array is
byte-identical. A further 248-array comparison covers a larger flat 96x48
loop in all four ST/adapt combinations: the small flat case exercises the
serial fallback, so it could not by itself establish compatibility of the
parallel `nz=1` path. The large finale also matches the baseline at 8, 16,
and 32 workers, including both repeated 32-worker runs.

Nine of the ten existing regression scripts passed, including the independent
P2G reference, flat-step lockstep, hydrostatic pressure jump, symmetry, the
independent 3D step, second-order convergence, momentum identity, escape,
spacetime-node, solids, adaptive-node, fused-loop, and adaptive-loop checks.
The momentum residual is 2.67e-8, pressure refinement ratios are 4.00 and 4.00,
and the full adaptive loop has zero represented-volume drift. The commands and
unabridged results are in `validation.txt`.

One inherited gate is red and remains explicitly open: `validate_st_loop.py`
on the stock graph reports peak droplets 41 versus the reference's 94, and
bubbles 56 versus 259; the bubble ratio is below its 0.4 lower bound. Its 3D
smoke passes. Re-running the unchanged script against the preserved main pack
produced the same failure and all 502 saved arrays matched the threaded run
byte for byte. Thus this is not introduced by threading. No tolerance was
changed, and the suite is not described as wholly green. The bounded follow-up
is to explain the stock-graph ST bubble-count discrepancy against the Python
reference; this PR establishes numerical equivalence to the existing build,
not a repair of that pre-existing validation gap.

The 20.04-second 3D bore, at 64x16x32 with two pillars and ST, adaptivity,
escape and multigrid enabled, compares another 1,146 arrays across its short
and full runs. All match byte for byte. Both builds conserve represented and
liquid volume exactly at every saved step, and all 167 actual solver states
have zero particles inside either pillar. Peak escaped counts are 1,270
droplets and 3,318 bubbles. The small scene measures 0.178 versus 0.095 seconds
per substep, about 1.88x, using the 20/167 wall difference. The large-state
headline remains the independently measured roughly 1.8x above.

This audit also exposed a second inherited defect. The resynchronized display
positions (`out_pos_synced`, used by the production frame writer) enter the
pillars in 125 frames, starting at zero-based substep 16: 1,225 particle-frame
occurrences in each build. These are not 1,225 distinct particles. The final
obstacle sweep protects the actual state, but the later time resynchronization
can move the display position back into a solid. An earlier handoff said that
display positions had been clamped too; this baseline does not support that
claim. The diagnostic graph adds a writer for `out_positions` so both paths
are audited separately. All 959 common outputs match the earlier run without
that extra writer, which checks that the instrumentation has not changed the
physical outputs. Both audited demo commands deliberately exit with a display
audit failure after saving their receipts, even when the identity comparison
passes. This remains a separate rendering/collision follow-up; neither the
coordinates nor the acceptance condition were altered for this PR.

[`threading_sidebyside.mp4`](threading_sidebyside.mp4) plays those original
display positions without hiding the defect and labels it throughout. The
panels advance at equal simulation time; their compute clocks use each run's
measured average step cost with startup excluded. They are not literal
per-frame stopwatch readings. The 167-frame video is 20.04 seconds long.
[`threading_phases.png`](threading_phases.png) shows the large-state cost
breakdown. [`receipts.json`](receipts.json) retains the build hashes, timing
repeats, array-manifest hashes and audit results. A manifest digest is SHA-256
of the sorted filename-to-file-hash JSON map, encoded as UTF-8 with compact
separators; full raw maps and arrays remain under ignored `out/threading/`.
[`validation.txt`](validation.txt) contains the existing tests' complete
output and the exact wrappers used, including the baseline recheck.

To reproduce, build main at `9eb7bc6` in a separate checkout and preserve its
entire installed `PFFlipSpike-1.0.0` pack, then build this branch with
`sim/bifrost/PFFlipSpike/build.bat`. Set `$baselinePack` below to that preserved
pack's `PFFlipSpikePackConfig.json`. Commands run from the Aniso root; each
label must be new because the driver refuses to reuse output directories.
The default SDK is Bifrost 3.1.0.8 for Maya 2027; `--bifrost` overrides it.

```powershell
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode gate --label original_gate --threads 32 --pack $baselinePack
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode gate --label new_gate32 --threads 32 --compare sim/bifrost/out/threading/original_gate/receipt.json
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode flat --label original_flat --threads 32 --pack $baselinePack
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode flat --label new_flat32 --threads 32 --compare sim/bifrost/out/threading/original_flat/receipt.json
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode bench --label original_bench --threads 32 --pack $baselinePack
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode bench --label new_bench32 --threads 32 --compare sim/bifrost/out/threading/original_bench/receipt.json
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode demo3d --label original_demo --threads 32 --pack $baselinePack
.\.venv\Scripts\python.exe sim/bifrost/threading_experiment.py --mode demo3d --label new_demo --threads 32 --compare sim/bifrost/out/threading/original_demo/receipt.json
.\.venv\Scripts\python.exe sim/bifrost/render_threading_demo.py sim/bifrost/out/threading/original_demo sim/bifrost/out/threading/new_demo sim/bifrost/out/threading/comparison.mp4
```

Repeat the gate with 1 and 8 workers and the benchmark with 8 and 16 workers,
using fresh labels. Run timing comparisons serially, on the same machine and
same canonical `out/loop3_D_finale_c1/` state; the driver records the derived
input hashes. Do not pair a historical timing with a new denominator. The
two demo commands currently return nonzero for the recorded display defect;
inspect their saved invariants and identity result rather than describing
that gate as passed. The renderer can intentionally show an identical pair
with a failed display audit, but labels the issue and refuses unequal states.
