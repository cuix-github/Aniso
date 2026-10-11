# Working in Aniso

Public repo: a C++ 3D-Gaussian-splatting renderer built from scratch, plus a
two-phase PF-FLIP/ST-FLIP fluid solver implemented in Autodesk Bifrost through
the PUBLIC SDK only. The two halves meet at milestone 7: a
tsunami in a captured scene, simulated particles rendered as Gaussians. The
current technical state, milestone by milestone, lives in `sim/README.md` —
read it before touching the sim side; `sim/README.md`'s "Open questions" is
kept current and supersedes anything older.

## Hard walls

- Public information only: the public Maya/Bifrost SDK, shipped nodes and
  public documentation, never non-public source. Everything must stay
  re-derivable from that public surface; when Bifrost behavior is unclear,
  ask me rather than guess.
- Licensed assets (Moana Island, Pixar Kitchen) and anything derived from
  them (renders, splats, trained models) never enter this public repo. Local
  assets live under `C:\Users\5d149\source\Aniso-assets\` with a manifest.

## Process

- Every change lands via its own PR that I review and merge, usually fast
  and often mid-session. The repo squash-merges: NEVER push onto a branch
  whose PR may have merged — check `gh pr view <N> --json state` first, and
  default to a fresh branch off `origin/main`. Issues and PRs share one
  counter: always say "issue #N" or "PR #N".
- Validation before features: the executable spec is
  `sim/reference2d/pfflip2d.py`. Every solver feature ships with its gate —
  unit lockstep against the spec, then a system gate through the real graph,
  then physics anchors. Compatibility paths (nz=1, st=0, adapt=0, dump_every=1)
  must stay bit-identical to what the old gates measured. When comparing the
  C++ node against the python spec, feed the spec the float32-rounded inputs
  the node's ports see, or boundary-straddling particles shift their stencils.
- Measurement discipline: profile before optimizing (`PFFLIP_PROFILE=1` makes
  the solver nodes print per-phase times to stderr). Difference the walls of
  two run lengths so startup cancels. A profiling pie must close — slices sum
  to the measured total. Never combine measurements taken in different
  pipeline configurations into one ratio; re-measure in the configuration
  being decided about, and say so when a number is an estimate. Before/after
  speed claims use the same machine, same state, same day — the canonical
  profiling state is the mid-bore dump in `sim/bifrost/out/loop3_D_finale_c1/`.
  Serialize timing runs; never overlap benchmarks.
- Comparison-experiment protocol (rules set 2026-10-11 after a
  dirty-worktree experiment compared a commit against itself and burned
  rounds chasing the contradiction). Non-negotiable, every A/B experiment:
  (1) verify the bench before measuring - assert the worktree/checkout state
  is exactly what the experiment claims (`git status` clean, the intended
  sha), never assume a cleanup from a previous step worked; (2) check the
  return code of every state-changing step, especially restores - a silent
  `git checkout -- .` failure poisons everything after it; (3) run a control
  first - reproduce the known result in the current environment before
  touching the variable, and if the control disagrees, stop; (4) change one
  variable per run, with each arm writing to its own output directory,
  never shared; (5) keep every arm's artifacts until conclusions land, so
  contradictions are settled by diffing files instead of re-running; and
  (6) when two experiments contradict each other, audit the bench before
  theorizing about the physics.
- The Bifrost boundary (2026-10-09): my dependence on
  Bifrost is about whole-workflow smoothness in the Maya ecosystem, not about
  any single stock node. So: Bifrost owns orchestration — graphs, ports,
  scene integration, playback; custom SDK nodes own compute-heavy inner
  loops. Stay on stock nodes until a measurement justifies custom code, then
  move the arithmetic without disturbing the workflow.

## The machine

Shared 5090 workstation with unbacked-up work on it. Heavy 3D runs go to
background/bedtime; the background runner kills tasks around the one-hour
mark, so long suites need chunked resume via full-state dumps. Watch disk:
per-substep dumps once grew to 31 GB (the `dump_every` port now gates them);
ask me before anything that grows tens of GB, and before deleting my data.

## Working style

Plain language; define a term the first time it is used. Show where each
derived number came from. Explain intuition first. Demos: violent splash
scenes only, simulations around 20 seconds not 5, honest compute clocks
(startup excluded and labeled).
