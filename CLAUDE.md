# Working in Aniso

Public repo: a C++ 3D-Gaussian-splatting renderer built from scratch, plus a
two-phase PF-FLIP/ST-FLIP fluid solver implemented in Autodesk Bifrost through
the PUBLIC SDK only. Owner: Xue (QAF). The two halves meet at milestone 7: a
tsunami in a captured scene, simulated particles rendered as Gaussians. The
current technical state, milestone by milestone, lives in `sim/README.md` —
read it before touching the sim side; `sim/README.md`'s "Open questions" is
kept current and supersedes anything older.

## Hard walls

- Public information only. Xue works at Autodesk on Maya; his side projects
  use the public Maya/Bifrost SDK and must stay re-derivable from public
  surface. He answers questions about Bifrost at conference-talk altitude —
  ask him, never request or copy internal source.
- Licensed assets (Moana Island, Pixar Kitchen) and anything derived from
  them (renders, splats, trained models) never enter this public repo. Local
  assets live under `C:\Users\5d149\source\Aniso-assets\` with a manifest.

## Process

- Every change lands via its own PR that Xue reviews and merges, usually fast
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
- The Bifrost boundary (Xue's own words, 2026-10-09): his dependence on
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
ask Xue before anything that grows tens of GB, and before deleting his data.

## Working with Xue

Plain language; define a term the first time it is used or expect to be asked.
He audits derived numbers — show where each number came from. Education
intuition-first; when he is struggling, supplement in Chinese (mixed register,
keep English technical terms). Demos: violent splash scenes only, simulations
around 20 seconds not 5, honest compute clocks (startup excluded and labeled).
When he shares why something matters, answer in kind before any how.
