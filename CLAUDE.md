# Working in Aniso

Public repo: a C++ 3D-Gaussian-splatting renderer built from scratch (CPU
reference renderer, CUDA port, walkable viewer, and the Maya-to-splats
pipeline). Its sister repo, Anisim
(github.com/cuix-github/Anisim), is the two-phase PF-FLIP/ST-FLIP fluid solver
in Bifrost; it lived here under `sim/` until 2026-10-11 and moved out with its
full history. The two meet at milestone 7: simulated particles rendered as
Gaussians inside a captured scene, exchanged as per-frame `.npy` particle
caches that Anisim writes and Aniso reads. Keep solver work in Anisim, with
Anisim's own CLAUDE.md, and renderer work here.

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
  counter: always say "issue #N" or "PR #N", and name the repo when it could
  be ambiguous ("Aniso PR #N", "Anisim PR #N").
- Validate against an independent implementation, the way every renderer
  step so far was proven: the loader against a numpy reader, the renderer
  against gsplat (dB agreement on identical splats), the CUDA port against
  the CPU renderer. A claim of correctness names what it was compared to.
- Measurement discipline: profile before optimizing. Difference the walls of
  two run lengths so startup cancels. A profiling pie must close — slices sum
  to the measured total. Never combine measurements taken in different
  configurations into one ratio; re-measure in the configuration being
  decided about, and say so when a number is an estimate. Before/after speed
  claims use the same machine, same scene, same day. Serialize timing runs;
  never overlap benchmarks.
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

## The machine

Shared 5090 workstation with unbacked-up work on it. No host-wide installs:
the CUDA toolkit and Python live inside the repo (`.toolchain/`, `.venv/`,
set up by `pipeline/setup_toolchain.py`). Heavy runs go to background or
bedtime; the background runner kills tasks around the one-hour mark. Ask me
before anything that grows tens of GB, and before deleting my data.

## Working style

Plain language; define a term the first time it is used. Show where each
derived number came from. Explain intuition first. When a step produces
something worth seeing, say how to run it and what to expect to see. Demos
use honest clocks (startup excluded and labeled).
