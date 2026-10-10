//-
// Milestone 5: the PF-FLIP step for Bifrost, 2D on a MAC grid.
//
// The graph assembles the staggered channels with stock nodes (two shifted
// splat_points_into_volume passes produce, per face grid, the weighted mean
// particle mass, mean mass-weighted momentum, and mean phase), samples them
// into flat arrays, and this node does the rest of one simulation step:
// normalize to face velocities, add gravity, solve the variable-coefficient
// pressure Poisson equation (Jacobi-preconditioned CG), correct, grid-to-
// particle with the per-phase FLIP blend, and RK2 advection with wall clamps.
//
// The executable spec is sim/reference2d/pfflip2d.py: this file mirrors its
// project(), g2p() and advect() line by line (double precision internally),
// so the two can be compared in lockstep on identical channel inputs.
//
// Flat layouts, row-major matching numpy's ravel():
//   u faces: (nx+1) x ny,  index i*ny + j
//   v faces: nx x (ny+1),  index i*(ny+1) + j
//   pressure: nx x ny,     index i*ny + j
// Particles are float3 with z carried through untouched.
//+

#ifndef PF_FLIP_SOLVE_H
#define PF_FLIP_SOLVE_H

#include "PFFlipSpikeExport.h"

#include <Amino/Core/Array.h>
#include <Amino/Core/Ptr.h>
#include <Amino/Core/String.h>
#include <Amino/Cpp/Annotate.h>
#include <Bifrost/Math/Types.h>

namespace PFFlip {
namespace Solve {

PF_FLIP_SPIKE_DECL
void step_2d(int   nx,
             int   ny,
             float dt,
             float gravity,
             float rho_liquid,
             float rho_air,
             float alpha_liquid,
             float alpha_air,
             int   max_iterations,
             float tolerance,
             const Amino::Array<float>& u_mass,
             const Amino::Array<float>& u_mom,
             const Amino::Array<float>& u_phase,
             const Amino::Array<float>& v_mass,
             const Amino::Array<float>& v_mom,
             const Amino::Array<float>& v_phase,
             const Amino::Array<Bifrost::Math::float3>& positions,
             const Amino::Array<Bifrost::Math::float3>& velocities,
             const Amino::Array<float>& particle_phase,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_positions,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_velocities,
             Amino::Ptr<Amino::Array<float>>& out_mass,
             Amino::Ptr<Amino::Array<float>>& out_mom_x,
             Amino::Ptr<Amino::Array<float>>& out_mom_y,
             Amino::Ptr<Amino::Array<float>>& out_u,
             Amino::Ptr<Amino::Array<float>>& out_v,
             Amino::Ptr<Amino::Array<float>>& pressure,
             int&   iterations_used,
             float& final_residual,
             float& max_divergence_after,
             float& max_speed)
    AMINO_ANNOTATE("Amino::Node");

// Milestone 6: the same step in 3D on a uniform MAC grid, with a geometric
// multigrid-preconditioned CG solve (preconditioner 0 = Jacobi, 1 = multigrid
// V-cycle). At nz = 1 the w faces are all walls and the step reduces exactly
// to the 2D method, which is the lockstep bridge used by validate_3d.py.
// Flat layouts, row-major like numpy ravel() on (nx[+1], ny[+1], nz[+1]):
//   u: (nx+1)*ny*nz   index (i*ny + j)*nz + k
//   v: nx*(ny+1)*nz   index (i*(ny+1) + j)*nz + k
//   w: nx*ny*(nz+1)   index (i*ny + j)*(nz+1) + k
//   p: nx*ny*nz       index (i*ny + j)*nz + k
PF_FLIP_SPIKE_DECL
void step_3d(int   nx,
             int   ny,
             int   nz,
             float dt,
             float gravity,
             float rho_liquid,
             float rho_air,
             float alpha_liquid,
             float alpha_air,
             int   max_iterations,
             float tolerance,
             int   preconditioner,
             int   escape,
             float esc_phi,
             float drag_droplet,
             float drag_bubble,
             float buoyancy,
             float rho0_face,
             int   st,
             int   st_seed,
             int   step_index,
             int   adapt,
             float coarse_gain,
             float ws_epsilon,
             const Amino::Array<float>& scale_in,
             const Amino::Array<Bifrost::Math::float3>& obstacles_min,
             const Amino::Array<Bifrost::Math::float3>& obstacles_max,
             const Amino::Array<float>& tau_in,
             const Amino::Array<float>& u_wt,
             const Amino::Array<float>& v_wt,
             const Amino::Array<float>& w_wt,
             const Amino::Array<float>& u_ws,
             const Amino::Array<float>& v_ws,
             const Amino::Array<float>& w_ws,
             const Amino::Array<float>& u2_mass,
             const Amino::Array<float>& u2_mom,
             const Amino::Array<float>& u2_phase,
             const Amino::Array<float>& u2_wt,
             const Amino::Array<float>& u2_ws,
             const Amino::Array<float>& v2_mass,
             const Amino::Array<float>& v2_mom,
             const Amino::Array<float>& v2_phase,
             const Amino::Array<float>& v2_wt,
             const Amino::Array<float>& v2_ws,
             const Amino::Array<float>& w2_mass,
             const Amino::Array<float>& w2_mom,
             const Amino::Array<float>& w2_phase,
             const Amino::Array<float>& w2_wt,
             const Amino::Array<float>& w2_ws,
             const Amino::Array<float>& u_mass,
             const Amino::Array<float>& u_mom,
             const Amino::Array<float>& u_phase,
             const Amino::Array<float>& v_mass,
             const Amino::Array<float>& v_mom,
             const Amino::Array<float>& v_phase,
             const Amino::Array<float>& w_mass,
             const Amino::Array<float>& w_mom,
             const Amino::Array<float>& w_phase,
             const Amino::Array<Bifrost::Math::float3>& positions,
             const Amino::Array<Bifrost::Math::float3>& velocities,
             const Amino::Array<float>& particle_phase,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_positions,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_velocities,
             Amino::Ptr<Amino::Array<float>>& out_mass,
             Amino::Ptr<Amino::Array<float>>& out_mom_x,
             Amino::Ptr<Amino::Array<float>>& out_mom_y,
             Amino::Ptr<Amino::Array<float>>& out_mom_z,
             Amino::Ptr<Amino::Array<float>>& out_u,
             Amino::Ptr<Amino::Array<float>>& out_v,
             Amino::Ptr<Amino::Array<float>>& out_w,
             Amino::Ptr<Amino::Array<float>>& pressure,
             Amino::Ptr<Amino::Array<float>>& out_escaped,
             Amino::Ptr<Amino::Array<float>>& tau_out,
             Amino::Ptr<Amino::Array<float>>& scale_out,
             Amino::Ptr<Amino::Array<bool>>& mask_fine,
             Amino::Ptr<Amino::Array<bool>>& mask_coarse,
             Amino::Ptr<Amino::Array<float>>& out_phase_state,
             Amino::Ptr<Amino::Array<float>>& out_wt,
             Amino::Ptr<Amino::Array<float>>& out_wtph,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_pos_synced,
             int&   iterations_used,
             float& final_residual,
             float& max_divergence_after,
             float& max_speed)
    AMINO_ANNOTATE("Amino::Node");

// Fused P2G (milestone 7 prep): one pass over the particles scatters every
// channel the step node consumes - mass, per-axis momentum, phase, and the
// ST temporal weight - onto all three MAC face grids and both size tiers,
// directly into flat arrays, replacing the per-channel stock splat + sample
// round-trips. The math is the reference's (pfflip2d.py _splat_group): Eq. 6
// kernel w = max(1 - d^2/r^2, 0)^3 on a (2 ceil r + 1)^d stencil, r = 1 fine
// and r = 2 coarse, face frames base = floor(p' + 0.5) with p' = pos minus
// the face offset. Channel semantics stay what the stock path produced, so
// step_3d is untouched: weighted mean with eps_mean added to the
// denominator, and ws = K/(K + ws_epsilon) so the step node's eps-inversion
// recovers the exact kernel sum K. st == 0 leaves the wt channels empty
// (plain-phi path); adapt == 0 leaves ws and every tier-2 channel empty and
// splats all particles into tier 1, matching the old graphs exactly.
PF_FLIP_SPIKE_DECL
void p2g_3d(int   nx,
            int   ny,
            int   nz,
            int   adapt,
            int   st,
            float eps_mean,
            float ws_epsilon,
            const Amino::Array<Bifrost::Math::float3>& positions,
            const Amino::Array<float>& scale,
            const Amino::Array<float>& mass,
            const Amino::Array<float>& momx,
            const Amino::Array<float>& momy,
            const Amino::Array<float>& momz,
            const Amino::Array<float>& wt,
            const Amino::Array<float>& phase,
            Amino::Ptr<Amino::Array<float>>& u_mass,
            Amino::Ptr<Amino::Array<float>>& u_mom,
            Amino::Ptr<Amino::Array<float>>& u_phase,
            Amino::Ptr<Amino::Array<float>>& u_wt,
            Amino::Ptr<Amino::Array<float>>& u_ws,
            Amino::Ptr<Amino::Array<float>>& v_mass,
            Amino::Ptr<Amino::Array<float>>& v_mom,
            Amino::Ptr<Amino::Array<float>>& v_phase,
            Amino::Ptr<Amino::Array<float>>& v_wt,
            Amino::Ptr<Amino::Array<float>>& v_ws,
            Amino::Ptr<Amino::Array<float>>& w_mass,
            Amino::Ptr<Amino::Array<float>>& w_mom,
            Amino::Ptr<Amino::Array<float>>& w_phase,
            Amino::Ptr<Amino::Array<float>>& w_wt,
            Amino::Ptr<Amino::Array<float>>& w_ws,
            Amino::Ptr<Amino::Array<float>>& u2_mass,
            Amino::Ptr<Amino::Array<float>>& u2_mom,
            Amino::Ptr<Amino::Array<float>>& u2_phase,
            Amino::Ptr<Amino::Array<float>>& u2_wt,
            Amino::Ptr<Amino::Array<float>>& u2_ws,
            Amino::Ptr<Amino::Array<float>>& v2_mass,
            Amino::Ptr<Amino::Array<float>>& v2_mom,
            Amino::Ptr<Amino::Array<float>>& v2_phase,
            Amino::Ptr<Amino::Array<float>>& v2_wt,
            Amino::Ptr<Amino::Array<float>>& v2_ws,
            Amino::Ptr<Amino::Array<float>>& w2_mass,
            Amino::Ptr<Amino::Array<float>>& w2_mom,
            Amino::Ptr<Amino::Array<float>>& w2_phase,
            Amino::Ptr<Amino::Array<float>>& w2_wt,
            Amino::Ptr<Amino::Array<float>>& w2_ws)
    AMINO_ANNOTATE("Amino::Node");

// Dump policy: pass the write path through only every Nth substep (step 0
// always dumps; every <= 1 means every substep). An empty path makes the
// downstream write node skip the dump, which keeps per-substep file I/O out
// of production runs without touching the write nodes themselves.
PF_FLIP_SPIKE_DECL
void dump_gate(int step_index,
               int every,
               const Amino::String& path,
               Amino::String& path_out)
    AMINO_ANNOTATE("Amino::Node");

} // namespace Solve
} // namespace PFFlip

#endif // PF_FLIP_SOLVE_H
