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
             Amino::Ptr<Amino::Array<float>>& out_phase_state,
             Amino::Ptr<Amino::Array<float>>& out_wt,
             Amino::Ptr<Amino::Array<float>>& out_wtph,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_pos_synced,
             int&   iterations_used,
             float& final_residual,
             float& max_divergence_after,
             float& max_speed)
    AMINO_ANNOTATE("Amino::Node");

} // namespace Solve
} // namespace PFFlip

#endif // PF_FLIP_SOLVE_H
