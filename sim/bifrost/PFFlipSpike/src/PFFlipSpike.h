//-
// PF-FLIP feasibility spike for Bifrost: a variable-coefficient pressure
// projection (the one kernel PF-FLIP needs that Bifrost does not ship)
// exercised on a built-in 1D two-phase hydrostatic test.
//
// The test: a closed 1D column, heavy liquid in the lower half, light air in
// the upper half (density ratio configurable, e.g. 1000). Gravity kicks every
// interior face velocity to g*dt. A correct two-phase projection must return
// the column exactly to rest (hydrostatic equilibrium). We solve the
// variable-coefficient Poisson equation with unpreconditioned conjugate
// gradient and report how still the fluid is afterwards.
//+

#ifndef PF_FLIP_SPIKE_H
#define PF_FLIP_SPIKE_H

#include "PFFlipSpikeExport.h"

#include <Amino/Core/Array.h>
#include <Amino/Core/Ptr.h>
#include <Amino/Cpp/Annotate.h>

namespace PFFlip {
namespace Spike {

PF_FLIP_SPIKE_DECL
void hydrostatic_projection_test(int   cells,
                                 float density_ratio,
                                 int   max_iterations,
                                 float tolerance,
                                 float& max_velocity_after,
                                 float& pressure_jump,
                                 float& final_residual,
                                 int&   iterations_used,
                                 Amino::Ptr<Amino::Array<float>>& pressure)
    AMINO_ANNOTATE("Amino::Node");

} // namespace Spike
} // namespace PFFlip

#endif // PF_FLIP_SPIKE_H
