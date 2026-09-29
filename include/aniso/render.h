#pragma once

#include "aniso/camera.h"
#include "aniso/gaussian.h"
#include "aniso/image.h"

namespace aniso {

struct RenderStats {
    std::size_t visible = 0;     // Gaussians that survived culling and were splatted
    std::size_t tilePairs = 0;   // Gaussian-tile overlaps, the unit of work in blending
    double projectMs = 0, sortMs = 0, blendMs = 0;
};

// Renders the scene from a camera, on the CPU, the way the reference rasterizer does:
//   1. For each Gaussian, build its 3D covariance from scale and rotation, move it into camera
//      space, and project it to a 2D covariance with the perspective Jacobian. Add a small
//      low-pass term so no splat is thinner than about a pixel. Invert it (the "conic") and
//      take a radius of three standard deviations.
//   2. Bin each splat into the 16x16-pixel tiles it overlaps and sort each tile's splats by
//      depth.
//   3. For each pixel, blend its tile's splats front to back:
//         alpha = min(0.99, opacity * exp(-0.5 * d^T conic d)),  C += T * alpha * colour,
//         T *= 1 - alpha,  stopping once T would fall below 1e-4.
// Colour is evaluated from the spherical harmonics along the direction from the camera centre to
// each Gaussian, up to shDegree (0 gives the view-independent colour only).
// Background is black. Tiles are blended in parallel across all hardware threads.
struct RenderOptions {
    int shDegree = 3;
};

Image render(const Scene& scene, const Camera& camera, RenderStats* stats = nullptr,
             const RenderOptions& options = {});

} // namespace aniso
