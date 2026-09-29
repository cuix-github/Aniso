#pragma once

#include "aniso/gaussian.h"

#include <filesystem>
#include <stdexcept>

namespace aniso {

struct PlyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Loads a scene written by the reference 3D Gaussian Splatting code (and by gsplat and most
// tools that follow it): binary little-endian PLY, one "vertex" element per Gaussian, all
// properties float32.
//
// The file stores the optimizer's raw values, so the loader converts them:
//   scale_i     is log(sigma)      -> scale    = exp(scale_i)
//   opacity     is logit(alpha)    -> opacity  = 1 / (1 + exp(-opacity))
//   rot_0..3    is (w, x, y, z)    -> rotation = normalized
//   f_dc_c      DC coefficient of channel c
//   f_rest_k    higher orders, stored channel-major: all of red, then green, then blue.
//               The loader reorders them to [coefficient][channel].
// Normals (nx, ny, nz) are written by the reference code but unused; they are ignored.
//
// Throws PlyError with a readable message on anything it does not understand.
Scene loadPly(const std::filesystem::path& path);

} // namespace aniso
