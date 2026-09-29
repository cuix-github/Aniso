#pragma once

#include "aniso/gaussian.h"

#include <array>

namespace aniso {

// Colour of Gaussian i seen along `dir` (unit vector from the camera centre to the Gaussian),
// from its spherical-harmonic coefficients up to `degree` (clamped to the scene's degree).
//
// Spherical harmonics are to colour-over-directions what a Fourier series is to a signal over
// time: degree 0 is a constant (the base colour), degree 1 adds a gradient across directions,
// and each higher degree adds finer variation. Degree 3 has 16 basis functions per channel.
// The result is offset by 0.5 and clamped at zero, as the reference code does.
std::array<float, 3> shColor(const Scene& scene, std::size_t i, const Vec3& dir, int degree);

} // namespace aniso
