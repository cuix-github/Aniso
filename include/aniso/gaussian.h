#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace aniso {

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

// Unit quaternion, stored w first, the same order as the 3DGS files (rot_0 is w).
struct Quat {
    float w = 1, x = 0, y = 0, z = 0;
};

// One 3D Gaussian with its parameters already "activated", i.e. in the units a renderer uses.
// The files store raw optimizer values instead; see ply_loader.h for the conversions.
struct Gaussian {
    Vec3 position;
    Vec3 scale;        // standard deviations along the local axes, in scene units
    Quat rotation;     // orientation of the local axes
    float opacity = 0; // peak opacity at the centre, in [0, 1]
};

// A whole scene. Colour is kept apart from the geometry because its size depends on the
// spherical-harmonics degree: (degree + 1)^2 coefficients per colour channel.
struct Scene {
    std::vector<Gaussian> gaussians;

    int shDegree = 0;
    // Laid out per Gaussian as [coefficient][channel]: coefficient 0 is the view-independent
    // "DC" term, then the higher orders. Stride per Gaussian is shCoeffsPerChannel() * 3.
    std::vector<float> sh;

    int shCoeffsPerChannel() const { return (shDegree + 1) * (shDegree + 1); }
    std::size_t size() const { return gaussians.size(); }
    const float* shOf(std::size_t i) const { return sh.data() + i * shCoeffsPerChannel() * 3; }
};

// The zeroth spherical-harmonic basis constant, 1 / (2 * sqrt(pi)).
inline constexpr float kShC0 = 0.28209479177387814f;

// View-independent colour of a Gaussian: 0.5 + C0 * DC coefficient, per channel.
inline std::array<float, 3> dcColor(const Scene& scene, std::size_t i) {
    const float* c = scene.shOf(i);
    return {0.5f + kShC0 * c[0], 0.5f + kShC0 * c[1], 0.5f + kShC0 * c[2]};
}

} // namespace aniso
