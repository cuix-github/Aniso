#include "aniso/sh.h"

#include <algorithm>

namespace aniso {
namespace {

// Real spherical-harmonic normalization constants, the same values and sign conventions as
// the reference rasterizer.
constexpr float C1 = 0.4886025119029199f;
constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                         -1.0925484305920792f, 0.5462742152960396f};
constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                         -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

} // namespace

std::array<float, 3> shColor(const Scene& scene, std::size_t i, const Vec3& dir, int degree) {
    degree = std::min(degree, scene.shDegree);
    const float* sh = scene.shOf(i); // [coefficient][channel]
    const float x = dir.x, y = dir.y, z = dir.z;

    // Basis function values for this direction, in the order the coefficients are stored.
    float basis[16];
    int count = 1;
    basis[0] = kShC0;
    if (degree >= 1) {
        basis[1] = -C1 * y;
        basis[2] = C1 * z;
        basis[3] = -C1 * x;
        count = 4;
    }
    if (degree >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        basis[4] = C2[0] * x * y;
        basis[5] = C2[1] * y * z;
        basis[6] = C2[2] * (2 * zz - xx - yy);
        basis[7] = C2[3] * x * z;
        basis[8] = C2[4] * (xx - yy);
        count = 9;
        if (degree >= 3) {
            basis[9] = C3[0] * y * (3 * xx - yy);
            basis[10] = C3[1] * x * y * z;
            basis[11] = C3[2] * y * (4 * zz - xx - yy);
            basis[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            basis[13] = C3[4] * x * (4 * zz - xx - yy);
            basis[14] = C3[5] * z * (xx - yy);
            basis[15] = C3[6] * x * (xx - 3 * yy);
            count = 16;
        }
    }

    std::array<float, 3> rgb = {0.5f, 0.5f, 0.5f};
    for (int k = 0; k < count; ++k)
        for (int c = 0; c < 3; ++c) rgb[c] += basis[k] * sh[k * 3 + c];
    for (float& v : rgb) v = std::max(0.0f, v);
    return rgb;
}

} // namespace aniso
