// Checks the projection maths on hand-built cameras, and the PNG writer's output size.

#include "aniso/camera.h"
#include "aniso/image.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)
bool near(double a, double b, double eps = 1e-4) { return std::fabs(a - b) <= eps; }

aniso::Camera identityCamera() {
    aniso::Camera c;
    c.width = 200; c.height = 100;
    c.fx = c.fy = 100; c.cx = 100; c.cy = 50;
    for (int i = 0; i < 3; ++i) c.R[i][i] = 1;
    return c;
}

void testProjection() {
    const aniso::Camera c = identityCamera();
    // On the optical axis: lands on the principal point.
    auto p = aniso::project(c, c.toCamera({0, 0, 5}));
    CHECK(p && near(p->u, 100) && near(p->v, 50) && near(p->depth, 5));
    // One unit right at depth 2: 100 * 1 / 2 = 50 pixels right. +y is down the image.
    p = aniso::project(c, c.toCamera({1, 1, 2}));
    CHECK(p && near(p->u, 150) && near(p->v, 100));
    // Behind the camera and too close are rejected.
    CHECK(!aniso::project(c, c.toCamera({0, 0, -1})));
    CHECK(!aniso::project(c, c.toCamera({0, 0, 0.1f})));
}

void testTransform() {
    // Camera turned 90 degrees about +y and moved: world +x becomes camera -z... checked by
    // round trip instead: the camera centre must map to the camera-space origin.
    aniso::Camera c = identityCamera();
    const double s = std::sqrt(0.5);
    c.R[0][0] = 0; c.R[0][2] = -1;
    c.R[2][0] = 1; c.R[2][2] = 0;
    c.t[0] = 3; c.t[1] = -2; c.t[2] = 7;
    (void)s;
    const aniso::Vec3 o = c.centre();
    const aniso::Vec3 back = c.toCamera(o);
    CHECK(near(back.x, 0) && near(back.y, 0) && near(back.z, 0));
}

void testPng() {
    aniso::Image img(3, 2);
    img.at(1, 1)[0] = 255;
    const auto path = std::filesystem::temp_directory_path() / "aniso_test.png";
    aniso::writePng(path, img);
    // signature 8 + IHDR 25 + IDAT (12 + zlib 2 + block 5 + 2*(1+9) + adler 4) + IEND 12
    CHECK(std::filesystem::file_size(path) == 8 + 25 + (12 + 2 + 5 + 20 + 4) + 12);
    std::filesystem::remove(path);
}

} // namespace

int main() {
    testProjection();
    testTransform();
    testPng();
    if (failures == 0) std::printf("all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
