// Renders tiny hand-built scenes whose result can be worked out by hand.

#include "aniso/render.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)

aniso::Camera camera() {
    aniso::Camera c;
    c.width = 64; c.height = 64;
    c.fx = c.fy = 64; c.cx = c.cy = 32;
    for (int i = 0; i < 3; ++i) c.R[i][i] = 1;
    return c;
}

// Adds a round Gaussian with a DC colour chosen so that 0.5 + C0 * dc equals rgb.
void add(aniso::Scene& s, aniso::Vec3 pos, float scale, float opacity, float r, float g, float b) {
    aniso::Gaussian gs;
    gs.position = pos;
    gs.scale = {scale, scale, scale};
    gs.opacity = opacity;
    s.gaussians.push_back(gs);
    for (float v : {r, g, b}) s.sh.push_back((v - 0.5f) / aniso::kShC0);
}

int px(const aniso::Image& img, int x, int y, int ch) { return img.rgb[(static_cast<std::size_t>(y) * img.width + x) * 3 + ch]; }

void testSingleGaussian() {
    aniso::Scene s;
    add(s, {0, 0, 4}, 0.2f, 0.8f, 1.0f, 0.0f, 0.0f);
    const auto img = aniso::render(s, camera());
    // The Gaussian projects to (32, 32) with a standard deviation of 0.2 * 64 / 4 = 3.2 pixels,
    // plus the 0.3 low-pass. Pixel (32, 32) is sampled at its centre (32.5, 32.5), half a pixel
    // off in each axis, so its weight is exp(-0.5 * 0.5 / variance) and the pixel is
    // opacity * weight * red over black.
    const float variance = 3.2f * 3.2f + 0.3f;
    const float expected = 0.8f * std::exp(-0.5f * (0.25f + 0.25f) / variance) * 255.0f;
    CHECK(std::abs(px(img, 32, 32, 0) - static_cast<int>(expected + 0.5f)) <= 1);
    CHECK(px(img, 32, 32, 1) == 0);
    // Far from it, nothing.
    CHECK(px(img, 2, 2, 0) == 0);
}

void testFrontHidesBack() {
    aniso::Scene s;
    add(s, {0, 0, 8}, 0.5f, 0.99f, 0.0f, 0.0f, 1.0f); // blue, behind
    add(s, {0, 0, 4}, 0.2f, 0.99f, 0.0f, 1.0f, 0.0f); // green, in front; listed second on purpose
    const auto img = aniso::render(s, camera());
    // Sorting by depth must put green first: the centre is almost pure green.
    CHECK(px(img, 32, 32, 1) > 240);
    CHECK(px(img, 32, 32, 2) < 10);
}

void testBehindCameraIsCulled() {
    aniso::Scene s;
    add(s, {0, 0, -4}, 0.5f, 0.99f, 1.0f, 1.0f, 1.0f);
    aniso::RenderStats st;
    const auto img = aniso::render(s, camera(), &st);
    CHECK(st.visible == 0);
    CHECK(px(img, 32, 32, 0) == 0);
}

} // namespace

int main() {
    testSingleGaussian();
    testFrontHidesBack();
    testBehindCameraIsCulled();
    if (failures == 0) std::printf("all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
