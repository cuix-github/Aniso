// The GPU renderer must agree with the CPU renderer, the reference, on hand-built scenes.

#include "aniso/gpu_render.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)

aniso::Camera camera(int w, int h) {
    aniso::Camera c;
    c.width = w; c.height = h;
    c.fx = c.fy = 0.8 * w; c.cx = w / 2.0; c.cy = h / 2.0;
    for (int i = 0; i < 3; ++i) c.R[i][i] = 1;
    return c;
}

double psnr(const aniso::Image& a, const aniso::Image& b) {
    double se = 0;
    for (std::size_t i = 0; i < a.rgb.size(); ++i) {
        const double d = (double(a.rgb[i]) - double(b.rgb[i])) / 255.0;
        se += d * d;
    }
    const double mse = se / a.rgb.size();
    return mse == 0 ? 1e9 : 10 * std::log10(1 / mse);
}

// A random cloud of coloured, rotated, stretched Gaussians in front of the camera, with
// view-dependent colour, so every stage of both renderers is exercised.
aniso::Scene randomScene(int n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1, 1), pos01(0, 1);
    aniso::Scene s;
    s.shDegree = 3;
    for (int i = 0; i < n; ++i) {
        aniso::Gaussian g;
        g.position = {u(rng) * 2, u(rng) * 1.2f, 3 + pos01(rng) * 4};
        g.scale = {0.02f + 0.1f * pos01(rng), 0.02f + 0.05f * pos01(rng), 0.01f + 0.03f * pos01(rng)};
        aniso::Quat q{u(rng), u(rng), u(rng), u(rng)};
        const float len = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        g.rotation = {q.w / len, q.x / len, q.y / len, q.z / len};
        g.opacity = 0.2f + 0.8f * pos01(rng);
        s.gaussians.push_back(g);
        for (int k = 0; k < 48; ++k) s.sh.push_back(k < 3 ? u(rng) * 1.5f : u(rng) * 0.15f);
    }
    return s;
}

} // namespace

int main() {
    if (!aniso::GpuRenderer::available()) {
        std::printf("no CUDA device: skipped\n");
        return EXIT_SUCCESS;
    }
    const aniso::Scene scene = randomScene(20000);
    aniso::GpuRenderer gpu(scene);
    for (auto [w, h] : {std::pair{320, 240}, std::pair{333, 217}}) { // odd size: partial tiles
        const auto cam = camera(w, h);
        const auto cpuImg = aniso::render(scene, cam);
        const auto gpuImg = gpu.render(cam);
        const double p = psnr(cpuImg, gpuImg);
        std::printf("%dx%d: GPU vs CPU PSNR %.1f dB\n", w, h, p);
        CHECK(p > 45.0);
    }
    // Degree 0 on both sides also agrees.
    aniso::RenderOptions dc;
    dc.shDegree = 0;
    const auto cam = camera(320, 240);
    CHECK(psnr(aniso::render(scene, cam, nullptr, dc), gpu.render(cam, nullptr, dc)) > 45.0);
    // Empty view: everything behind the camera gives black, not a crash.
    aniso::Camera behind = cam;
    behind.R[2][2] = -1;
    behind.R[0][0] = -1;
    const auto black = gpu.render(behind);
    bool allBlack = true;
    for (auto v : black.rgb) allBlack = allBlack && v == 0;
    CHECK(allBlack);
    if (failures == 0) std::printf("all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
