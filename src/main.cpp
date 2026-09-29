// aniso info <scene.ply>
//   Loads a Gaussian-splat scene and prints what is in it: the count, the SH degree, the
//   bounds, and the spread of scales, opacities, and colours. Used to check the loader
//   against tools/reference_stats.py, which reads the same file with numpy.

#include "aniso/ply_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Stats {
    double min = 0, max = 0, mean = 0;
};

Stats stats(std::vector<float> v) {
    Stats s;
    if (v.empty()) return s;
    auto [lo, hi] = std::minmax_element(v.begin(), v.end());
    s.min = *lo;
    s.max = *hi;
    double sum = 0;
    for (float x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    return s;
}

void print(const char* label, const Stats& s) {
    std::printf("  %-14s min %12.6f   max %12.6f   mean %12.6f\n", label, s.min, s.max, s.mean);
}

int info(const std::string& path) {
    const auto t0 = std::chrono::steady_clock::now();
    const aniso::Scene scene = aniso::loadPly(path);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const std::size_t n = scene.size();
    std::printf("%s\n", path.c_str());
    std::printf("  gaussians      %zu\n", n);
    std::printf("  sh degree      %d (%d coefficients per channel)\n", scene.shDegree, scene.shCoeffsPerChannel());
    std::printf("  load time      %.1f ms\n", ms);

    std::vector<float> px(n), py(n), pz(n), smax(n), op(n), r(n), g(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        const aniso::Gaussian& gs = scene.gaussians[i];
        px[i] = gs.position.x;
        py[i] = gs.position.y;
        pz[i] = gs.position.z;
        smax[i] = std::max({gs.scale.x, gs.scale.y, gs.scale.z});
        op[i] = gs.opacity;
        const auto c = aniso::dcColor(scene, i);
        r[i] = c[0];
        g[i] = c[1];
        b[i] = c[2];
    }
    print("position.x", stats(px));
    print("position.y", stats(py));
    print("position.z", stats(pz));
    print("largest scale", stats(smax));
    print("opacity", stats(op));
    print("dc colour r", stats(r));
    print("dc colour g", stats(g));
    print("dc colour b", stats(b));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "info") {
        try {
            return info(argv[2]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    std::fprintf(stderr, "usage: aniso info <scene.ply>\n");
    return 2;
}
