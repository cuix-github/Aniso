// aniso info <scene.ply>
//   Prints what is in a scene: the count, the SH degree, the bounds, and the spread of scales,
//   opacities, and colours. Checked against tools/reference_stats.py, which reads the same
//   file with numpy.
//
// aniso cameras <colmap-sparse-dir>
//   Lists the cameras of a COLMAP reconstruction: image name, size, focal length, position.
//
// aniso dots <scene.ply> <colmap-sparse-dir> <image-name> <out.png>
//   Projects every Gaussian's centre through that photo's camera and draws it as one pixel in
//   its view-independent colour, nearest in front. If the camera maths is right, the dots
//   outline the same view as the photo.
//
// aniso render <scene.ply> <colmap-sparse-dir> <image-name> <out.png> [--width N] [--sh-degree D]
//   Renders the scene from that photo's camera, optionally at a different width (the
//   Tanks and Temples photos are half the recorded camera size, so --width 980 matches them)
//   and with view-dependent colour limited to degree D (default 3, the full colour; 0 is the
//   view-independent colour only).

#include "aniso/camera.h"
#include "aniso/image.h"
#include "aniso/ply_loader.h"
#include "aniso/render.h"

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

int cameras(const std::string& sparseDir) {
    const auto cams = aniso::loadColmapCameras(sparseDir);
    std::printf("%zu cameras in %s\n", cams.size(), sparseDir.c_str());
    for (const auto& c : cams) {
        const aniso::Vec3 o = c.centre();
        std::printf("  %-14s %4dx%-4d  fx %8.2f  fy %8.2f  centre (%8.3f, %8.3f, %8.3f)\n", c.imageName.c_str(),
                    c.width, c.height, c.fx, c.fy, o.x, o.y, o.z);
    }
    return 0;
}

int dots(const std::string& plyPath, const std::string& sparseDir, const std::string& imageName,
         const std::string& outPath) {
    const aniso::Scene scene = aniso::loadPly(plyPath);
    const auto cams = aniso::loadColmapCameras(sparseDir);
    const auto it = std::find_if(cams.begin(), cams.end(), [&](const aniso::Camera& c) { return c.imageName == imageName; });
    if (it == cams.end()) throw std::runtime_error("no camera for image " + imageName);
    const aniso::Camera& cam = *it;

    aniso::Image img(cam.width, cam.height);
    std::vector<float> nearest(static_cast<std::size_t>(cam.width) * cam.height, 1e30f);
    std::size_t visible = 0;
    for (std::size_t i = 0; i < scene.size(); ++i) {
        const auto p = aniso::project(cam, cam.toCamera(scene.gaussians[i].position));
        if (!p) continue;
        const int x = static_cast<int>(p->u), y = static_cast<int>(p->v);
        if (x < 0 || y < 0 || x >= cam.width || y >= cam.height) continue;
        ++visible;
        float& d = nearest[static_cast<std::size_t>(y) * cam.width + x];
        if (p->depth >= d) continue;
        d = p->depth;
        const auto c = aniso::dcColor(scene, i);
        std::uint8_t* px = img.at(x, y);
        for (int k = 0; k < 3; ++k) px[k] = static_cast<std::uint8_t>(std::clamp(c[k], 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    aniso::writePng(outPath, img);
    std::printf("%s: %zu of %zu Gaussians land in the %dx%d frame of %s\n", outPath.c_str(), visible, scene.size(),
                cam.width, cam.height, imageName.c_str());
    return 0;
}

int renderCmd(const std::string& plyPath, const std::string& sparseDir, const std::string& imageName,
              const std::string& outPath, int width, int shDegree) {
    const aniso::Scene scene = aniso::loadPly(plyPath);
    const auto cams = aniso::loadColmapCameras(sparseDir);
    const auto it = std::find_if(cams.begin(), cams.end(), [&](const aniso::Camera& c) { return c.imageName == imageName; });
    if (it == cams.end()) throw std::runtime_error("no camera for image " + imageName);
    const aniso::Camera cam = width > 0 ? aniso::resized(*it, width) : *it;

    aniso::RenderStats st;
    aniso::RenderOptions options;
    options.shDegree = shDegree;
    const aniso::Image img = aniso::render(scene, cam, &st, options);
    aniso::writePng(outPath, img);
    std::printf("%s: %dx%d, SH degree %d, %zu splats, %zu splat-tile pairs\n", outPath.c_str(), cam.width, cam.height,
                std::min(shDegree, scene.shDegree), st.visible, st.tilePairs);
    std::printf("  project %.0f ms, sort and bin %.0f ms, blend %.0f ms, total %.0f ms\n", st.projectMs, st.sortMs,
                st.blendMs, st.projectMs + st.sortMs + st.blendMs);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.size() == 2 && args[0] == "info") return info(args[1]);
        if (args.size() == 2 && args[0] == "cameras") return cameras(args[1]);
        if (args.size() == 5 && args[0] == "dots") return dots(args[1], args[2], args[3], args[4]);
        if (args.size() >= 5 && args.size() % 2 == 1 && args[0] == "render") {
            int width = 0, shDegree = 3;
            for (std::size_t k = 5; k + 1 < args.size(); k += 2) {
                if (args[k] == "--width") width = std::stoi(args[k + 1]);
                else if (args[k] == "--sh-degree") shDegree = std::stoi(args[k + 1]);
                else throw std::runtime_error("unknown option " + args[k]);
            }
            return renderCmd(args[1], args[2], args[3], args[4], width, shDegree);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage:\n"
                 "  aniso info <scene.ply>\n"
                 "  aniso cameras <colmap-sparse-dir>\n"
                 "  aniso dots <scene.ply> <colmap-sparse-dir> <image-name> <out.png>\n"
                 "  aniso render <scene.ply> <colmap-sparse-dir> <image-name> <out.png> [--width N] [--sh-degree 0..3]\n");
    return 2;
}
