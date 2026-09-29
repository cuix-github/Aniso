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
// aniso path <scene.ply> <path.txt> <WxH> <hfov-degrees> [--gpu] [--sh-degree D]
//   Renders one frame per line of path.txt ("px py pz  fx fy fz  ux uy uz": position, forward,
//   up, in world space) and streams raw RGB frames to stdout for a video encoder.
//   tools/flythrough.py builds the path and pipes the frames into ffmpeg.
//
// aniso render <scene.ply> <colmap-sparse-dir> <image-name> <out.png> [--width N] [--sh-degree D]
//   Renders the scene from that photo's camera, optionally at a different width (the
//   Tanks and Temples photos are half the recorded camera size, so --width 980 matches them)
//   and with view-dependent colour limited to degree D (default 3, the full colour; 0 is the
//   view-independent colour only). --gpu uses the CUDA renderer, in builds that have it.

#include "aniso/camera.h"
#include "aniso/image.h"
#include "aniso/ply_loader.h"
#include "aniso/render.h"
#ifdef ANISO_WITH_CUDA
#include "aniso/gpu_render.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

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
              const std::string& outPath, int width, int shDegree, bool gpu) {
    const aniso::Scene scene = aniso::loadPly(plyPath);
    const auto cams = aniso::loadColmapCameras(sparseDir);
    const auto it = std::find_if(cams.begin(), cams.end(), [&](const aniso::Camera& c) { return c.imageName == imageName; });
    if (it == cams.end()) throw std::runtime_error("no camera for image " + imageName);
    const aniso::Camera cam = width > 0 ? aniso::resized(*it, width) : *it;

    aniso::RenderStats st;
    aniso::RenderOptions options;
    options.shDegree = shDegree;
    aniso::Image img(1, 1);
    if (gpu) {
#ifdef ANISO_WITH_CUDA
        aniso::GpuRenderer renderer(scene);
        renderer.render(cam, nullptr, options); // first call pays for CUDA start-up; time the second
        img = renderer.render(cam, &st, options);
#else
        throw std::runtime_error("this build has no CUDA renderer; build with build_cuda.bat");
#endif
    } else {
        img = aniso::render(scene, cam, &st, options);
    }
    aniso::writePng(outPath, img);
    std::printf("%s: %dx%d, SH degree %d, %zu splats, %zu splat-tile pairs\n", outPath.c_str(), cam.width, cam.height,
                std::min(shDegree, scene.shDegree), st.visible, st.tilePairs);
    std::printf("  %s: project %.1f ms, sort and bin %.1f ms, blend %.1f ms, total %.1f ms\n", gpu ? "GPU" : "CPU",
                st.projectMs, st.sortMs, st.blendMs, st.projectMs + st.sortMs + st.blendMs);
    return 0;
}

// A camera looking along `forward` from `position`, with `up` roughly up, in COLMAP's convention
// (x right, y down, z forward).
aniso::Camera poseCamera(const aniso::Vec3& position, aniso::Vec3 f, const aniso::Vec3& up, int width, int height,
                         double hfovDeg) {
    auto norm = [](aniso::Vec3 v) {
        const float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return aniso::Vec3{v.x / l, v.y / l, v.z / l};
    };
    auto cross = [](aniso::Vec3 a, aniso::Vec3 b) {
        return aniso::Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    };
    f = norm(f);
    const aniso::Vec3 r = norm(cross(f, up)), d = cross(f, r);
    const aniso::Vec3 rows[3] = {r, d, f};
    aniso::Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = 0.5 * width / std::tan(hfovDeg * 3.14159265358979 / 360.0);
    c.cx = width / 2.0;
    c.cy = height / 2.0;
    for (int i = 0; i < 3; ++i) {
        c.R[i][0] = rows[i].x; c.R[i][1] = rows[i].y; c.R[i][2] = rows[i].z;
        c.t[i] = -(rows[i].x * position.x + rows[i].y * position.y + rows[i].z * position.z);
    }
    return c;
}

// aniso path: renders one frame per line of a path file and streams raw RGB (8 bits, rows top to
// bottom, no header) to stdout, for piping into a video encoder.
int pathCmd(const std::string& plyPath, const std::string& pathFile, int width, int height, double hfov, bool gpu,
            int shDegree) {
    const aniso::Scene scene = aniso::loadPly(plyPath);
    std::ifstream in(pathFile);
    if (!in) throw std::runtime_error("cannot open " + pathFile);
    // Nine numbers per line (position, forward, up), and an optional tenth: the splat scale.
    std::vector<std::array<float, 10>> poses;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        std::array<float, 10> p{};
        for (int k = 0; k < 9; ++k) s >> p[k];
        if (!s) throw std::runtime_error("bad path line: " + line);
        if (!(s >> p[9])) p[9] = 1.0f;
        poses.push_back(p);
    }
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    aniso::RenderOptions options;
    options.shDegree = shDegree;
#ifdef ANISO_WITH_CUDA
    std::unique_ptr<aniso::GpuRenderer> renderer;
    if (gpu) renderer = std::make_unique<aniso::GpuRenderer>(scene);
#else
    if (gpu) throw std::runtime_error("this build has no CUDA renderer; build with build_cuda.bat");
#endif
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& p : poses) {
        const aniso::Camera cam = poseCamera({p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], p[8]}, width,
                                             height, hfov);
        options.splatScale = p[9];
#ifdef ANISO_WITH_CUDA
        const aniso::Image img = renderer ? renderer->render(cam, nullptr, options) : aniso::render(scene, cam, nullptr, options);
#else
        const aniso::Image img = aniso::render(scene, cam, nullptr, options);
#endif
        std::fwrite(img.rgb.data(), 1, img.rgb.size(), stdout);
    }
    std::fflush(stdout);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "rendered %zu frames at %dx%d on the %s in %.1f s (%.1f ms per frame, including output)\n",
                 poses.size(), width, height, gpu ? "GPU" : "CPU", s, 1000.0 * s / std::max<std::size_t>(1, poses.size()));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.size() == 2 && args[0] == "info") return info(args[1]);
        if (args.size() == 2 && args[0] == "cameras") return cameras(args[1]);
        if (args.size() == 5 && args[0] == "dots") return dots(args[1], args[2], args[3], args[4]);
        if (args.size() >= 5 && args[0] == "path") {
            bool gpu = false;
            int shDegree = 3;
            for (std::size_t k = 5; k < args.size(); ++k) {
                if (args[k] == "--gpu") gpu = true;
                else if (args[k] == "--sh-degree" && k + 1 < args.size()) shDegree = std::stoi(args[++k]);
                else throw std::runtime_error("unknown option " + args[k]);
            }
            const auto x = args[3].find('x');
            if (x == std::string::npos) throw std::runtime_error("size must look like 1920x1080");
            return pathCmd(args[1], args[2], std::stoi(args[3].substr(0, x)), std::stoi(args[3].substr(x + 1)),
                           std::stod(args[4]), gpu, shDegree);
        }
        if (args.size() >= 5 && args[0] == "render") {
            int width = 0, shDegree = 3;
            bool gpu = false;
            for (std::size_t k = 5; k < args.size(); ++k) {
                if (args[k] == "--gpu") gpu = true;
                else if (args[k] == "--width" && k + 1 < args.size()) width = std::stoi(args[++k]);
                else if (args[k] == "--sh-degree" && k + 1 < args.size()) shDegree = std::stoi(args[++k]);
                else throw std::runtime_error("unknown option " + args[k]);
            }
            return renderCmd(args[1], args[2], args[3], args[4], width, shDegree, gpu);
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
                 "  aniso render <scene.ply> <colmap-sparse-dir> <image-name> <out.png> [--width N] [--sh-degree 0..3] [--gpu]\n"
                 "  aniso path <scene.ply> <path.txt> <WxH> <hfov-degrees> [--gpu] [--sh-degree 0..3]   (raw RGB to stdout)\n");
    return 2;
}
