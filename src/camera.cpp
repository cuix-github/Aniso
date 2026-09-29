#include "aniso/camera.h"

#include "aniso/ply_loader.h" // PlyError is reused as the general "bad input file" error

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

namespace aniso {

Vec3 Camera::toCamera(const Vec3& p) const {
    const double x = p.x, y = p.y, z = p.z;
    return {static_cast<float>(R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]),
            static_cast<float>(R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]),
            static_cast<float>(R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2])};
}

Vec3 Camera::centre() const {
    return {static_cast<float>(-(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2])),
            static_cast<float>(-(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2])),
            static_cast<float>(-(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]))};
}

std::optional<Projected> project(const Camera& cam, const Vec3& p, float nearPlane) {
    if (p.z < nearPlane) return std::nullopt;
    return Projected{static_cast<float>(cam.fx * p.x / p.z + cam.cx),
                     static_cast<float>(cam.fy * p.y / p.z + cam.cy), p.z};
}

namespace {

template <typename T>
T readPod(std::istream& in, const std::string& file) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw PlyError(file + ": unexpected end of file");
    return value;
}

struct Intrinsics {
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

// COLMAP camera model ids and their parameter counts (from COLMAP's models.h).
int paramCount(int model) {
    static const int counts[] = {3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12};
    return (model >= 0 && model < 11) ? counts[model] : -1;
}

std::map<std::uint32_t, Intrinsics> readCameras(const std::filesystem::path& path) {
    const std::string file = path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw PlyError(file + ": cannot open");
    std::map<std::uint32_t, Intrinsics> cams;
    const auto count = readPod<std::uint64_t>(in, file);
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto id = readPod<std::uint32_t>(in, file);
        const auto model = readPod<std::int32_t>(in, file);
        Intrinsics c;
        c.width = static_cast<int>(readPod<std::uint64_t>(in, file));
        c.height = static_cast<int>(readPod<std::uint64_t>(in, file));
        const int n = paramCount(model);
        if (n < 0) throw PlyError(file + ": unknown camera model " + std::to_string(model));
        std::vector<double> params(n);
        for (double& p : params) p = readPod<double>(in, file);
        if (model == 0) {        // SIMPLE_PINHOLE: f, cx, cy
            c.fx = c.fy = params[0];
            c.cx = params[1];
            c.cy = params[2];
        } else if (model == 1) { // PINHOLE: fx, fy, cx, cy
            c.fx = params[0];
            c.fy = params[1];
            c.cx = params[2];
            c.cy = params[3];
        } else {
            throw PlyError(file + ": camera model " + std::to_string(model) +
                           " has lens distortion; undistort the images first");
        }
        cams[id] = c;
    }
    return cams;
}

void quatToMatrix(double w, double x, double y, double z, double R[3][3]) {
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n; x /= n; y /= n; z /= n;
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z);     R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z);     R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y);     R[2][1] = 2 * (y * z + w * x);     R[2][2] = 1 - 2 * (x * x + y * y);
}

} // namespace

std::vector<Camera> loadColmapCameras(const std::filesystem::path& sparseDir) {
    const auto intrinsics = readCameras(sparseDir / "cameras.bin");

    const auto path = sparseDir / "images.bin";
    const std::string file = path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw PlyError(file + ": cannot open");

    std::vector<Camera> out;
    const auto count = readPod<std::uint64_t>(in, file);
    for (std::uint64_t i = 0; i < count; ++i) {
        readPod<std::uint32_t>(in, file); // image id, unused
        double q[4], tv[3];
        for (double& v : q) v = readPod<double>(in, file);
        for (double& v : tv) v = readPod<double>(in, file);
        const auto cameraId = readPod<std::uint32_t>(in, file);
        std::string name;
        for (char ch; (ch = readPod<char>(in, file)) != '\0';) name.push_back(ch);
        const auto points = readPod<std::uint64_t>(in, file);
        in.seekg(static_cast<std::streamoff>(points * 24), std::ios::cur); // x, y, point3D id

        auto it = intrinsics.find(cameraId);
        if (it == intrinsics.end()) throw PlyError(file + ": image " + name + " refers to a missing camera");
        Camera c;
        c.imageName = name;
        c.width = it->second.width;
        c.height = it->second.height;
        c.fx = it->second.fx; c.fy = it->second.fy;
        c.cx = it->second.cx; c.cy = it->second.cy;
        quatToMatrix(q[0], q[1], q[2], q[3], c.R);
        for (int k = 0; k < 3; ++k) c.t[k] = tv[k];
        out.push_back(std::move(c));
    }
    std::sort(out.begin(), out.end(), [](const Camera& a, const Camera& b) { return a.imageName < b.imageName; });
    return out;
}

} // namespace aniso
