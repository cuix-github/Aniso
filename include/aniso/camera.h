#pragma once

#include "aniso/gaussian.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace aniso {

// A pinhole camera as COLMAP describes one photo: intrinsics (focal lengths and principal
// point in pixels) and a world-to-camera transform. COLMAP's camera looks down +z, with +x to
// the right and +y down the image, so a projected point needs no axis flips.
struct Camera {
    std::string imageName;
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double R[3][3] = {};    // world to camera rotation
    double t[3] = {};       // world to camera translation: X_cam = R * X_world + t

    Vec3 toCamera(const Vec3& p) const;
    // World position of the camera centre, -R^T t.
    Vec3 centre() const;
};

// A point in pixel coordinates plus its depth along the camera's viewing axis.
struct Projected {
    float u = 0, v = 0, depth = 0;
};

// Projects a camera-space point. Empty if it is behind the camera or closer than nearPlane.
std::optional<Projected> project(const Camera& cam, const Vec3& pCamera, float nearPlane = 0.2f);

// Reads COLMAP's binary sparse model (cameras.bin and images.bin in the given folder) and
// returns one Camera per registered image, sorted by image name. Supports the PINHOLE and
// SIMPLE_PINHOLE camera models, which is what undistorted datasets use.
std::vector<Camera> loadColmapCameras(const std::filesystem::path& sparseDir);

} // namespace aniso
