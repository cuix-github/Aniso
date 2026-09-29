#include "aniso/render.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace aniso {
namespace {

constexpr int kTile = 16;

struct Splat {
    float u = 0, v = 0;                 // centre in pixels
    float conicA = 0, conicB = 0, conicC = 0; // inverse 2D covariance [[A, B], [B, C]]
    float opacity = 0;
    float depth = 0;
    float rgb[3] = {};
    int radius = 0;
};

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Rotation matrix of a unit quaternion (w, x, y, z).
void rotation(const Quat& q, float M[3][3]) {
    const float w = q.w, x = q.x, y = q.y, z = q.z;
    M[0][0] = 1 - 2 * (y * y + z * z); M[0][1] = 2 * (x * y - w * z);     M[0][2] = 2 * (x * z + w * y);
    M[1][0] = 2 * (x * y + w * z);     M[1][1] = 1 - 2 * (x * x + z * z); M[1][2] = 2 * (y * z - w * x);
    M[2][0] = 2 * (x * z - w * y);     M[2][1] = 2 * (y * z + w * x);     M[2][2] = 1 - 2 * (x * x + y * y);
}

// Projects one Gaussian. Returns false if it is culled.
bool makeSplat(const Scene& scene, std::size_t i, const Camera& cam, Splat& s) {
    const Gaussian& g = scene.gaussians[i];
    Vec3 p = cam.toCamera(g.position);
    if (p.z < 0.2f) return false;

    // 3D covariance: Sigma = M M^T with M = R * diag(scale).
    float R[3][3];
    rotation(g.rotation, R);
    const float sc[3] = {g.scale.x, g.scale.y, g.scale.z};
    float M[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) M[r][c] = R[r][c] * sc[c];
    float Sigma[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) Sigma[r][c] = M[r][0] * M[c][0] + M[r][1] * M[c][1] + M[r][2] * M[c][2];

    // Clamp the point's direction the way the reference does, so the linearization stays sane
    // for Gaussians far outside the view.
    const float limX = 1.3f * static_cast<float>(0.5 * cam.width / cam.fx);
    const float limY = 1.3f * static_cast<float>(0.5 * cam.height / cam.fy);
    const float tx = std::clamp(p.x / p.z, -limX, limX) * p.z;
    const float ty = std::clamp(p.y / p.z, -limY, limY) * p.z;

    // Perspective Jacobian at the centre (2x3), times the world-to-camera rotation W.
    const float fx = static_cast<float>(cam.fx), fy = static_cast<float>(cam.fy);
    const float J[2][3] = {{fx / p.z, 0, -fx * tx / (p.z * p.z)}, {0, fy / p.z, -fy * ty / (p.z * p.z)}};
    float T[2][3];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            T[r][c] = J[r][0] * static_cast<float>(cam.R[0][c]) + J[r][1] * static_cast<float>(cam.R[1][c]) +
                      J[r][2] * static_cast<float>(cam.R[2][c]);

    // 2D covariance T Sigma T^T, plus the 0.3-pixel low-pass the reference adds.
    float TS[2][3];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) TS[r][c] = T[r][0] * Sigma[0][c] + T[r][1] * Sigma[1][c] + T[r][2] * Sigma[2][c];
    const float a = TS[0][0] * T[0][0] + TS[0][1] * T[0][1] + TS[0][2] * T[0][2] + 0.3f;
    const float b = TS[0][0] * T[1][0] + TS[0][1] * T[1][1] + TS[0][2] * T[1][2];
    const float c = TS[1][0] * T[1][0] + TS[1][1] * T[1][1] + TS[1][2] * T[1][2] + 0.3f;

    const float det = a * c - b * b;
    if (det <= 0) return false;
    s.conicA = c / det;
    s.conicB = -b / det;
    s.conicC = a / det;

    // Radius: three standard deviations along the larger eigenvalue.
    const float mid = 0.5f * (a + c);
    const float lambda = mid + std::sqrt(std::max(0.1f, mid * mid - det));
    s.radius = static_cast<int>(std::ceil(3.0f * std::sqrt(lambda)));

    s.u = fx * p.x / p.z + static_cast<float>(cam.cx);
    s.v = fy * p.y / p.z + static_cast<float>(cam.cy);
    if (s.u + s.radius < 0 || s.v + s.radius < 0 || s.u - s.radius >= cam.width || s.v - s.radius >= cam.height)
        return false;

    s.depth = p.z;
    s.opacity = g.opacity;
    const auto rgb = dcColor(scene, i);
    for (int k = 0; k < 3; ++k) s.rgb[k] = std::max(0.0f, rgb[k]);
    return true;
}

template <typename Fn>
void parallelFor(std::size_t n, Fn fn) {
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) fn(i);
        });
    }
    for (auto& th : pool) th.join();
}

} // namespace

Image render(const Scene& scene, const Camera& cam, RenderStats* stats) {
    RenderStats local;
    RenderStats& st = stats ? *stats : local;

    // 1. Project.
    auto t0 = Clock::now();
    std::vector<Splat> splats(scene.size());
    std::vector<char> alive(scene.size(), 0);
    parallelFor((scene.size() + 4095) / 4096, [&](std::size_t chunk) {
        const std::size_t end = std::min(scene.size(), (chunk + 1) * 4096);
        for (std::size_t i = chunk * 4096; i < end; ++i) alive[i] = makeSplat(scene, i, cam, splats[i]) ? 1 : 0;
    });
    std::vector<std::uint32_t> order;
    for (std::size_t i = 0; i < scene.size(); ++i)
        if (alive[i]) order.push_back(static_cast<std::uint32_t>(i));
    st.visible = order.size();
    st.projectMs = msSince(t0);

    // 2. Sort once by depth, then bin into tiles; each tile's list inherits the depth order.
    t0 = Clock::now();
    std::sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return splats[x].depth < splats[y].depth; });
    const int tilesX = (cam.width + kTile - 1) / kTile, tilesY = (cam.height + kTile - 1) / kTile;
    std::vector<std::vector<std::uint32_t>> tiles(static_cast<std::size_t>(tilesX) * tilesY);
    for (std::uint32_t idx : order) {
        const Splat& s = splats[idx];
        const int x0 = std::max(0, static_cast<int>(s.u - s.radius) / kTile);
        const int y0 = std::max(0, static_cast<int>(s.v - s.radius) / kTile);
        const int x1 = std::min(tilesX - 1, static_cast<int>(s.u + s.radius) / kTile);
        const int y1 = std::min(tilesY - 1, static_cast<int>(s.v + s.radius) / kTile);
        for (int ty = y0; ty <= y1; ++ty)
            for (int tx = x0; tx <= x1; ++tx) tiles[static_cast<std::size_t>(ty) * tilesX + tx].push_back(idx);
        st.tilePairs += static_cast<std::size_t>(x1 - x0 + 1) * (y1 - y0 + 1);
    }
    st.sortMs = msSince(t0);

    // 3. Blend each tile's pixels front to back.
    t0 = Clock::now();
    Image img(cam.width, cam.height);
    parallelFor(tiles.size(), [&](std::size_t t) {
        const int tx = static_cast<int>(t % tilesX), ty = static_cast<int>(t / tilesX);
        const auto& list = tiles[t];
        for (int y = ty * kTile; y < std::min(cam.height, (ty + 1) * kTile); ++y) {
            for (int x = tx * kTile; x < std::min(cam.width, (tx + 1) * kTile); ++x) {
                const float px = x + 0.5f, py = y + 0.5f;
                float T = 1.0f, C[3] = {0, 0, 0};
                for (std::uint32_t idx : list) {
                    const Splat& s = splats[idx];
                    const float dx = px - s.u, dy = py - s.v;
                    const float power = -0.5f * (s.conicA * dx * dx + s.conicC * dy * dy) - s.conicB * dx * dy;
                    if (power > 0) continue;
                    const float alpha = std::min(0.99f, s.opacity * std::exp(power));
                    if (alpha < 1.0f / 255.0f) continue;
                    const float nextT = T * (1 - alpha);
                    if (nextT < 1e-4f) break;
                    for (int k = 0; k < 3; ++k) C[k] += s.rgb[k] * alpha * T;
                    T = nextT;
                }
                std::uint8_t* out = img.at(x, y);
                for (int k = 0; k < 3; ++k) out[k] = static_cast<std::uint8_t>(std::clamp(C[k], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
    });
    st.blendMs = msSince(t0);
    return img;
}

} // namespace aniso
