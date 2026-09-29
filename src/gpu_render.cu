#include "aniso/gpu_render.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aniso {
namespace {

constexpr int kTile = 16;
constexpr int kBlock = kTile * kTile;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}
#define CU(call) check((call), #call)

// A device buffer that only grows.
template <typename T>
struct Buffer {
    T* ptr = nullptr;
    std::size_t cap = 0;
    void reserve(std::size_t n) {
        if (n <= cap) return;
        if (ptr) cudaFree(ptr);
        CU(cudaMalloc(&ptr, std::max<std::size_t>(n, 1) * sizeof(T)));
        cap = n;
    }
    ~Buffer() {
        if (ptr) cudaFree(ptr);
    }
};

struct CameraParams {
    float R[9];  // world to camera, row major
    float t[3];
    float eye[3];
    float fx, fy, cx, cy;
    int width, height, tilesX, tilesY;
    int shDegree, shCoeffs;
};

__constant__ float kC1 = 0.4886025119029199f;
__constant__ float kC2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f,
                             0.5462742152960396f};
__constant__ float kC3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                             -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

// Same basis, order, and conventions as src/sh.cpp.
__device__ float3 shColor(const float* sh, int degree, float x, float y, float z) {
    float b[16];
    int count = 1;
    b[0] = 0.28209479177387814f;
    if (degree >= 1) {
        b[1] = -kC1 * y; b[2] = kC1 * z; b[3] = -kC1 * x;
        count = 4;
    }
    if (degree >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        b[4] = kC2[0] * x * y; b[5] = kC2[1] * y * z; b[6] = kC2[2] * (2 * zz - xx - yy);
        b[7] = kC2[3] * x * z; b[8] = kC2[4] * (xx - yy);
        count = 9;
        if (degree >= 3) {
            b[9] = kC3[0] * y * (3 * xx - yy); b[10] = kC3[1] * x * y * z; b[11] = kC3[2] * y * (4 * zz - xx - yy);
            b[12] = kC3[3] * z * (2 * zz - 3 * xx - 3 * yy); b[13] = kC3[4] * x * (4 * zz - xx - yy);
            b[14] = kC3[5] * z * (xx - yy); b[15] = kC3[6] * x * (xx - 3 * yy);
            count = 16;
        }
    }
    float3 c = make_float3(0.5f, 0.5f, 0.5f);
    for (int k = 0; k < count; ++k) {
        c.x += b[k] * sh[k * 3 + 0];
        c.y += b[k] * sh[k * 3 + 1];
        c.z += b[k] * sh[k * 3 + 2];
    }
    return make_float3(fmaxf(c.x, 0.f), fmaxf(c.y, 0.f), fmaxf(c.z, 0.f));
}

// Step 1: project one Gaussian, the same maths as makeSplat() in src/render.cpp.
__global__ void preprocess(int n, const float3* pos, const float3* scale, const float4* rot, const float* opacity,
                           const float* sh, CameraParams cam, float2* uv, float4* conicOpacity, float* depth,
                           float3* rgb, uint4* rect, uint32_t* tilesTouched) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    tilesTouched[i] = 0;

    const float3 g = pos[i];
    const float px = cam.R[0] * g.x + cam.R[1] * g.y + cam.R[2] * g.z + cam.t[0];
    const float py = cam.R[3] * g.x + cam.R[4] * g.y + cam.R[5] * g.z + cam.t[1];
    const float pz = cam.R[6] * g.x + cam.R[7] * g.y + cam.R[8] * g.z + cam.t[2];
    if (pz < 0.2f) return;

    const float4 q = rot[i]; // w, x, y, z
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    const float Rq[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    const float s[3] = {scale[i].x, scale[i].y, scale[i].z};
    float M[3][3], S[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) M[r][c] = Rq[r][c] * s[c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) S[r][c] = M[r][0] * M[c][0] + M[r][1] * M[c][1] + M[r][2] * M[c][2];

    const float limX = 1.3f * 0.5f * cam.width / cam.fx, limY = 1.3f * 0.5f * cam.height / cam.fy;
    const float tx = fminf(limX, fmaxf(-limX, px / pz)) * pz;
    const float ty = fminf(limY, fmaxf(-limY, py / pz)) * pz;
    const float J[2][3] = {{cam.fx / pz, 0, -cam.fx * tx / (pz * pz)}, {0, cam.fy / pz, -cam.fy * ty / (pz * pz)}};
    float T[2][3];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) T[r][c] = J[r][0] * cam.R[c] + J[r][1] * cam.R[3 + c] + J[r][2] * cam.R[6 + c];
    float TS[2][3];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) TS[r][c] = T[r][0] * S[0][c] + T[r][1] * S[1][c] + T[r][2] * S[2][c];
    const float a = TS[0][0] * T[0][0] + TS[0][1] * T[0][1] + TS[0][2] * T[0][2] + 0.3f;
    const float b = TS[0][0] * T[1][0] + TS[0][1] * T[1][1] + TS[0][2] * T[1][2];
    const float c = TS[1][0] * T[1][0] + TS[1][1] * T[1][1] + TS[1][2] * T[1][2] + 0.3f;
    const float det = a * c - b * b;
    if (det <= 0) return;

    const float mid = 0.5f * (a + c);
    const float lambda = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    const int radius = (int)ceilf(3.0f * sqrtf(lambda));
    const float u = cam.fx * px / pz + cam.cx, v = cam.fy * py / pz + cam.cy;
    if (u + radius < 0 || v + radius < 0 || u - radius >= cam.width || v - radius >= cam.height) return;

    const int x0 = max(0, (int)(u - radius) / kTile), y0 = max(0, (int)(v - radius) / kTile);
    const int x1 = min(cam.tilesX - 1, (int)(u + radius) / kTile), y1 = min(cam.tilesY - 1, (int)(v + radius) / kTile);
    if (x1 < x0 || y1 < y0) return;

    float3 d = make_float3(g.x - cam.eye[0], g.y - cam.eye[1], g.z - cam.eye[2]);
    const float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    rgb[i] = shColor(sh + (size_t)i * cam.shCoeffs * 3, cam.shDegree, d.x / len, d.y / len, d.z / len);
    uv[i] = make_float2(u, v);
    conicOpacity[i] = make_float4(c / det, -b / det, a / det, opacity[i]);
    depth[i] = pz;
    rect[i] = make_uint4(x0, y0, x1, y1);
    tilesTouched[i] = (uint32_t)((x1 - x0 + 1) * (y1 - y0 + 1));
}

// Step 3: one key per tile a Gaussian touches.
__global__ void duplicate(int n, const uint32_t* offsets, const uint32_t* tilesTouched, const uint4* rect,
                          const float* depth, int tilesX, uint64_t* keys, uint32_t* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || tilesTouched[i] == 0) return;
    uint32_t out = offsets[i] - tilesTouched[i]; // offsets is an inclusive sum
    const uint32_t depthBits = __float_as_uint(depth[i]);
    const uint4 r = rect[i];
    for (uint32_t ty = r.y; ty <= r.w; ++ty)
        for (uint32_t tx = r.x; tx <= r.z; ++tx) {
            keys[out] = ((uint64_t)(ty * tilesX + tx) << 32) | depthBits;
            values[out] = (uint32_t)i;
            ++out;
        }
}

// Where each tile's run of keys starts and ends in the sorted list.
__global__ void tileRanges(int total, const uint64_t* keys, uint2* ranges) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t tile = (uint32_t)(keys[i] >> 32);
    if (i == 0) ranges[tile].x = 0;
    else {
        const uint32_t prev = (uint32_t)(keys[i - 1] >> 32);
        if (prev != tile) {
            ranges[prev].y = i;
            ranges[tile].x = i;
        }
    }
    if (i == total - 1) ranges[tile].y = total;
}

// Step 5: one block per tile, one thread per pixel, the same blending as the CPU renderer.
__global__ void blend(const uint2* ranges, const uint32_t* values, const float2* uv, const float4* conicOpacity,
                      const float3* rgb, int width, int height, int tilesX, uint8_t* out) {
    const int tile = blockIdx.y * tilesX + blockIdx.x;
    const int x = blockIdx.x * kTile + threadIdx.x, y = blockIdx.y * kTile + threadIdx.y;
    const bool inside = x < width && y < height;
    const float px = x + 0.5f, py = y + 0.5f;

    __shared__ float2 sUv[kBlock];
    __shared__ float4 sCo[kBlock];
    __shared__ float3 sRgb[kBlock];

    const uint2 range = ranges[tile];
    const int rounds = ((int)(range.y - range.x) + kBlock - 1) / kBlock;
    int todo = (int)(range.y - range.x);
    const int t = threadIdx.y * kTile + threadIdx.x;

    float T = 1.0f;
    float3 C = make_float3(0, 0, 0);
    bool done = !inside;
    for (int r = 0; r < rounds; ++r, todo -= kBlock) {
        if (__syncthreads_count(done) == kBlock) break;
        const int idx = (int)range.x + r * kBlock + t;
        if (idx < (int)range.y) {
            const uint32_t g = values[idx];
            sUv[t] = uv[g];
            sCo[t] = conicOpacity[g];
            sRgb[t] = rgb[g];
        }
        __syncthreads();
        for (int j = 0; !done && j < min(kBlock, todo); ++j) {
            const float dx = px - sUv[j].x, dy = py - sUv[j].y;
            const float4 co = sCo[j];
            const float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
            if (power > 0) continue;
            const float alpha = fminf(0.99f, co.w * __expf(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float nextT = T * (1 - alpha);
            if (nextT < 1e-4f) {
                done = true;
                continue;
            }
            C.x += sRgb[j].x * alpha * T;
            C.y += sRgb[j].y * alpha * T;
            C.z += sRgb[j].z * alpha * T;
            T = nextT;
        }
    }
    if (inside) {
        uint8_t* o = out + ((size_t)y * width + x) * 3;
        o[0] = (uint8_t)(fminf(fmaxf(C.x, 0.f), 1.f) * 255.f + 0.5f);
        o[1] = (uint8_t)(fminf(fmaxf(C.y, 0.f), 1.f) * 255.f + 0.5f);
        o[2] = (uint8_t)(fminf(fmaxf(C.z, 0.f), 1.f) * 255.f + 0.5f);
    }
}

int bitsFor(uint32_t v) {
    int b = 0;
    while ((1u << b) <= v && b < 32) ++b;
    return b;
}

} // namespace

struct GpuRenderer::Impl {
    int n = 0, shDegree = 0, shCoeffs = 1;
    Buffer<float3> pos, scale;
    Buffer<float4> rot;
    Buffer<float> opacity, sh;

    Buffer<float2> uv;
    Buffer<float4> conicOpacity;
    Buffer<float> depth;
    Buffer<float3> rgb;
    Buffer<uint4> rect;
    Buffer<uint32_t> tilesTouched, offsets;
    Buffer<uint64_t> keys, keysAlt;
    Buffer<uint32_t> values, valuesAlt;
    Buffer<uint2> ranges;
    Buffer<uint8_t> image, temp;
    cudaEvent_t ev[4];
};

bool GpuRenderer::available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

GpuRenderer::GpuRenderer(const Scene& scene) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.n = static_cast<int>(scene.size());
    m.shDegree = scene.shDegree;
    m.shCoeffs = scene.shCoeffsPerChannel();
    std::vector<float3> p(m.n), s(m.n);
    std::vector<float4> r(m.n);
    std::vector<float> o(m.n);
    for (int i = 0; i < m.n; ++i) {
        const Gaussian& g = scene.gaussians[i];
        p[i] = make_float3(g.position.x, g.position.y, g.position.z);
        s[i] = make_float3(g.scale.x, g.scale.y, g.scale.z);
        r[i] = make_float4(g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z);
        o[i] = g.opacity;
    }
    m.pos.reserve(m.n); m.scale.reserve(m.n); m.rot.reserve(m.n); m.opacity.reserve(m.n);
    m.sh.reserve(scene.sh.size());
    CU(cudaMemcpy(m.pos.ptr, p.data(), p.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.scale.ptr, s.data(), s.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.rot.ptr, r.data(), r.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.opacity.ptr, o.data(), o.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.sh.ptr, scene.sh.data(), scene.sh.size() * sizeof(float), cudaMemcpyHostToDevice));
    m.uv.reserve(m.n); m.conicOpacity.reserve(m.n); m.depth.reserve(m.n); m.rgb.reserve(m.n);
    m.rect.reserve(m.n); m.tilesTouched.reserve(m.n); m.offsets.reserve(m.n);
    for (auto& e : m.ev) CU(cudaEventCreate(&e));
}

GpuRenderer::~GpuRenderer() {
    if (impl_)
        for (auto& e : impl_->ev) cudaEventDestroy(e);
}

Image GpuRenderer::render(const Camera& camera, RenderStats* stats, const RenderOptions& options) {
    Impl& m = *impl_;
    CameraParams cp{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) cp.R[r * 3 + c] = static_cast<float>(camera.R[r][c]);
        cp.t[r] = static_cast<float>(camera.t[r]);
    }
    const Vec3 eye = camera.centre();
    cp.eye[0] = eye.x; cp.eye[1] = eye.y; cp.eye[2] = eye.z;
    cp.fx = static_cast<float>(camera.fx); cp.fy = static_cast<float>(camera.fy);
    cp.cx = static_cast<float>(camera.cx); cp.cy = static_cast<float>(camera.cy);
    cp.width = camera.width; cp.height = camera.height;
    cp.tilesX = (camera.width + kTile - 1) / kTile;
    cp.tilesY = (camera.height + kTile - 1) / kTile;
    cp.shDegree = std::min(options.shDegree, m.shDegree);
    cp.shCoeffs = m.shCoeffs;
    const int tileCount = cp.tilesX * cp.tilesY;

    CU(cudaEventRecord(m.ev[0]));
    const int threads = 256, blocks = (m.n + threads - 1) / threads;
    preprocess<<<blocks, threads>>>(m.n, m.pos.ptr, m.scale.ptr, m.rot.ptr, m.opacity.ptr, m.sh.ptr, cp, m.uv.ptr,
                                    m.conicOpacity.ptr, m.depth.ptr, m.rgb.ptr, m.rect.ptr, m.tilesTouched.ptr);
    CU(cudaGetLastError());

    std::size_t tempBytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, tempBytes, m.tilesTouched.ptr, m.offsets.ptr, m.n);
    m.temp.reserve(tempBytes);
    cub::DeviceScan::InclusiveSum(m.temp.ptr, tempBytes, m.tilesTouched.ptr, m.offsets.ptr, m.n);
    uint32_t total = 0;
    CU(cudaMemcpy(&total, m.offsets.ptr + m.n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CU(cudaEventRecord(m.ev[1]));

    m.keys.reserve(total); m.keysAlt.reserve(total); m.values.reserve(total); m.valuesAlt.reserve(total);
    m.ranges.reserve(tileCount);
    CU(cudaMemset(m.ranges.ptr, 0, tileCount * sizeof(uint2)));
    if (total > 0) {
        duplicate<<<blocks, threads>>>(m.n, m.offsets.ptr, m.tilesTouched.ptr, m.rect.ptr, m.depth.ptr, cp.tilesX,
                                       m.keys.ptr, m.values.ptr);
        cub::DoubleBuffer<uint64_t> k(m.keys.ptr, m.keysAlt.ptr);
        cub::DoubleBuffer<uint32_t> v(m.values.ptr, m.valuesAlt.ptr);
        const int endBit = 32 + bitsFor(static_cast<uint32_t>(tileCount));
        tempBytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, k, v, total, 0, endBit);
        m.temp.reserve(tempBytes);
        cub::DeviceRadixSort::SortPairs(m.temp.ptr, tempBytes, k, v, total, 0, endBit);
        tileRanges<<<(total + threads - 1) / threads, threads>>>(total, k.Current(), m.ranges.ptr);
        CU(cudaGetLastError());
        CU(cudaEventRecord(m.ev[2]));
        m.image.reserve(static_cast<std::size_t>(camera.width) * camera.height * 3);
        blend<<<dim3(cp.tilesX, cp.tilesY), dim3(kTile, kTile)>>>(m.ranges.ptr, v.Current(), m.uv.ptr,
                                                                   m.conicOpacity.ptr, m.rgb.ptr, camera.width,
                                                                   camera.height, cp.tilesX, m.image.ptr);
    } else {
        CU(cudaEventRecord(m.ev[2]));
        m.image.reserve(static_cast<std::size_t>(camera.width) * camera.height * 3);
        CU(cudaMemset(m.image.ptr, 0, static_cast<std::size_t>(camera.width) * camera.height * 3));
    }
    CU(cudaGetLastError());
    CU(cudaEventRecord(m.ev[3]));

    Image img(camera.width, camera.height);
    CU(cudaMemcpy(img.rgb.data(), m.image.ptr, img.rgb.size(), cudaMemcpyDeviceToHost));

    if (stats) {
        float a = 0, b = 0, c = 0;
        cudaEventElapsedTime(&a, m.ev[0], m.ev[1]);
        cudaEventElapsedTime(&b, m.ev[1], m.ev[2]);
        cudaEventElapsedTime(&c, m.ev[2], m.ev[3]);
        stats->projectMs = a;
        stats->sortMs = b;
        stats->blendMs = c;
        stats->tilePairs = total;
        stats->visible = 0; // not counted on the GPU
    }
    return img;
}

} // namespace aniso
