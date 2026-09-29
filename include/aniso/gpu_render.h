#pragma once

#include "aniso/camera.h"
#include "aniso/gaussian.h"
#include "aniso/image.h"
#include "aniso/render.h"

#include <memory>

namespace aniso {

// The CUDA renderer. Same algorithm and the same maths as the CPU renderer, organized the way
// the reference rasterizer is:
//   1. one thread per Gaussian projects it and counts the 16x16-pixel tiles it overlaps;
//   2. a prefix sum over those counts gives each Gaussian its slots in one flat list;
//   3. each Gaussian writes one 64-bit key per tile it touches: tile number in the high 32 bits,
//      depth in the low 32 (positive floats order like their bit patterns);
//   4. one radix sort orders every key, which groups keys by tile and orders each tile by depth;
//   5. one thread block per tile blends its 256 pixels front to back, loading splats into
//      shared memory in batches and stopping early once every pixel is opaque.
// The scene is uploaded once, in the constructor; each render only sends a camera.
class GpuRenderer {
public:
    explicit GpuRenderer(const Scene& scene);
    ~GpuRenderer();
    GpuRenderer(const GpuRenderer&) = delete;
    GpuRenderer& operator=(const GpuRenderer&) = delete;

    // Timings in stats are GPU time per stage, measured with CUDA events.
    Image render(const Camera& camera, RenderStats* stats = nullptr, const RenderOptions& options = {});

    // True if a CUDA device is present and usable.
    static bool available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aniso
