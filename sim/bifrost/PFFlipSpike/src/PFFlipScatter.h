#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PFFlip {
namespace Detail {

// A scatter cannot simply parallelize its particle loop: particles share grid
// destinations, and atomics or per-thread sums change floating-point addition
// order. Instead, assign disjoint x slabs of the *destination* to workers.
// Each slab's list is filled in original particle order. A boundary particle
// appears in both neighboring lists, but each list writes only its own faces.
// Every face therefore receives exactly the old serial sequence of additions,
// independent of scheduling, worker count, phase, or particle size.
//
// This is a one-dimensional ownership partition, not grid adaptivity. Four
// x indices per slab give the 256-cell benchmark enough work units for 32
// workers without allocating a full grid per worker. Extra storage is only
// the particle-index lists (O(particles), with bounded boundary duplication).
class OrderedScatterSlabs {
public:
    template<class X, class Radius>
    OrderedScatterSlabs(size_t n, int nx, X x, Radius radius) : particles_(n), nx_(nx) {
#ifdef _OPENMP
        if (omp_get_max_threads() > 1 && n >= 4096 && nx >= 8)
            count_ = (nx + 1 + width - 1) / width;
#endif
        if (count_ == 1) return;
        offsets_.assign(static_cast<size_t>(count_) + 1, 0);
        auto bounds = [&](size_t i) {
            // Across the staggered grids x offsets are 0 or 0.5. Include a
            // conservative integer interval containing every nonzero kernel
            // contribution; the original stencil and kernel still do the
            // exact clipping. Clamp before converting so distant particles
            // cannot overflow a bin index.
            const double px = x(i), r = radius(i);
            const int lo = static_cast<int>(std::clamp(std::floor(px - r - 0.5), 0.0, double(nx)));
            const int hi = static_cast<int>(std::clamp(std::ceil(px + r), 0.0, double(nx)));
            return std::pair<int, int>(lo / width, hi / width);
        };
        for (size_t i = 0; i < n; ++i) {
            const auto b = bounds(i);
            for (int s = b.first; s <= b.second; ++s) ++offsets_[s + 1];
        }
        for (int s = 0; s < count_; ++s) offsets_[s + 1] += offsets_[s];
        indices_.resize(offsets_.back());
        auto next = offsets_;
        for (size_t i = 0; i < n; ++i) {
            const auto b = bounds(i);
            for (int s = b.first; s <= b.second; ++s) indices_[next[s]++] = i;
        }
    }

    int count() const { return count_; }
    int lo(int slab) const { return count_ == 1 ? 0 : slab * width; }
    int hi(int slab) const { return count_ == 1 ? nx_ + 1 : std::min((slab + 1) * width, nx_ + 1); }
    size_t begin(int slab) const { return count_ == 1 ? 0 : offsets_[slab]; }
    size_t end(int slab) const { return count_ == 1 ? particles_ : offsets_[slab + 1]; }
    size_t particle(size_t slot) const { return count_ == 1 ? slot : indices_[slot]; }

private:
    static constexpr int width = 4;
    size_t particles_;
    int nx_, count_ = 1;
    std::vector<size_t> offsets_, indices_;
};

} // namespace Detail
} // namespace PFFlip
