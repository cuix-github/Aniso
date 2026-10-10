#include "PFFlipSolve.h"
#include "PFFlipScatter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The fused particle-to-grid scatter. See the header for the contract; the
// executable spec is pfflip2d.py's _face_frames/_splat_group extended to the
// third axis exactly the way step_3d extends the 2D step: at nz = 1 the
// z fractional offsets are identically zero, every dk != 0 cell is culled by
// the bounds check, and the kernel reduces bit-for-bit to the 2D one.
// Destination slabs run concurrently, but accumulation at each face retains
// the serial input-particle order. Double precision internally, float at ports.

namespace {

using Vec = std::vector<double>;

struct Acc {
    Vec K, m, mom, ph, wtv;
    void init(size_t n, bool needWt) {
        K.assign(n, 0.0);
        m.assign(n, 0.0);
        mom.assign(n, 0.0);
        ph.assign(n, 0.0);
        if (needWt) wtv.assign(n, 0.0);
    }
};

} // namespace

namespace PFFlip {
namespace Solve {

void p2g_3d(int nx, int ny, int nz, int adapt, int st, float eps_mean,
            float ws_epsilon,
            const Amino::Array<Bifrost::Math::float3>& positions,
            const Amino::Array<float>& scale,
            const Amino::Array<float>& mass,
            const Amino::Array<float>& momx,
            const Amino::Array<float>& momy,
            const Amino::Array<float>& momz,
            const Amino::Array<float>& wt,
            const Amino::Array<float>& phase,
            Amino::Ptr<Amino::Array<float>>& u_mass,
            Amino::Ptr<Amino::Array<float>>& u_mom,
            Amino::Ptr<Amino::Array<float>>& u_phase,
            Amino::Ptr<Amino::Array<float>>& u_wt,
            Amino::Ptr<Amino::Array<float>>& u_ws,
            Amino::Ptr<Amino::Array<float>>& v_mass,
            Amino::Ptr<Amino::Array<float>>& v_mom,
            Amino::Ptr<Amino::Array<float>>& v_phase,
            Amino::Ptr<Amino::Array<float>>& v_wt,
            Amino::Ptr<Amino::Array<float>>& v_ws,
            Amino::Ptr<Amino::Array<float>>& w_mass,
            Amino::Ptr<Amino::Array<float>>& w_mom,
            Amino::Ptr<Amino::Array<float>>& w_phase,
            Amino::Ptr<Amino::Array<float>>& w_wt,
            Amino::Ptr<Amino::Array<float>>& w_ws,
            Amino::Ptr<Amino::Array<float>>& u2_mass,
            Amino::Ptr<Amino::Array<float>>& u2_mom,
            Amino::Ptr<Amino::Array<float>>& u2_phase,
            Amino::Ptr<Amino::Array<float>>& u2_wt,
            Amino::Ptr<Amino::Array<float>>& u2_ws,
            Amino::Ptr<Amino::Array<float>>& v2_mass,
            Amino::Ptr<Amino::Array<float>>& v2_mom,
            Amino::Ptr<Amino::Array<float>>& v2_phase,
            Amino::Ptr<Amino::Array<float>>& v2_wt,
            Amino::Ptr<Amino::Array<float>>& v2_ws,
            Amino::Ptr<Amino::Array<float>>& w2_mass,
            Amino::Ptr<Amino::Array<float>>& w2_mom,
            Amino::Ptr<Amino::Array<float>>& w2_phase,
            Amino::Ptr<Amino::Array<float>>& w2_wt,
            Amino::Ptr<Amino::Array<float>>& w2_ws) {
    const int NX = std::max(2, nx), NY = std::max(2, ny), NZ = std::max(1, nz);
    const bool stOn = st != 0;
    const bool adOn = adapt != 0;
    const int ntier = adOn ? 2 : 1;

    // PFFLIP_PROFILE=1 prints scatter/pack wall times to stderr, like the
    // step node's PFPROF lines.
    static const bool prof = []() {
        const char* e = std::getenv("PFFLIP_PROFILE");
        return e && e[0] == '1';
    }();
    using Clock = std::chrono::steady_clock;
    const Clock::time_point tBegin = Clock::now();
    Clock::time_point tScatter{};

    const int dims[3][3] = {{NX + 1, NY, NZ}, {NX, NY + 1, NZ}, {NX, NY, NZ + 1}};
    const double off[3][3] = {{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    size_t ng[3];
    Acc acc[3][2];
    for (int g = 0; g < 3; ++g) {
        ng[g] = static_cast<size_t>(dims[g][0]) * dims[g][1] * dims[g][2];
        for (int t = 0; t < ntier; ++t) acc[g][t].init(ng[g], stOn);
    }

    const size_t np = positions.size();
    const Detail::OrderedScatterSlabs slabs(np, NX,
        [&](size_t i) { return double(positions[i].x); },
        [&](size_t i) { return adOn && i < scale.size() && scale[i] >= 1.5f ? 2.0 : 1.0; });
#pragma omp parallel for schedule(dynamic, 1) if(slabs.count() > 1)
    for (int slab = 0; slab < slabs.count(); ++slab) {
        for (size_t slot = slabs.begin(slab); slot < slabs.end(slab); ++slot) {
            const size_t i = slabs.particle(slot);
            const double sc = (adOn && i < scale.size())
                ? static_cast<double>(scale[i]) : 1.0;
            const int tier = (adOn && sc >= 1.5) ? 1 : 0;
            const double r = tier ? 2.0 : 1.0;
            const int span = tier ? 2 : 1;
            const double inv_r2 = 1.0 / (r * r);
            const double q[3] = {
                i < momx.size() ? static_cast<double>(momx[i]) : 0.0,
                i < momy.size() ? static_cast<double>(momy[i]) : 0.0,
                i < momz.size() ? static_cast<double>(momz[i]) : 0.0};
            const double mi = i < mass.size() ? static_cast<double>(mass[i]) : 0.0;
            const double phi = i < phase.size() ? static_cast<double>(phase[i]) : 0.0;
            const double wi = (stOn && i < wt.size())
                ? static_cast<double>(wt[i]) : 0.0;
            const double P[3] = {positions[i].x, positions[i].y, positions[i].z};

            for (int g = 0; g < 3; ++g) {
                Acc& a = acc[g][tier];
                double f[3];
                int b[3];
                for (int ax = 0; ax < 3; ++ax) {
                    const double p = P[ax] - off[g][ax];
                    b[ax] = static_cast<int>(std::floor(p + 0.5));
                    f[ax] = p - b[ax];
                }
                for (int di = -span; di <= span; ++di) {
                    const int ii = b[0] + di;
                    if (ii < slabs.lo(slab) || ii >= slabs.hi(slab) || ii >= dims[g][0]) continue;
                    const double dx2 = (f[0] - di) * (f[0] - di);
                    for (int dj = -span; dj <= span; ++dj) {
                        const int jj = b[1] + dj;
                        if (jj < 0 || jj >= dims[g][1]) continue;
                        const double dy2 = (f[1] - dj) * (f[1] - dj);
                        for (int dk = -span; dk <= span; ++dk) {
                            const int kk = b[2] + dk;
                            if (kk < 0 || kk >= dims[g][2]) continue;
                            const double d2 =
                                (dx2 + dy2 + (f[2] - dk) * (f[2] - dk)) * inv_r2;
                            const double base = 1.0 - d2;
                            if (base <= 0.0) continue;
                            const double w3 = base * base * base;
                            const size_t idx =
                                (static_cast<size_t>(ii) * dims[g][1] + jj) *
                                    dims[g][2] + kk;
                            a.K[idx] += w3;
                            a.m[idx] += w3 * mi;
                            a.mom[idx] += w3 * q[g];
                            a.ph[idx] += w3 * phi;
                            if (stOn) a.wtv[idx] += w3 * wi;
                        }
                    }
                }
            }
        }
    }

    if (prof) tScatter = Clock::now();

    const double em = eps_mean, ew = ws_epsilon;
    auto mean = [&](const Vec& A, const Vec& K) {
        auto out = Amino::newMutablePtr<Amino::Array<float>>(A.size());
        for (size_t c = 0; c < A.size(); ++c)
            (*out)[c] = static_cast<float>(A[c] / (K[c] + em));
        return out.toImmutable();
    };
    auto wsChan = [&](const Vec& K) {
        auto out = Amino::newMutablePtr<Amino::Array<float>>(K.size());
        for (size_t c = 0; c < K.size(); ++c)
            (*out)[c] = static_cast<float>(K[c] / (K[c] + ew));
        return out.toImmutable();
    };
    auto empty = []() {
        return Amino::newMutablePtr<Amino::Array<float>>(0).toImmutable();
    };

    Amino::Ptr<Amino::Array<float>>* outs[2][3][5] = {
        {{&u_mass, &u_mom, &u_phase, &u_wt, &u_ws},
         {&v_mass, &v_mom, &v_phase, &v_wt, &v_ws},
         {&w_mass, &w_mom, &w_phase, &w_wt, &w_ws}},
        {{&u2_mass, &u2_mom, &u2_phase, &u2_wt, &u2_ws},
         {&v2_mass, &v2_mom, &v2_phase, &v2_wt, &v2_ws},
         {&w2_mass, &w2_mom, &w2_phase, &w2_wt, &w2_ws}}};
    for (int t = 0; t < 2; ++t)
        for (int g = 0; g < 3; ++g) {
            if (t >= ntier) {
                for (int c = 0; c < 5; ++c) *outs[t][g][c] = empty();
                continue;
            }
            const Acc& a = acc[g][t];
            *outs[t][g][0] = mean(a.m, a.K);
            *outs[t][g][1] = mean(a.mom, a.K);
            *outs[t][g][2] = mean(a.ph, a.K);
            *outs[t][g][3] = stOn ? mean(a.wtv, a.K) : empty();
            *outs[t][g][4] = adOn ? wsChan(a.K) : empty();
        }

    if (prof) {
        const auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const Clock::time_point tEnd = Clock::now();
        std::fprintf(stderr, "P2GPROF np=%zu scatter=%.1f pack=%.1f total=%.1f\n",
                     np, ms(tBegin, tScatter), ms(tScatter, tEnd),
                     ms(tBegin, tEnd));
        std::fflush(stderr);
    }
}

void dump_gate(int step_index, int every, const Amino::String& path,
               Amino::String& path_out) {
    // Pass the write path through only every Nth substep (step 0 always);
    // an empty path makes the downstream write node skip the dump cheaply.
    path_out = (every <= 1 || step_index % std::max(1, every) == 0)
        ? path : Amino::String();
}

} // namespace Solve
} // namespace PFFlip
