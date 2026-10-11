#include "PFFlipSolve.h"
#include "PFFlipScatter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <vector>

// Milestone 6: the PF-FLIP step in 3D with a multigrid-preconditioned solve.
// Same structure as the 2D node (which mirrors sim/reference2d/pfflip2d.py);
// at nz = 1 the w faces are all walls and this reduces exactly to the 2D
// method, which validate_3d.py uses as the lockstep bridge. Double precision
// internally. The multigrid is a geometric V-cycle on the variable-beta
// Poisson operator: damped-Jacobi smoothing, cell-average restriction,
// constant prolongation, face-averaged coarse coefficients, explicit 1/h^2
// per level; used inside flexible PCG so mild asymmetry cannot stall CG.

namespace {

using Vec = std::vector<double>;

struct Level {
    int nx, ny, nz;
    double inv_h2;
    Vec bu, bv, bw;        // face coefficients, walls zero
    Vec diagInv;           // inverse diagonal of A
    Vec x, b, r;           // work vectors (cells)
    size_t nc() const { return static_cast<size_t>(nx) * ny * nz; }
    size_t iu(int i, int j, int k) const { return (static_cast<size_t>(i) * ny + j) * nz + k; }
    size_t iv(int i, int j, int k) const { return (static_cast<size_t>(i) * (ny + 1) + j) * nz + k; }
    size_t iw(int i, int j, int k) const { return (static_cast<size_t>(i) * ny + j) * (nz + 1) + k; }
    size_t ic(int i, int j, int k) const { return (static_cast<size_t>(i) * ny + j) * nz + k; }
};

void applyA(const Level& L, const Vec& x, Vec& out) {
    const int NX = L.nx, NY = L.ny, NZ = L.nz;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                const size_t c = L.ic(i, j, k);
                double acc = 0.0;
                if (i + 1 < NX) acc += L.bu[L.iu(i + 1, j, k)] * (x[L.ic(i + 1, j, k)] - x[c]);
                if (i > 0)      acc -= L.bu[L.iu(i, j, k)] * (x[c] - x[L.ic(i - 1, j, k)]);
                if (j + 1 < NY) acc += L.bv[L.iv(i, j + 1, k)] * (x[L.ic(i, j + 1, k)] - x[c]);
                if (j > 0)      acc -= L.bv[L.iv(i, j, k)] * (x[c] - x[L.ic(i, j - 1, k)]);
                if (k + 1 < NZ) acc += L.bw[L.iw(i, j, k + 1)] * (x[L.ic(i, j, k + 1)] - x[c]);
                if (k > 0)      acc -= L.bw[L.iw(i, j, k)] * (x[c] - x[L.ic(i, j, k - 1)]);
                out[c] = -acc * L.inv_h2;
            }
}

void buildDiag(Level& L) {
    L.diagInv.assign(L.nc(), 0.0);
    for (int i = 0; i < L.nx; ++i)
        for (int j = 0; j < L.ny; ++j)
            for (int k = 0; k < L.nz; ++k) {
                const double dg = (L.bu[L.iu(i, j, k)] + L.bu[L.iu(i + 1, j, k)] +
                                   L.bv[L.iv(i, j, k)] + L.bv[L.iv(i, j + 1, k)] +
                                   L.bw[L.iw(i, j, k)] + L.bw[L.iw(i, j, k + 1)]) * L.inv_h2;
                L.diagInv[L.ic(i, j, k)] = dg > 0.0 ? 1.0 / dg : 0.0;
            }
}

void smooth(Level& L, Vec& x, const Vec& b, int sweeps, Vec& tmp) {
    const double omega = 0.8;
    for (int s = 0; s < sweeps; ++s) {
        applyA(L, x, tmp);
        const long long n = static_cast<long long>(x.size());
#pragma omp parallel for schedule(static)
        for (long long c = 0; c < n; ++c)
            x[c] += omega * L.diagInv[c] * (b[c] - tmp[c]);
    }
}

int half(int n) { return n > 1 ? (n + 1) / 2 : 1; }

// Average the <=4 fine faces a coarse face covers; clipped at odd boundaries.
double faceAvgU(const Level& f, int I, int J, int K, int cny, int cnz) {
    (void)cny; (void)cnz;
    const int i = std::min(2 * I, f.nx);
    double s = 0.0; int n = 0;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
            const int j = 2 * J + a, k = 2 * K + b;
            if (j < f.ny && k < f.nz) { s += f.bu[f.iu(i, j, k)]; ++n; }
        }
    return n ? s / n : 0.0;
}
double faceAvgV(const Level& f, int I, int J, int K) {
    const int j = std::min(2 * J, f.ny);
    double s = 0.0; int n = 0;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
            const int i = 2 * I + a, k = 2 * K + b;
            if (i < f.nx && k < f.nz) { s += f.bv[f.iv(i, j, k)]; ++n; }
        }
    return n ? s / n : 0.0;
}
double faceAvgW(const Level& f, int I, int J, int K) {
    const int k = std::min(2 * K, f.nz);
    double s = 0.0; int n = 0;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
            const int i = 2 * I + a, j = 2 * J + b;
            if (i < f.nx && j < f.ny) { s += f.bw[f.iw(i, j, k)]; ++n; }
        }
    return n ? s / n : 0.0;
}

std::vector<Level> buildHierarchy(Level&& fine) {
    std::vector<Level> ls;
    ls.push_back(std::move(fine));
    while (ls.size() < 12) {
        const Level& f = ls.back();
        if (f.nc() <= 64) break;
        Level c;
        c.nx = half(f.nx); c.ny = half(f.ny); c.nz = half(f.nz);
        c.inv_h2 = f.inv_h2 * 0.25;
        c.bu.assign(static_cast<size_t>(c.nx + 1) * c.ny * c.nz, 0.0);
        c.bv.assign(static_cast<size_t>(c.nx) * (c.ny + 1) * c.nz, 0.0);
        c.bw.assign(static_cast<size_t>(c.nx) * c.ny * (c.nz + 1), 0.0);
        for (int I = 0; I <= c.nx; ++I)
            for (int J = 0; J < c.ny; ++J)
                for (int K = 0; K < c.nz; ++K)
                    c.bu[c.iu(I, J, K)] = (I == 0 || I == c.nx) ? 0.0 : faceAvgU(f, I, J, K, c.ny, c.nz);
        for (int I = 0; I < c.nx; ++I)
            for (int J = 0; J <= c.ny; ++J)
                for (int K = 0; K < c.nz; ++K)
                    c.bv[c.iv(I, J, K)] = (J == 0 || J == c.ny) ? 0.0 : faceAvgV(f, I, J, K);
        for (int I = 0; I < c.nx; ++I)
            for (int J = 0; J < c.ny; ++J)
                for (int K = 0; K <= c.nz; ++K)
                    c.bw[c.iw(I, J, K)] = (K == 0 || K == c.nz) ? 0.0 : faceAvgW(f, I, J, K);
        buildDiag(c);
        c.x.assign(c.nc(), 0.0); c.b.assign(c.nc(), 0.0); c.r.assign(c.nc(), 0.0);
        ls.push_back(std::move(c));
    }
    return ls;
}

void vcycle(std::vector<Level>& ls, size_t l, Vec& tmp) {
    Level& L = ls[l];
    if (l + 1 == ls.size()) {
        smooth(L, L.x, L.b, 40, tmp);
        return;
    }
    smooth(L, L.x, L.b, 2, tmp);
    applyA(L, L.x, tmp);
    for (size_t c = 0; c < L.nc(); ++c) L.r[c] = L.b[c] - tmp[c];
    Level& C = ls[l + 1];
    std::fill(C.b.begin(), C.b.end(), 0.0);
    std::fill(C.x.begin(), C.x.end(), 0.0);
    std::vector<int> cnt(C.nc(), 0);
    for (int i = 0; i < L.nx; ++i)
        for (int j = 0; j < L.ny; ++j)
            for (int k = 0; k < L.nz; ++k) {
                const size_t cc = C.ic(i / 2, std::min(j / 2, C.ny - 1), std::min(k / 2, C.nz - 1));
                C.b[cc] += L.r[L.ic(i, j, k)];
                ++cnt[cc];
            }
    for (size_t c = 0; c < C.nc(); ++c)
        if (cnt[c]) C.b[c] /= cnt[c];
    vcycle(ls, l + 1, tmp);
    for (int i = 0; i < L.nx; ++i)
        for (int j = 0; j < L.ny; ++j)
            for (int k = 0; k < L.nz; ++k)
                L.x[L.ic(i, j, k)] +=
                    C.x[C.ic(i / 2, std::min(j / 2, C.ny - 1), std::min(k / 2, C.nz - 1))];
    smooth(L, L.x, L.b, 2, tmp);
}

// Trilinear sampling of a staggered grid with per-axis index clipping,
// mirroring the 2D _sample_faces (and collapsing in any dimension of size 1).
double sampleGrid(const Vec& g, int gx, int gy, int gz,
                  double offx, double offy, double offz,
                  double px, double py, double pz) {
    const double qx = px - offx, qy = py - offy, qz = pz - offz;
    const double bx = std::floor(qx), by = std::floor(qy), bz = std::floor(qz);
    const double fx = qx - bx, fy = qy - by, fz = qz - bz;
    double out = 0.0;
    for (int di = 0; di <= 1; ++di)
        for (int dj = 0; dj <= 1; ++dj)
            for (int dk = 0; dk <= 1; ++dk) {
                const double w = (di ? fx : 1.0 - fx) * (dj ? fy : 1.0 - fy) *
                                 (dk ? fz : 1.0 - fz);
                const int ii = std::clamp(static_cast<int>(bx) + di, 0, gx - 1);
                const int jj = std::clamp(static_cast<int>(by) + dj, 0, gy - 1);
                const int kk = std::clamp(static_cast<int>(bz) + dk, 0, gz - 1);
                out += w * g[(static_cast<size_t>(ii) * gy + jj) * gz + kk];
            }
    return out;
}

// ---- ST-FLIP helpers: temporal kernel, its Gauss-Legendre normalization, smoothstep.
double wtKernel(double tau) {
    const double d = 1.0 - (tau - 0.5) * (tau - 0.5);
    return d > 0.0 ? (35.0 / 16.0) * d * d * d : 0.0;
}
// numpy.polynomial.legendre.leggauss(8), both scaled by 0.5 (as the 2D reference).
const double GL_N[8] = {-0.4801449282487681, -0.3983332387068134, -0.2627662049581645,
                        -0.0917173212478249, 0.0917173212478249, 0.2627662049581645,
                        0.3983332387068134, 0.4801449282487681};
const double GL_W[8] = {0.0506142681451881, 0.1111905172266872, 0.1568533229389436,
                        0.1813418916891810, 0.1813418916891810, 0.1568533229389436,
                        0.1111905172266872, 0.0506142681451881};
double wtNorm(double xi) {
    double z = 0.0;
    for (int i = 0; i < 8; ++i) z += GL_W[i] * wtKernel(xi * GL_N[i]);
    return std::max(z, 1e-9);
}

} // namespace

namespace PFFlip {
namespace Solve {

void step_3d(int nx, int ny, int nz, float dt_in, float gravity, float rho_liquid,
             float rho_air, float alpha_liquid, float alpha_air, int max_iterations,
             float tolerance, int preconditioner, int escape, float esc_phi,
             float drag_droplet, float drag_bubble, float buoyancy, float rho0_face,
             int st, int st_seed, int step_index,
             int adapt, float coarse_gain, float ws_epsilon,
             const Amino::Array<float>& scale_in,
             const Amino::Array<Bifrost::Math::float3>& obstacles_min,
             const Amino::Array<Bifrost::Math::float3>& obstacles_max,
             const Amino::Array<float>& tau_in,
             const Amino::Array<float>& u_wt, const Amino::Array<float>& v_wt,
             const Amino::Array<float>& w_wt,
             const Amino::Array<float>& u_ws, const Amino::Array<float>& v_ws,
             const Amino::Array<float>& w_ws,
             const Amino::Array<float>& u2_mass, const Amino::Array<float>& u2_mom,
             const Amino::Array<float>& u2_phase, const Amino::Array<float>& u2_wt,
             const Amino::Array<float>& u2_ws,
             const Amino::Array<float>& v2_mass, const Amino::Array<float>& v2_mom,
             const Amino::Array<float>& v2_phase, const Amino::Array<float>& v2_wt,
             const Amino::Array<float>& v2_ws,
             const Amino::Array<float>& w2_mass, const Amino::Array<float>& w2_mom,
             const Amino::Array<float>& w2_phase, const Amino::Array<float>& w2_wt,
             const Amino::Array<float>& w2_ws,
             const Amino::Array<float>& u_mass, const Amino::Array<float>& u_mom,
             const Amino::Array<float>& u_phase, const Amino::Array<float>& v_mass,
             const Amino::Array<float>& v_mom, const Amino::Array<float>& v_phase,
             const Amino::Array<float>& w_mass, const Amino::Array<float>& w_mom,
             const Amino::Array<float>& w_phase,
             const Amino::Array<Bifrost::Math::float3>& positions,
             const Amino::Array<Bifrost::Math::float3>& velocities,
             const Amino::Array<float>& particle_phase,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_positions,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_velocities,
             Amino::Ptr<Amino::Array<float>>& out_mass,
             Amino::Ptr<Amino::Array<float>>& out_mom_x,
             Amino::Ptr<Amino::Array<float>>& out_mom_y,
             Amino::Ptr<Amino::Array<float>>& out_mom_z,
             Amino::Ptr<Amino::Array<float>>& out_u,
             Amino::Ptr<Amino::Array<float>>& out_v,
             Amino::Ptr<Amino::Array<float>>& out_w,
             Amino::Ptr<Amino::Array<float>>& pressure,
             Amino::Ptr<Amino::Array<float>>& out_escaped,
             Amino::Ptr<Amino::Array<float>>& tau_out,
             Amino::Ptr<Amino::Array<float>>& scale_out,
             Amino::Ptr<Amino::Array<bool>>& mask_fine,
             Amino::Ptr<Amino::Array<bool>>& mask_coarse,
             Amino::Ptr<Amino::Array<float>>& out_phase_state,
             Amino::Ptr<Amino::Array<float>>& out_wt,
             Amino::Ptr<Amino::Array<float>>& out_wtph,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_pos_synced,
             int& iterations_used, float& final_residual, float& max_divergence_after,
             float& max_speed) {
    const int NX = std::max(2, nx), NY = std::max(2, ny), NZ = std::max(1, nz);
    const double dt = dt_in;
    const size_t np = positions.size();

    // PFFLIP_PROFILE=1 prints per-phase wall times to stderr (spike 1); the
    // phase boundaries are the ---- section markers below.
    static const bool prof = []() {
        const char* e = std::getenv("PFFLIP_PROFILE");
        return e && e[0] == '1';
    }();
    using Clock = std::chrono::steady_clock;
    Clock::time_point tMark = Clock::now();
    const Clock::time_point tBegin = tMark;
    double msAsm = 0.0, msSolve = 0.0, msCorr = 0.0, msPart = 0.0;
    auto lap = [&](double& acc) {
        if (!prof) return;
        const Clock::time_point now = Clock::now();
        acc = std::chrono::duration<double, std::milli>(now - tMark).count();
        tMark = now;
    };

    Level L;
    L.nx = NX; L.ny = NY; L.nz = NZ; L.inv_h2 = 1.0;
    const size_t nu = static_cast<size_t>(NX + 1) * NY * NZ;
    const size_t nv = static_cast<size_t>(NX) * (NY + 1) * NZ;
    const size_t nw = static_cast<size_t>(NX) * NY * (NZ + 1);

    // ---- channels -> face velocities (u = mom/mass where mass > 0).
    Vec u(nu, 0.0), v(nv, 0.0), w(nw, 0.0);
    // Two-tier combination: the stock splat returns weighted MEANS per tier; the
    // weight SUMS are recovered from the epsilon-inversion channel ws, where the
    // graph splats the constant 1 with add_to_denominator = eps, so
    // mean = S/(S+eps)  =>  S = eps*mean/(1-mean). Combined mean of q is then
    // (S_A q_A + S_B q_B)/(S_A + S_B). With adapt off (or no tier-B channels) the
    // tier-A channels pass straight through.
    const bool twoTier = adapt != 0 && u2_mass.size() > 0;
    const double epsw = ws_epsilon > 0.0f ? ws_epsilon : 16.0;
    auto wsum = [&](const Amino::Array<float>& ws, size_t k) {
        const double a = k < ws.size()
            ? std::clamp(static_cast<double>(ws[k]), 0.0, 0.999999) : 0.0;
        return epsw * a / (1.0 - a);
    };
    auto at = [](const Amino::Array<float>& arr, size_t k) {
        return k < arr.size() ? static_cast<double>(arr[k]) : 0.0;
    };
    Vec chM_u(nu), chO_u(nu), chP_u(nu), chW_u(nu);
    Vec chM_v(nv), chO_v(nv), chP_v(nv), chW_v(nv);
    Vec chM_w(nw), chO_w(nw), chP_w(nw), chW_w(nw);
    const bool hasWt = u_wt.size() > 0;
    auto combine = [&](size_t n, const Amino::Array<float>& mA, const Amino::Array<float>& oA,
                       const Amino::Array<float>& pA, const Amino::Array<float>& wtA,
                       const Amino::Array<float>& wsA, const Amino::Array<float>& mB,
                       const Amino::Array<float>& oB, const Amino::Array<float>& pB,
                       const Amino::Array<float>& wtB, const Amino::Array<float>& wsB,
                       Vec& M, Vec& O, Vec& P, Vec& W) {
        for (size_t k = 0; k < n; ++k) {
            if (!twoTier) {
                M[k] = at(mA, k); O[k] = at(oA, k); P[k] = at(pA, k);
                W[k] = hasWt ? at(wtA, k) : 1.0;
                continue;
            }
            const double sA = wsum(wsA, k), sB = wsum(wsB, k);
            const double den = sA + sB;
            if (den < 1e-12) { M[k] = O[k] = P[k] = 0.0; W[k] = hasWt ? 0.0 : 1.0; continue; }
            M[k] = (sA * at(mA, k) + sB * at(mB, k)) / den;
            O[k] = (sA * at(oA, k) + sB * at(oB, k)) / den;
            P[k] = (sA * at(pA, k) + sB * at(pB, k)) / den;
            W[k] = hasWt ? (sA * at(wtA, k) + sB * at(wtB, k)) / den : 1.0;
        }
    };
    combine(nu, u_mass, u_mom, u_phase, u_wt, u_ws,
            u2_mass, u2_mom, u2_phase, u2_wt, u2_ws, chM_u, chO_u, chP_u, chW_u);
    combine(nv, v_mass, v_mom, v_phase, v_wt, v_ws,
            v2_mass, v2_mom, v2_phase, v2_wt, v2_ws, chM_v, chO_v, chP_v, chW_v);
    combine(nw, w_mass, w_mom, w_phase, w_wt, w_ws,
            w2_mass, w2_mom, w2_phase, w2_wt, w2_ws, chM_w, chO_w, chP_w, chW_w);
    auto normalize = [](Vec& g, const Vec& mass, const Vec& mom) {
        for (size_t c = 0; c < g.size(); ++c)
            g[c] = mass[c] > 0.0 ? mom[c] / std::max(mass[c], 1e-30) : 0.0;
    };
    normalize(u, chM_u, chO_u);
    normalize(v, chM_v, chO_v);
    normalize(w, chM_w, chO_w);

    const size_t nObs = std::min(obstacles_min.size(), obstacles_max.size());
    // apply fn(face-kind, i, j, k) over every face whose centre lies inside a box.
    auto forSolidFaces = [&](auto&& fn) {
        for (size_t b = 0; b < nObs; ++b) {
            const auto& m0 = obstacles_min[b];
            const auto& m1 = obstacles_max[b];
            for (int i = std::max(0, (int)std::ceil(m0.x));
                 i <= std::min(NX, (int)std::floor(m1.x)); ++i)
                for (int j = std::max(0, (int)std::ceil(m0.y - 0.5));
                     j <= std::min(NY - 1, (int)std::floor(m1.y - 0.5)); ++j)
                    for (int k = std::max(0, (int)std::ceil(m0.z - 0.5));
                         k <= std::min(NZ - 1, (int)std::floor(m1.z - 0.5)); ++k)
                        fn(0, i, j, k);
            for (int i = std::max(0, (int)std::ceil(m0.x - 0.5));
                 i <= std::min(NX - 1, (int)std::floor(m1.x - 0.5)); ++i)
                for (int j = std::max(0, (int)std::ceil(m0.y));
                     j <= std::min(NY, (int)std::floor(m1.y)); ++j)
                    for (int k = std::max(0, (int)std::ceil(m0.z - 0.5));
                         k <= std::min(NZ - 1, (int)std::floor(m1.z - 0.5)); ++k)
                        fn(1, i, j, k);
            for (int i = std::max(0, (int)std::ceil(m0.x - 0.5));
                 i <= std::min(NX - 1, (int)std::floor(m1.x - 0.5)); ++i)
                for (int j = std::max(0, (int)std::ceil(m0.y - 0.5));
                     j <= std::min(NY - 1, (int)std::floor(m1.y - 0.5)); ++j)
                    for (int k = std::max(0, (int)std::ceil(m0.z));
                         k <= std::min(NZ, (int)std::floor(m1.z)); ++k)
                        fn(2, i, j, k);
        }
    };
    auto walls = [&](Vec& uu, Vec& vv, Vec& ww) {
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) { uu[L.iu(0, j, k)] = 0.0; uu[L.iu(NX, j, k)] = 0.0; }
        for (int i = 0; i < NX; ++i)
            for (int k = 0; k < NZ; ++k) { vv[L.iv(i, 0, k)] = 0.0; vv[L.iv(i, NY, k)] = 0.0; }
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NY; ++j) { ww[L.iw(i, j, 0)] = 0.0; ww[L.iw(i, j, NZ)] = 0.0; }
        forSolidFaces([&](int kind, int i, int j, int k) {
            if (kind == 0) uu[L.iu(i, j, k)] = 0.0;
            else if (kind == 1) vv[L.iv(i, j, k)] = 0.0;
            else ww[L.iw(i, j, k)] = 0.0;
        });
    };
    walls(u, v, w);
    Vec uStar = u, vStar = v, wStar = w;

    // ---- gravity on y faces, then walls.
    for (size_t c = 0; c < nv; ++c) v[c] += gravity * dt;
    walls(u, v, w);

    // ---- face coefficients beta = 1/rho(phi), walls zero.
    L.bu.assign(nu, 0.0); L.bv.assign(nv, 0.0); L.bw.assign(nw, 0.0);
    const double rl = rho_liquid, rg = rho_air;
    // With ST graphs the phase channel arrives premultiplied by the temporal weight
    // and must be divided by the splatted weight channel; an empty wt channel means
    // the plain scheme (wt = 1), keeping every existing graph and driver valid.
    auto beta = [&](Vec& bb, const Vec& ph, const Vec& wt) {
        for (size_t c = 0; c < bb.size(); ++c) {
            double p = ph[c];
            if (hasWt) p = wt[c] > 1e-12 ? p / wt[c] : 0.0;
            bb[c] = 1.0 / (rg + (rl - rg) * std::clamp(p, 0.0, 1.0));
        }
    };
    beta(L.bu, chP_u, chW_u); beta(L.bv, chP_v, chW_v); beta(L.bw, chP_w, chW_w);
    walls(L.bu, L.bv, L.bw);
    buildDiag(L);
    L.x.assign(L.nc(), 0.0); L.b.assign(L.nc(), 0.0); L.r.assign(L.nc(), 0.0);

    // ---- right-hand side b = -div(u)/dt.
    Vec rhs(L.nc());
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                rhs[L.ic(i, j, k)] = -((u[L.iu(i + 1, j, k)] - u[L.iu(i, j, k)]) +
                                       (v[L.iv(i, j + 1, k)] - v[L.iv(i, j, k)]) +
                                       (w[L.iw(i, j, k + 1)] - w[L.iw(i, j, k)])) / dt;

    lap(msAsm);
    // ---- flexible PCG with Jacobi (0) or multigrid V-cycle (1) preconditioner.
    std::vector<Level> ls;
    Vec tmp(L.nc());
    if (preconditioner == 1) ls = buildHierarchy(std::move(L));
    Level& F = preconditioner == 1 ? ls[0] : L;

    const size_t ncell = F.nc();
    Vec p(ncell, 0.0), r = rhs, z(ncell), d(ncell), q(ncell), rPrev(ncell, 0.0);
    auto precond = [&](const Vec& rr, Vec& zz) {
        if (preconditioner == 1) {
            ls[0].b = rr;
            std::fill(ls[0].x.begin(), ls[0].x.end(), 0.0);
            vcycle(ls, 0, tmp);
            zz = ls[0].x;
        } else {
            for (size_t c = 0; c < ncell; ++c) zz[c] = F.diagInv[c] * rr[c];
        }
    };
    precond(r, z);
    d = z;
    double rz = 0.0, rr2 = 0.0;
    for (size_t c = 0; c < ncell; ++c) { rz += r[c] * z[c]; rr2 += r[c] * r[c]; }
    double r0 = std::sqrt(rr2);
    if (r0 == 0.0) r0 = 1.0;
    int it = 0;
    for (it = 0; it < max_iterations; ++it) {
        rr2 = 0.0;
        for (size_t c = 0; c < ncell; ++c) rr2 += r[c] * r[c];
        if (std::sqrt(rr2) / r0 < tolerance) break;
        applyA(F, d, q);
        double dq = 0.0;
        for (size_t c = 0; c < ncell; ++c) dq += d[c] * q[c];
        if (dq == 0.0) break;
        const double a = rz / dq;
        rPrev = r;
        const long long ncl = static_cast<long long>(ncell);
#pragma omp parallel for schedule(static)
        for (long long c = 0; c < ncl; ++c) { p[c] += a * d[c]; r[c] -= a * q[c]; }
        precond(r, z);
        double rzNew = 0.0;   // flexible (Polak-Ribiere) update tolerates the V-cycle
        for (size_t c = 0; c < ncell; ++c) rzNew += (r[c] - rPrev[c]) * z[c];
        const double betaCG = std::max(0.0, rzNew / rz);
        for (size_t c = 0; c < ncell; ++c) d[c] = z[c] + betaCG * d[c];
        rz = 0.0;
        for (size_t c = 0; c < ncell; ++c) rz += r[c] * z[c];
        if (rz == 0.0) break;
    }

    lap(msSolve);
    // ---- velocity correction on interior faces, then walls.
    for (int i = 1; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                u[F.iu(i, j, k)] -= dt * F.bu[F.iu(i, j, k)] *
                                    (p[F.ic(i, j, k)] - p[F.ic(i - 1, j, k)]);
    for (int i = 0; i < NX; ++i)
        for (int j = 1; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                v[F.iv(i, j, k)] -= dt * F.bv[F.iv(i, j, k)] *
                                    (p[F.ic(i, j, k)] - p[F.ic(i, j - 1, k)]);
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 1; k < NZ; ++k)
                w[F.iw(i, j, k)] -= dt * F.bw[F.iw(i, j, k)] *
                                    (p[F.ic(i, j, k)] - p[F.ic(i, j, k - 1)]);
    walls(u, v, w);

    double maxDiv = 0.0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                maxDiv = std::max(maxDiv, std::fabs(
                    (u[F.iu(i + 1, j, k)] - u[F.iu(i, j, k)]) +
                    (v[F.iv(i, j + 1, k)] - v[F.iv(i, j, k)]) +
                    (w[F.iw(i, j, k + 1)] - w[F.iw(i, j, k)])));

    lap(msCorr);
    // ---- particle machinery. Sizes: with adapt on, particles carry a scale
    // (1 fine, 2 coarse); masses scale as scale^d (d = 2 flat, 3 otherwise) with a
    // calibrated coarse gain; the FLIP blend uses (1-alpha)/scale^2; deep-air fine
    // air merges 4-to-1 (8-to-1 in 3D) and coarse splits near the interface or
    // walls, mirroring the 2D reference's rules.
    const bool flat = NZ == 1;
    const bool stOn = st != 0;
    const bool adOn = adapt != 0;
    const double dExp = flat ? 2.0 : 3.0;
    auto scaleOf = [&](size_t pi) {
        return adOn && pi < scale_in.size() ? static_cast<double>(scale_in[pi]) : 1.0;
    };

    // count splats (Eq. 6 kernel) at given positions; value = scale^d per particle.
    auto countSplat = [&](const std::vector<double>& qx, const std::vector<double>& qy,
                          const std::vector<double>& qz, const std::vector<char>& isLiq,
                          const std::vector<double>& scl, Vec cntU[2], Vec cntV[2],
                          Vec cntW[2]) {
        const size_t n = qx.size();
        const Detail::OrderedScatterSlabs slabs(n, NX,
            [&](size_t i) { return qx[i]; }, [](size_t) { return 1.0; });
#pragma omp parallel for schedule(dynamic, 1) if(slabs.count() > 1)
        for (int slab = 0; slab < slabs.count(); ++slab) {
            for (size_t slot = slabs.begin(slab); slot < slabs.end(slab); ++slot) {
                const size_t pi = slabs.particle(slot);
                const int ph = isLiq[pi] ? 1 : 0;
                const double val = std::pow(scl[pi], dExp);
                const int ngrids = NZ == 1 ? 2 : 3;
                for (int grid = 0; grid < ngrids; ++grid) {
                    const double ox = grid == 0 ? 0.0 : 0.5;
                    const double oy = grid == 1 ? 0.0 : 0.5;
                    const double oz = grid == 2 ? 0.0 : 0.5;
                    const int gx = grid == 0 ? NX + 1 : NX;
                    const int gy = grid == 1 ? NY + 1 : NY;
                    const int gzd = grid == 2 ? NZ + 1 : NZ;
                    Vec* cnt = grid == 0 ? cntU : (grid == 1 ? cntV : cntW);
                    const double ax = qx[pi] - ox, ay = qy[pi] - oy, az = qz[pi] - oz;
                    const int bx = static_cast<int>(std::floor(ax + 0.5));
                    const int by = static_cast<int>(std::floor(ay + 0.5));
                    const int bz = static_cast<int>(std::floor(az + 0.5));
                    const double fx = ax - bx, fy = ay - by, fz = az - bz;
                    for (int di = -1; di <= 1; ++di)
                        for (int dj = -1; dj <= 1; ++dj)
                            for (int dk = -1; dk <= 1; ++dk) {
                                const int i = bx + di, j = by + dj, k = bz + dk;
                                if (i < slabs.lo(slab) || i >= slabs.hi(slab) || i >= gx ||
                                    j < 0 || j >= gy || k < 0 || k >= gzd)
                                    continue;
                                const double d2 = (fx - di) * (fx - di) + (fy - dj) * (fy - dj) +
                                                  (fz - dk) * (fz - dk);
                                const double wgt = std::max(1.0 - d2, 0.0);
                                if (wgt > 0.0)
                                    cnt[ph][(static_cast<size_t>(i) * gy + j) * gzd + k] +=
                                        wgt * wgt * wgt * val;
                            }
                }
            }
        }
    };
    auto sampleCnt = [&](const Vec& cu, const Vec& cv, const Vec& cw,
                         double px, double py, double pz) {
        const double su2 = sampleGrid(cu, NX + 1, NY, NZ, 0.0, 0.5, 0.5, px, py, pz);
        const double sv2 = sampleGrid(cv, NX, NY + 1, NZ, 0.5, 0.0, 0.5, px, py, pz);
        if (NZ == 1) return 0.5 * (su2 + sv2);
        const double sw2 = sampleGrid(cw, NX, NY, NZ + 1, 0.5, 0.5, 0.0, px, py, pz);
        return (su2 + sv2 + sw2) / 3.0;
    };

    std::vector<char> escFlag(np, 0);
    if ((escape != 0 || adOn) && np > 0) {
        std::vector<double> qx(np), qy(np), qz(np), scl(np);
        std::vector<char> isLiq(np);
#pragma omp parallel for schedule(static)
        for (long long pi = 0; pi < static_cast<long long>(np); ++pi) {
            qx[pi] = positions[pi].x; qy[pi] = positions[pi].y; qz[pi] = positions[pi].z;
            scl[pi] = scaleOf(pi);
            isLiq[pi] = (pi < particle_phase.size() ? particle_phase[pi] : 0.0f) >= 0.5f;
        }
        Vec cntU[2] = {Vec(nu, 0.0), Vec(nu, 0.0)};
        Vec cntV[2] = {Vec(nv, 0.0), Vec(nv, 0.0)};
        Vec cntW[2] = {Vec(nw, 0.0), Vec(nw, 0.0)};
        countSplat(qx, qy, qz, isLiq, scl, cntU, cntV, cntW);
        if (escape != 0) {
            const double thresh = (1.0 - esc_phi) * rho0_face;
#pragma omp parallel for schedule(static)
            for (long long pi = 0; pi < static_cast<long long>(np); ++pi) {
                const int other = isLiq[pi] ? 0 : 1;
                const double frac = sampleCnt(cntU[other], cntV[other], cntW[other],
                                              qx[pi], qy[pi], qz[pi]);
                escFlag[pi] = frac > thresh ? 1 : 0;
            }
        }
    }

    Vec du(nu), dv(nv), dw(nw);
    const long long lnu = static_cast<long long>(nu), lnv = static_cast<long long>(nv);
    const long long lnw = static_cast<long long>(nw);
#pragma omp parallel for schedule(static)
    for (long long c = 0; c < lnu; ++c) du[c] = u[c] - uStar[c];
#pragma omp parallel for schedule(static)
    for (long long c = 0; c < lnv; ++c) dv[c] = v[c] - vStar[c];
#pragma omp parallel for schedule(static)
    for (long long c = 0; c < lnw; ++c) dw[c] = w[c] - wStar[c];

    auto sU = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX + 1, NY, NZ, 0.0, 0.5, 0.5, x, y, zc); };
    auto sV = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX, NY + 1, NZ, 0.5, 0.0, 0.5, x, y, zc); };
    auto sW = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX, NY, NZ + 1, 0.5, 0.5, 0.0, x, y, zc); };

    const double lo = 0.51, hx = NX - 0.51, hy = NY - 0.51, hz = NZ - 0.51;
    const long long lnp = static_cast<long long>(np);

    std::vector<double> nvx(np), nvy(np), nvz(np);
#pragma omp parallel for schedule(static)
    for (long long pi = 0; pi < lnp; ++pi) {
        const double px = positions[pi].x, py = positions[pi].y, pz = positions[pi].z;
        double vx = pi < static_cast<long long>(velocities.size()) ? velocities[pi].x : 0.0;
        double vy = pi < static_cast<long long>(velocities.size()) ? velocities[pi].y : 0.0;
        double vz = pi < static_cast<long long>(velocities.size()) ? velocities[pi].z : 0.0;
        const bool liquid = (static_cast<size_t>(pi) < particle_phase.size()
                             ? particle_phase[pi] : 0.0f) >= 0.5f;
        double a = liquid ? alpha_liquid : alpha_air;
        if (adOn) {
            const double sc = scaleOf(pi);
            a = 1.0 - (1.0 - a) / (sc * sc);
        }
        if (!escFlag[pi]) {
            vx = a * (vx + sU(du, px, py, pz)) + (1.0 - a) * sU(u, px, py, pz);
            vy = a * (vy + sV(dv, px, py, pz)) + (1.0 - a) * sV(v, px, py, pz);
            vz = flat ? 0.0 : a * (vz + sW(dw, px, py, pz)) + (1.0 - a) * sW(w, px, py, pz);
        } else {
            const double gx = sU(u, px, py, pz), gy = sV(v, px, py, pz);
            const double gz = flat ? 0.0 : sW(w, px, py, pz);
            const double drag = liquid ? drag_droplet : drag_bubble;
            vy += (liquid ? gravity : buoyancy * std::fabs(gravity)) * dt;
            vx += drag * dt * (gx - vx);
            vy += drag * dt * (gy - vy);
            vz = flat ? 0.0 : vz + drag * dt * (gz - vz);
        }
        nvx[pi] = vx; nvy[pi] = vy; nvz[pi] = vz;
    }

    std::vector<double> tauNew(np, 0.0), wtNewV(np, 1.0);
    if (stOn) {
        std::mt19937 rng(static_cast<unsigned>(st_seed) * 2654435761u ^
                         static_cast<unsigned>(step_index) * 2246822519u);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        // Keep the original RNG stream in input order. Only the independent
        // velocity attenuation and eight-point normalization run in parallel;
        // per-thread RNGs would change trajectories with the worker count.
        for (size_t pi = 0; pi < np; ++pi) tauNew[pi] = uni(rng);
#pragma omp parallel for schedule(static)
        for (long long pi = 0; pi < lnp; ++pi) {
            const double sp = std::max({std::fabs(nvx[pi]), std::fabs(nvy[pi]),
                                        std::fabs(nvz[pi])});
            const double c = std::clamp(sp * dt, 0.0, 1.0);
            const double xi = c * c * (3.0 - 2.0 * c);
            const double tp = xi * tauNew[pi];
            tauNew[pi] = tp;
            wtNewV[pi] = wtKernel(tp) / wtNorm(xi);
        }
    }

    // working vectors sized np, then merge/split edits them.
    std::vector<double> Px(np), Py(np), Pz(np), Vx(np), Vy(np), Vz(np);
    std::vector<double> Tau(np), Wt(np), Scl(np), Ph01(np);
    std::vector<char> Esc(np);
#pragma omp parallel for schedule(dynamic, 1024)
    for (long long pi = 0; pi < lnp; ++pi) {
        double px = positions[pi].x, py = positions[pi].y, pz = positions[pi].z;
        double vx = nvx[pi], vy = nvy[pi], vz = nvz[pi];
        const bool liquid = (static_cast<size_t>(pi) < particle_phase.size()
                             ? particle_phase[pi] : 0.0f) >= 0.5f;
        const double tauOld = static_cast<size_t>(pi) < tau_in.size()
            ? static_cast<double>(tau_in[pi]) : 0.0;
        const double dtAct = stOn
            ? std::clamp(dt * (1.0 + tauNew[pi] - tauOld), 0.0, 2.0 * dt) : dt;
        if (escFlag[pi]) {
            px += dtAct * vx; py += dtAct * vy;
            if (!flat) pz += dtAct * vz;
        } else if (stOn) {
            double rem = dtAct;
            for (int r = 0; rem > 1e-12 && r < 64; ++r) {
                const double v1x = sU(u, px, py, pz), v1y = sV(v, px, py, pz);
                const double v1z = flat ? 0.0 : sW(w, px, py, pz);
                const double sp = std::max({std::fabs(v1x), std::fabs(v1y),
                                            std::fabs(v1z), 1e-9});
                const double dts = std::min(rem, 1.0 / sp);
                const double mx = px + 0.5 * dts * v1x, my = py + 0.5 * dts * v1y;
                const double mz = flat ? pz : pz + 0.5 * dts * v1z;
                const double cx = std::max(mx, 0.51), cy = std::max(my, 0.51);
                const double cz = flat ? pz : std::max(mz, 0.51);
                px += dts * sU(u, cx, cy, cz);
                py += dts * sV(v, cx, cy, cz);
                if (!flat) pz += dts * sW(w, cx, cy, cz);
                rem -= dts;
            }
        } else {
            const double mx = px + 0.5 * dt * sU(u, px, py, pz);
            const double my = py + 0.5 * dt * sV(v, px, py, pz);
            const double mz = flat ? pz : pz + 0.5 * dt * sW(w, px, py, pz);
            const double cx = std::max(mx, 0.51), cy = std::max(my, 0.51);
            const double cz = flat ? pz : std::max(mz, 0.51);
            px += dt * sU(u, cx, cy, cz);
            py += dt * sV(v, cx, cy, cz);
            if (!flat) pz += dt * sW(w, cx, cy, cz);
        }
        if (px < lo || px > hx) { px = std::clamp(px, lo, hx); vx = 0.0; }
        if (py < lo || py > hy) { py = std::clamp(py, lo, hy); vy = 0.0; }
        if (!flat && (pz < lo || pz > hz)) { pz = std::clamp(pz, lo, hz); vz = 0.0; }
        for (size_t b = 0; b < nObs; ++b) {
            const auto& m0 = obstacles_min[b];
            const auto& m1 = obstacles_max[b];
            const bool inZ = flat || (pz > m0.z && pz < m1.z);
            if (px > m0.x && px < m1.x && py > m0.y && py < m1.y && inZ) {
                double pen[6] = {px - m0.x, m1.x - px, py - m0.y, m1.y - py,
                                 flat ? 1e30 : pz - m0.z, flat ? 1e30 : m1.z - pz};
                int side = 0;
                for (int q2 = 1; q2 < 6; ++q2) if (pen[q2] < pen[side]) side = q2;
                if (side == 0) { px = m0.x - 0.01; vx = 0.0; }
                else if (side == 1) { px = m1.x + 0.01; vx = 0.0; }
                else if (side == 2) { py = m0.y - 0.01; vy = 0.0; }
                else if (side == 3) { py = m1.y + 0.01; vy = 0.0; }
                else if (side == 4) { pz = m0.z - 0.01; vz = 0.0; }
                else { pz = m1.z + 0.01; vz = 0.0; }
            }
        }
        Px[pi] = px; Py[pi] = py; Pz[pi] = pz;
        Vx[pi] = vx; Vy[pi] = vy; Vz[pi] = vz;
        Tau[pi] = tauNew[pi]; Wt[pi] = wtNewV[pi];
        Scl[pi] = scaleOf(pi); Ph01[pi] = liquid ? 1.0 : 0.0;
        Esc[pi] = escFlag[pi];
    }

    if (adOn && np > 0) {
        // fresh liquid-count field at the new positions, as the reference does.
        std::vector<char> isLiq(Px.size());
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(Px.size()); ++i)
            isLiq[i] = Ph01[i] >= 0.5 ? 1 : 0;
        Vec cntU[2] = {Vec(nu, 0.0), Vec(nu, 0.0)};
        Vec cntV[2] = {Vec(nv, 0.0), Vec(nv, 0.0)};
        Vec cntW[2] = {Vec(nw, 0.0), Vec(nw, 0.0)};
        countSplat(Px, Py, Pz, isLiq, Scl, cntU, cntV, cntW);
        std::vector<double> lf(Px.size());
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(Px.size()); ++i)
            lf[i] = sampleCnt(cntU[1], cntV[1], cntW[1], Px[i], Py[i], Pz[i]) /
                    std::max(static_cast<double>(rho0_face), 1e-9);

        const int groupN = flat ? 4 : 8;
        const double wallM = 2.0;
        // ---- split coarse near interface or walls
        std::vector<size_t> splitIdx;
        for (size_t i = 0; i < Px.size(); ++i) {
            if (Scl[i] < 1.5) continue;
            const bool nearWall = Px[i] < wallM || Px[i] > NX - wallM ||
                                  Py[i] < wallM || Py[i] > NY - wallM ||
                                  (!flat && (Pz[i] < wallM || Pz[i] > NZ - wallM));
            if (lf[i] > 0.05 || nearWall) splitIdx.push_back(i);
        }
        if (!splitIdx.empty()) {
            std::vector<char> drop(Px.size(), 0);
            for (size_t i : splitIdx) drop[i] = 1;
            auto keepFilter = [&](auto& vec) {
                size_t wr = 0;
                for (size_t i = 0; i < drop.size(); ++i)
                    if (!drop[i]) vec[wr++] = vec[i];
                vec.resize(wr);
            };
            std::vector<double> sp_[10];
            // stash split parents before filtering
            std::vector<double> pxs, pys, pzs, vxs, vys, vzs, tas, phs;
            for (size_t i : splitIdx) {
                pxs.push_back(Px[i]); pys.push_back(Py[i]); pzs.push_back(Pz[i]);
                vxs.push_back(Vx[i]); vys.push_back(Vy[i]); vzs.push_back(Vz[i]);
                tas.push_back(Tau[i]); phs.push_back(Ph01[i]);
            }
            keepFilter(Px); keepFilter(Py); keepFilter(Pz);
            keepFilter(Vx); keepFilter(Vy); keepFilter(Vz);
            keepFilter(Tau); keepFilter(Wt); keepFilter(Scl); keepFilter(Ph01);
            {
                size_t wr = 0;
                for (size_t i = 0; i < drop.size(); ++i)
                    if (!drop[i]) Esc[wr++] = Esc[i];
                Esc.resize(wr);
            }
            for (size_t m2 = 0; m2 < pxs.size(); ++m2) {
                for (int ci = 0; ci < groupN; ++ci) {
                    const double ox = (ci & 1) ? 0.5 : -0.5;
                    const double oy = (ci & 2) ? 0.5 : -0.5;
                    const double oz = flat ? 0.0 : ((ci & 4) ? 0.5 : -0.5);
                    Px.push_back(std::clamp(pxs[m2] + ox, lo, hx));
                    Py.push_back(std::clamp(pys[m2] + oy, lo, hy));
                    Pz.push_back(flat ? pzs[m2] : std::clamp(pzs[m2] + oz, lo, hz));
                    Vx.push_back(vxs[m2]); Vy.push_back(vys[m2]); Vz.push_back(vzs[m2]);
                    Tau.push_back(tas[m2]);
                    Wt.push_back(wtKernel(tas[m2]) / wtNorm(1.0));
                    Scl.push_back(1.0); Ph01.push_back(phs[m2]); Esc.push_back(0);
                }
            }
        }
        // ---- merge deep-air fine, groupN-to-1 per 2-cell block
        std::vector<size_t> cand;
        for (size_t i = 0; i < Px.size(); ++i)
            if (Scl[i] < 1.5 && Ph01[i] < 0.5 &&
                (i < lf.size() ? lf[i] : 1.0) < 0.02)
                cand.push_back(i);
        if (static_cast<int>(cand.size()) >= groupN) {
            std::stable_sort(cand.begin(), cand.end(), [&](size_t a2, size_t b2) {
                const long long ka =
                    (static_cast<long long>(Px[a2] / 2) * 100000 +
                     static_cast<long long>(Py[a2] / 2)) * 100000 +
                    (flat ? 0 : static_cast<long long>(Pz[a2] / 2));
                const long long kb =
                    (static_cast<long long>(Px[b2] / 2) * 100000 +
                     static_cast<long long>(Py[b2] / 2)) * 100000 +
                    (flat ? 0 : static_cast<long long>(Pz[b2] / 2));
                return ka < kb;
            });
            auto key = [&](size_t i) {
                return (static_cast<long long>(Px[i] / 2) * 100000 +
                        static_cast<long long>(Py[i] / 2)) * 100000 +
                       (flat ? 0 : static_cast<long long>(Pz[i] / 2));
            };
            std::vector<char> drop(Px.size(), 0);
            std::vector<double> mpx, mpy, mpz, mvx, mvy, mvz, mta;
            size_t i = 0;
            while (i < cand.size()) {
                size_t j = i;
                while (j < cand.size() && key(cand[j]) == key(cand[i])) ++j;
                const size_t take = ((j - i) / groupN) * groupN;
                for (size_t g2 = 0; g2 < take; g2 += groupN) {
                    double ax = 0, ay = 0, az = 0, bx = 0, by = 0, bz = 0, tt = 0;
                    for (int m2 = 0; m2 < groupN; ++m2) {
                        const size_t id = cand[i + g2 + m2];
                        drop[id] = 1;
                        ax += Px[id]; ay += Py[id]; az += Pz[id];
                        bx += Vx[id]; by += Vy[id]; bz += Vz[id];
                        tt += Tau[id];
                    }
                    mpx.push_back(ax / groupN); mpy.push_back(ay / groupN);
                    mpz.push_back(az / groupN);
                    mvx.push_back(bx / groupN); mvy.push_back(by / groupN);
                    mvz.push_back(bz / groupN);
                    mta.push_back(tt / groupN);
                }
                i = j;
            }
            if (!mpx.empty()) {
                auto keepFilter = [&](auto& vec) {
                    size_t wr = 0;
                    for (size_t k2 = 0; k2 < drop.size(); ++k2)
                        if (!drop[k2]) vec[wr++] = vec[k2];
                    vec.resize(wr);
                };
                keepFilter(Px); keepFilter(Py); keepFilter(Pz);
                keepFilter(Vx); keepFilter(Vy); keepFilter(Vz);
                keepFilter(Tau); keepFilter(Wt); keepFilter(Scl); keepFilter(Ph01);
                {
                    size_t wr = 0;
                    for (size_t k2 = 0; k2 < drop.size(); ++k2)
                        if (!drop[k2]) Esc[wr++] = Esc[k2];
                    Esc.resize(wr);
                }
                for (size_t m2 = 0; m2 < mpx.size(); ++m2) {
                    Px.push_back(mpx[m2]); Py.push_back(mpy[m2]); Pz.push_back(mpz[m2]);
                    Vx.push_back(mvx[m2]); Vy.push_back(mvy[m2]); Vz.push_back(mvz[m2]);
                    Tau.push_back(mta[m2]);
                    Wt.push_back(wtKernel(mta[m2]) / wtNorm(1.0));
                    Scl.push_back(2.0); Ph01.push_back(0.0); Esc.push_back(0);
                }
            }
        }
    }

    // ---- final obstacle sweep: merge centroids and split children may land
    // inside a box; push every particle out along least penetration.
    if (nObs > 0) {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(Px.size()); ++i) {
            for (size_t b = 0; b < nObs; ++b) {
                const auto& m0 = obstacles_min[b];
                const auto& m1 = obstacles_max[b];
                const bool inZ = flat || (Pz[i] > m0.z && Pz[i] < m1.z);
                if (Px[i] > m0.x && Px[i] < m1.x && Py[i] > m0.y && Py[i] < m1.y && inZ) {
                    double pen[6] = {Px[i] - m0.x, m1.x - Px[i], Py[i] - m0.y,
                                     m1.y - Py[i], flat ? 1e30 : Pz[i] - m0.z,
                                     flat ? 1e30 : m1.z - Pz[i]};
                    int side = 0;
                    for (int q2 = 1; q2 < 6; ++q2) if (pen[q2] < pen[side]) side = q2;
                    if (side == 0) { Px[i] = m0.x - 0.01; Vx[i] = 0.0; }
                    else if (side == 1) { Px[i] = m1.x + 0.01; Vx[i] = 0.0; }
                    else if (side == 2) { Py[i] = m0.y - 0.01; Vy[i] = 0.0; }
                    else if (side == 3) { Py[i] = m1.y + 0.01; Vy[i] = 0.0; }
                    else if (side == 4) { Pz[i] = m0.z - 0.01; Vz[i] = 0.0; }
                    else { Pz[i] = m1.z + 0.01; Vz[i] = 0.0; }
                }
            }
        }
    }

    // ---- pack outputs at the final particle count.
    const size_t nf = Px.size();
    auto outPos = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(nf);
    auto outVel = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(nf);
    auto outSync = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(nf);
    auto outM = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outMx = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outMy = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outMz = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outTau = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outScl = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outPh = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outWt = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outWtPh = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outEsc = Amino::newMutablePtr<Amino::Array<float>>(nf);
    auto outMF = Amino::newMutablePtr<Amino::Array<bool>>(nf);
    auto outMC = Amino::newMutablePtr<Amino::Array<bool>>(nf);
    double maxSpeed = 0.0;
    for (size_t i = 0; i < nf; ++i) {
        (*outPos)[i] = {static_cast<float>(Px[i]), static_cast<float>(Py[i]),
                        static_cast<float>(Pz[i])};
        (*outVel)[i] = {static_cast<float>(Vx[i]), static_cast<float>(Vy[i]),
                        static_cast<float>(Vz[i])};
        const double sdt = Tau[i] * dt;
        // The time-resync can carry a display position back inside a box the
        // sweep above just pushed the actual position out of; clamp the
        // display position too (position only - no state or velocity change).
        double sx = Px[i] - sdt * Vx[i];
        double sy = Py[i] - sdt * Vy[i];
        double sz = flat ? Pz[i] : Pz[i] - sdt * Vz[i];
        for (size_t b = 0; b < nObs; ++b) {
            const auto& m0 = obstacles_min[b];
            const auto& m1 = obstacles_max[b];
            const bool inZ = flat || (sz > m0.z && sz < m1.z);
            if (sx > m0.x && sx < m1.x && sy > m0.y && sy < m1.y && inZ) {
                double pen[6] = {sx - m0.x, m1.x - sx, sy - m0.y, m1.y - sy,
                                 flat ? 1e30 : sz - m0.z, flat ? 1e30 : m1.z - sz};
                int side = 0;
                for (int q2 = 1; q2 < 6; ++q2) if (pen[q2] < pen[side]) side = q2;
                if (side == 0) sx = m0.x - 0.01;
                else if (side == 1) sx = m1.x + 0.01;
                else if (side == 2) sy = m0.y - 0.01;
                else if (side == 3) sy = m1.y + 0.01;
                else if (side == 4) sz = m0.z - 0.01;
                else sz = m1.z + 0.01;
            }
        }
        (*outSync)[i] = {static_cast<float>(sx), static_cast<float>(sy),
                         static_cast<float>(sz)};
        const double rhoP = Ph01[i] >= 0.5 ? rl : rg;
        const double mEff = rhoP * std::pow(Scl[i], dExp) *
                            (Scl[i] > 1.5 ? coarse_gain : 1.0);
        const double wp = Wt[i];
        (*outM)[i] = static_cast<float>(wp * mEff);
        (*outMx)[i] = static_cast<float>(wp * mEff * Vx[i]);
        (*outMy)[i] = static_cast<float>(wp * mEff * Vy[i]);
        (*outMz)[i] = static_cast<float>(wp * mEff * Vz[i]);
        (*outTau)[i] = static_cast<float>(Tau[i]);
        (*outScl)[i] = static_cast<float>(Scl[i]);
        (*outPh)[i] = static_cast<float>(Ph01[i]);
        (*outWt)[i] = static_cast<float>(wp);
        (*outWtPh)[i] = static_cast<float>(wp * Ph01[i]);
        (*outEsc)[i] = Esc[i] ? 1.0f : 0.0f;
        (*outMF)[i] = Scl[i] < 1.5;
        (*outMC)[i] = Scl[i] >= 1.5;
        maxSpeed = std::max({maxSpeed, std::fabs(Vx[i]), std::fabs(Vy[i]),
                             std::fabs(Vz[i])});
    }

    auto outU = Amino::newMutablePtr<Amino::Array<float>>(nu);
    auto outV = Amino::newMutablePtr<Amino::Array<float>>(nv);
    auto outW = Amino::newMutablePtr<Amino::Array<float>>(nw);
    auto outP = Amino::newMutablePtr<Amino::Array<float>>(ncell);
    for (size_t c = 0; c < nu; ++c) (*outU)[c] = static_cast<float>(u[c]);
    for (size_t c = 0; c < nv; ++c) (*outV)[c] = static_cast<float>(v[c]);
    for (size_t c = 0; c < nw; ++c) (*outW)[c] = static_cast<float>(w[c]);
    for (size_t c = 0; c < ncell; ++c) (*outP)[c] = static_cast<float>(p[c]);

    out_positions = outPos.toImmutable();
    out_velocities = outVel.toImmutable();
    out_mass = outM.toImmutable();
    out_mom_x = outMx.toImmutable();
    out_mom_y = outMy.toImmutable();
    out_mom_z = outMz.toImmutable();
    out_u = outU.toImmutable();
    out_v = outV.toImmutable();
    out_w = outW.toImmutable();
    pressure = outP.toImmutable();
    out_escaped = outEsc.toImmutable();
    tau_out = outTau.toImmutable();
    scale_out = outScl.toImmutable();
    mask_fine = outMF.toImmutable();
    mask_coarse = outMC.toImmutable();
    out_phase_state = outPh.toImmutable();
    out_wt = outWt.toImmutable();
    out_wtph = outWtPh.toImmutable();
    out_pos_synced = outSync.toImmutable();
    iterations_used = it;
    final_residual = static_cast<float>(std::sqrt(rr2) / r0);
    max_divergence_after = static_cast<float>(maxDiv);
    max_speed = static_cast<float>(maxSpeed);
    lap(msPart);
    if (prof) {
        std::fprintf(stderr,
                     "PFPROF step=%d grid=%dx%dx%d np=%zu iters=%d asm=%.1f "
                     "solve=%.1f corr=%.1f part=%.1f total=%.1f\n",
                     step_index, NX, NY, NZ, np, it, msAsm, msSolve, msCorr,
                     msPart,
                     std::chrono::duration<double, std::milli>(Clock::now() -
                                                               tBegin).count());
        std::fflush(stderr);
    }
}

} // namespace Solve
} // namespace PFFlip
