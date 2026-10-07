#include "PFFlipSolve.h"

#include <algorithm>
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
        for (size_t c = 0; c < x.size(); ++c)
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
             const Amino::Array<float>& tau_in,
             const Amino::Array<float>& u_wt, const Amino::Array<float>& v_wt,
             const Amino::Array<float>& w_wt,
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
             Amino::Ptr<Amino::Array<float>>& out_wt,
             Amino::Ptr<Amino::Array<float>>& out_wtph,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_pos_synced,
             int& iterations_used, float& final_residual, float& max_divergence_after,
             float& max_speed) {
    const int NX = std::max(2, nx), NY = std::max(2, ny), NZ = std::max(1, nz);
    const double dt = dt_in;
    const size_t np = positions.size();

    Level L;
    L.nx = NX; L.ny = NY; L.nz = NZ; L.inv_h2 = 1.0;
    const size_t nu = static_cast<size_t>(NX + 1) * NY * NZ;
    const size_t nv = static_cast<size_t>(NX) * (NY + 1) * NZ;
    const size_t nw = static_cast<size_t>(NX) * NY * (NZ + 1);

    // ---- channels -> face velocities (u = mom/mass where mass > 0).
    Vec u(nu, 0.0), v(nv, 0.0), w(nw, 0.0);
    auto normalize = [](Vec& g, const Amino::Array<float>& mass, const Amino::Array<float>& mom) {
        const size_t n = std::min({g.size(), mass.size(), mom.size()});
        for (size_t c = 0; c < n; ++c)
            g[c] = mass[c] > 0.0f
                 ? static_cast<double>(mom[c]) / std::max(static_cast<double>(mass[c]), 1e-30)
                 : 0.0;
    };
    normalize(u, u_mass, u_mom);
    normalize(v, v_mass, v_mom);
    normalize(w, w_mass, w_mom);

    auto walls = [&](Vec& uu, Vec& vv, Vec& ww) {
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) { uu[L.iu(0, j, k)] = 0.0; uu[L.iu(NX, j, k)] = 0.0; }
        for (int i = 0; i < NX; ++i)
            for (int k = 0; k < NZ; ++k) { vv[L.iv(i, 0, k)] = 0.0; vv[L.iv(i, NY, k)] = 0.0; }
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NY; ++j) { ww[L.iw(i, j, 0)] = 0.0; ww[L.iw(i, j, NZ)] = 0.0; }
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
    auto beta = [&](Vec& bb, const Amino::Array<float>& ph, const Amino::Array<float>& wt) {
        for (size_t c = 0; c < bb.size(); ++c) {
            double p = c < ph.size() ? static_cast<double>(ph[c]) : 0.0;
            if (wt.size() > 0) {
                const double wc = c < wt.size() ? static_cast<double>(wt[c]) : 0.0;
                p = wc > 1e-12 ? p / wc : 0.0;
            }
            bb[c] = 1.0 / (rg + (rl - rg) * std::clamp(p, 0.0, 1.0));
        }
    };
    beta(L.bu, u_phase, u_wt); beta(L.bv, v_phase, v_wt); beta(L.bw, w_phase, w_wt);
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
        for (size_t c = 0; c < ncell; ++c) { p[c] += a * d[c]; r[c] -= a * q[c]; }
        precond(r, z);
        double rzNew = 0.0;   // flexible (Polak-Ribiere) update tolerates the V-cycle
        for (size_t c = 0; c < ncell; ++c) rzNew += (r[c] - rPrev[c]) * z[c];
        const double betaCG = std::max(0.0, rzNew / rz);
        for (size_t c = 0; c < ncell; ++c) d[c] = z[c] + betaCG * d[c];
        rz = 0.0;
        for (size_t c = 0; c < ncell; ++c) rz += r[c] * z[c];
        if (rz == 0.0) break;
    }

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

    // ---- g2p (per-phase FLIP blend), escape forces, then advection. With ST on,
    // each particle advances by dt*(1 + tau_new - tau_old) clamped to [0, 2dt],
    // sub-stepped at local CFL 1; tau draws are serial for determinism and exported,
    // so lockstep comparisons replay the exact same jitter in the reference.
    Vec du(nu), dv(nv), dw(nw);
    for (size_t c = 0; c < nu; ++c) du[c] = u[c] - uStar[c];
    for (size_t c = 0; c < nv; ++c) dv[c] = v[c] - vStar[c];
    for (size_t c = 0; c < nw; ++c) dw[c] = w[c] - wStar[c];

    auto sU = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX + 1, NY, NZ, 0.0, 0.5, 0.5, x, y, zc); };
    auto sV = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX, NY + 1, NZ, 0.5, 0.0, 0.5, x, y, zc); };
    auto sW = [&](const Vec& g, double x, double y, double zc) {
        return sampleGrid(g, NX, NY, NZ + 1, 0.5, 0.5, 0.0, x, y, zc); };

    const bool flat = NZ == 1;
    const bool stOn = st != 0;
    const double lo = 0.51, hx = NX - 0.51, hy = NY - 0.51, hz = NZ - 0.51;

    // ---- escaped-particle detection (unchanged from the escape PR): per-phase
    // count splats with the Eq. 6 kernel, other-phase density vs calibrated rho0.
    std::vector<char> escFlag(np, 0);
    if (escape != 0 && np > 0) {
        Vec cntU[2] = {Vec(nu, 0.0), Vec(nu, 0.0)};
        Vec cntV[2] = {Vec(nv, 0.0), Vec(nv, 0.0)};
        auto splatCounts = [&](double offx, double offy, double offz,
                               int gx, int gy, int gz, Vec* cnt) {
            for (size_t pi = 0; pi < np; ++pi) {
                const int ph = (pi < particle_phase.size() ? particle_phase[pi] : 0.0f) >= 0.5f;
                const double qx = positions[pi].x - offx;
                const double qy = positions[pi].y - offy;
                const double qz = positions[pi].z - offz;
                const int bx = static_cast<int>(std::floor(qx + 0.5));
                const int by = static_cast<int>(std::floor(qy + 0.5));
                const int bz = static_cast<int>(std::floor(qz + 0.5));
                const double fx = qx - bx, fy = qy - by, fz = qz - bz;
                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int dk = -1; dk <= 1; ++dk) {
                            const int i = bx + di, j = by + dj, k = bz + dk;
                            if (i < 0 || i >= gx || j < 0 || j >= gy || k < 0 || k >= gz)
                                continue;
                            const double d2 = (fx - di) * (fx - di) + (fy - dj) * (fy - dj) +
                                              (fz - dk) * (fz - dk);
                            const double wgt = std::max(1.0 - d2, 0.0);
                            if (wgt > 0.0)
                                cnt[ph][(static_cast<size_t>(i) * gy + j) * gz + k] +=
                                    wgt * wgt * wgt;
                        }
            }
        };
        splatCounts(0.0, 0.5, 0.5, NX + 1, NY, NZ, cntU);
        splatCounts(0.5, 0.0, 0.5, NX, NY + 1, NZ, cntV);
        const double thresh = (1.0 - esc_phi) * rho0_face;
        for (size_t pi = 0; pi < np; ++pi) {
            const bool liquid = (pi < particle_phase.size() ? particle_phase[pi] : 0.0f) >= 0.5f;
            const int other = liquid ? 0 : 1;
            const double px = positions[pi].x, py = positions[pi].y, pz = positions[pi].z;
            const double frac =
                0.5 * (sampleGrid(cntU[other], NX + 1, NY, NZ, 0.0, 0.5, 0.5, px, py, pz) +
                       sampleGrid(cntV[other], NX, NY + 1, NZ, 0.5, 0.0, 0.5, px, py, pz));
            escFlag[pi] = frac > thresh ? 1 : 0;
        }
    }

    std::vector<double> nvx(np), nvy(np), nvz(np);
    for (size_t pi = 0; pi < np; ++pi) {
        const double px = positions[pi].x, py = positions[pi].y, pz = positions[pi].z;
        double vx = pi < velocities.size() ? velocities[pi].x : 0.0;
        double vy = pi < velocities.size() ? velocities[pi].y : 0.0;
        double vz = pi < velocities.size() ? velocities[pi].z : 0.0;
        const bool liquid = (pi < particle_phase.size() ? particle_phase[pi] : 0.0f) >= 0.5f;
        const double a = liquid ? alpha_liquid : alpha_air;
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

    std::vector<double> tauNew(np, 0.0), wtNew(np, 1.0);
    if (stOn) {
        std::mt19937 rng(static_cast<unsigned>(st_seed) * 2654435761u ^
                         static_cast<unsigned>(step_index) * 2246822519u);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        for (size_t pi = 0; pi < np; ++pi) {
            const double sp = std::max({std::fabs(nvx[pi]), std::fabs(nvy[pi]),
                                        std::fabs(nvz[pi])});
            const double c = std::clamp(sp * dt, 0.0, 1.0);
            const double xi = c * c * (3.0 - 2.0 * c);
            const double tp = xi * uni(rng);
            tauNew[pi] = tp;
            wtNew[pi] = wtKernel(tp) / wtNorm(xi);
        }
    }

    auto outPos = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(np);
    auto outVel = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(np);
    auto outSync = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(np);
    auto outM = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outMx = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outMy = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outMz = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outTau = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outWt = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outWtPh = Amino::newMutablePtr<Amino::Array<float>>(np);

    double maxSpeed = 0.0;
    for (size_t pi = 0; pi < np; ++pi) {
        double px = positions[pi].x, py = positions[pi].y, pz = positions[pi].z;
        double vx = nvx[pi], vy = nvy[pi], vz = nvz[pi];
        const bool liquid = (pi < particle_phase.size() ? particle_phase[pi] : 0.0f) >= 0.5f;
        const double tauOld = pi < tau_in.size() ? static_cast<double>(tau_in[pi]) : 0.0;
        const double dtAct = stOn
            ? std::clamp(dt * (1.0 + tauNew[pi] - tauOld), 0.0, 2.0 * dt) : dt;

        if (escFlag[pi]) {
            px += dtAct * vx;
            py += dtAct * vy;
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

        (*outPos)[pi] = {static_cast<float>(px), static_cast<float>(py),
                         static_cast<float>(pz)};
        (*outVel)[pi] = {static_cast<float>(vx), static_cast<float>(vy),
                         static_cast<float>(vz)};
        const double sdt = tauNew[pi] * dt;
        (*outSync)[pi] = {static_cast<float>(px - sdt * vx),
                          static_cast<float>(py - sdt * vy),
                          static_cast<float>(flat ? pz : pz - sdt * vz)};
        const float m = liquid ? rho_liquid : rho_air;
        const double wp = wtNew[pi];
        (*outM)[pi] = static_cast<float>(wp * m);
        (*outMx)[pi] = static_cast<float>(wp * m * vx);
        (*outMy)[pi] = static_cast<float>(wp * m * vy);
        (*outMz)[pi] = static_cast<float>(wp * m * vz);
        (*outTau)[pi] = static_cast<float>(tauNew[pi]);
        (*outWt)[pi] = static_cast<float>(wp);
        (*outWtPh)[pi] = static_cast<float>(wp * (liquid ? 1.0 : 0.0));
        maxSpeed = std::max({maxSpeed, std::fabs(vx), std::fabs(vy), std::fabs(vz)});
    }

    auto outEsc = Amino::newMutablePtr<Amino::Array<float>>(np);
    for (size_t pi = 0; pi < np; ++pi) (*outEsc)[pi] = escFlag[pi] ? 1.0f : 0.0f;

    auto outU = Amino::newMutablePtr<Amino::Array<float>>(nu);
    auto outV = Amino::newMutablePtr<Amino::Array<float>>(nv);
    auto outW = Amino::newMutablePtr<Amino::Array<float>>(nw);
    auto outP = Amino::newMutablePtr<Amino::Array<float>>(ncell);
    for (size_t c = 0; c < nu; ++c) (*outU)[c] = static_cast<float>(u[c]);
    for (size_t c = 0; c < nv; ++c) (*outV)[c] = static_cast<float>(v[c]);
    for (size_t c = 0; c < nw; ++c) (*outW)[c] = static_cast<float>(w[c]);
    for (size_t c = 0; c < ncell; ++c) (*outP)[c] = static_cast<float>(p[c]);

    out_positions = outPos.toImmutable();
    tau_out = outTau.toImmutable();
    out_wt = outWt.toImmutable();
    out_wtph = outWtPh.toImmutable();
    out_pos_synced = outSync.toImmutable();
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
    iterations_used = it;
    final_residual = static_cast<float>(std::sqrt(rr2) / r0);
    max_divergence_after = static_cast<float>(maxDiv);
    max_speed = static_cast<float>(maxSpeed);
}

} // namespace Solve
} // namespace PFFlip
