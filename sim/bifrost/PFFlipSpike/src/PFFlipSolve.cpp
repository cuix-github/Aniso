#include "PFFlipSolve.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Mirrors sim/reference2d/pfflip2d.py exactly; see the header. All grid math is
// double precision, as the numpy reference is, so lockstep comparisons on
// identical inputs agree to float rounding.

namespace {

using Vec = std::vector<double>;

struct Grid2 {
    int nx, ny;                       // logical dims of this array
    Vec a;
    Grid2(int nx_, int ny_) : nx(nx_), ny(ny_), a(static_cast<size_t>(nx_) * ny_, 0.0) {}
    double& at(int i, int j) { return a[static_cast<size_t>(i) * ny + j]; }
    double  at(int i, int j) const { return a[static_cast<size_t>(i) * ny + j]; }
};

// python _sample_faces: bilinear with per-axis index clipping.
double sampleFaces(const Grid2& g, double offx, double offy, double px, double py) {
    const double qx = px - offx, qy = py - offy;
    const double bx = std::floor(qx), by = std::floor(qy);
    const double fx = qx - bx, fy = qy - by;
    double out = 0.0;
    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            const double w = (di ? fx : 1.0 - fx) * (dj ? fy : 1.0 - fy);
            const int ii = std::clamp(static_cast<int>(bx) + di, 0, g.nx - 1);
            const int jj = std::clamp(static_cast<int>(by) + dj, 0, g.ny - 1);
            out += w * g.at(ii, jj);
        }
    }
    return out;
}

} // namespace

namespace PFFlip {
namespace Solve {

void step_2d(int nx, int ny, float dt_in, float gravity, float rho_liquid, float rho_air,
             float alpha_liquid, float alpha_air, int max_iterations, float tolerance,
             const Amino::Array<float>& u_mass, const Amino::Array<float>& u_mom,
             const Amino::Array<float>& u_phase, const Amino::Array<float>& v_mass,
             const Amino::Array<float>& v_mom, const Amino::Array<float>& v_phase,
             const Amino::Array<Bifrost::Math::float3>& positions,
             const Amino::Array<Bifrost::Math::float3>& velocities,
             const Amino::Array<float>& particle_phase,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_positions,
             Amino::Ptr<Amino::Array<Bifrost::Math::float3>>& out_velocities,
             Amino::Ptr<Amino::Array<float>>& out_mass,
             Amino::Ptr<Amino::Array<float>>& out_mom_x,
             Amino::Ptr<Amino::Array<float>>& out_mom_y,
             Amino::Ptr<Amino::Array<float>>& out_u,
             Amino::Ptr<Amino::Array<float>>& out_v,
             Amino::Ptr<Amino::Array<float>>& pressure,
             int& iterations_used, float& final_residual, float& max_divergence_after,
             float& max_speed) {
    const int NX = std::max(2, nx), NY = std::max(2, ny);
    const double dt = dt_in;
    const size_t nu = static_cast<size_t>(NX + 1) * NY;
    const size_t nv = static_cast<size_t>(NX) * (NY + 1);
    const size_t np = positions.size();

    Grid2 u(NX + 1, NY), v(NX, NY + 1);

    // ---- normalize channels to face velocities (p2g tail): u = mom/mass where mass>0.
    for (size_t k = 0; k < nu && k < u_mass.size() && k < u_mom.size(); ++k)
        u.a[k] = u_mass[k] > 0.0f ? static_cast<double>(u_mom[k]) / std::max(static_cast<double>(u_mass[k]), 1e-30) : 0.0;
    for (size_t k = 0; k < nv && k < v_mass.size() && k < v_mom.size(); ++k)
        v.a[k] = v_mass[k] > 0.0f ? static_cast<double>(v_mom[k]) / std::max(static_cast<double>(v_mass[k]), 1e-30) : 0.0;

    auto enforceWalls = [&]() {
        for (int j = 0; j < NY; ++j) { u.at(0, j) = 0.0; u.at(NX, j) = 0.0; }
        for (int i = 0; i < NX; ++i) { v.at(i, 0) = 0.0; v.at(i, NY) = 0.0; }
    };
    enforceWalls();

    // u_star: the grid state particles compare against in the FLIP delta.
    Grid2 uStar = u, vStar = v;

    // ---- gravity on y faces, then walls (python: v += g*dt; _enforce_walls()).
    for (size_t k = 0; k < nv; ++k) v.a[k] += gravity * dt;
    enforceWalls();

    // ---- face coefficients beta = 1/rho(phi), walls beta = 0.
    Grid2 bu(NX + 1, NY), bv(NX, NY + 1);
    const double rl = rho_liquid, rg = rho_air;
    for (size_t k = 0; k < nu; ++k) {
        const double ph = std::clamp(k < u_phase.size() ? static_cast<double>(u_phase[k]) : 0.0, 0.0, 1.0);
        bu.a[k] = 1.0 / (rg + (rl - rg) * ph);
    }
    for (size_t k = 0; k < nv; ++k) {
        const double ph = std::clamp(k < v_phase.size() ? static_cast<double>(v_phase[k]) : 0.0, 0.0, 1.0);
        bv.a[k] = 1.0 / (rg + (rl - rg) * ph);
    }
    for (int j = 0; j < NY; ++j) { bu.at(0, j) = 0.0; bu.at(NX, j) = 0.0; }
    for (int i = 0; i < NX; ++i) { bv.at(i, 0) = 0.0; bv.at(i, NY) = 0.0; }

    // ---- right-hand side b = -div(u)/dt.
    Grid2 rhs(NX, NY);
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            rhs.at(i, j) = -((u.at(i + 1, j) - u.at(i, j)) + (v.at(i, j + 1) - v.at(i, j))) / dt;

    // ---- A(p): python's operator, verbatim.
    auto applyA = [&](const Grid2& p, Grid2& out) {
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                double gpuR = (i + 1 <= NX - 1) ? (p.at(i + 1, j) - p.at(i, j)) * bu.at(i + 1, j) : 0.0;
                double gpuL = (i >= 1) ? (p.at(i, j) - p.at(i - 1, j)) * bu.at(i, j) : 0.0;
                double gpvT = (j + 1 <= NY - 1) ? (p.at(i, j + 1) - p.at(i, j)) * bv.at(i, j + 1) : 0.0;
                double gpvB = (j >= 1) ? (p.at(i, j) - p.at(i, j - 1)) * bv.at(i, j) : 0.0;
                out.at(i, j) = -((gpuR - gpuL) + (gpvT - gpvB));
            }
        }
    };

    // ---- Jacobi-preconditioned conjugate gradient, matching the reference.
    Grid2 diagInv(NX, NY);
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            const double dg = bu.at(i + 1, j) + bu.at(i, j) + bv.at(i, j + 1) + bv.at(i, j);
            diagInv.at(i, j) = dg > 0.0 ? 1.0 / std::max(dg, 1e-300) : 0.0;
        }

    Grid2 p(NX, NY), r = rhs, z(NX, NY), d(NX, NY), q(NX, NY);
    const size_t ncell = p.a.size();
    for (size_t k = 0; k < ncell; ++k) z.a[k] = diagInv.a[k] * r.a[k];
    d = z;
    double rz = 0.0, rr = 0.0;
    for (size_t k = 0; k < ncell; ++k) { rz += r.a[k] * z.a[k]; rr += r.a[k] * r.a[k]; }
    double r0 = std::sqrt(rr);
    if (r0 == 0.0) r0 = 1.0;
    int it = 0;
    for (it = 0; it < max_iterations; ++it) {
        rr = 0.0;
        for (size_t k = 0; k < ncell; ++k) rr += r.a[k] * r.a[k];
        if (std::sqrt(rr) / r0 < tolerance) break;
        applyA(d, q);
        double dq = 0.0;
        for (size_t k = 0; k < ncell; ++k) dq += d.a[k] * q.a[k];
        if (dq == 0.0) break;
        const double a = rz / dq;
        for (size_t k = 0; k < ncell; ++k) { p.a[k] += a * d.a[k]; r.a[k] -= a * q.a[k]; }
        double rzNew = 0.0;
        for (size_t k = 0; k < ncell; ++k) { z.a[k] = diagInv.a[k] * r.a[k]; rzNew += r.a[k] * z.a[k]; }
        const double betaCG = rzNew / rz;
        for (size_t k = 0; k < ncell; ++k) d.a[k] = z.a[k] + betaCG * d.a[k];
        rz = rzNew;
    }

    // ---- velocity correction on interior faces, then walls.
    for (int i = 1; i <= NX - 1; ++i)
        for (int j = 0; j < NY; ++j)
            u.at(i, j) -= dt * bu.at(i, j) * (p.at(i, j) - p.at(i - 1, j));
    for (int i = 0; i < NX; ++i)
        for (int j = 1; j <= NY - 1; ++j)
            v.at(i, j) -= dt * bv.at(i, j) * (p.at(i, j) - p.at(i, j - 1));
    enforceWalls();

    // ---- diagnostics.
    double maxDiv = 0.0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            maxDiv = std::max(maxDiv, std::fabs((u.at(i + 1, j) - u.at(i, j)) + (v.at(i, j + 1) - v.at(i, j))));

    // ---- g2p: per-phase FLIP blend, then RK2 advection with wall clamps.
    Grid2 du(NX + 1, NY), dv(NX, NY + 1);
    for (size_t k = 0; k < nu; ++k) du.a[k] = u.a[k] - uStar.a[k];
    for (size_t k = 0; k < nv; ++k) dv.a[k] = v.a[k] - vStar.a[k];

    auto outPos = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(np);
    auto outVel = Amino::newMutablePtr<Amino::Array<Bifrost::Math::float3>>(np);
    auto outM   = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outMx  = Amino::newMutablePtr<Amino::Array<float>>(np);
    auto outMy  = Amino::newMutablePtr<Amino::Array<float>>(np);

    auto sampleU = [&](const Grid2& g, double px, double py) { return sampleFaces(g, 0.0, 0.5, px, py); };
    auto sampleV = [&](const Grid2& g, double px, double py) { return sampleFaces(g, 0.5, 0.0, px, py); };

    const double lo = 0.51, hix = NX - 0.51, hiy = NY - 0.51;
    double maxSpeed = 0.0;
    for (size_t k = 0; k < np; ++k) {
        double px = positions[k].x, py = positions[k].y;
        double vx = k < velocities.size() ? velocities[k].x : 0.0;
        double vy = k < velocities.size() ? velocities[k].y : 0.0;
        const bool liquid = (k < particle_phase.size() ? particle_phase[k] : 0.0f) >= 0.5f;
        const double a = liquid ? alpha_liquid : alpha_air;

        // FLIP blend against the pre-gravity grid state.
        const double dvx = sampleU(du, px, py), dvy = sampleV(dv, px, py);
        const double picx = sampleU(u, px, py), picy = sampleV(v, px, py);
        vx = a * (vx + dvx) + (1.0 - a) * picx;
        vy = a * (vy + dvy) + (1.0 - a) * picy;

        // RK2 through the corrected grid field.
        const double m1x = px + 0.5 * dt * sampleU(u, px, py);
        const double m1y = py + 0.5 * dt * sampleV(v, px, py);
        const double cmx = std::max(m1x, 0.51), cmy = std::max(m1y, 0.51);
        px += dt * sampleU(u, cmx, cmy);
        py += dt * sampleV(v, cmx, cmy);

        // Wall clamps; a clamped axis loses its velocity component.
        if (px < lo || px > hix) { px = std::clamp(px, lo, hix); vx = 0.0; }
        if (py < lo || py > hiy) { py = std::clamp(py, lo, hiy); vy = 0.0; }

        (*outPos)[k] = {static_cast<float>(px), static_cast<float>(py), positions[k].z};
        (*outVel)[k] = {static_cast<float>(vx), static_cast<float>(vy), 0.0f};
        const float m = liquid ? rho_liquid : rho_air;
        (*outM)[k]  = m;
        (*outMx)[k] = m * static_cast<float>(vx);
        (*outMy)[k] = m * static_cast<float>(vy);
        maxSpeed = std::max(maxSpeed, std::max(std::fabs(vx), std::fabs(vy)));
    }

    auto outU = Amino::newMutablePtr<Amino::Array<float>>(nu);
    auto outV = Amino::newMutablePtr<Amino::Array<float>>(nv);
    auto outP = Amino::newMutablePtr<Amino::Array<float>>(ncell);
    for (size_t k = 0; k < nu; ++k) (*outU)[k] = static_cast<float>(u.a[k]);
    for (size_t k = 0; k < nv; ++k) (*outV)[k] = static_cast<float>(v.a[k]);
    for (size_t k = 0; k < ncell; ++k) (*outP)[k] = static_cast<float>(p.a[k]);

    out_positions  = outPos.toImmutable();
    out_velocities = outVel.toImmutable();
    out_mass       = outM.toImmutable();
    out_mom_x      = outMx.toImmutable();
    out_mom_y      = outMy.toImmutable();
    out_u          = outU.toImmutable();
    out_v          = outV.toImmutable();
    pressure       = outP.toImmutable();
    iterations_used      = it;
    final_residual       = static_cast<float>(std::sqrt(rr) / r0);
    max_divergence_after = static_cast<float>(maxDiv);
    max_speed            = static_cast<float>(maxSpeed);
}

} // namespace Solve
} // namespace PFFlip
