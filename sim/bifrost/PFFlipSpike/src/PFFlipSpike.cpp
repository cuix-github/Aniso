#include "PFFlipSpike.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// y = A x for the 1D variable-coefficient Laplacian with solid walls:
// (A x)_i = beta_{i-1/2} (x_i - x_{i-1}) + beta_{i+1/2} (x_i - x_{i+1}),
// wall faces carry beta = 0 (no flow through walls).
void applyA(const std::vector<double>& beta, const std::vector<double>& x, std::vector<double>& y) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double v = 0.0;
        if (i > 0) v += beta[i] * (x[i] - x[i - 1]);
        if (i + 1 < n) v += beta[i + 1] * (x[i] - x[i + 1]);
        y[i] = v;
    }
}

} // namespace

namespace PFFlip {
namespace Spike {

void hydrostatic_projection_test(int cells, float density_ratio, int max_iterations, float tolerance,
                                 float& max_velocity_after, float& pressure_jump, float& final_residual,
                                 int& iterations_used, Amino::Ptr<Amino::Array<float>>& pressure) {
    const int n = std::max(4, cells);
    const double dt = 1.0, dx = 1.0, g = -9.8;
    const double rhoHeavy = std::max(1.0, static_cast<double>(density_ratio)), rhoLight = 1.0;

    // Cell densities: heavy phase at the bottom half (low indices), light on top.
    std::vector<double> rho(n);
    for (int i = 0; i < n; ++i) rho[i] = i < n / 2 ? rhoHeavy : rhoLight;

    // Face coefficients beta = 1/rho_face; n+1 faces, walls at 0 and n get beta 0.
    std::vector<double> beta(n + 1, 0.0);
    for (int f = 1; f < n; ++f) beta[f] = 1.0 / (0.5 * (rho[f - 1] + rho[f]));

    // After the gravity kick, every interior face velocity is g*dt; walls stay 0.
    std::vector<double> uStar(n + 1, 0.0);
    for (int f = 1; f < n; ++f) uStar[f] = g * dt;

    // Right-hand side b_i = -div(u*)_i * dx / dt  (sign chosen so A is positive semi-definite).
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = -(uStar[i + 1] - uStar[i]) * dx / dt;

    // Unpreconditioned conjugate gradient on A p = b.
    std::vector<double> p(n, 0.0), r(b), d(b), q(n);
    double rr = 0.0;
    for (int i = 0; i < n; ++i) rr += r[i] * r[i];
    const double rr0 = std::max(rr, 1e-300);
    int it = 0;
    for (; it < max_iterations && std::sqrt(rr / rr0) > tolerance; ++it) {
        applyA(beta, d, q);
        double dq = 0.0;
        for (int i = 0; i < n; ++i) dq += d[i] * q[i];
        if (dq == 0.0) break;
        const double alpha = rr / dq;
        double rrNew = 0.0;
        for (int i = 0; i < n; ++i) {
            p[i] += alpha * d[i];
            r[i] -= alpha * q[i];
            rrNew += r[i] * r[i];
        }
        const double betaCG = rrNew / rr;
        for (int i = 0; i < n; ++i) d[i] = r[i] + betaCG * d[i];
        rr = rrNew;
    }

    // Velocity correction: u = u* - dt * beta_face * (p_i - p_{i-1}) / dx.
    double maxU = 0.0;
    for (int f = 1; f < n; ++f) {
        const double u = uStar[f] - dt * beta[f] * (p[f] - p[f - 1]) / dx;
        maxU = std::max(maxU, std::fabs(u));
    }

    max_velocity_after = static_cast<float>(maxU);
    pressure_jump      = static_cast<float>(p[0] - p[n - 1]);
    final_residual     = static_cast<float>(std::sqrt(rr / rr0));
    iterations_used    = it;

    auto out = Amino::newMutablePtr<Amino::Array<float>>(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) (*out)[static_cast<size_t>(i)] = static_cast<float>(p[i]);
    pressure = out.toImmutable();
}

} // namespace Spike
} // namespace PFFlip
