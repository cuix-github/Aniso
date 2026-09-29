// Checks spherical-harmonic colour against values worked out by hand.

#include "aniso/sh.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)
bool near(float a, float b) { return std::fabs(a - b) <= 1e-5f; }

// One Gaussian at degree 3, all coefficients zero except the ones a test sets.
aniso::Scene scene() {
    aniso::Scene s;
    s.shDegree = 3;
    s.gaussians.resize(1);
    s.sh.assign(16 * 3, 0.0f);
    return s;
}

void testDcOnly() {
    aniso::Scene s = scene();
    s.sh[0] = 0.4f; // red DC
    // With only the DC term, every direction gives 0.5 + C0 * dc.
    for (aniso::Vec3 d : {aniso::Vec3{1, 0, 0}, aniso::Vec3{0, 0, -1}}) {
        const auto c = aniso::shColor(s, 0, d, 3);
        CHECK(near(c[0], 0.5f + aniso::kShC0 * 0.4f));
        CHECK(near(c[1], 0.5f));
    }
}

void testDegreeOneFollowsDirection() {
    aniso::Scene s = scene();
    s.sh[1 * 3 + 1] = 0.5f; // green, coefficient 1, whose basis is -C1 * y
    const float C1 = 0.4886025119029199f;
    const auto up = aniso::shColor(s, 0, {0, -1, 0}, 3);
    const auto down = aniso::shColor(s, 0, {0, 1, 0}, 3);
    const auto side = aniso::shColor(s, 0, {1, 0, 0}, 3);
    CHECK(near(up[1], 0.5f + C1 * 0.5f));   // brighter looking one way
    CHECK(near(down[1], 0.5f - C1 * 0.5f)); // darker looking the other
    CHECK(near(side[1], 0.5f));             // unchanged side on
    // Limiting the degree to 0 ignores it.
    CHECK(near(aniso::shColor(s, 0, {0, -1, 0}, 0)[1], 0.5f));
}

void testClampAtZero() {
    aniso::Scene s = scene();
    s.sh[2] = -10.0f; // blue DC strongly negative
    CHECK(aniso::shColor(s, 0, {0, 0, 1}, 3)[2] == 0.0f);
}

} // namespace

int main() {
    testDcOnly();
    testDegreeOneFollowsDirection();
    testClampAtZero();
    if (failures == 0) std::printf("all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
