// Writes small synthetic scenes with known raw values, loads them, and checks every
// conversion the loader performs. No test framework: a failed check prints and exits 1.

#include "aniso/ply_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)

bool near(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// Writes a reference-layout PLY with the given SH degree. Gaussian i gets raw values
// derived from i so every field is distinguishable.
std::filesystem::path writeScene(const std::string& name, int degree, int count) {
    const int restCount = 3 * ((degree + 1) * (degree + 1) - 1);
    std::vector<std::string> props = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int k = 0; k < restCount; ++k) props.push_back("f_rest_" + std::to_string(k));
    for (const char* p : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) props.push_back(p);

    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";
    for (const auto& p : props) out << "property float " << p << "\n";
    out << "end_header\n";

    for (int i = 0; i < count; ++i) {
        std::vector<float> v;
        v.insert(v.end(), {1.0f + i, 2.0f + i, 3.0f + i});   // position
        v.insert(v.end(), {0.0f, 0.0f, 0.0f});                // normals, ignored
        v.insert(v.end(), {0.1f, 0.2f, 0.3f});                // f_dc
        for (int k = 0; k < restCount; ++k) v.push_back(100.0f + k); // f_rest, channel-major
        v.push_back(0.0f);                                    // opacity logit 0 -> 0.5
        v.insert(v.end(), {0.0f, std::log(2.0f), std::log(0.5f)}); // log scales
        v.insert(v.end(), {2.0f, 0.0f, 0.0f, 0.0f});          // unnormalized quaternion
        out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
    return path;
}

void testDegree(int degree) {
    const auto path = writeScene("aniso_test_deg" + std::to_string(degree) + ".ply", degree, 3);
    const aniso::Scene s = aniso::loadPly(path);

    CHECK(s.size() == 3);
    CHECK(s.shDegree == degree);
    const auto& g = s.gaussians[1];
    CHECK(near(g.position.x, 2) && near(g.position.y, 3) && near(g.position.z, 4));
    CHECK(near(g.scale.x, 1) && near(g.scale.y, 2) && near(g.scale.z, 0.5f));
    CHECK(near(g.opacity, 0.5f));
    CHECK(near(g.rotation.w, 1) && near(g.rotation.x, 0));

    const float* sh = s.shOf(1);
    CHECK(near(sh[0], 0.1f) && near(sh[1], 0.2f) && near(sh[2], 0.3f));
    const int per = s.shCoeffsPerChannel() - 1;
    if (per > 0) {
        // Coefficient 1 of red is f_rest_0, of green f_rest_<per>, of blue f_rest_<2*per>.
        CHECK(near(sh[3 + 0], 100.0f));
        CHECK(near(sh[3 + 1], 100.0f + per));
        CHECK(near(sh[3 + 2], 100.0f + 2 * per));
        // Last coefficient of blue is the last f_rest.
        CHECK(near(sh[per * 3 + 2], 100.0f + 3 * per - 1));
    }
    const auto c = aniso::dcColor(s, 1);
    CHECK(near(c[0], 0.5f + aniso::kShC0 * 0.1f));
    std::filesystem::remove(path);
}

void testRejectsAscii() {
    const auto path = std::filesystem::temp_directory_path() / "aniso_test_ascii.ply";
    std::ofstream(path) << "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    bool threw = false;
    try {
        aniso::loadPly(path);
    } catch (const aniso::PlyError&) {
        threw = true;
    }
    CHECK(threw);
    std::filesystem::remove(path);
}

} // namespace

int main() {
    for (int degree : {0, 1, 3}) testDegree(degree);
    testRejectsAscii();
    if (failures == 0) std::printf("all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
