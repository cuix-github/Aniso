#include "aniso/ply_loader.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aniso {
namespace {

struct Header {
    std::size_t vertexCount = 0;
    std::vector<std::string> properties; // in file order; all float32
    std::streamoff bodyOffset = 0;
};

Header readHeader(std::istream& in, const std::string& name) {
    Header h;
    std::string line;
    if (!std::getline(in, line) || line.rfind("ply", 0) != 0) {
        throw PlyError(name + ": not a PLY file");
    }
    bool inVertex = false;
    bool sawFormat = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream words(line);
        std::string word;
        words >> word;
        if (word == "format") {
            std::string format;
            words >> format;
            if (format != "binary_little_endian") {
                throw PlyError(name + ": only binary_little_endian PLY is supported, got " + format);
            }
            sawFormat = true;
        } else if (word == "element") {
            std::string element;
            std::size_t count = 0;
            words >> element >> count;
            inVertex = (element == "vertex");
            if (inVertex) {
                h.vertexCount = count;
            } else if (count != 0) {
                throw PlyError(name + ": unexpected element '" + element + "'");
            }
        } else if (word == "property") {
            std::string type, prop;
            words >> type >> prop;
            if (!inVertex) continue;
            if (type != "float" && type != "float32") {
                throw PlyError(name + ": property '" + prop + "' has type " + type + ", expected float");
            }
            h.properties.push_back(prop);
        } else if (word == "end_header") {
            if (!sawFormat) throw PlyError(name + ": missing format line");
            h.bodyOffset = in.tellg();
            return h;
        }
    }
    throw PlyError(name + ": header has no end_header");
}

int shDegreeFromRestCount(std::size_t restCount, const std::string& name) {
    for (int degree = 0; degree <= 4; ++degree) {
        const std::size_t perChannel = static_cast<std::size_t>((degree + 1) * (degree + 1));
        if (restCount == 3 * (perChannel - 1)) return degree;
    }
    throw PlyError(name + ": " + std::to_string(restCount) + " f_rest properties do not match any SH degree");
}

} // namespace

Scene loadPly(const std::filesystem::path& path) {
    const std::string name = path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw PlyError(name + ": cannot open");

    const Header h = readHeader(in, name);

    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t i = 0; i < h.properties.size(); ++i) column[h.properties[i]] = i;
    auto col = [&](const std::string& prop) {
        auto it = column.find(prop);
        if (it == column.end()) throw PlyError(name + ": missing property '" + prop + "'");
        return it->second;
    };

    const std::size_t cx = col("x"), cy = col("y"), cz = col("z");
    const std::size_t cOpacity = col("opacity");
    const std::size_t cScale[3] = {col("scale_0"), col("scale_1"), col("scale_2")};
    const std::size_t cRot[4] = {col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")};
    const std::size_t cDc[3] = {col("f_dc_0"), col("f_dc_1"), col("f_dc_2")};

    std::size_t restCount = 0;
    while (column.count("f_rest_" + std::to_string(restCount))) ++restCount;

    Scene scene;
    scene.shDegree = shDegreeFromRestCount(restCount, name);
    const std::size_t coeffs = static_cast<std::size_t>(scene.shCoeffsPerChannel());
    const std::size_t restPerChannel = coeffs - 1;
    std::vector<std::size_t> cRest(restCount);
    for (std::size_t k = 0; k < restCount; ++k) cRest[k] = col("f_rest_" + std::to_string(k));

    const std::size_t stride = h.properties.size();
    std::vector<float> body(h.vertexCount * stride);
    in.seekg(h.bodyOffset);
    in.read(reinterpret_cast<char*>(body.data()), static_cast<std::streamsize>(body.size() * sizeof(float)));
    if (static_cast<std::size_t>(in.gcount()) != body.size() * sizeof(float)) {
        throw PlyError(name + ": file is shorter than the header promises (" +
                       std::to_string(h.vertexCount) + " Gaussians)");
    }

    scene.gaussians.resize(h.vertexCount);
    scene.sh.resize(h.vertexCount * coeffs * 3);

    for (std::size_t i = 0; i < h.vertexCount; ++i) {
        const float* v = body.data() + i * stride;
        Gaussian& g = scene.gaussians[i];

        g.position = {v[cx], v[cy], v[cz]};
        g.scale = {std::exp(v[cScale[0]]), std::exp(v[cScale[1]]), std::exp(v[cScale[2]])};
        g.opacity = 1.0f / (1.0f + std::exp(-v[cOpacity]));

        Quat q{v[cRot[0]], v[cRot[1]], v[cRot[2]], v[cRot[3]]};
        const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        g.rotation = n > 0 ? Quat{q.w / n, q.x / n, q.y / n, q.z / n} : Quat{};

        float* sh = scene.sh.data() + i * coeffs * 3;
        for (int c = 0; c < 3; ++c) sh[c] = v[cDc[c]];                     // coefficient 0
        for (int c = 0; c < 3; ++c) {
            for (std::size_t k = 0; k < restPerChannel; ++k) {                // coefficients 1..
                sh[(k + 1) * 3 + c] = v[cRest[c * restPerChannel + k]];
            }
        }
    }
    return scene;
}

} // namespace aniso
