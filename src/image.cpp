#include "aniso/image.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

namespace aniso {
namespace {

std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t crc = 0) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) out.push_back(static_cast<std::uint8_t>(v >> s));
}

void chunk(std::ofstream& f, const char type[4], const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> buf;
    put32(buf, static_cast<std::uint32_t>(data.size()));
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    put32(buf, crc32(buf.data() + 4, buf.size() - 4));
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

} // namespace

void writePng(const std::filesystem::path& path, const Image& img) {
    // Raw scanlines, each prefixed with filter type 0.
    std::vector<std::uint8_t> raw;
    const std::size_t rowBytes = static_cast<std::size_t>(img.width) * 3;
    raw.reserve((rowBytes + 1) * img.height);
    for (int y = 0; y < img.height; ++y) {
        raw.push_back(0);
        const auto* row = img.rgb.data() + y * rowBytes;
        raw.insert(raw.end(), row, row + rowBytes);
    }

    // zlib stream of stored (uncompressed) deflate blocks, at most 65535 bytes each.
    std::vector<std::uint8_t> z = {0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    for (std::size_t pos = 0; pos < raw.size() || pos == 0;) {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(len & 0xFF));
        z.push_back(static_cast<std::uint8_t>(len >> 8));
        z.push_back(static_cast<std::uint8_t>(~len & 0xFF));
        z.push_back(static_cast<std::uint8_t>((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
        if (last) break;
    }
    put32(z, (b << 16) | a);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path.string() + ": cannot write");
    const std::uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    f.write(reinterpret_cast<const char*>(signature), 8);

    std::vector<std::uint8_t> ihdr;
    put32(ihdr, static_cast<std::uint32_t>(img.width));
    put32(ihdr, static_cast<std::uint32_t>(img.height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0}); // 8-bit, truecolour, no interlace
    chunk(f, "IHDR", ihdr);
    chunk(f, "IDAT", z);
    chunk(f, "IEND", {});
}

} // namespace aniso
