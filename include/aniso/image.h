#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace aniso {

// An 8-bit RGB image, rows top to bottom.
struct Image {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgb;

    Image(int w, int h) : width(w), height(h), rgb(static_cast<std::size_t>(w) * h * 3, 0) {}
    std::uint8_t* at(int x, int y) { return rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3; }
};

// Writes a PNG with no external library: uncompressed deflate blocks, so files are large but
// the writer stays about fifty lines and has no dependencies.
void writePng(const std::filesystem::path& path, const Image& image);

} // namespace aniso
