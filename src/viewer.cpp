// aniso_viewer <scene.ply> <colmap-sparse-dir> [start-image]
//
// A first-person window for walking through a splat scene. Plain Win32, no libraries.
//
//   W A S D         move forward, left, back, right
//   Q E             move down, up
//   Shift           move four times faster
//   right mouse     hold and drag to look around
//   mouse wheel     change walking speed
//   N P             jump to the next or previous photo camera
//   Esc             quit
//
// The CPU renderer is too slow for full resolution at game frame rates, so while the camera
// moves the viewer renders a quarter-width image and stretches it; once the camera has been
// still for a moment it renders once at full resolution.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "aniso/camera.h"
#include "aniso/ply_loader.h"
#include "aniso/render.h"
#ifdef ANISO_WITH_CUDA
#include "aniso/gpu_render.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

using aniso::Vec3;
using Clock = std::chrono::steady_clock;

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 scale(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 a) { return scale(a, 1.0f / std::sqrt(dot(a, a))); }

// Rows of a world-to-camera rotation are the camera's axes in world space: right, down, forward.
Vec3 row(const aniso::Camera& c, int r) {
    return {static_cast<float>(c.R[r][0]), static_cast<float>(c.R[r][1]), static_cast<float>(c.R[r][2])};
}

// A free-flying camera: position plus yaw and pitch about the scene's up direction.
struct FlyCamera {
    Vec3 position;
    Vec3 up;          // scene up, estimated from the photo cameras
    Vec3 h0, h1;      // horizontal basis: yaw 0 looks along h0, positive yaw turns toward h1
    float yaw = 0, pitch = 0;

    Vec3 forward() const {
        const Vec3 horizontal = add(scale(h0, std::cos(yaw)), scale(h1, std::sin(yaw)));
        return normalize(add(scale(horizontal, std::cos(pitch)), scale(up, std::sin(pitch))));
    }
    Vec3 right() const { return normalize(cross(forward(), up)); }

    // Look along a photo camera's direction from its position.
    void lookFrom(const aniso::Camera& c) {
        position = c.centre();
        const Vec3 f = row(c, 2);
        pitch = std::asin(std::clamp(dot(f, up), -1.0f, 1.0f));
        const Vec3 fh = normalize(add(f, scale(up, -dot(f, up))));
        yaw = std::atan2(dot(fh, h1), dot(fh, h0));
    }

    // A COLMAP-style camera with these intrinsics and the current pose.
    aniso::Camera toCamera(const aniso::Camera& intrinsics) const {
        aniso::Camera c = intrinsics;
        const Vec3 f = forward(), r = right(), d = cross(f, r); // down = forward x right
        const Vec3 rows[3] = {r, d, f};
        for (int i = 0; i < 3; ++i) {
            c.R[i][0] = rows[i].x; c.R[i][1] = rows[i].y; c.R[i][2] = rows[i].z;
            c.t[i] = -dot(rows[i], position);
        }
        return c;
    }
};

struct Viewer {
    aniso::Scene scene;
    std::vector<aniso::Camera> photos;
    aniso::Camera intrinsics;  // window-sized intrinsics of the starting photo camera
    FlyCamera fly;
    std::size_t photoIndex = 0;
    float speed = 1.0f;

    HWND hwnd = nullptr;
    bool looking = false;
    POINT lookCentre{};
    Clock::time_point lastChange = Clock::now();
    bool fullResShown = false;

    aniso::Image frame{1, 1};
#ifdef ANISO_WITH_CUDA
    std::unique_ptr<aniso::GpuRenderer> gpu; // when present, every frame renders at full resolution
#endif
    std::vector<std::uint8_t> bgra;
    double lastFrameMs = 0;
};

Viewer* g = nullptr;

void markChanged() {
    g->lastChange = Clock::now();
    g->fullResShown = false;
}

void jumpToPhoto(std::size_t i) {
    g->photoIndex = i % g->photos.size();
    g->fly.lookFrom(g->photos[g->photoIndex]);
    markChanged();
}

void renderFrame(bool full) {
    RECT rc;
    GetClientRect(g->hwnd, &rc);
    bool onGpu = false;
#ifdef ANISO_WITH_CUDA
    onGpu = g->gpu != nullptr;
#endif
    const int width = (full || onGpu) ? g->intrinsics.width : std::max(64, g->intrinsics.width / 4);
    const aniso::Camera cam = g->fly.toCamera(aniso::resized(g->intrinsics, width));
    const auto t0 = Clock::now();
#ifdef ANISO_WITH_CUDA
    if (onGpu) g->frame = g->gpu->render(cam);
    else
#endif
        g->frame = aniso::render(g->scene, cam);
    g->lastFrameMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    const auto& img = g->frame;
    g->bgra.resize(static_cast<std::size_t>(img.width) * img.height * 4);
    for (std::size_t p = 0, n = static_cast<std::size_t>(img.width) * img.height; p < n; ++p) {
        g->bgra[p * 4 + 0] = img.rgb[p * 3 + 2];
        g->bgra[p * 4 + 1] = img.rgb[p * 3 + 1];
        g->bgra[p * 4 + 2] = img.rgb[p * 3 + 0];
        g->bgra[p * 4 + 3] = 255;
    }
    InvalidateRect(g->hwnd, nullptr, FALSE);

    char title[256];
    std::snprintf(title, sizeof title, "Aniso viewer  |  %s  %dx%d  %.1f ms (%.0f fps)  |  speed %.2f  |  near %s",
                  onGpu ? "GPU" : "CPU", img.width, img.height, g->lastFrameMs, 1000.0 / std::max(1.0, g->lastFrameMs), g->speed,
                  g->photos[g->photoIndex].imageName.c_str());
    SetWindowTextA(g->hwnd, title);
}

void paint(HDC dc) {
    RECT rc;
    GetClientRect(g->hwnd, &rc);
    const auto& img = g->frame;
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = img.width;
    bi.bmiHeader.biHeight = -img.height; // top-down rows
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, COLORONCOLOR);
    StretchDIBits(dc, 0, 0, rc.right, rc.bottom, 0, 0, img.width, img.height, g->bgra.data(), &bi, DIB_RGB_COLORS,
                  SRCCOPY);
}

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        if (g && !g->bgra.empty()) paint(dc);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_RBUTTONDOWN: {
        g->looking = true;
        SetCapture(hwnd);
        ShowCursor(FALSE);
        RECT rc;
        GetClientRect(hwnd, &rc);
        g->lookCentre = {rc.right / 2, rc.bottom / 2};
        POINT screen = g->lookCentre;
        ClientToScreen(hwnd, &screen);
        SetCursorPos(screen.x, screen.y);
        return 0;
    }
    case WM_RBUTTONUP:
        g->looking = false;
        ReleaseCapture();
        ShowCursor(TRUE);
        return 0;
    case WM_MOUSEMOVE:
        if (g->looking) {
            const int dx = static_cast<short>(LOWORD(lp)) - g->lookCentre.x;
            const int dy = static_cast<short>(HIWORD(lp)) - g->lookCentre.y;
            if (dx != 0 || dy != 0) {
                g->fly.yaw += dx * 0.003f;
                g->fly.pitch = std::clamp(g->fly.pitch - dy * 0.003f, -1.5f, 1.5f);
                POINT screen = g->lookCentre;
                ClientToScreen(hwnd, &screen);
                SetCursorPos(screen.x, screen.y);
                markChanged();
            }
        }
        return 0;
    case WM_MOUSEWHEEL:
        g->speed *= GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1.25f : 0.8f;
        markChanged();
        return 0;
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) DestroyWindow(hwnd);
        if (wp == 'N') jumpToPhoto(g->photoIndex + 1);
        if (wp == 'P') jumpToPhoto(g->photoIndex + g->photos.size() - 1);
        return 0;
    case WM_SIZE:
        markChanged();
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

// Scene up: the average of the photo cameras' up directions (COLMAP's camera y points down).
Vec3 estimateUp(const std::vector<aniso::Camera>& cams) {
    Vec3 sum{};
    for (const auto& c : cams) sum = add(sum, scale(row(c, 1), -1.0f));
    return normalize(sum);
}

int run(const std::string& plyPath, const std::string& sparseDir, const std::string& startImage) {
    Viewer v;
    g = &v;
    std::printf("loading %s ...\n", plyPath.c_str());
    v.scene = aniso::loadPly(plyPath);
#ifdef ANISO_WITH_CUDA
    if (aniso::GpuRenderer::available()) v.gpu = std::make_unique<aniso::GpuRenderer>(v.scene);
#endif
    v.photos = aniso::loadColmapCameras(sparseDir);
    if (v.photos.empty()) throw std::runtime_error("no cameras in " + sparseDir);

    auto it = std::find_if(v.photos.begin(), v.photos.end(),
                           [&](const aniso::Camera& c) { return c.imageName == startImage; });
    v.photoIndex = it == v.photos.end() ? 0 : static_cast<std::size_t>(it - v.photos.begin());
    v.intrinsics = aniso::resized(v.photos[v.photoIndex], 1280);

    v.fly.up = estimateUp(v.photos);
    const Vec3 f0 = row(v.photos[v.photoIndex], 2);
    v.fly.h0 = normalize(add(f0, scale(v.fly.up, -dot(f0, v.fly.up))));
    v.fly.h1 = normalize(cross(v.fly.h0, v.fly.up));
    jumpToPhoto(v.photoIndex);

    // Walking speed: a tenth of the spread of the photo cameras per second.
    Vec3 lo = v.photos[0].centre(), hi = lo;
    for (const auto& c : v.photos) {
        const Vec3 p = c.centre();
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    v.speed = 0.1f * std::sqrt(dot(add(hi, scale(lo, -1)), add(hi, scale(lo, -1))));

    WNDCLASSA wc{};
    wc.lpfnWndProc = wndProc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = "AnisoViewer";
    RegisterClassA(&wc);
    RECT rc{0, 0, v.intrinsics.width, v.intrinsics.height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    v.hwnd = CreateWindowA("AnisoViewer", "Aniso viewer", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                           CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance,
                           nullptr);
    std::printf("%zu Gaussians, %zu photo cameras. WASD move, QE down/up, Shift fast, hold right mouse to look,\n"
                "wheel changes speed, N/P jump between photo cameras, Esc quits.\n",
                v.scene.size(), v.photos.size());

    auto last = Clock::now();
    for (;;) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }

        const auto now = Clock::now();
        const float dt = std::min(0.25f, std::chrono::duration<float>(now - last).count());
        last = now;

        if (GetForegroundWindow() == v.hwnd) {
            auto down = [](int key) { return (GetAsyncKeyState(key) & 0x8000) != 0; };
            Vec3 move{};
            if (down('W')) move = add(move, v.fly.forward());
            if (down('S')) move = add(move, scale(v.fly.forward(), -1));
            if (down('D')) move = add(move, v.fly.right());
            if (down('A')) move = add(move, scale(v.fly.right(), -1));
            if (down('E')) move = add(move, v.fly.up);
            if (down('Q')) move = add(move, scale(v.fly.up, -1));
            if (dot(move, move) > 0) {
                const float boost = down(VK_SHIFT) ? 4.0f : 1.0f;
                v.fly.position = add(v.fly.position, scale(normalize(move), v.speed * boost * dt));
                markChanged();
            }
        }

        const bool moving = std::chrono::duration<double>(now - v.lastChange).count() < 0.2;
        if (moving) {
            renderFrame(false);
        } else if (!v.fullResShown) {
            renderFrame(true);
            v.fullResShown = true;
        } else {
            MsgWaitForMultipleObjects(0, nullptr, FALSE, 15, QS_ALLINPUT);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: aniso_viewer <scene.ply> <colmap-sparse-dir> [start-image]\n");
        return 2;
    }
    try {
        return run(argv[1], argv[2], argc > 3 ? argv[3] : "");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
