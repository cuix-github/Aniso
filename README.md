# Aniso

A 3D Gaussian Splatting renderer written from scratch in C++, as a way to understand the technique from the inside. It starts on the CPU, where every step can be read and checked, and moves to the GPU later.

A Gaussian-splat scene is millions of soft, anisotropic 3D Gaussians, each with a position, a shape, an opacity, and a colour that varies with viewing direction. Rendering them is rasterization rather than ray tracing: project each Gaussian to a 2D ellipse, sort by depth, and blend front to back.

## Status

Four steps work. **Loading**: `aniso info` reads a scene trained by the reference 3DGS code; its numbers match an independent numpy reader exactly on the Tanks and Temples *train* scene (741,883 Gaussians). **Cameras**: `aniso dots` projects every Gaussian's centre through a photo's camera; the dots land on the train's lettering. **Rendering**: `aniso render` draws the scene from any photo's camera on the CPU, in about 0.3 seconds at 980×545 across all cores. **Walking**: `aniso_viewer` opens a window to fly through the scene like a first-person game. **View-dependent colour**: the full degree-3 spherical harmonics, so colour changes with the viewing angle.

Against the real photos, at the photos' size (PSNR, higher is closer):

| Photo | View-independent colour | Full colour |
|---|---|---|
| 00001 | 21.7 dB | 22.2 dB |
| 00050 | 22.0 dB | 22.5 dB |
| 00150 | 19.5 dB | 20.3 dB |
| 00250 | 24.2 dB | 25.2 dB |

Full colour costs about 2 ms more per frame. The locomotive comes out sharp, lettering readable. The sky is blotchy and the frame edges smear, which is typical of this scene: the sky is effectively at infinity, and the edges were barely covered by the photos.

A difference image of the two shows where view-dependent colour matters: the handrails and the painted metal of the locomotive change with the viewing angle, while the gravel and dirt do not.

**On the GPU**: a CUDA renderer with the same maths, organized like the reference rasterizer (per-Gaussian projection, one radix sort keyed by tile and depth, one thread block per 16×16 tile). Its images match the CPU renderer to within two brightness levels (77 to 86 dB). At 1280 wide:

| Scene | CPU | GPU (RTX 5090) |
|---|---|---|
| *train*, 0.74 million Gaussians | 421 ms | 2.3 ms |
| Kitchen, 2.2 million Gaussians | 845 ms | 2.6 ms |

With the GPU build, the viewer renders every frame at full resolution, a few milliseconds each.

## Build

Needs CMake 3.20+ and a C++20 compiler. On Windows with Visual Studio 2022, from cmd:

```
build.bat
```

It configures, builds in Release, and runs the tests, using the CMake bundled with Visual Studio. Elsewhere:

```
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

The CUDA renderer is optional. It needs an NVIDIA GPU and the project-local CUDA 12.8 toolkit, which `python pipeline\setup_toolchain.py` sets up once (nothing is installed system-wide). Then, from cmd:

```
build_cuda.bat
```

That builds into `build-cuda\` with Ninja and runs the tests, including one that checks the GPU renderer against the CPU renderer. `aniso render ... --gpu` uses the GPU, and `build-cuda\aniso_viewer.exe` uses it automatically.

## Try it

Download a trained scene into `data/` (ignored by git). The smallest official one is the *train* scene after 7,000 iterations, about 180 MB, mirrored in Voxel51's dataset on Hugging Face:

```
curl -L -o data/train/point_cloud_7000.ply https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/FO_dataset/train/point_cloud/iteration_7000/point_cloud.ply
build/Release/aniso info data/train/point_cloud_7000.ply
python tools/reference_stats.py data/train/point_cloud_7000.ply   # the cross-check, needs numpy
```

The cameras and photos come from the input archive published with the reference code (`tandt_db.zip`, about 680 MB; only `tandt/train/` is needed):

```
curl -L -o data/tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
# extract tandt/train/ into data/, then:
build/Release/aniso cameras data/tandt/train/sparse/0
build/Release/aniso dots data/train/point_cloud_7000.ply data/tandt/train/sparse/0 00001.jpg renders/dots_00001.png
```

Open `renders/dots_00001.png` next to `data/tandt/train/images/00001.jpg`. The dot image is at the camera's full resolution (1959×1090); the photos in the archive are half size, so scale one to the other before overlaying.

To render, and compare with the photo (the comparison needs Pillow and numpy):

```
build\Release\aniso.exe render data\train\point_cloud_7000.ply data\tandt\train\sparse\0 00001.jpg renders\render_00001.png --width 980
python tools\compare.py renders\render_00001.png data\tandt\train\images\00001.jpg renders\compare_00001.png
start renders\compare_00001.png
```

`--width 980` renders at the photos' size, so the two can be compared pixel for pixel; without it the render uses the camera's recorded size.

To see where view-dependent colour matters, render with and without it and diff the two:

```
build\Release\aniso.exe render data\train\point_cloud_7000.ply data\tandt\train\sparse\0 00001.jpg renders\r_sh0.png --width 980 --sh-degree 0
build\Release\aniso.exe render data\train\point_cloud_7000.ply data\tandt\train\sparse\0 00001.jpg renders\r_sh3.png --width 980
python tools\diff.py renders\r_sh0.png renders\r_sh3.png renders\diff_sh.png
start renders\diff_sh.png
```

To walk through the scene yourself:

```
build\Release\aniso_viewer.exe data\train\point_cloud_7000.ply data\tandt\train\sparse\0 00001.jpg
```

W A S D move, Q and E go down and up, Shift moves four times faster, holding the right mouse button and dragging looks around, the mouse wheel changes walking speed, N and P jump to the next or previous photo camera, Esc quits. On the CPU, the viewer renders a quarter-width image while you move (about 10 frames per second) and sharpens to full resolution, 1280 wide, when you stop. The title bar shows the frame time and the nearest photo camera. Walk away from where the photographer stood, behind the train or up in the air, and the scene falls apart into floating fragments and smears: that is the edge of what the photos saw.

To make a smooth 1080p fly-through video with the GPU renderer (needs the CUDA build and `ffmpeg` on the PATH; a 12-second clip renders and encodes in under ten seconds):

```
python tools\flythrough.py data\train\point_cloud_7000.ply data\tandt\train\sparse\0 renders\train_orbit.mp4 --mode orbit
start renders\train_orbit.mp4
```

It circles the subject along a smoothed version of the photographers' path, always looking at the point the photos look at most. `--seconds`, `--fps`, `--size`, and `--fov` adjust the clip. Behind it, `aniso path` renders one frame per camera pose and streams raw frames straight into ffmpeg.

For directed video, `tools\shots.py` renders a shot list: slow moves (straight dollies or horizontal arcs) with one fixed point of interest per shot, eased in and out so the camera never jerks, and hard cuts between shots. A shot can include a *reveal*, which shrinks every Gaussian partway through and grows it back, so the scene visibly comes apart into the ellipsoids it is made of: large flat ones on a tabletop, tiny ones on cereal rings, black gaps between them. `tools\shots\kitchen.json` is a 68-second shot list for the splat kitchen trained in `pipeline\`:

```
python tools\shots.py data\kitchen\train_30k\point_cloud.ply tools\shots\kitchen.json renders\kitchen_shots.mp4
```

The reveal is also available on its own, as the `splatScale` render option and as an optional tenth column in `aniso path` files.

To see the scene from a moving camera, render a run of consecutive photo cameras into a GIF (about half a minute for 60 frames):

```
python tools\orbit.py data\train\point_cloud_7000.ply data\tandt\train\sparse\0 renders\orbit.gif --first 1 --count 60
start renders\orbit.gif
```

## What the file holds

The file stores the optimizer's raw values, not the ones a renderer uses, and the loader converts them. Scales are stored as logarithms, so the loader takes `exp`. Opacity is stored as a logit, so it goes through a sigmoid. Rotation is a quaternion with `w` first, normalized on load. Colour is spherical-harmonic coefficients: `f_dc_*` is the view-independent term, and the view-dependent ones in `f_rest_*` are stored all red, then all green, then all blue; the loader reorders them per coefficient. The view-independent colour is `0.5 + 0.2821 * f_dc`, and it can fall outside [0, 1] on its own, because the view-dependent terms add or subtract on top; the final colour is clamped after they are applied.

## How cameras work here

COLMAP describes each photo by a pinhole camera: focal lengths and a principal point in pixels, and a world-to-camera rotation and translation, so a world point maps to camera space as `R * X + t`. The camera looks down +z with +x right and +y down the image, so projecting is just `u = fx * x / z + cx`, `v = fy * y / z + cy`. The trained Gaussians live in the same world frame as COLMAP's reconstruction, because the reference code trains directly in it.

## Plan

Each step ends in something that can be looked at and checked. They are tracked as issues.

1. Load a trained scene. Done.
2. Read the scene's cameras and project each Gaussian's centre as a dot; the dots should outline the reference photo. Done.
3. Render on the CPU: project each Gaussian's covariance to a 2D ellipse, sort by depth, blend front to back, view-independent colour first. Compare with the reference image. Done.
4. View-dependent colour from the full spherical harmonics. Done.
5. A scene of our own: a real bowl, photographed, reconstructed with COLMAP and trained with gsplat, rendered here and matching gsplat's own render.

After that the renderer becomes the base for pouring a simulated liquid into that captured bowl, and for a GPU version written in HIP.
