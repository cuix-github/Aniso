# Aniso

A 3D Gaussian Splatting renderer written from scratch in C++, as a way to understand the technique from the inside. It starts on the CPU, where every step can be read and checked, and moves to the GPU later.

A Gaussian-splat scene is millions of soft, anisotropic 3D Gaussians, each with a position, a shape, an opacity, and a colour that varies with viewing direction. Rendering them is rasterization rather than ray tracing: project each Gaussian to a 2D ellipse, sort by depth, and blend front to back.

## Status

Step one works: loading a scene trained by the reference 3DGS code. `aniso info` prints what is in a file, and its numbers match an independent numpy reader exactly on the Tanks and Temples *train* scene (741,883 Gaussians, spherical harmonics of degree 3, loaded in about 150 ms).

## Build

Needs CMake 3.20+ and a C++20 compiler. On Windows, Visual Studio 2022 ships both.

```
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

## Try it

Download a trained scene into `data/` (ignored by git). The smallest official one is the *train* scene after 7,000 iterations, about 180 MB, mirrored in Voxel51's dataset on Hugging Face:

```
curl -L -o data/train/point_cloud_7000.ply https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/FO_dataset/train/point_cloud/iteration_7000/point_cloud.ply
build/Release/aniso info data/train/point_cloud_7000.ply
python tools/reference_stats.py data/train/point_cloud_7000.ply   # the cross-check, needs numpy
```

## What the file holds

The file stores the optimizer's raw values, not the ones a renderer uses, and the loader converts them. Scales are stored as logarithms, so the loader takes `exp`. Opacity is stored as a logit, so it goes through a sigmoid. Rotation is a quaternion with `w` first, normalized on load. Colour is spherical-harmonic coefficients: `f_dc_*` is the view-independent term, and the view-dependent ones in `f_rest_*` are stored all red, then all green, then all blue; the loader reorders them per coefficient. The view-independent colour is `0.5 + 0.2821 * f_dc`, and it can fall outside [0, 1] on its own, because the view-dependent terms add or subtract on top; the final colour is clamped after they are applied.

## Plan

Each step ends in something that can be looked at and checked. They are tracked as issues.

1. Load a trained scene. Done.
2. Read the scene's cameras and project each Gaussian's centre as a dot; the dots should outline the reference photo.
3. Render on the CPU: project each Gaussian's covariance to a 2D ellipse, sort by depth, blend front to back, view-independent colour first. Compare with the reference image.
4. View-dependent colour from the full spherical harmonics.
5. A scene of our own: a real bowl, photographed, reconstructed with COLMAP and trained with gsplat, rendered here and matching gsplat's own render.

After that the renderer becomes the base for pouring a simulated liquid into that captured bowl, and for a GPU version written in HIP.
