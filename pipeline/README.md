# From a DCC scene to Gaussian splats

This folder turns a 3D scene built in a DCC into a Gaussian-splat scene: render it from many cameras in Maya, train splats on those renders with gsplat, and check the result in Aniso against views the training never saw. The first scene is the Pixar Kitchen Set.

Everything runs on Windows, with Maya 2027 and an NVIDIA GPU. All downloads, renders, and trained scenes live in `data/`, which git ignores. The Kitchen Set's licence allows personal, non-commercial testing only and forbids distributing the asset or anything derived from it, so none of its renders or splats belong in this repository or anywhere public.

## One-time setup

```
python pipeline\setup_toolchain.py
```

It puts NVIDIA's CUDA 12.8 components in `.toolchain\` and a Python environment with PyTorch and gsplat in `.venv\`, both ignored by git; nothing is installed system-wide. The script explains why each piece is needed. The first time gsplat runs, it compiles its CUDA code, which takes about a minute.

## Steps

1. **Build the dataset in Maya.** `kitchen.py` imports the USD, gives every object a slightly glossy version of its own colour, lights the room, places cameras, saves a Maya scene with one renderable camera per view, and writes the cameras and starting points in COLMAP's binary format:

   ```
   "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" pipeline\maya\kitchen.py data\kitchen\Kitchen_set\Kitchen_set.usd data\kitchen\dataset --views 320
   ```

2. **Render every view** with Maya Hardware 2.0:

   ```
   "C:\Program Files\Autodesk\Maya2027\bin\Render.exe" -r hw2 -rd C:\full\path\to\data\kitchen\dataset\images data\kitchen\dataset\kitchen.mb
   ```

3. **Check the dataset.** Projects the starting points through each camera over its image and makes a contact sheet. If the points sit on the surfaces they came from, the camera export is right.

   ```
   python pipeline\check_dataset.py data\kitchen\dataset data\kitchen\dataset_check --max 32
   ```

4. **Train.** Every eighth view is held out. At the end the held-out views are rendered and scored, and the scene is written as `point_cloud.ply`.

   ```
   .toolchain\run.bat .venv\Scripts\python.exe pipeline\train_gsplat.py data\kitchen\dataset data\kitchen\train_30k --steps 30000
   ```

5. **Look at it in Aniso**, from a held-out camera or walking:

   ```
   build\Release\aniso.exe render data\kitchen\train_30k\point_cloud.ply data\kitchen\dataset\sparse\0 view_000.png renders\kitchen_000.png
   build\Release\aniso_viewer.exe data\kitchen\train_30k\point_cloud.ply data\kitchen\dataset\sparse\0 view_000.png
   ```

## Decisions and why

- **No COLMAP run.** COLMAP recovers unknown cameras from real photos. Maya placed every camera, so they are known exactly and are written straight into COLMAP's file format, which trainers read. The starting points come from the mesh vertices instead of a reconstruction.
- **Hardware 2.0, not Arnold.** Arnold batch rendering needs its own authorization, which the available Maya login does not provide; Arnold aborts rather than render with a watermark. Hardware 2.0 is Maya's viewport renderer run in batch: rasterized, with shadow maps and ambient occlusion, noise-free and fast (320 views in about 40 seconds).
- **One fixed lighting setup.** Splats bake lighting into colour, so every view must see the same light: a soft ambient, a warm key light with shadow maps, a cool fill, the ceiling fixture.
- **Slightly glossy materials.** The Kitchen Set has no materials, only a flat colour per object. Each gets a blinn with a soft highlight so there is view-dependent appearance for the spherical harmonics to learn.
- **Cameras that see something.** The set is open on some sides. Before a camera is accepted, nine rays are cast across its view; it is rejected if anything is closer than 60 cm or if more than two rays escape the set.
- **Our own training loop on gsplat's rasterizer**, instead of gsplat's example trainer, which pulls in many dependencies. The whole render, compare, adjust loop fits in one readable file.
- **Metres, Y up.** Maya's world, converted from centimetres.
