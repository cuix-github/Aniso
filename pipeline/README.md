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

6. **Score it independently with Aniso**: renders every held-out view with Aniso's own renderer, scores it against the true image and against gsplat's render of the same splats, and makes a sheet of true | Aniso | difference:

   ```
   python pipeline\eval_aniso.py data\kitchen\dataset data\kitchen\train_30k --sheet 6
   ```

## First results (2026-09-29)

320 views at 960×540, 280 trained and 40 held out. Training took 6.6 minutes on an RTX 5090 for 30,000 steps and grew the scene from 150,000 starting points to 2.19 million Gaussians (544 MB).

| Held-out measure | Value |
|---|---|
| Aniso render vs Maya's true image | 29.9 dB mean, 21.1 worst, 34.6 best |
| Aniso vs gsplat, same splats | 66.3 dB (effectively identical) |

Where splats struggle here: thin edges, such as tile grout and object outlines, carry most of the error; soft blotches appear on large flat areas like the floor; and objects much closer to the camera than any training view came, such as a chair back in the worst view, smear into streaks. Splats are only as good as the distances the training saw.

**How many views does it need?** The same 40 held-out views, with training on fewer views spread evenly around the room (`--train-views N`), 30,000 steps each:

| Training views | Held-out PSNR | PSNR on its own training views | Gaussians |
|---|---|---|---|
| 35 | 22.4 dB | 40.0 dB | 1.55 M |
| 70 | 26.4 dB | 36.7 dB | 1.81 M |
| 140 | 29.1 dB | 37.5 dB | 2.14 M |
| 280 | 29.9 dB | 35.0 dB | 2.19 M |

Quality climbs steeply up to about 140 views and then flattens: doubling to 280 adds under 1 dB. With few views the scene fits its training views almost perfectly and generalizes poorly; the gap between the two columns is overfitting, and it narrows as views are added. (Training PSNR is from the last logged step, one view each, so it is indicative only.)

On the CPU, Aniso needs about 0.85 seconds per full 1280×720 frame of the kitchen, which makes walking choppy; the CUDA renderer is the fix.

## Decisions and why

- **No COLMAP run.** COLMAP recovers unknown cameras from real photos. Maya placed every camera, so they are known exactly and are written straight into COLMAP's file format, which trainers read. The starting points come from the mesh vertices instead of a reconstruction.
- **Hardware 2.0, not Arnold.** Arnold batch rendering needs its own authorization, which the available Maya login does not provide; Arnold aborts rather than render with a watermark. Hardware 2.0 is Maya's viewport renderer run in batch: rasterized, with shadow maps and ambient occlusion, noise-free and fast (320 views in about 40 seconds).
- **One fixed lighting setup.** Splats bake lighting into colour, so every view must see the same light: a soft ambient, a warm key light with shadow maps, a cool fill, the ceiling fixture.
- **Slightly glossy materials.** The Kitchen Set has no materials, only a flat colour per object. Each gets a blinn with a soft highlight so there is view-dependent appearance for the spherical harmonics to learn.
- **Cameras that see something.** The set is open on some sides. Before a camera is accepted, nine rays are cast across its view; it is rejected if anything is closer than 60 cm or if more than two rays escape the set.
- **Our own training loop on gsplat's rasterizer**, instead of gsplat's example trainer, which pulls in many dependencies. The whole render, compare, adjust loop fits in one readable file.
- **Metres, Y up.** Maya's world, converted from centimetres.
