"""Trains a Gaussian-splat scene from a COLMAP-format dataset with gsplat, and writes it in the
reference 3DGS .ply layout that Aniso reads.

    python pipeline/train_gsplat.py <dataset-dir> <out-dir> [--steps 30000] [--test-every 8]

The dataset needs images/ and sparse/0/{cameras,images,points3D}.bin. Every eighth image (in
name order) is held out and never trained on; at the end the held-out views are rendered and
scored, and their names are written to held_out.txt for independent checks with Aniso.

The loop, in one sentence: render the current Gaussians from one training photo's camera,
compare with the photo (L1 plus structural similarity), backpropagate, let Adam nudge every
parameter, and let gsplat's default strategy split, clone and prune Gaussians as it goes.

Needs torch with CUDA, gsplat, numpy, Pillow. On Windows run it through .toolchain/run.bat so
gsplat can compile its CUDA code on first use.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_dataset import qmat, read_cameras, read_images, read_points  # noqa: E402

from gsplat import rasterization  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402

C0 = 0.28209479177387814


def ssim(a, b):
    """Structural similarity of two [1, 3, H, W] images with an 11x11 Gaussian window."""
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    blur = lambda t: F.conv2d(t, w, padding=5, groups=3)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a ** 2
    s_bb = blur(b * b) - mu_b ** 2
    s_ab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def load_dataset(root, device):
    sp = os.path.join(root, "sparse", "0")
    cams = read_cameras(os.path.join(sp, "cameras.bin"))
    images = read_images(os.path.join(sp, "images.bin"))
    xyz, rgb = read_points(os.path.join(sp, "points3D.bin"))
    views = []
    for name, q, t, cid in images:
        w, h, fx, fy, cx, cy = cams[cid]
        m = np.eye(4)
        m[:3, :3] = qmat(q)
        m[:3, 3] = t
        img = np.asarray(Image.open(os.path.join(root, "images", name)).convert("RGB"), np.float32) / 255.0
        views.append(dict(name=name, viewmat=torch.tensor(m, dtype=torch.float32, device=device),
                          K=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device),
                          w=w, h=h, image=torch.tensor(img, device=device)))
    return views, xyz, rgb


def init_gaussians(xyz, rgb, device, sh_degree):
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    # Initial size: the mean distance to the three nearest neighbours, computed in chunks.
    d3 = torch.empty(len(means), device=device)
    for s in range(0, len(means), 2048):
        d = torch.cdist(means[s:s + 2048], means)
        d3[s:s + 2048] = d.topk(4, largest=False).values[:, 1:].mean(1)
    scales = torch.log(d3.clamp_min(1e-4))[:, None].repeat(1, 3)
    n = len(means)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1
    colours = torch.tensor(rgb, dtype=torch.float32, device=device) / 255.0
    k = (sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(((colours - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, k - 1, 3, device=device)),
    })
    return params


def render(params, view, sh_degree):
    colours = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colours, view["viewmat"][None], view["K"][None],
        view["w"], view["h"], sh_degree=sh_degree, packed=False)
    return img[0], info


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def save_ply(params, path):
    """The reference 3DGS layout: x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3,
    with f_rest stored channel by channel."""
    means = params["means"].detach().cpu().numpy()
    n = len(means)
    f_dc = params["sh0"].detach().cpu().numpy().reshape(n, 3)
    f_rest = params["shN"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()
    cols = [means, np.zeros((n, 3), np.float32), f_dc, f_rest,
            params["opacities"].detach().cpu().numpy()[:, None], params["scales"].detach().cpu().numpy(),
            F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()]
    data = np.concatenate(cols, 1).astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] +
             ["f_rest_%d" % i for i in range(f_rest.shape[1])] +
             ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n).encode())
        for nm in names:
            f.write(("property float %s\n" % nm).encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"

    views, xyz, rgb = load_dataset(a.dataset, dev)
    test = [v for i, v in enumerate(views) if i % a.test_every == 0]
    train = [v for i, v in enumerate(views) if i % a.test_every != 0]
    with open(os.path.join(a.out, "held_out.txt"), "w") as f:
        f.write("\n".join(v["name"] for v in test) + "\n")

    centres = torch.stack([torch.linalg.inv(v["viewmat"])[:3, 3] for v in views])
    scene_scale = float((centres - centres.mean(0)).norm(dim=1).max()) * 1.1

    params = init_gaussians(xyz, rgb, dev, a.sh_degree)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    print("train %d views, held out %d, starting Gaussians %d, scene scale %.2f m"
          % (len(train), len(test), len(params["means"]), scene_scale), flush=True)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    log = []
    for step in range(a.steps):
        v = train[rng.integers(len(train))]
        deg = min(step // 1000, a.sh_degree)
        img, info = render(params, v, deg)
        strategy.step_pre_backward(params, opts, state, step, info)
        l1 = (img - v["image"]).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], v["image"].permute(2, 0, 1)[None]))
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info)
        if step % 1000 == 0 or step == a.steps - 1:
            with torch.no_grad():
                p = psnr(img.clamp(0, 1), v["image"])
            log.append(dict(step=step, loss=loss.item(), train_psnr=p, gaussians=len(params["means"]),
                            seconds=round(time.time() - t0, 1)))
            print("step %5d  loss %.4f  train PSNR %.2f  Gaussians %d  %.0f s"
                  % (step, loss.item(), p, len(params["means"]), time.time() - t0), flush=True)

    with torch.no_grad():
        scores = []
        for v in test:
            img, _ = render(params, v, a.sh_degree)
            img = img.clamp(0, 1)
            scores.append(dict(name=v["name"], psnr=psnr(img, v["image"])))
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(a.out, "gsplat_" + v["name"]))
    mean = sum(s["psnr"] for s in scores) / len(scores)
    print("held-out PSNR (gsplat's own render): mean %.2f dB over %d views" % (mean, len(scores)), flush=True)

    save_ply(params, os.path.join(a.out, "point_cloud.ply"))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(dict(steps=a.steps, train_views=len(train), held_out_views=len(test), scene_scale=scene_scale,
                       gaussians=len(params["means"]), minutes=round((time.time() - t0) / 60, 1),
                       held_out_psnr_mean=mean, held_out_scores=scores, log=log), f, indent=1)
    print("wrote %s" % os.path.join(a.out, "point_cloud.ply"))


if __name__ == "__main__":
    main()
