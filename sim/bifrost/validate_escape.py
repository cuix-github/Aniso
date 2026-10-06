"""Validates the escaped-particle treatment in the Bifrost node against the 2D
reference, the same two layers as every other feature.

  lockstep   a mid-splash plunge state (droplets and bubbles in flight) is packaged as
             channels; the node (nz=1, escape on) and the python reference take the
             identical state through one step. The escaped MASKS must match particle
             for particle, and velocities/positions to float tolerance.
  loop       the full plunge runs through the Bifrost 3D loop graph at nz=1 with
             escape on, against a python run from the identical seed at the same fixed
             dt: droplet/bubble count curves overlaid, plus a two-row video
             (python above, Bifrost below) with the same markers as the 2D demo.

Run from sim/bifrost:  ..\\..\\.venv\\Scripts\\python.exe validate_escape.py [lockstep|loop]
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference2d"))
from pfflip2d import Sim  # noqa: E402
from validate_step import channels  # noqa: E402
from validate_3d import run_node  # noqa: E402

RESULTS = os.path.join(HERE, "..", "reference2d", "results", "validation")
NX, NY, DT = 120, 160, 0.012
ESC = {"escape": 1, "esc_phi": 0.3, "drag_droplet": 1.0, "drag_bubble": 8.0, "buoyancy": 2.0}
try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()


def mask(x, y):
    return (y < 36) | ((np.abs(x - 60) < 10) & (y > 70) & (y < 130))


def mid_splash_sim():
    s = Sim(NX, NY, escape=True)
    s.seed(mask)
    s.calibrate()
    for _ in range(290):          # t = 3.48, droplets and bubbles in flight
        s.step(DT)
    return s


def lockstep():
    print("=== lockstep (node escape treatment vs python, identical state)")
    s = mid_splash_sim()
    d_in = os.path.join(HERE, "out", "esc_in")
    d_out = os.path.join(HERE, "out", "esc_out")
    os.makedirs(d_in, exist_ok=True)
    ch = {k: v.astype(np.float32) for k, v in channels(s).items()}
    for k, v in ch.items():
        np.save(os.path.join(d_in, k + ".npy"), v.ravel())
    for k in ("w_mass", "w_mom", "w_phase"):
        np.save(os.path.join(d_in, k + ".npy"), np.zeros(NX * NY * 2, np.float32))
    np.save(os.path.join(d_in, "positions.npy"),
            np.concatenate([s.pos, np.full((len(s.pos), 1), 0.5)], 1).astype(np.float32))
    np.save(os.path.join(d_in, "velocities.npy"),
            np.concatenate([s.vel, np.zeros((len(s.pos), 1))], 1).astype(np.float32))
    np.save(os.path.join(d_in, "particle_phase.npy"), s.typ.astype(np.float32))

    run_node(d_in, d_out, nx=NX, ny=NY, nz=1, dt=DT, preconditioner=1,
             rho0_face=s.rho0_face, **ESC)

    # python: the same float32 state and the SAME channels through one step, with the
    # escape pieces applied exactly as Sim.step does them.
    ref = Sim(NX, NY, escape=True)
    ref.pos = np.load(os.path.join(d_in, "positions.npy"))[:, :2].astype(float)
    ref.vel = np.load(os.path.join(d_in, "velocities.npy"))[:, :2].astype(float)
    ref.typ = s.typ.copy()
    ref.rho0_face = s.rho0_face
    um = ch["u_mass"].astype(float)
    uo = ch["u_mom"].astype(float)
    vm = ch["v_mass"].astype(float)
    vo = ch["v_mom"].astype(float)
    ref.u = np.where(um > 0, uo / np.maximum(um, 1e-30), 0.0)
    ref.v = np.where(vm > 0, vo / np.maximum(vm, 1e-30), 0.0)
    ref._enforce_walls()
    u_star, v_star = ref.u.copy(), ref.v.copy()
    ref.v += ref.g * DT
    ref._enforce_walls()
    phi_u = np.clip(ch["u_phase"].astype(float), 0, 1)
    phi_v = np.clip(ch["v_phase"].astype(float), 0, 1)
    ref.mu, ref.mv = np.empty_like(ref.u), np.empty_like(ref.v)
    ref.phase = lambda raw: phi_u if raw is ref.mu else phi_v
    ref.project(DT)
    ones_air = (ref.typ == 0).astype(float)
    ones_liq = 1.0 - ones_air
    fu, fv = ref._face_frames()
    au, lu = ref._splat(ref.u.shape, fu, [ones_air, ones_liq])
    av, lv = ref._splat(ref.v.shape, fv, [ones_air, ones_liq])
    thresh = (1.0 - ref.esc_phi) * ref.rho0_face
    frac_air = 0.5 * (ref._sample_faces(au, np.array([0.0, 0.5]), ref.pos) +
                      ref._sample_faces(av, np.array([0.5, 0.0]), ref.pos))
    frac_liq = 0.5 * (ref._sample_faces(lu, np.array([0.0, 0.5]), ref.pos) +
                      ref._sample_faces(lv, np.array([0.5, 0.0]), ref.pos))
    drop = (ref.typ == 1) & (frac_air > thresh)
    bub = (ref.typ == 0) & (frac_liq > thresh)
    esc = drop | bub
    ref.escaped = esc
    ref.g2p(u_star, v_star, skip=esc)
    if esc.any():
        vg = ref.sample_velocity(ref.pos)
        ref.vel[drop, 1] += ref.g * DT
        ref.vel[drop] += ref.drag_droplet * DT * (vg[drop] - ref.vel[drop])
        ref.vel[bub, 1] += ref.buoyancy * abs(ref.g) * DT
        ref.vel[bub] += ref.drag_bubble * DT * (vg[bub] - ref.vel[bub])
    ref.advect(DT, ballistic=esc)

    b_esc = np.load(os.path.join(d_out, "out_escaped.npy")) > 0.5
    b_pos = np.load(os.path.join(d_out, "out_positions.npy"))[:, :2]
    b_vel = np.load(os.path.join(d_out, "out_velocities.npy"))[:, :2]
    mismatch = int((b_esc != ref.escaped).sum())
    print(f"  escaped python: {int(ref.escaped.sum())}, node: {int(b_esc.sum())}, "
          f"mask mismatches: {mismatch} of {len(b_esc)}")
    dp = np.abs(b_pos - ref.pos).max()
    dv = np.abs(b_vel - ref.vel).max()
    print(f"  max |diff| positions {dp:.3e}, velocities {dv:.3e}")
    ok = mismatch <= 2 and dp < 2e-3 and dv < 2e-2
    print("lockstep:", "PASS" if ok else "FAIL")
    return ok


def render(pos, typ, escaped, scale=4):
    img = np.full((NX * scale, NY * scale, 3), 255, np.uint8)
    pix = np.clip((pos * scale).astype(np.int64), 0, [NX * scale - 1, NY * scale - 1])
    for sel, colour in ((~escaped & (typ == 0), (223, 230, 239)),
                        (~escaped & (typ == 1), (25, 80, 190))):
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(pix[sel, 0] + dx, 0, NX * scale - 1),
                    np.clip(pix[sel, 1] + dy, 0, NY * scale - 1)] = colour
    im = Image.fromarray(np.rot90(img))
    d = ImageDraw.Draw(im)
    H = im.height
    for sel, fill, ring in ((escaped & (typ == 1), (255, 120, 0), (120, 50, 0)),
                            (escaped & (typ == 0), (255, 255, 255), (10, 30, 70))):
        for x, y in pos[sel]:
            d.ellipse([x * scale - 4, H - y * scale - 4, x * scale + 4, H - y * scale + 4],
                      fill=fill, outline=ring, width=2)
    return im


def loop():
    print("=== loop (full plunge through the Bifrost graph, escape on)")
    from demo3d import run_bifrost3
    import demo3d
    sub = 500
    # identical seeding: 2D-style 2x2 per cell at z=0.5, straight from the reference.
    s0 = Sim(NX, NY, escape=True)
    s0.seed(mask)
    s0.calibrate()

    d = os.path.join(HERE, "out", "loop3_esc2d")
    os.makedirs(d, exist_ok=True)
    pos3 = np.concatenate([s0.pos, np.full((len(s0.pos), 1), 0.5)], 1).astype(np.float32)
    typ = s0.typ.copy()
    mass = np.where(typ == 1, 1000.0, 1.0).astype(np.float32)
    zeros = np.zeros(len(pos3), np.float32)

    def lattice(ni, nj, nk):
        i, j, k = np.meshgrid(np.arange(ni) + 0.5, np.arange(nj) + 0.5,
                              np.arange(nk) + 0.5, indexing="ij")
        return np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(np.float32)

    arrs = {"positions": pos3, "velocities": np.zeros_like(pos3),
            "particle_phase": typ.astype(np.float32), "mass": mass,
            "momx": zeros, "momy": zeros, "momz": zeros,
            "probes_u": lattice(NX + 1, NY, 1), "probes_v": lattice(NX, NY + 1, 1),
            "probes_w": lattice(NX, NY, 2)}
    for k, v in arrs.items():
        np.save(os.path.join(d, k + ".npy"), v)
    args = [os.path.join(HERE, "run_sim_3d.bat"),
            "--set-port", "nx", str(NX), "--set-port", "ny", str(NY),
            "--set-port", "nz", "1", "--set-port", "dt", str(DT),
            "--set-port", "substeps", str(sub), "--set-port", "preconditioner", "1",
            "--set-port", "rho0_face", str(s0.rho0_face),
            "--set-port", "pos_pattern", (d + "/frame.####").replace("\\", "/"),
            "--set-port", "esc_pattern", (d + "/esc.####").replace("\\", "/"),
            "--set-port", "diag_pattern", (d + "/diag.####").replace("\\", "/"),
            "--set-port", "path_final_positions", (d + "/final.npy").replace("\\", "/")]
    for k, v in ESC.items():
        args += ["--set-port", k, str(v)]
    for k in arrs:
        args += ["--set-port", "path_" + k, os.path.join(d, k + ".npy").replace("\\", "/")]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(os.path.join(d, f"frame.{sub - 1:04d}.npy")):
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-2000:])
        raise SystemExit("bifcmd escape loop failed")

    # python twin at the same fixed dt from the same seed.
    sp = Sim(NX, NY, escape=True)
    sp.seed(mask)
    sp.calibrate()
    p_frames, p_esc = [], []
    for _ in range(sub):
        sp.step(DT)
        p_frames.append(sp.pos.copy())
        p_esc.append(sp.escaped.copy())

    cb, cp = [], []
    frames_dir = os.path.join(d, "png")
    os.makedirs(frames_dir, exist_ok=True)
    head, pad = 40, 8
    out_i = 0
    for k in range(sub):
        be = np.load(os.path.join(d, f"esc.{k:04d}.npy")) > 0.5
        cb.append((int((be & (typ == 1)).sum()), int((be & (typ == 0)).sum())))
        cp.append((int((p_esc[k] & (typ == 1)).sum()), int((p_esc[k] & (typ == 0)).sum())))
        if k % 4 == 0:
            bp = np.load(os.path.join(d, f"frame.{k:04d}.npy"))[:, :2]
            rows = [("python reference", p_frames[k], p_esc[k]),
                    ("Bifrost, full loop", bp, be)]
            imgs = [render(p, typ, e) for _, p, e in rows]
            w, h = imgs[0].size
            sheet = Image.new("RGB", (w * 2 + pad * 3, h + head + pad), "white")
            dr = ImageDraw.Draw(sheet)
            for i, ((lab, *_), im) in enumerate(zip(rows, imgs)):
                dr.text((pad + i * (w + pad), 8), lab, fill=(10, 10, 10), font=FONT)
                sheet.paste(im, (pad + i * (w + pad), head))
            sheet.save(os.path.join(frames_dir, f"fr{out_i:04d}.png"))
            out_i += 1
    cb, cp = np.array(cb), np.array(cp)
    print(f"  peak droplets python {cp[:, 0].max()}, bifrost {cb[:, 0].max()}; "
          f"peak bubbles python {cp[:, 1].max()}, bifrost {cb[:, 1].max()}")
    video = os.path.join(RESULTS, "escape_bifrost_vs_python.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                    "-i", os.path.join(frames_dir, "fr%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video],
                   check=True, capture_output=True)
    print("wrote", os.path.normpath(video))
    ok = (cb[:, 0].max() > 30 and cb[:, 1].max() > 30 and
          0.4 < cb[:, 0].max() / max(cp[:, 0].max(), 1) < 2.5 and
          0.4 < cb[:, 1].max() / max(cp[:, 1].max(), 1) < 2.5)
    print("loop:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("lockstep", "all"):
        ok = lockstep() and ok
    if which in ("loop", "all"):
        ok = loop() and ok
    print("OVERALL:", "PASS" if ok else "FAIL")
