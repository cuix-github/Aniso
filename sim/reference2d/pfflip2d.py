"""A minimal 2D reference implementation of the core PF-FLIP loop, following Section 3 of
"Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation" (Braun, Bender,
Thuerey, SIGGRAPH 2025). This is the executable spec for the Bifrost build: no adaptivity, no
escaped-particle droplets, no divergence correction — just the two-phase heart of the method.

Both phases are particles. Each step: splat mass and momentum to MAC faces with the paper's
smooth kernel (Eq. 6), read the phase field off the accumulated face masses (Eq. 7), add
gravity, solve the variable-coefficient pressure Poisson equation (Eq. 8) with face
coefficients 1/rho(phi), correct velocities, FLIP-update the particles with a per-phase blend
(Eq. 12, which doubles as viscosity via Eq. 13), and advect with RK2 through the grid field.

Scenes (run this file):  python pfflip2d.py [hydrostatic|dam_break|rayleigh_taylor|all]
Outputs go to results/: a frame-strip PNG per scene plus printed pass/fail style numbers.
Needs numpy and Pillow only.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

# ------------------------------------------------------------------ simulation core


class Sim:
    def __init__(self, nx, ny, rho_l=1000.0, rho_g=1.0, alpha_l=0.97, alpha_g=0.9, g=-9.8,
                 escape=False, esc_phi=0.3, drag_droplet=1.0, drag_bubble=8.0, buoyancy=2.0,
                 st=False, sub_advect=False, adapt=False):
        self.nx, self.ny, self.dx = nx, ny, 1.0
        self.rho_l, self.rho_g = rho_l, rho_g
        self.alpha = {1: alpha_l, 0: alpha_g}
        self.g = g
        # Escaped-particle treatment (the paper's droplets and bubbles), OFF by default
        # so every validated behaviour is unchanged. A liquid particle where the phase
        # field says air is a droplet (ballistic, gravity + drag); an air particle where
        # it says liquid is a bubble (buoyancy + drag). Both skip the FLIP blend and the
        # grid advection until they rejoin their own phase.
        self.escape, self.esc_phi = escape, esc_phi
        self.drag_droplet, self.drag_bubble, self.buoyancy = drag_droplet, drag_bubble, buoyancy
        self.escaped = np.zeros(0, dtype=bool)
        # ST-FLIP (Braun et al. 2026) spatiotemporal sampling, OFF by default. Each
        # particle carries a time jitter tau in (-1/2, 1/2] slab units; splats are
        # weighted by the paper's one-sided temporal poly6 kernel (peak at +1/2), and
        # advection folds the jitter change into one per-particle step of length
        # dt*(1 + tau_new - tau_old), sub-stepped at local CFL <= 1. Spatial kernel
        # deviation from the paper (ours is radial, theirs separable) noted; the
        # milestone-4 finding says kernel shape barely moves the fields.
        self.st = st
        self.sub_advect = sub_advect
        # Adaptive particles (paper Section 5, 2D spec): air particles in deep air
        # merge 4-to-1 into scale-2 particles (mass x4, splat radius x2); coarse
        # particles split back near the interface or walls. The per-size blend
        # correction equalizes viscosity: (1-alpha_coarse) = (1-alpha_fine)/scale^2.
        self.adapt = adapt
        self.scale = np.zeros(0)
        self.coarse_gain = 1.0
        self.st_rng = np.random.default_rng(7)
        self.tau = np.zeros(0)
        self.u = np.zeros((nx + 1, ny))   # x-velocity on vertical faces
        self.v = np.zeros((nx, ny + 1))   # y-velocity on horizontal faces
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.typ = np.zeros(0, dtype=np.int32)   # 1 liquid, 0 air
        self.rho0_face = None                    # calibrated splat mass of a full face

    # --- seeding: 2x2 particles per cell, jittered, everywhere; type from a mask function
    def seed(self, liquid_mask, jitter=0.35, seed=1):
        rng = np.random.default_rng(seed)
        cx, cy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        offs = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        ps, ts = [], []
        for ox, oy in offs:
            p = np.stack([cx + ox, cy + oy], -1).reshape(-1, 2).astype(np.float64)
            p += rng.uniform(-jitter, jitter, p.shape) * 0.5
            ps.append(p)
            ts.append(liquid_mask(p[:, 0], p[:, 1]).astype(np.int32))
        self.pos = np.concatenate(ps)
        self.typ = np.concatenate(ts)
        self.vel = np.zeros_like(self.pos)
        self.scale = np.ones(len(self.pos))
        self.tau = np.zeros(len(self.pos))      # particles start synchronized
        self.xi = np.ones(len(self.pos))        # per-particle jitter strength

    def masses(self):
        m = np.where(self.typ == 1, self.rho_l, self.rho_g)
        if self.adapt:
            m = m * self.scale ** 2 * np.where(self.scale > 1.5, self.coarse_gain, 1.0)
        return m

    # --- Eq. 6 kernel weights against a set of face centres, vectorised over a 3x3 stencil
    def _splat(self, centres_shape, face_of, values_list):
        """Splat each entry of values_list (per-particle scalars) onto faces. face_of maps a
        particle to its base face index. Uniform runs use the r = dx kernel on a 3x3
        stencil; adaptive runs splat each size group with its own radius."""
        out = [np.zeros(centres_shape) for _ in values_list]
        if self.adapt and len(self.scale) == len(face_of[0]):
            for sc in np.unique(self.scale):
                sel = self.scale == sc
                self._splat_group(out, centres_shape,
                                  (face_of[0][sel], face_of[1][sel]),
                                  [np.asarray(v)[sel] if np.ndim(v) else v
                                   for v in values_list], float(sc))
        else:
            self._splat_group(out, centres_shape, face_of, values_list, 1.0)
        return out

    @staticmethod
    def _splat_group(out, centres_shape, face_of, values_list, r):
        base, frac = face_of
        span = int(np.ceil(r))
        for di in range(-span, span + 1):
            for dj in range(-span, span + 1):
                ii = base[:, 0] + di
                jj = base[:, 1] + dj
                ok = (ii >= 0) & (ii < centres_shape[0]) & (jj >= 0) & (jj < centres_shape[1])
                d2 = ((frac[:, 0] - di) ** 2 + (frac[:, 1] - dj) ** 2) / r ** 2
                w = np.maximum(1.0 - d2, 0.0) ** 3
                w = np.where(ok, w, 0.0)
                iis = np.clip(ii, 0, centres_shape[0] - 1)
                jjs = np.clip(jj, 0, centres_shape[1] - 1)
                for o, val in zip(out, values_list):
                    np.add.at(o, (iis, jjs), w * val)

    def _face_frames(self):
        """Base indices and fractional offsets of every particle against u-faces and v-faces."""
        pu = self.pos - np.array([0.0, 0.5])   # u-face centres are at (i, j+0.5)
        pv = self.pos - np.array([0.5, 0.0])   # v-face centres are at (i+0.5, j)
        bu = np.floor(pu + 0.5).astype(np.int64)
        bv = np.floor(pv + 0.5).astype(np.int64)
        return (bu, pu - bu), (bv, pv - bv)

    @staticmethod
    def _wt(tau):
        """ST-FLIP temporal kernel: one-sided poly6 peaking at the slab end (+1/2)."""
        return (35.0 / 16.0) * np.maximum(1.0 - (tau - 0.5) ** 2, 0.0) ** 3

    _GL_NODES = 0.5 * np.polynomial.legendre.leggauss(8)[0]
    _GL_WEIGHTS = 0.5 * np.polynomial.legendre.leggauss(8)[1]

    def _wt_norm(self, xi):
        """E[wT(xi*T)], T~U(-1/2,1/2): the weight renormalization for attenuated
        jitter, so calm (xi<1) and fast (xi=1) regions deposit consistent mass."""
        taus = xi[:, None] * self._GL_NODES[None, :]
        return (self._wt(taus) * self._GL_WEIGHTS[None, :]).sum(1)

    def p2g(self):
        m = self.masses()
        if self.st:
            m = m * self._wt(self.tau) / np.maximum(self._wt_norm(self.xi), 1e-9)
        fu, fv = self._face_frames()
        mu, pu = self._splat(self.u.shape, fu, [m, m * self.vel[:, 0]])
        mv, pv = self._splat(self.v.shape, fv, [m, m * self.vel[:, 1]])
        self.mu, self.mv = mu, mv
        with np.errstate(divide="ignore", invalid="ignore"):
            self.u = np.where(mu > 0, pu / np.maximum(mu, 1e-300), 0.0)
            self.v = np.where(mv > 0, pv / np.maximum(mv, 1e-300), 0.0)
        self._enforce_walls()

    def _enforce_walls(self):
        self.u[0, :] = self.u[-1, :] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0

    # --- Eq. 7: the phase field from raw splatted face mass
    def phase(self, raw):
        if self.rho0_face is None:
            raise RuntimeError("call calibrate() after seeding")
        eta = np.log(self.rho_l / self.rho_g)
        rho_min = eta * self.rho_g * self.rho0_face
        phi = np.sqrt(np.maximum(raw - rho_min, 0.0) / (self.rho0_face * self.rho_l))
        return np.minimum(phi, 1.0)

    def calibrate(self):
        """rho0_face: the raw mass a face accumulates from a uniform all-liquid seeding,
        with the same temporal weighting the simulation will use (averaged over a few
        jitter realizations when ST sampling is on, per the ST-FLIP m0 recipe)."""
        probe = Sim(8, 8, self.rho_l, self.rho_g, st=self.st)
        probe.seed(lambda x, y: np.ones_like(x, dtype=bool), jitter=0.0)
        fu, _ = probe._face_frames()
        reps = 5 if self.st else 1
        acc = []
        for _ in range(reps):
            m = probe.masses()
            if self.st:
                probe.tau = probe.st_rng.uniform(-0.5, 0.5, len(probe.pos))
                m = m * self._wt(probe.tau)   # full-jitter reference scale (xi = 1)
            (mu,) = probe._splat(probe.u.shape, fu, [m])
            acc.append(np.median(mu[2:-2, 2:-2]))
        self.rho0_face = float(np.mean(acc)) / self.rho_l
        if self.adapt:
            # measure the raw face mass an all-coarse uniform seeding produces, and
            # gain-correct coarse masses so fine and coarse fields read identically.
            probe2 = Sim(8, 8, self.rho_l, self.rho_g)
            probe2.seed(lambda x, y: np.ones_like(x, dtype=bool), jitter=0.0)
            keep = (probe2.pos[:, 0] % 2 < 1) & (probe2.pos[:, 1] % 2 < 1)
            probe2.pos = probe2.pos[keep] + 0.5
            probe2.typ = probe2.typ[keep]
            probe2.adapt = True
            probe2.scale = np.full(len(probe2.pos), 2.0)
            probe2.coarse_gain = 1.0
            m2 = probe2.masses() * 4.0 / 4.0
            fu2, _ = probe2._face_frames()
            (mu2,) = probe2._splat(probe2.u.shape, fu2, [probe2.masses()])
            raw2 = np.median(mu2[2:-2, 2:-2]) / self.rho_l
            self.coarse_gain = float(np.mean(acc)) / self.rho_l / max(raw2, 1e-9)

    def project(self, dt, tol=1e-6, max_iter=4000):
        phi_u = self.phase(self.mu)
        phi_v = self.phase(self.mv)
        rho_u = self.rho_g + (self.rho_l - self.rho_g) * phi_u
        rho_v = self.rho_g + (self.rho_l - self.rho_g) * phi_v
        bu = 1.0 / rho_u
        bv = 1.0 / rho_v
        bu[0, :] = bu[-1, :] = 0.0     # closed walls
        bv[:, 0] = bv[:, -1] = 0.0

        div = (self.u[1:, :] - self.u[:-1, :]) + (self.v[:, 1:] - self.v[:, :-1])
        b = -div / dt

        def A(p):
            gpu = np.zeros_like(self.u)
            gpv = np.zeros_like(self.v)
            gpu[1:-1, :] = (p[1:, :] - p[:-1, :]) * bu[1:-1, :]
            gpv[:, 1:-1] = (p[:, 1:] - p[:, :-1]) * bv[:, 1:-1]
            return -((gpu[1:, :] - gpu[:-1, :]) + (gpv[:, 1:] - gpv[:, :-1]))

        diag = bu[1:, :] + bu[:-1, :] + bv[:, 1:] + bv[:, :-1]
        inv_diag = np.where(diag > 0, 1.0 / np.maximum(diag, 1e-300), 0.0)

        p = np.zeros((self.nx, self.ny))
        r = b - A(p)
        z = inv_diag * r
        d = z.copy()
        rz = float((r * z).sum())
        r0 = np.sqrt(float((r * r).sum())) or 1.0
        it = 0
        for it in range(max_iter):
            if np.sqrt(float((r * r).sum())) / r0 < tol:
                break
            q = A(d)
            a = rz / float((d * q).sum())
            p += a * d
            r -= a * q
            z = inv_diag * r
            rz_new = float((r * z).sum())
            d = z + (rz_new / rz) * d
            rz = rz_new
        self.iters = it

        self.u[1:-1, :] -= dt * bu[1:-1, :] * (p[1:, :] - p[:-1, :])
        self.v[:, 1:-1] -= dt * bv[:, 1:-1] * (p[:, 1:] - p[:, :-1])
        self._enforce_walls()

    def _sample_faces(self, grid, offset, pts):
        q = pts - offset
        b = np.floor(q).astype(np.int64)
        f = q - b
        out = np.zeros(len(pts))
        for di in (0, 1):
            for dj in (0, 1):
                w = (f[:, 0] if di else 1 - f[:, 0]) * (f[:, 1] if dj else 1 - f[:, 1])
                ii = np.clip(b[:, 0] + di, 0, grid.shape[0] - 1)
                jj = np.clip(b[:, 1] + dj, 0, grid.shape[1] - 1)
                out += w * grid[ii, jj]
        return out

    def sample_velocity(self, pts):
        return np.stack([self._sample_faces(self.u, np.array([0.0, 0.5]), pts),
                         self._sample_faces(self.v, np.array([0.5, 0.0]), pts)], -1)

    def g2p(self, u_old, v_old, skip=None):
        du, dv = self.u - u_old, self.v - v_old
        delta = np.stack([self._sample_faces(du, np.array([0.0, 0.5]), self.pos),
                          self._sample_faces(dv, np.array([0.5, 0.0]), self.pos)], -1)
        pic = self.sample_velocity(self.pos)
        a1 = np.where(self.typ == 1, self.alpha[1], self.alpha[0])
        if self.adapt:
            a1 = 1.0 - (1.0 - a1) / self.scale ** 2
        a = a1[:, None]
        blended = a * (self.vel + delta) + (1 - a) * pic
        if skip is not None:
            blended[skip] = self.vel[skip]
        self.vel = blended

    def advect(self, dt, ballistic=None):
        old = self.pos
        mid = self.pos + 0.5 * dt * self.sample_velocity(self.pos)
        self.pos = self.pos + dt * self.sample_velocity(np.clip(mid, 0.51, None))
        if ballistic is not None and ballistic.any():
            self.pos[ballistic] = old[ballistic] + dt * self.vel[ballistic]
        lo = 0.51
        hix, hiy = self.nx - 0.51, self.ny - 0.51
        for k, hi in ((0, hix), (1, hiy)):
            below, above = self.pos[:, k] < lo, self.pos[:, k] > hi
            self.pos[:, k] = np.clip(self.pos[:, k], lo, hi)
            self.vel[below | above, k] = 0.0

    def step(self, dt):
        self.p2g()
        u_star, v_star = self.u.copy(), self.v.copy()
        self.v += self.g * dt          # gravity on y-faces
        self._enforce_walls()
        self.project(dt)
        esc = None
        if self.escape:
            # Escape detection from the OTHER phase's local number density, which is
            # self-excluding: a particle's own splat cannot mask it (phi-based
            # detection fails for droplets because a lone liquid particle's own mass,
            # through Eq. 7's square root, still reads as phi ~ 0.45).
            w_rep = self.scale ** 2 if self.adapt else np.ones(len(self.typ))
            ones_air = (self.typ == 0).astype(float) * w_rep
            ones_liq = ((self.typ == 1).astype(float)) * w_rep
            fu, fv = self._face_frames()
            au, lu = self._splat(self.u.shape, fu, [ones_air, ones_liq])
            av, lv = self._splat(self.v.shape, fv, [ones_air, ones_liq])
            thresh = (1.0 - self.esc_phi) * self.rho0_face
            frac_air = 0.5 * (self._sample_faces(au, np.array([0.0, 0.5]), self.pos) +
                              self._sample_faces(av, np.array([0.5, 0.0]), self.pos))
            frac_liq = 0.5 * (self._sample_faces(lu, np.array([0.0, 0.5]), self.pos) +
                              self._sample_faces(lv, np.array([0.5, 0.0]), self.pos))
            drop = (self.typ == 1) & (frac_air > thresh)
            bub = (self.typ == 0) & (frac_liq > thresh)
            esc = drop | bub
            self.escaped = esc
        self.g2p(u_star, v_star, skip=esc)
        if esc is not None and esc.any():
            vg = self.sample_velocity(self.pos)
            d = drop
            self.vel[d, 1] += self.g * dt
            self.vel[d] += self.drag_droplet * dt * (vg[d] - self.vel[d])
            b = bub
            self.vel[b, 1] += self.buoyancy * abs(self.g) * dt
            self.vel[b] += self.drag_bubble * dt * (vg[b] - self.vel[b])
        if self.st or self.sub_advect:
            self._advect_st(dt, ballistic=esc)
        else:
            self.advect(dt, ballistic=esc)
        if self.adapt:
            self._adapt_particles()

    def _liq_frac(self):
        ones_liq = (self.typ == 1).astype(float) * self.scale ** 2
        fu, fv = self._face_frames()
        (lu,) = self._splat(self.u.shape, fu, [ones_liq])
        (lv,) = self._splat(self.v.shape, fv, [ones_liq])
        return 0.5 * (self._sample_faces(lu, np.array([0.0, 0.5]), self.pos) +
                      self._sample_faces(lv, np.array([0.5, 0.0]), self.pos)) / self.rho0_face

    def _adapt_particles(self, merge_thresh=0.02, split_thresh=0.05, wall_margin=2.0):
        lf = self._liq_frac()
        # ---- split coarse near the interface or walls
        near_wall = ((self.pos[:, 0] < wall_margin) | (self.pos[:, 0] > self.nx - wall_margin) |
                     (self.pos[:, 1] < wall_margin) | (self.pos[:, 1] > self.ny - wall_margin))
        to_split = (self.scale > 1.5) & ((lf > split_thresh) | near_wall)
        if to_split.any():
            base = self.pos[to_split]
            offs = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
            new_pos = (base[:, None, :] + offs[None, :, :]).reshape(-1, 2)
            new_vel = np.repeat(self.vel[to_split], 4, axis=0)
            keep = ~to_split
            self.pos = np.concatenate([self.pos[keep], new_pos])
            self.vel = np.concatenate([self.vel[keep], new_vel])
            self.typ = np.concatenate([self.typ[keep], np.repeat(self.typ[to_split], 4)])
            self.scale = np.concatenate([self.scale[keep], np.ones(len(new_pos))])
            self.tau = np.concatenate([self.tau[keep], np.repeat(self.tau[to_split], 4)])
            self.xi = np.concatenate([self.xi[keep] if len(self.xi) == len(keep) else
                                      np.ones(keep.sum()), np.ones(len(new_pos))])
            lf = np.concatenate([lf[keep], np.full(len(new_pos), 1.0)])  # keep them fine this step
        # ---- merge deep-air fine 4-to-1 per 2x2 block
        cand = (self.scale < 1.5) & (self.typ == 0) & (lf < merge_thresh)
        if cand.sum() >= 4:
            idx = np.where(cand)[0]
            blocks = (self.pos[idx, 0] // 2).astype(np.int64) * 100000 + \
                     (self.pos[idx, 1] // 2).astype(np.int64)
            order = np.argsort(blocks, kind="stable")
            idx, blocks = idx[order], blocks[order]
            _, starts, counts = np.unique(blocks, return_index=True, return_counts=True)
            merged_members, new_p, new_v, new_tau = [], [], [], []
            for st_i, ct in zip(starts, counts):
                take = (ct // 4) * 4
                for g in range(0, take, 4):
                    mem = idx[st_i + g: st_i + g + 4]
                    merged_members.append(mem)
                    new_p.append(self.pos[mem].mean(0))
                    new_v.append(self.vel[mem].mean(0))
                    new_tau.append(self.tau[mem].mean())
            if merged_members:
                drop = np.concatenate(merged_members)
                keep = np.ones(len(self.pos), bool)
                keep[drop] = False
                self.pos = np.concatenate([self.pos[keep], np.array(new_p)])
                self.vel = np.concatenate([self.vel[keep], np.array(new_v)])
                self.typ = np.concatenate([self.typ[keep],
                                           np.zeros(len(new_p), dtype=np.int32)])
                self.scale = np.concatenate([self.scale[keep], np.full(len(new_p), 2.0)])
                self.tau = np.concatenate([self.tau[keep], np.array(new_tau)])
                self.xi = np.concatenate([self.xi[keep] if len(self.xi) == len(keep) else
                                          np.ones(int(keep.sum())), np.ones(len(new_p))])

    def _advect_st(self, dt, ballistic=None, cfl_local=1.0, max_rounds=64):
        """ST-FLIP advection: each particle advances by dt*(1 + tau_new - tau_old),
        folding the jitter change into one step, sub-stepped at local CFL <= 1."""
        if self.st:
            # Velocity-adaptive jitter attenuation (the paper's calm-water fix):
            # full jitter where a particle crosses a cell per step, none where the
            # flow is still, via a smoothstep of the local per-particle CFL.
            cfl_p = np.abs(self.vel).max(1) * dt / self.dx
            t = np.clip(cfl_p, 0.0, 1.0)
            self.xi = t * t * (3.0 - 2.0 * t)
            tau_new = self.xi * self.st_rng.uniform(-0.5, 0.5, len(self.pos))
            remaining = dt * (1.0 + tau_new - self.tau)
            self.tau = tau_new
        else:
            remaining = np.full(len(self.pos), float(dt))
        if ballistic is not None and ballistic.any():
            # escaped particles fly on their own velocity for their full jittered step
            self.pos[ballistic] += remaining[ballistic, None] * self.vel[ballistic]
            remaining = np.where(ballistic, 0.0, remaining)
        for _ in range(max_rounds):
            active = remaining > 1e-12
            if not active.any():
                break
            v = self.sample_velocity(self.pos)
            speed = np.maximum(np.abs(v).max(1), 1e-9)
            dt_sub = np.where(active, np.minimum(remaining, cfl_local * self.dx / speed), 0.0)
            mid = self.pos + 0.5 * dt_sub[:, None] * v
            v2 = self.sample_velocity(np.clip(mid, 0.51, None))
            self.pos = self.pos + dt_sub[:, None] * v2
            remaining = remaining - dt_sub
        lo = 0.51
        hix, hiy = self.nx - 0.51, self.ny - 0.51
        for k, hi in ((0, hix), (1, hiy)):
            below, above = self.pos[:, k] < lo, self.pos[:, k] > hi
            self.pos[:, k] = np.clip(self.pos[:, k], lo, hi)
            self.vel[below | above, k] = 0.0

    def run(self, frames, dt_frame, cfl=0.5, on_frame=None):
        for f in range(frames):
            remaining = dt_frame
            while remaining > 1e-9:
                vmax = max(1e-6, float(np.abs(self.vel).max()))
                dt = min(remaining, cfl * self.dx / vmax, dt_frame / 2)
                self.step(dt)
                remaining -= dt
            if on_frame:
                on_frame(f, self)


# ------------------------------------------------------------------ rendering helpers

def draw(sim, scale=3):
    img = np.full((sim.nx * scale, sim.ny * scale, 3), 245, np.uint8)
    pix = (sim.pos * scale).astype(np.int64)
    pix[:, 0] = np.clip(pix[:, 0], 0, sim.nx * scale - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, sim.ny * scale - 1)
    air = sim.typ == 0
    img[pix[air, 0], pix[air, 1]] = (210, 210, 215)
    liq = sim.typ == 1
    img[pix[liq, 0], pix[liq, 1]] = (30, 90, 200)
    return Image.fromarray(np.rot90(img))


def save_strip(frames_imgs, path, cols):
    w, h = frames_imgs[0].size
    rows = (len(frames_imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(frames_imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)
    print("wrote", path)


# ------------------------------------------------------------------ scenes

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def hydrostatic():
    sim = Sim(64, 64)
    sim.seed(lambda x, y: y < 32)
    sim.calibrate()
    vmax_log = []
    sim.run(30, 0.05, on_frame=lambda f, s: vmax_log.append(float(np.abs(s.vel).max())))
    print("hydrostatic: max |v| over last 5 frames = %.4g (want: small, no blow-up); iters/frame ~%d"
          % (max(vmax_log[-5:]), sim.iters))
    return max(vmax_log[-5:]) < 1.0


def dam_break():
    sim = Sim(160, 80)
    sim.seed(lambda x, y: (x < 40) & (y < 56))
    sim.calibrate()
    shots, fronts = [], []

    def cb(f, s):
        liq_x = s.pos[s.typ == 1, 0]
        fronts.append(float(np.percentile(liq_x, 99.5)))
        if f % 4 == 0:
            shots.append(draw(s))

    sim.run(41, 0.06, on_frame=cb)
    save_strip(shots, os.path.join(RESULTS, "dam_break.png"), cols=3)
    mono = all(b >= a - 1.0 for a, b in zip(fronts, fronts[1:]))
    print("dam_break: front x from %.1f to %.1f of 160 (want: advancing); monotonic-ish: %s"
          % (fronts[0], max(fronts), mono))
    return mono and max(fronts) > 80  # mildly damped by the FLIP blend; tune alpha when matching Bifrost


def rayleigh_taylor():
    sim = Sim(64, 192, rho_l=19.0, rho_g=1.0, alpha_l=0.97, alpha_g=0.97)  # Atwood 0.9, as the paper's RT setup
    mid, amp = 96, 3.0
    sim.seed(lambda x, y: y > mid + amp * np.cos(2 * np.pi * x / 64))       # heavy phase on top
    sim.calibrate()
    shots, tips = [], []

    def cb(f, s):
        heavy_y = s.pos[s.typ == 1, 1]
        tips.append(float(np.percentile(heavy_y, 0.5)))
        if f % 5 == 0:
            shots.append(draw(s, scale=2))

    sim.run(46, 0.12, on_frame=cb)
    save_strip(shots, os.path.join(RESULTS, "rayleigh_taylor.png"), cols=5)
    print("rayleigh_taylor: heavy-phase tip fell from y=%.1f to y=%.1f (want: fingers growing downward)"
          % (tips[0], min(tips)))
    return min(tips) < tips[0] - 30


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    ok = True
    for name, fn in (("hydrostatic", hydrostatic), ("dam_break", dam_break),
                     ("rayleigh_taylor", rayleigh_taylor)):
        if which in (name, "all"):
            print("===", name)
            ok = fn() and ok
    print("total %.1f s, overall: %s" % (time.time() - t0, "PASS" if ok else "CHECK OUTPUTS"))
