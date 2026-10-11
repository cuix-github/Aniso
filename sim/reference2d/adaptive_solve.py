"""Milestone 6.5a, stage 1: the adaptive pressure solve's math, in 2D numpy.

The discretization is finite-volume on a graded quadtree: cells of size
2^level (fine level 0), 2:1 balanced, flux between neighbors i,j of
beta * A / d with A the shared (finer) face length and d the axis distance
between cell centers (s for same-level faces, 1.5 s_fine across a junction,
the classic graded-octree stencil). A T-junction is decomposed into its fine
sub-faces, each connecting the coarse cell to one fine neighbor, so the
operator is symmetric and conservative by construction: every coefficient
appears once as +c to the pair and -c to each diagonal, row sums are zero,
and the flux leaving one cell is exactly the flux entering the other.

Refined everywhere, the assembly reduces to the uniform 5-point operator -
validate_adaptive_solve.py checks that to rounding, anchors the uniform
operator against pfflip2d.project itself, and measures everything else.

Pure numpy; the solver is Jacobi-preconditioned CG with the constant null
space projected out (all-Neumann walls, as the uniform reference).
"""
import numpy as np


# ---- grid construction ----------------------------------------------------

def build_cells(nx, ny, levels, target):
    """Graded quadtree over an nx x ny fine grid (both multiples of
    2^levels). target[i, j] is the finest level required at fine cell
    (i, j): 0 = must stay fine, 1 = may be 2x, ... A cell at level l is
    kept only if every fine cell it covers allows level >= l. Returns a
    list of (level, i, j) with i, j in that level's own index units."""
    assert nx % (1 << levels) == 0 and ny % (1 << levels) == 0
    cells = [(levels, i, j) for i in range(nx >> levels)
             for j in range(ny >> levels)]
    # coarse-to-fine sweep: split any cell whose footprint demands finer
    for cur in range(levels, 0, -1):
        nxt = []
        for (l, i, j) in cells:
            if l != cur:
                nxt.append((l, i, j))
                continue
            s = 1 << l
            if target[i * s:(i + 1) * s, j * s:(j + 1) * s].min() < l:
                nxt += [(l - 1, 2 * i + a, 2 * j + b)
                        for a in (0, 1) for b in (0, 1)]
            else:
                nxt.append((l, i, j))
        cells = nxt
    return _grade(cells, nx, ny)


def _grade(cells, nx, ny):
    """Enforce 2:1 balance: no cell may touch a neighbor more than one
    level finer. Splits offenders until a fixpoint."""
    while True:
        index = {c: True for c in cells}
        split = set()
        for (l, i, j) in cells:
            if l == 0:
                continue
            s = 1 << l
            # a neighbor at level <= l-2 across any edge means imbalance;
            # brute edge scan at fine resolution of the four sides
            fi0, fi1 = i * s, (i + 1) * s
            fj0, fj1 = j * s, (j + 1) * s
            for (fi, fj_lo, fj_hi, horiz) in (
                    (fi0 - 1, fj0, fj1, True), (fi1, fj0, fj1, True)):
                if 0 <= fi < nx:
                    for fj in range(fj_lo, fj_hi):
                        if _level_at(index, fi, fj, l) <= l - 2:
                            split.add((l, i, j))
                            break
            for (fj, fi_lo, fi_hi) in ((fj0 - 1, fi0, fi1), (fj1, fi0, fi1)):
                if 0 <= fj < ny:
                    for fi in range(fi_lo, fi_hi):
                        if _level_at(index, fi, fj, l) <= l - 2:
                            split.add((l, i, j))
                            break
        if not split:
            return cells
        nxt = []
        for c in cells:
            if c in split:
                (l, i, j) = c
                nxt += [(l - 1, 2 * i + a, 2 * j + b)
                        for a in (0, 1) for b in (0, 1)]
            else:
                nxt.append(c)
        cells = nxt


def _level_at(index, fi, fj, lmax):
    """Level of the cell covering fine coordinate (fi, fj), searching from
    fine upward; lmax bounds the search."""
    for l in range(0, lmax + 3):
        if (l, fi >> l, fj >> l) in index:
            return l
    return lmax + 3


# ---- assembly ---------------------------------------------------------------

class Operator:
    """Symmetric graded-grid Poisson operator with Jacobi-PCG."""

    def __init__(self, cells, beta_face):
        """beta_face(x, y, axis, span) -> coefficient at a face whose
        midpoint is (x, y), axis 0 for x-faces and 1 for y-faces, spanning
        `span` unit faces (so array-backed betas can average), fine units."""
        self.cells = cells
        self.index = {c: n for n, c in enumerate(cells)}
        n = len(cells)
        self.size = np.array([float(1 << l) for (l, _, _) in cells])
        self.vol = self.size ** 2
        self.center = np.array([[(i + 0.5) * (1 << l), (j + 0.5) * (1 << l)]
                                for (l, i, j) in cells])
        # Variational assembly: each face contributes the energy
        # (beta A / d) * Delta^2, where Delta is an interpolated pressure
        # difference across the face (weights sum to zero), so
        # A = G^T beta G is symmetric PSD by construction and A(const) = 0.
        # Across a junction, the coarse-side sample is offset-corrected:
        # the fine center sits s/2 off the coarse axis, a quarter of the
        # coarse cell, so the sample is 3/4 the junction's coarse cell plus
        # 1/4 its lateral same-level neighbor on the fine cell's side. That
        # alignment is what removes the naive stencil's first-order lateral
        # error (measured at dt*g/3 in the hydrostatic gate before the fix).
        # The lateral neighbor may itself sit at another level (a corner of
        # the refinement region): the sample then interpolates toward that
        # neighbor's center with a weight set by the true center distance
        # (s/2 over D, exact for linear fields in every graded case), and a
        # refined lateral region contributes its aligned child pair, half
        # the lateral weight each. Only at domain walls, where no lateral
        # exists at all, does the sample fall back to the plain coarse
        # value - locally first order, with zero lateral gradient there
        # under the Neumann condition.
        self.faces = []   # (ent [(idx, w)], beta*A/d, axis, area, kind, fx, fy)

        def lateral(lkey, lat_dir, lat_axis):
            """Candidates for the coarse cell lkey's lateral neighbor in
            direction lat_dir along lat_axis: same level, one coarser, or
            the aligned pair of its refined children. Returns (entries
            summing to 1, center distance D in fine units) or None."""
            (lc, ki, kj) = lkey
            di, dj = (0, lat_dir) if lat_axis == 1 else (lat_dir, 0)
            same = (lc, ki + di, kj + dj)
            if same in self.index:
                return [(self.index[same], 1.0)], float(1 << lc)
            coarser = (lc + 1, (ki + di) >> 1, (kj + dj) >> 1)
            if coarser in self.index and (lc + 1, ki >> 1, kj >> 1) != coarser:
                return [(self.index[coarser], 1.0)], 1.5 * (1 << lc)
            fl = lc - 1
            if fl >= 0:
                if lat_axis == 1:
                    row = 2 * (kj + dj) if lat_dir > 0 else 2 * kj - 1
                    pair = [(fl, 2 * ki, row), (fl, 2 * ki + 1, row)]
                else:
                    col = 2 * (ki + di) if lat_dir > 0 else 2 * ki - 1
                    pair = [(fl, col, 2 * kj), (fl, col, 2 * kj + 1)]
                if all(k in self.index for k in pair):
                    return ([(self.index[k], 0.5) for k in pair],
                            1.5 * (1 << fl))
            return None

        def junction(me, s, nbc_key, lat_dir, lat_axis, bface, axis, sgn,
                     fx, fy):
            # weights oriented so Delta = p(positive side) - p(negative side)
            nbc = self.index[nbc_key]
            found = lateral(nbc_key, lat_dir, lat_axis)
            ent = [(me, -sgn)]
            if found is not None:
                lat_ent, D = found
                w = (s / 2.0) / D
                ent += [(nbc, (1.0 - w) * sgn)]
                ent += [(idx, w * frac * sgn) for (idx, frac) in lat_ent]
            else:
                ent += [(nbc, 1.0 * sgn)]
            self.faces.append((ent, bface * s / (1.5 * s), axis, s,
                               "junction", fx, fy))

        for (l, i, j) in cells:
            me = self.index[(l, i, j)]
            s = 1 << l
            fj0 = j * s                      # my fine j origin
            fi0 = i * s
            nb = self.index.get((l, i + 1, j))
            if nb is not None:
                self.faces.append(
                    ([(me, -1.0), (nb, 1.0)],
                     beta_face((i + 1) * s, (j + 0.5) * s, 0, s), 0, s, "same",
                     (i + 1) * s, (j + 0.5) * s))
            for (fi, sign) in (((i + 1) * s, +1), (i * s - 1, -1)):
                key = (l + 1, fi >> (l + 1), fj0 >> (l + 1))
                if key in self.index:
                    x = (i + 1) * s if sign > 0 else i * s
                    up = (fj0 >> (l + 1) << (l + 1)) != fj0  # upper child?
                    junction(me, s, key, 1 if up else -1, 1,
                             beta_face(x, (j + 0.5) * s, 0, s), 0, float(sign),
                             x, (j + 0.5) * s)
            nb = self.index.get((l, i, j + 1))
            if nb is not None:
                self.faces.append(
                    ([(me, -1.0), (nb, 1.0)],
                     beta_face((i + 0.5) * s, (j + 1) * s, 1, s), 1, s, "same",
                     (i + 0.5) * s, (j + 1) * s))
            for (fj, sign) in (((j + 1) * s, +1), (j * s - 1, -1)):
                key = (l + 1, fi0 >> (l + 1), fj >> (l + 1))
                if key in self.index:
                    y = (j + 1) * s if sign > 0 else j * s
                    right = (fi0 >> (l + 1) << (l + 1)) != fi0
                    junction(me, s, key, 1 if right else -1, 0,
                             beta_face((i + 0.5) * s, y, 1, s), 1, float(sign),
                             (i + 0.5) * s, y)

        rows, cols, coefs = [], [], []
        for ent, c, _axis, _area, _kind, _fx, _fy in self.faces:
            for (a, wa) in ent:
                for (b, wb) in ent:
                    rows.append(a), cols.append(b), coefs.append(c * wa * wb)
        self.rows = np.array(rows, np.int64)
        self.cols = np.array(cols, np.int64)
        self.coefs = np.array(coefs)
        self.diag = np.zeros(n)
        mask = self.rows == self.cols
        np.add.at(self.diag, self.rows[mask], self.coefs[mask])

    def rhs(self, u, v, dt):
        """The adjoint-consistent divergence of uniform face fields: the
        variational projection's b is G^T (A u*), so a junction face
        distributes its flux over all its interpolation cells with the same
        weights the gradient uses. On an all-fine grid this reduces exactly
        to the geometric -div/dt. Mixing conventions makes the system
        inconsistent (measured as 6 percent hydrostatic junction residuals
        from an operator whose own consistency error was 4e-4)."""
        b = np.zeros(len(self.cells))
        for ent, _c, axis, area, _kind, fx, fy in self.faces:
            if axis == 0:
                ys = (fy - area / 2.0 + np.arange(area)).astype(int)
                ustar = float(u[int(round(fx)), ys].mean())
            else:
                xs = (fx - area / 2.0 + np.arange(area)).astype(int)
                ustar = float(v[xs, int(round(fy))].mean())
            for (idx, w) in ent:
                b[idx] += area * ustar * w / dt
        return b

    def matvec(self, x):
        out = np.zeros_like(x)
        np.add.at(out, self.rows, self.coefs * x[self.cols])
        return out

    def solve(self, b, tol=1e-10, max_iter=20000):
        # all-Neumann: null space is the constant vector, so project the
        # incompatible part out of b (A's row sums are zero, range _|_ 1)
        b = b - b.mean()
        dinv = np.where(self.diag > 0, 1.0 / np.maximum(self.diag, 1e-300), 0.0)
        x = np.zeros_like(b)
        r = b.copy()
        z = dinv * r
        d = z.copy()
        rz = r @ z
        r0 = max(np.sqrt(r @ r), 1e-300)
        it = 0
        for it in range(max_iter):
            if np.sqrt(r @ r) / r0 < tol:
                break
            q = self.matvec(d)
            a = rz / max(d @ q, 1e-300)
            x += a * d
            r -= a * q
            z = dinv * r
            rz_new = r @ z
            d = z + (rz_new / max(rz, 1e-300)) * d
            rz = rz_new
        x -= x.mean()
        return x, it


# ---- stage 2: multigrid on the adaptive hierarchy -------------------------

def coarsen(cells):
    """One hierarchy level up: every cell collapses into its parent (cells
    already at the coarser level pass through as themselves). Grading is
    preserved because all relative level differences are unchanged."""
    out = {}
    for (l, i, j) in cells:
        out[(l + 1, i >> 1, j >> 1)] = True
    return list(out.keys())


class Multigrid:
    """Flexible (Polak-Ribiere) PCG with a geometric V-cycle preconditioner
    over the collapsed-quadtree hierarchy. Every level is a stage-1 Operator
    assembled by the same variational rules, so the validated junction
    treatment holds on all levels; transfers are the volume-integrated
    naturals: residuals sum into parents, corrections inject into children.
    Smoother and constants mirror the production 3D solver (damped Jacobi
    omega = 0.8, 2+2 sweeps, 40 at the coarsest)."""

    def __init__(self, cells, beta_face, max_levels=12, coarsest=64):
        self.ops = [Operator(cells, beta_face)]
        self.parent = []
        level_cells = cells
        while len(self.ops) < max_levels and len(level_cells) > coarsest:
            coarse = coarsen(level_cells)
            cmap = {c: k for k, c in enumerate(coarse)}
            self.parent.append(np.array(
                [cmap[(l + 1, i >> 1, j >> 1)] for (l, i, j) in level_cells],
                np.int64))
            self.ops.append(Operator(coarse, beta_face))
            level_cells = coarse

    def _smooth(self, op, x, b, sweeps):
        dinv = np.where(op.diag > 0, 1.0 / np.maximum(op.diag, 1e-300), 0.0)
        for _ in range(sweeps):
            x += 0.8 * dinv * (b - op.matvec(x))
        return x

    def _vcycle(self, k, b):
        op = self.ops[k]
        x = np.zeros_like(b)
        if k + 1 == len(self.ops):
            return self._smooth(op, x, b, 40)
        x = self._smooth(op, x, b, 2)
        r = b - op.matvec(x)
        bc = np.zeros(len(self.ops[k + 1].cells))
        np.add.at(bc, self.parent[k], r)
        x += self._vcycle(k + 1, bc)[self.parent[k]]
        return self._smooth(op, x, b, 2)

    def solve(self, b, tol=1e-9, max_iter=2000):
        op = self.ops[0]
        b = b - b.mean()
        x = np.zeros_like(b)
        r = b.copy()
        z = self._vcycle(0, r)
        d = z.copy()
        rz = r @ z
        r0 = max(np.sqrt(r @ r), 1e-300)
        it = 0
        for it in range(max_iter):
            if np.sqrt(r @ r) / r0 < tol:
                break
            q = op.matvec(d)
            dq = d @ q
            if dq == 0.0:
                break
            a = rz / dq
            x += a * d
            r_prev = r.copy()
            r -= a * q
            z = self._vcycle(0, r)
            rz_new = max(0.0, (r - r_prev) @ z)   # flexible: tolerates V-cycle
            d = z + (rz_new / max(rz, 1e-300)) * d
            rz = r @ z
            if rz == 0.0:
                break
        x -= x.mean()
        return x, it
