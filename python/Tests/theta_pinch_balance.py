"""ThetaPinchPressureBalance, written again in numpy as a manta.EquilibriumModel.

A port, not a new model: the same closed-form balance on the transport's own
p_h, the same 16-point Gauss rule per cell, the same patch-of-cell-means
recovery of p_x and the same safeguarded Newton for the total pressure C
(PhysicsCases/ThetaPinchPressureBalance.cpp has the reasoning). So a run with
this model is the run with the C++ one to round-off, which is what lets
test_field_coupling.py check the Python EquilibriumModel path against a model
whose numbers are already pinned.

On every flux surface p + B^2 = C, so with B = Phi_a / V'

    V'           = Phi_a / sqrt(C - p)
    dV'/dx       = (Phi_a / 2) p_x (C - p)^(-3/2)
    <|grad x|^2> = 4 pi A / V'^2,     A(x) = Int_0^x V'
    B            = Phi_a / V'

and the wall fixes Int_0^1 V' dx = pi a^2, one equation for C.
"""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

import manta

XG, WG = leggauss(16)


def lagrange(nodes, x):
    """The Lagrange basis on `nodes` at x, and its derivative."""
    n = len(nodes)
    l = np.ones(n)
    dl = np.zeros(n)
    for j in range(n):
        for m in range(n):
            if m != j:
                l[j] *= (x - nodes[m]) / (nodes[j] - nodes[m])
        for r in range(n):
            if r == j:
                continue
            term = 1.0 / (nodes[j] - nodes[r])
            for m in range(n):
                if m != j and m != r:
                    term *= (x - nodes[m]) / (nodes[j] - nodes[m])
            dl[j] += term
    return l, dl


class Layout:
    """Where p_h lives, and the gradient recovery: fixed for a run."""

    def __init__(self, points, lo, hi):
        self.points = np.asarray(points, dtype=float)
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.nCells = len(lo)
        if len(points) % self.nCells:
            raise ValueError("transport nodes do not divide among the cells")
        self.nPer = len(points) // self.nCells
        self.nodes = self.points.reshape(self.nCells, self.nPer)
        self._build_recovery()

    def basis(self, i, x):
        return lagrange(self.nodes[i], x)

    def _build_recovery(self):
        nPer, nCells = self.nPer, self.nCells
        m = (nPer + 1) // 2
        width = min(2 * m + 1, nCells)
        d = min(nPer, width - 1)
        self.patch, self.fit = [], []
        for i in range(nCells):
            first = min(max(i - m, 0), nCells - width)
            P = list(range(first, first + width))
            c0 = 0.5 * (self.lo[i] + self.hi[i])
            h = self.hi[i] - self.lo[i]
            V = np.zeros((len(P), d + 1))
            E = np.zeros((len(P), len(P) * nPer))
            for ci, c in enumerate(P):
                mid, half = 0.5 * (self.lo[c] + self.hi[c]), 0.5 * (self.hi[c] - self.lo[c])
                for xq, wq in zip(XG, WG):
                    x = mid + half * xq
                    w = 0.5 * wq
                    V[ci] += w * ((x - c0) / h) ** np.arange(d + 1)
                    E[ci, ci * nPer : (ci + 1) * nPer] += w * self.basis(c, x)[0]
            self.patch.append(P)
            self.fit.append(np.linalg.solve(V.T @ V, V.T @ E))

    def derivative_weights(self, i, x):
        c0 = 0.5 * (self.lo[i] + self.hi[i])
        h = self.hi[i] - self.lo[i]
        fit = self.fit[i]
        d = fit.shape[0] - 1
        sx = (x - c0) / h
        dphi = np.array([0.0] + [m * sx ** (m - 1) / h for m in range(1, d + 1)])
        return fit.T @ dphi

    def pressure(self, p, i, x, derivative=False):
        v = self.basis(i, x)[0] @ p[i * self.nPer : (i + 1) * self.nPer]
        if not derivative:
            return v
        patch = np.concatenate([p[c * self.nPer : (c + 1) * self.nPer] for c in self.patch[i]])
        return v, self.derivative_weights(i, x) @ patch

    def cell_integral(self, p, C, i, a, b, power):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        if not half > 0.0:
            return 0.0
        return sum(
            half * wq * (C - self.pressure(p, i, mid + half * xq)) ** (-power)
            for xq, wq in zip(XG, WG)
        )

    def cell_of(self, x):
        i = int(np.searchsorted(self.lo, x, side="right")) - 1
        i = min(max(i, 0), self.nCells - 1)
        return i, (i > 0 and x == self.lo[i])


class Balance:
    """An equilibrium: the nodal pressure and the total pressure C."""

    def __init__(self, layout, p, C):
        self.layout, self.p, self.C = layout, p, C


class PythonPressureBalance(manta.EquilibriumModel):
    geometry = [
        manta.GeometrySlot("Vprime", "area per unit flux label, dA/dx", "a^2"),
        manta.GeometrySlot("dVprime_dx", "d V' / dx", "a^2"),
        manta.GeometrySlot("gradx2", "<|grad x|^2>", "a^-2"),
        manta.GeometrySlot("B", "axial magnetic field", "B0"),
    ]
    label = "Phi/Phi_wall"
    name = "ThetaPinch"

    def __init__(self, cell_boundaries, pressure_aux=0, wall_radius=1.0, vacuum_field=1.0):
        super().__init__()
        self.lo = np.asarray(cell_boundaries[:-1], dtype=float)
        self.hi = np.asarray(cell_boundaries[1:], dtype=float)
        self.pressure_aux = pressure_aux
        self.B02 = vacuum_field**2
        self.wallA = math.pi * wall_radius**2
        self.Phi = self.wallA * vacuum_field
        self.solves = 0

    def Solve(self, states, points, weights, t, warm):
        self.solves += 1
        points = np.asarray(points)
        if warm is not None and np.array_equal(warm.layout.points, points):
            L = warm.layout
        else:
            L = Layout(points, self.lo, self.hi)
        p = np.array(states["Aux"][:, self.pressure_aux], dtype=float)

        def integral(C, power):
            return sum(L.cell_integral(p, C, i, L.lo[i], L.hi[i], power) for i in range(L.nCells))

        def F(C):
            return self.Phi * integral(C, 0.5) - self.wallA

        def dF(C):
            return -0.5 * self.Phi * integral(C, 1.5)

        pmax = max(
            p.max(),
            max(
                L.pressure(p, i, 0.5 * (L.lo[i] + L.hi[i]) + 0.5 * (L.hi[i] - L.lo[i]) * xq)
                for i in range(L.nCells)
                for xq in XG
            ),
        )
        lo = pmax + 1e-14 * max(abs(pmax), self.B02)
        hi = max(warm.C if warm is not None else lo, pmax + self.B02)
        grow = 0
        while F(hi) > 0.0:
            grow += 1
            if grow > 100:
                raise RuntimeError("could not bracket the total pressure")
            lo, hi = hi, pmax + 2.0 * (hi - pmax)

        C = warm.C if (warm is not None and lo < warm.C < hi) else hi
        for _ in range(200):
            f = F(C)
            if f > 0.0:
                lo = C
            else:
                hi = C
            nxt = C - f / dF(C)
            if not (lo < nxt < hi):
                nxt = 0.5 * (lo + hi)
            step, C = nxt - C, nxt
            if abs(step) <= 1e-15 * abs(C) or hi - lo <= 1e-15 * abs(C):
                return Balance(L, p, C)
        raise RuntimeError("the wall constraint did not converge")

    def _slots_in_cell(self, e, A, i, x):
        L, p, C, Phi = e.layout, e.p, e.C, self.Phi
        pv, px = L.pressure(p, i, x, derivative=True)
        gap = C - pv
        if not gap > 0.0:
            raise RuntimeError(f"p = {pv} reaches the total pressure C = {C} at x = {x}")
        Vp = Phi / math.sqrt(gap)
        Ax = A[i] + Phi * L.cell_integral(p, C, i, L.lo[i], x, 0.5)
        return np.array([Vp, 0.5 * Phi * px * gap**-1.5, 4.0 * math.pi * Ax / (Vp * Vp), Phi / Vp])

    def Geometry(self, e, points):
        L = e.layout
        A = np.zeros(L.nCells + 1)
        for i in range(L.nCells):
            A[i + 1] = A[i] + self.Phi * L.cell_integral(e.p, e.C, i, L.lo[i], L.hi[i], 0.5)
        g = np.empty((4, len(points)))
        for j, x in enumerate(points):
            i, on_face = L.cell_of(x)
            s = self._slots_in_cell(e, A, i, x)
            if on_face:
                s = 0.5 * (s + self._slots_in_cell(e, A, i - 1, x))
            g[:, j] = s
        return g
