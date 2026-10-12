"""The toy model as an expensive, noisy black box, and MaNTA's view of it.

A turbulent transport model returns every flux at a point from one call and
says nothing about derivatives. `ToyFlux` stands in for one: a batched call
returns sigma and S at every point, sigma carrying *stochastic* relative noise
-- a fresh draw per point on every call, as a time-averaged turbulence
simulation would give -- and it counts what it is asked for.

`BlackBoxCase` hands it to MaNTA. Its Jacobian comes from one of

* "analytic"  -- the box's exact, noise-free derivatives: the control arm;
* "full"      -- forward differences in every u and every q direction,
                 1 + 2 nVars box calls per build, or 2 nVars with the base
                 point reused;
* "gradients" -- forward differences in the q directions only, dropping
                 d sigma/du, as TGYRO does: 1 + nVars, or nVars.

**A Jacobian may be wrong, but it must not be singular.** MaNTA never assembles
the Jacobian, so an error in it costs Newton iterations and not accuracy -- but
only while the Newton step J~^-1 F can still point anywhere. A null space would
confine every step to the range of J~, and a solution with a component outside
it would be unreachable. Dropping d sigma/du leaves a diffusion operator with
diffusivity d sigma/dq and a Dirichlet end, which is nonsingular; losing
d sigma/dq, or giving it the wrong sign, degenerates the HDG trace system. A
noisy difference can do exactly that, so every build keeps d sigma/dq above a
floor -- `dq_floor` times the batch's largest -- and counts each time it has to.
That guard is written for one channel. With several, the object to keep
definite is the nVars x nVars block d sigma_i/dq_j, which is a design question
of its own.
"""

import numpy as np

import manta

from toy_model import A, B, C, D, U1


class ToyFlux:
    """sigma = A u^(-3/2) q (1 + noise * xi), S = D exp(-(x - C)^2 / B).

    `u`, `q` are (nPoints, nVars); `x` is (nPoints,). Returns (sigma, S), both
    (nPoints, nVars). xi ~ N(0, 1), drawn afresh for every point of every call.
    """

    nVars = 1

    def __init__(self, noise=0.0, seed=0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.points = 0

    def __call__(self, u, q, x):
        self.calls += 1
        self.points += u.shape[0]
        sigma = A * u ** -1.5 * q
        if self.noise:
            sigma = sigma * (1.0 + self.noise * self.rng.standard_normal(sigma.shape))
        source = np.reshape(D * np.exp(-(x - C) ** 2 / B), (-1, 1))
        return sigma, source

    def derivatives(self, u, q, x):
        """Exact, noise-free (d sigma/du, d sigma/dq), each (nPoints, nVars, nVars).

        Not counted: a real black box has no such thing. It is the control arm.
        """
        n = u.shape[0]
        dsdu = np.reshape(-1.5 * A * u ** -2.5 * q, (n, 1, 1))
        dsdq = np.reshape(A * u ** -1.5, (n, 1, 1))
        return dsdu, dsdq


class BlackBoxCase(manta.TransportSystem):
    variables = [manta.Field("u", "toy-model profile", lower=manta.Neumann,
                             upper=manta.Dirichlet)]

    regrid = manta.Regrid.InPlace

    def __init__(self, box, jacobian="full", dz=1.0e-6, du=1.0e-6,
                 dq_floor=1.0e-3, reuse_base=True):
        super().__init__()
        if jacobian not in ("analytic", "full", "gradients"):
            raise ValueError(f"unknown jacobian {jacobian!r}")
        self.box = box
        self.jacobian = jacobian
        self.dz = dz                # q step: h_q = dz * u, a step in u'/u
        self.du = du                # u step: h_u = du * u
        self.dq_floor = dq_floor
        self.reuse_base = reuse_base
        self._last = None           # (t, u, q, x, sigma, S) of the last value call
        self.reset_counts()

    def reset_counts(self):
        self.value_calls = 0        # box calls for ComputePhysics
        self.fd_calls = 0           # box calls made only to difference
        self.base_reused = 0        # builds whose base point was the last value call
        self.builds = 0
        self.floored_points = 0     # points where the guard raised d sigma/dq

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return U1

    def InitialValue(self, index, x):
        return U1

    def InitialDerivative(self, index, x):
        return 0.0

    @staticmethod
    def _unpack(states, positions):
        u = np.array(states["Variable"], dtype=float)
        q = np.array(states["Derivative"], dtype=float)
        return u, q, np.array(positions, dtype=float)

    def ComputePhysics(self, states, positions, t):
        u, q, x = self._unpack(states, positions)
        sigma, source = self.box(u, q, x)
        self.value_calls += 1
        self._last = (t, u, q, x, sigma, source)
        return [list(sigma.T), list(source.T), []]

    def _base(self, u, q, x, t):
        last = self._last
        if (self.reuse_base and last is not None and last[0] == t
                and np.array_equal(last[1], u) and np.array_equal(last[2], q)
                and np.array_equal(last[3], x)):
            self.base_reused += 1
            return last[4], last[5]
        self.fd_calls += 1
        return self.box(u, q, x)

    def _difference(self, u, q, x, t):
        """Forward-differenced (dsigma/du, dsigma/dq, dS/du, dS/dq), (n, nVars, nVars)."""
        n, nv = u.shape
        dsdu = np.zeros((n, nv, nv)); dsdq = np.zeros((n, nv, nv))
        dSdu = np.zeros((n, nv, nv)); dSdq = np.zeros((n, nv, nv))
        sigma0, S0 = self._base(u, q, x, t)
        directions = [("q", j) for j in range(nv)]
        if self.jacobian == "full":
            directions += [("u", j) for j in range(nv)]
        for kind, j in directions:
            up, qp = u.copy(), q.copy()
            if kind == "q":
                h = self.dz * np.abs(u[:, j])
                qp[:, j] += h
            else:
                h = self.du * np.abs(u[:, j])
                up[:, j] += h
            sigma, S = self.box(up, qp, x)
            self.fd_calls += 1
            ds = (sigma - sigma0) / h[:, None]
            dS = (S - S0) / h[:, None]
            if kind == "q":
                dsdq[:, :, j], dSdq[:, :, j] = ds, dS
            else:
                dsdu[:, :, j], dSdu[:, :, j] = ds, dS
        return dsdu, dsdq, dSdu, dSdq

    def _guard(self, dsdq):
        assert dsdq.shape[1] == 1, "the guard is written for one channel"
        d = dsdq[:, 0, 0]
        floor = self.dq_floor * np.max(np.abs(d))
        low = d < floor
        self.floored_points += int(np.count_nonzero(low))
        d[low] = floor

    def ComputePhysicsDerivatives(self, states, positions, t):
        u, q, x = self._unpack(states, positions)
        n, nv = u.shape
        self.builds += 1
        if self.jacobian == "analytic":
            dsdu, dsdq = self.box.derivatives(u, q, x)
            dSdu = np.zeros_like(dsdu); dSdq = np.zeros_like(dsdq)
        else:
            dsdu, dsdq, dSdu, dSdq = self._difference(u, q, x, t)
            self._guard(dsdq)

        # One dict per variable i, holding row i of each block: the derivative
        # with respect to every variable's u, q and sigma, at every point.
        def rows(du, dq):
            return [{
                "Variable": du[:, i, :],
                "Derivative": dq[:, i, :],
                "Flux": np.zeros((n, nv)),
                "Aux": np.zeros((n, 0)),
                "Scalars": np.zeros(0),
            } for i in range(nv)]

        return [rows(dsdu, dsdq), rows(dSdu, dSdq), []]
