"""TGYRO's flux-matching iteration.

From Candy, Holland, Waltz, Fahey & Belli, Phys. Plasmas 16 (2009) 060704
(`refs/TGYRO.pdf`), and its open source, github.com/gafusion/gacode,
`tgyro/src/tgyro_iteration_standard.f90` -- the default method, which this
follows line for line. Where the letter and the code differ, the code wins.

TGYRO solves for the steady state directly. It does not discretise the PDE: it
asks that the flux the model returns at each of a few radii equal the flux the
sources require there,

    sigma_hat(r_j; u_j, u'_j) = -int_0^{r_j} S dx,     j = 1..N,

with the integral by the trapezoidal rule on the radii themselves
(`tgyro_volume_int.f90`, `use_trap`). The radii are r_j = j/N, plus the axis
r_0 = 0, and the outermost one is the matching radius where the profile is held
at its boundary value.

**The unknowns are logarithmic gradients**, z_j = -d ln u / dr at each radius,
not profile values. A profile is rebuilt from them by integrating inwards from
the boundary value with the trapezoidal rule (`math_scaleintv(..., 'log')`), the
axis gradient being zero. So TGYRO needs positive profiles: a problem whose
boundary value is zero is shifted by `problem.tgyro_offset`, which is only
honest for physics invariant under u -> u + c, and Park's and Jardin's are.

**The iteration**, per pass:

  * the target Jacobian dT/dz, by forward differences of the targets -- cheap,
    no model calls;
  * the flux Jacobian dQ/dz, *block diagonal by assumption*: one model call per
    evolved channel, perturbing that channel's z by dz at every radius at once,
    with the profiles held fixed;
  * a Newton step on (dQ/dz - dT/dz) dz = -relax * (Q - T), each component
    capped at dz_max;
  * a model call at the new point, and then, for every component whose own
    residual |Q - T| rose, the step is undone for that component and its
    relaxation divided by `relax_factor` (reset to 0.75 relax_factor with a
    doubled step if it falls below relax_factor^-3); if anything was undone,
    one more model call.

So an iteration costs nchannels + 1 or nchannels + 2 calls per radius. The
defaults are the code's: LOC_DX = 0.1, LOC_DX_MAX = 1.0, LOC_RELAX = 2.0, all
with the minor radius 1, and residual method 2 (|Q - T|). The letter used
dz = 0.3/a for gyrokinetic fluxes, whose time-averaging noise a small step
cannot see past; these models are noiseless.

**TGYRO has no convergence test** -- it runs a fixed number of iterations. Here
the loop stops when max |Q - T| falls below `tol` times max |T|, so that its
cost can be stated at a given accuracy.
"""

import numpy as np


class Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Grid:
    def __init__(self, problem, nradii, offset):
        self.p = problem
        self.N = nradii
        self.r = np.linspace(0.0, 1.0, nradii + 1)
        self.dr = 1.0 / nradii
        self.offset = offset
        self.wb = problem.ub + offset        # held at the matching radius

    def profile(self, z):
        """w at every radius from z at every radius (z[:, 0] = 0 on the axis),
        integrating d ln w / dr = -z inwards from the boundary by trapezoids."""
        lnw = np.empty_like(z)
        lnw[:, -1] = np.log(self.wb)
        for j in range(self.N - 1, -1, -1):
            lnw[:, j] = lnw[:, j + 1] + 0.5 * (z[:, j] + z[:, j + 1]) * self.dr
        return np.exp(lnw)

    def flux(self, z, W):
        """Q at the radii r_1..r_N. One model call per radius."""
        U = W - self.offset[:, None]
        Q = -z * W
        return self.p.flux(self.r[1:], U[:, 1:], Q[:, 1:])

    def target(self, z, W):
        """-int_0^r S dx at r_1..r_N, by trapezoids on the radii."""
        U = W - self.offset[:, None]
        S0, S1 = self.p.source(self.r, U)
        S = S0 + S1 * U
        cum = np.cumsum(0.5 * (S[:, :-1] + S[:, 1:]) * self.dr, axis=1)
        return -cum


def solve(problem, nradii, dz=0.1, dz_max=1.0, relax_factor=2.0, tol=1e-10,
          max_iter=500):
    nv = problem.nvars
    offset = np.full(nv, problem.tgyro_offset, dtype=float)
    g = _Grid(problem, nradii, offset)
    start = problem.evals

    # The initial gradients, from the initial profile at the radii.
    w0 = problem.initial(g.r) + offset[:, None]
    hfine = 1e-6
    wp = problem.initial(np.minimum(g.r + hfine, 1.0)) + offset[:, None]
    wm = problem.initial(np.maximum(g.r - hfine, 0.0)) + offset[:, None]
    span = np.minimum(g.r + hfine, 1.0) - np.maximum(g.r - hfine, 0.0)
    z = -(wp - wm) / span / w0
    z[:, 0] = 0.0

    W = g.profile(z)
    f = g.flux(z, W)
    t = g.target(z, W)
    res = np.abs(f - t)
    relax = np.ones((nv, nradii))
    history = []
    converged = False
    it = 0
    for it in range(max_iter + 1):
        scale = max(np.max(np.abs(t)), 1e-300)
        history.append((problem.evals - start, float(np.max(res) / scale)))
        if np.max(res) <= tol * scale:
            converged = True
            break
        if it == max_iter or not np.all(np.isfinite(res)):
            break

        z0, f0, t0, res0 = z.copy(), f, t, res
        W0 = W
        n = nv * nradii

        # Target Jacobian: forward differences, every component (cheap).
        jg = np.zeros((n, n))
        for p in range(n):
            zp = z0.copy()
            s, j = divmod(p, nradii)
            zp[s, j + 1] += dz
            jg[:, p] = ((g.target(zp, g.profile(zp)) - t0) / dz).reshape(-1)

        # Flux Jacobian: block diagonal in radius, one call per channel.
        jf = np.zeros((n, n))
        for ip in range(nv):
            zp = z0.copy()
            zp[ip, 1:] += dz
            fp = g.flux(zp, W0)                 # profiles held fixed
            for j in range(nradii):
                for s in range(nv):
                    jf[s * nradii + j, ip * nradii + j] = (fp[s, j] - f0[s, j]) / dz

        b = np.linalg.solve(jf - jg, (-(f0 - t0) * relax).reshape(-1))
        b = np.clip(b, -dz_max, dz_max).reshape(nv, nradii)

        z = z0.copy()
        z[:, 1:] += b
        W = g.profile(z)
        t = g.target(z, W)
        f = g.flux(z, W)
        res = np.abs(f - t)

        corrected = False
        for s in range(nv):
            for j in range(nradii):
                if res0[s, j] < res[s, j] and relax_factor > 1.0:
                    corrected = True
                    z[s, j + 1] = z0[s, j + 1]
                    relax[s, j] /= relax_factor
                    if relax[s, j] < relax_factor ** -3:
                        relax[s, j] = 0.75 * relax_factor
                        z[s, j + 1] = z0[s, j + 1] + 2.0 * b[s, j]
                else:
                    relax[s, j] = 1.0
        if corrected:
            W = g.profile(z)
            t = g.target(z, W)
            f = g.flux(z, W)
            res = np.abs(f - t)

    zf = z.copy()

    def profile(x):
        """Each channel at x: z linear between radii, integrated exactly."""
        x = np.asarray(x, dtype=float)
        lnw_r = np.log(g.profile(zf))
        j = np.clip((x / g.dr).astype(int), 0, nradii - 1)
        frac = (x - g.r[j]) / g.dr
        out = []
        for s in range(nv):
            zx = zf[s, j] + frac * (zf[s, j + 1] - zf[s, j])
            lnw = lnw_r[s, j + 1] + 0.5 * (zx + zf[s, j + 1]) * (g.r[j + 1] - x)
            out.append(np.exp(lnw) - offset[s])
        return np.array(out)

    return Result(profile=profile, evals=problem.evals - start, iterations=it,
                  converged=converged, residual=history[-1][1], history=history,
                  points=nradii)
