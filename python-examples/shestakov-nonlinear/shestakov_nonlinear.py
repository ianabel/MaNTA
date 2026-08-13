"""Shestakov's analytic test problem, from Journal of Computational Physics
**185** (2003) 399-426, Section 2.1, with its corrigendum at JCP **186** (2003)
360 (`refs/LoDestroMethod.pdf` and `refs/ShestakovCorrection.pdf`).

Degenerate nonlinear diffusion with a closed-form steady state. The diffusivity
is the inverse-square density scale length, so the flux depends on the gradient
*cubed*:

    D = l_n^-2 = (n_x / n)^2,   Gamma = -D n_x = -D0 n_x^3 / n^2

In the conservative form MaNTA integrates, `a d_t u - d_x[sigma_hat] = S`, the
sign flips -- Shestakov writes `d_t n + d_x Gamma = S` -- so

    sigma_hat = -Gamma = D0 q^3 / u^2,      q = du/dx

    S(x) = S0 for x < d, 0 otherwise        (S0 = 1, d = 0.1)
    sigma_hat(0) = 0                        (Neumann)
    n(Lx) = n_b                             (Dirichlet; Lx = 1)

This is the case the paper uses to show a *semi-implicit* scheme -- one that
lags `D` -- going unstable: here `D ~ (d_x n)^p n^q` with **p = 2, q = -2**, and
Shestakov's analysis gives instability for `p > 1` once `n^2 D dt > 2/(p-1)`.
His Fig. 1 shows a self-similar cooling wave.

Neither of the corrigendum's caveats applies to this problem, which is worth
knowing before using it to test anything. Its short-wavelength instability of
the *fully* implicit scheme needs `-2 < n^2 dt D (p+1) < 0`, and `p + 1 = 3 > 0`
here; its warning about a singular matrix needs `D` to change sign across the
mesh, and `D = (n_x/n)^2` is a square.

## The steady state, which the paper misprints

Integrating `d_x Gamma = S` from the zero-flux axis gives `Gamma = S0 min(x, d)`,
and substituting `w = n^(1/3)` linearises what is left:

    w_x = -(1/3) (Gamma/D0)^(1/3),   w(Lx) = n_b^(1/3)

so with `n = w^3`,

    n_e(x) = [ n_b^(1/3) + (1/3)(S0 d/D0)^(1/3) (Lx - x) ]^3                x >= d
    n_e(x) = [ n_b^(1/3) + (1/3)(S0/D0)^(1/3) (
                  0.75 (d^(4/3) - x^(4/3)) + d^(1/3)(Lx - d) ) ]^3          x <  d

**The paper's brace assigns these two branches the other way round.** As
printed, `(Lx - x)^3` is labelled `x < d`; substituting that into the steady
equation leaves a residual of exactly `-S`, i.e. it solves the wrong problem.
Swapped, the residual is finite-difference noise. The `S0 d / 27 D0` prefactor
multiplies *both* branches, and the whole thing above reduces to the paper's
when `n_b = 0`.

The paper also inverts its similarity variable `eta = x^4 / (64 D0 t)` as
`x_f = eta_f t^(1/4)`, dropping a factor `(64 D0)^(1/4)`. Restored, the printed
`eta_f = 3.339` reproduces the paper's own tabulated front positions (0.785,
0.618, 0.321) to three figures; as printed it is out by 3-8%.

## Why n_b is not zero here, and why it needs SuppressAlgebraicError

Shestakov's Section 2.1 sets `n_b = 0`. As `u -> 0` the flux `D0 q^3 / u^2` is a
0/0 -- finite in the true solution -- whose Jacobian entry
`d(sigma_hat)/du = -2 D0 q^3 / u^3` diverges outright, and `sigma` sits in IDA's
local error test. The step then dies in the error test however small `h` gets,
with the Newton converging happily each time.

`SuppressAlgebraicError = true` takes `sigma`, `q`, `lambda` and `phi` out of
that test and is what makes this problem runnable at all. It moves the wall a
long way but does not remove it: `n_b >= 1e-2` then works at every resolution
tried, where without it nothing below 0.07 works anywhere; his Section 2.2 value
of 1e-3 works at most resolutions; and `n_b = 0` runs at exactly one, so it is
widened rather than cured. `benchmark.py` maps the whole grid, and the key costs
restart fidelity and aux accuracy -- see docs/running.rst.
"""

import numpy as np

import manta

S0 = 1.0                  # Shestakov's source strength
SOURCE_WIDTH = 0.1        # ... his d
D0 = 1.0                  # ... his D_0
LX = 1.0                  # ... his L_x
BOUNDARY_DENSITY = 0.01   # ... his n(Lx); his Sec 2.1 uses 0, which see above


def ExactSolution(x, n_b=BOUNDARY_DENSITY):
    """The steady state, for a general Dirichlet value `n_b`."""
    x = np.asarray(x, dtype=float)
    d, w0 = SOURCE_WIDTH, n_b ** (1.0 / 3.0)
    outer = w0 + (S0 * d / D0) ** (1.0 / 3.0) * (LX - x) / 3.0
    inner = w0 + (S0 / D0) ** (1.0 / 3.0) * (
        0.75 * (d ** (4.0 / 3.0) - x ** (4.0 / 3.0))
        + d ** (1.0 / 3.0) * (LX - d)) / 3.0
    return np.where(x < d, inner, outer) ** 3


class ShestakovNonlinear(manta.TransportSystem):
    variables = [manta.Field("n", "density", "", lower=manta.Neumann,
                             upper=manta.Dirichlet)]

    # A registered case is built by the factory as `(config, grid)`; both are
    # defaulted so `benchmark.py` can build one directly for a `manta.Runner`.
    def __init__(self, config=None, grid=None, n_b=BOUNDARY_DENSITY):
        super().__init__()
        self.n_b = n_b
        self.reset_counts()

    # PERFORMANCE.md measures MaNTA by the number of calls into a
    # TransportSystem, so this case counts its own.
    def reset_counts(self):
        self.nFlux = 0        # SigmaFn point-evaluations
        self.nDeriv = 0       # derivative point-evaluations

    # --- boundaries --------------------------------------------------------
    def LowerBoundary(self, index, t):
        return 0.0            # zero flux on the axis

    def UpperBoundary(self, index, t):
        return self.n_b

    # --- physics -----------------------------------------------------------
    def SigmaFn(self, index, state, x, t):
        self.nFlux += 1
        return D0 * state.q[0] ** 3 / state.u[0] ** 2

    def Sources(self, index, state, x, t):
        return S0 if x < SOURCE_WIDTH else 0.0

    # --- derivatives -------------------------------------------------------
    # Both blocks are nonzero here, unlike the other two benchmarks: the flux
    # depends on `u` as well as on `q`, and it is this second one that goes
    # singular at a zero Dirichlet value. Shestakov never forms either -- he
    # lags `D` over previous iterates and needs no Jacobian at all, which is
    # exactly why his scheme is untroubled by `n_b = 0` and this is not.
    def dSigmaFn_dq(self, index, state, x, t):
        self.nDeriv += 1
        return np.array([3.0 * D0 * state.q[0] ** 2 / state.u[0] ** 2])

    def dSigmaFn_du(self, index, state, x, t):
        return np.array([-2.0 * D0 * state.q[0] ** 3 / state.u[0] ** 3])

    # --- initial condition -------------------------------------------------
    # Shestakov starts from n = 1, which MaNTA cannot: it contradicts the
    # Dirichlet value at x = Lx, and being constant it makes D = 0 and
    # d(sigma_hat)/dq = 0 everywhere, so the trace system degenerates and
    # IDACalcIC fails. This one keeps his n(0) = 1 and zero axis flux, meets
    # the Dirichlet value, and vanishes like (Lx - x)^3 so that the flux stays
    # finite at the wall the way the true solution does.
    def _initial(self, x):
        s = (LX - x) ** 3 * (1.0 + 3.0 * x)
        return self.n_b + (1.0 - self.n_b) * s

    def InitialValue(self, index, x):
        return self._initial(x)

    def InitialDerivative(self, index, x):
        return (1.0 - self.n_b) * (-3.0 * (LX - x) ** 2 * (1.0 + 3.0 * x)
                                   + 3.0 * (LX - x) ** 3)


def registerTransportSystems():
    manta.registerPhysicsCase("ShestakovNonlinear", ShestakovNonlinear)
