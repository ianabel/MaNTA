"""Park's spatial-accuracy benchmark, from Computer Physics Communications 214
(2017) 1-5, Section 3 (`refs/ParkEfficientSolver.pdf`).

Steady cylindrical energy transport with constant density and diffusivity. It
has a closed-form solution, which is why Park uses it to measure the order of
his IDO scheme; it is reproduced here so MaNTA can be measured the same way and
against the same published numbers.

Writing `V' = x` for a cylinder and taking `n = chi = 1`, Park's Eq. (1) in the
conservative form MaNTA integrates -- `a d_t u - d_x[sigma_hat] = S` -- is

    d_t u - d_x[ x q ] = S,      q = du/dx

    S(x) = 4 x (1 - x^2) exp(1 - x^2)
    u(x) = exp(1 - x^2) - 1                  (the steady state)

    sigma_hat(0) = 0                         (Neumann; automatic, sigma_hat = x q)
    u(1) = 0                                 (Dirichlet)

The problem is linear -- `chi` is constant -- so the nonlinear iteration every
scheme in `refs/` is built around does no work here. That is deliberate: this
benchmark isolates *spatial* accuracy per call. For the nonlinear half see
`../jardin-critical-gradient/`.

A caution before anyone ports numbers out of the paper. Park prints the source
as `S = S0 (1-rho^2) exp(1-rho^2)` against the solution
`T = S0/(n chi) (exp(1-rho^2) - 1)`. **Those two disagree by a factor of four**:
substituting the solution into the steady equation gives `S = 4 S0 (...)`. Both
satisfy the stated boundary conditions, so nothing catches it. The consistent
pair is what is written above and implemented below.
"""

import numpy as np

import manta


def ExactSolution(x):
    """The steady state. Independent of t -- the run relaxes onto it."""
    return np.exp(1.0 - x * x) - 1.0


class ParkConvergence(manta.TransportSystem):
    # Neumann on the axis, Dirichlet at the wall. Naming the variable rather
    # than taking `numbered_spec` is what puts a `T` group in the netCDF output.
    variables = [manta.Field("T", "temperature", "", lower=manta.Neumann,
                             upper=manta.Dirichlet)]

    # A registered case is built by the factory as `(config, grid)`; both are
    # defaulted so `benchmark.py` can build one directly for a `manta.Runner`.
    def __init__(self, config=None, grid=None):
        super().__init__()
        self.reset_counts()

    # PERFORMANCE.md measures MaNTA by the number of calls into a
    # TransportSystem, so this case counts its own. `benchmark.py` reads them;
    # a `manta run.conf` run simply ignores them.
    def reset_counts(self):
        # SigmaFn alone is counted: `residual` evaluates SigmaFn and Sources on
        # the same states at the same abscissae in one ComputePhysics call, so
        # one counter is the number of physics point-evaluations.
        self.nFlux = 0        # SigmaFn point-evaluations
        self.nDeriv = 0       # derivative point-evaluations

    # --- boundaries --------------------------------------------------------
    def LowerBoundary(self, index, t):
        return 0.0            # zero flux on the axis

    def UpperBoundary(self, index, t):
        return ExactSolution(1.0)

    # --- physics -----------------------------------------------------------
    def SigmaFn(self, index, state, x, t):
        self.nFlux += 1
        return x * state.q[0]

    def Sources(self, index, state, x, t):
        return 4.0 * x * (1.0 - x * x) * np.exp(1.0 - x * x)

    # --- derivatives -------------------------------------------------------
    # Only the one nonzero block is written. An absent derivative hook means
    # that block is identically zero, which is already what the zeroed
    # out-parameter gives, so dSigmaFn_du and the three dSources_* hooks would
    # be four functions returning zeros. Leaving them out also makes nDeriv an
    # exact count of derivative point-evaluations.
    def dSigmaFn_dq(self, index, state, x, t):
        self.nDeriv += 1
        return np.array([x])

    # --- initial condition -------------------------------------------------
    # Deliberately not the answer: the run has to find it.
    def InitialValue(self, index, x):
        return 0.0

    def InitialDerivative(self, index, x):
        return 0.0


def registerTransportSystems():
    manta.registerPhysicsCase("ParkConvergence", ParkConvergence)
