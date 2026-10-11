"""Nonlinear diffusion with a localised source and a closed-form steady state.

In the form MaNTA integrates, `a d_t u - d_x[sigma_hat] = S`, with a = 1,

    sigma_hat = A u^(-3/2) q,            q = du/dx
    S(x)      = D exp( -(x - C)^2 / B )

on x in [0, 1], with zero gradient on the axis and a fixed edge value:

    u'(0, t) = 0                         (Neumann)
    u(1, t)  = U1                        (Dirichlet)
    u(x, 0)  = U1

The diffusivity A u^(-3/2) falls as u rises, so the source heats the core
against a conductivity that weakens where it is hottest.

## The steady state

Integrating the steady equation once from the axis, where the flux vanishes,
gives `A u^(-3/2) u' = -F(x)` with `F(x) = int_0^x S`. Then `w = u^(-1/2)` has
`w' = F / 2A`, so

    u(x)^(-1/2) = U1^(-1/2) - G(x),      G(x) = (1/2A) int_x^1 F(s) ds

and both integrals of the Gaussian are closed-form; `ExactSteadyState` has
them. It agrees with nested quadrature to 5e-16 over [0, 1]. A steady state
exists while `G(0) < U1^(-1/2)`; at the defaults G(0) = 0.73 against 1.83.

This is written vectorised -- `ComputePhysics` and `ComputePhysicsDerivatives`
take every point of a batch at once -- because it is the test problem for a
surrogate flux, which is evaluated a batch at a time.
"""

import numpy as np
import scipy.special

import manta

A = 6.0        # flux coefficient
B = 0.02       # source width^2
C = 0.3        # source centre
D = 50.0       # source height
U1 = 0.3       # edge value u(1)


def ExactSteadyState(x):
    """The steady state of the problem above, for the module's parameters."""
    x = np.asarray(x, dtype=float)
    rb = np.sqrt(B)
    y = (x - C) / rb
    G = (B * D / (4 * A)) * (np.exp(-(1 - C) ** 2 / B) - np.exp(-y ** 2)) \
        + (D * np.sqrt(B * np.pi) / (4 * A)) * (
            (C - 1) * scipy.special.erf((C - 1) / rb)
            + (1 - x) * scipy.special.erf(C / rb)
            - (x - C) * scipy.special.erf(y))
    return 1.0 / (U1 ** -0.5 - G) ** 2


class ToyModel(manta.TransportSystem):
    variables = [manta.Field("u", "toy-model profile", lower=manta.Neumann,
                             upper=manta.Dirichlet)]

    # Nothing here depends on where the case is evaluated.
    regrid = manta.Regrid.InPlace

    # A registered case is built by the factory as `(config, grid)`; both are
    # defaulted so `__main__` below can build one directly for a manta.Runner.
    def __init__(self, config=None, grid=None):
        super().__init__()

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return U1

    # states["Variable"] and states["Derivative"] are (nPoints, nVars).
    # Returns [fluxes, sources, aux], each a list over variables.
    def ComputePhysics(self, states, positions, t):
        u = np.asarray(states["Variable"])[:, 0]
        q = np.asarray(states["Derivative"])[:, 0]
        x = np.asarray(positions)
        flux = A * u ** -1.5 * q
        source = D * np.exp(-(x - C) ** 2 / B)
        return [[flux], [source], []]

    # One dict per variable for the flux and for the source, each holding the
    # derivative with respect to every variable's u ("Variable"), q
    # ("Derivative") and sigma ("Flux") at every point.
    def ComputePhysicsDerivatives(self, states, positions, t):
        u = np.asarray(states["Variable"])[:, 0]
        q = np.asarray(states["Derivative"])[:, 0]
        n = u.shape[0]

        def block(dvar, dder):
            return {
                "Variable": np.reshape(dvar, (n, 1)),
                "Derivative": np.reshape(dder, (n, 1)),
                "Flux": np.zeros((n, 1)),
                "Aux": np.zeros((n, 0)),
                "Scalars": np.zeros(0),
            }

        dflux = block(-1.5 * A * u ** -2.5 * q, A * u ** -1.5)
        # The source depends on x alone.
        dsource = block(np.zeros(n), np.zeros(n))
        return [[dflux], [dsource], []]

    def InitialValue(self, index, x):
        return U1

    def InitialDerivative(self, index, x):
        return 0.0


def registerTransportSystems():
    manta.registerPhysicsCase("ToyModel", ToyModel)


if __name__ == "__main__":
    # The steady state against ExactSteadyState, refining the mesh at k = 3.
    # The error falls as h^(k+1) = h^4. Six cells are 0.17 wide against a
    # source of width sqrt(B) = 0.14, so the coarsest row is under-resolved.
    x = np.linspace(0.0, 1.0, 401)
    exact = ExactSteadyState(x)
    for cells in (6, 12, 24, 48):
        runner = manta.Runner(ToyModel())
        runner.configure({
            "OutputFilename": "steady",
            "PolynomialDegree": 3,
            "GridSize": cells,
            "LowerBoundary": 0.0,
            "UpperBoundary": 1.0,
            "Relative_tolerance": 1.0e-8,
            "Absolute_tolerance": 1.0e-8,
            "delta_t": 1.0,
            "SteadyStateTolerance": 1.0e-11,
            "WriteOutput": False,
        })
        runner.run_ss()
        u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
        error = np.max(np.abs(u - exact)) / np.max(exact)
        print(f"{cells:3d} cells, k = 3: relative L_inf error {error:.2e}")
