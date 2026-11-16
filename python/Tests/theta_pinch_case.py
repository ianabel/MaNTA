"""ThetaPinchDiffusion, written again as a Python case reading the geometry.

The C++ case (PhysicsCases/ThetaPinchDiffusion.cpp) is an
AutodiffTransportSystem; this one writes its derivatives by hand, geometry
derivatives included, so that a run of it against the C++ field model checks
the Python half of the coupling: ``state.geom`` by slot name, and the three
``d*_dGeometry`` hooks the coupled Jacobian is built from.

Isothermal density transport on the normalised flux x, evolving N = V' n:

    sigma_hat = V' D <|grad x|^2> dn/dx,   n = N / V',
                dn/dx = (q - n dV'/dx) / V',   D = D0 (Bref / B)^gamma
    S         = V' S0 exp(-x^2 / w^2)
    G         = p - T N / V' = 0          (the pressure, as the aux variable)

With n eliminated, sigma_hat = D g2 (q - N V'_x / V').
"""

import math

import numpy as np

import manta


class PythonThetaPinch(manta.TransportSystem):
    variables = [
        manta.Field(
            "N",
            "particles per unit flux per unit length, V' n",
            "n0 a^2",
            lower=manta.Neumann,
            upper=manta.Dirichlet,
        )
    ]
    aux = [manta.Aux("p", "pressure, in units where 2 mu0 = 1", "B0^2")]
    regrid = manta.Regrid.InPlace
    readsGeometry = True
    geometryDerivatives = manta.GeometryDerivatives.Analytic

    def __init__(self, temperature=0.3, D0=0.5, gamma=2.0, Bref=1.0,
                 source=1.0, width=0.3, edge=0.1, centre=1.0, wall_radius=1.0):
        super().__init__()
        self.T, self.D0, self.gamma, self.Bref = temperature, D0, gamma, Bref
        self.S0, self.width = source, width
        self.edge, self.centre = edge, centre
        self.wallA = math.pi * wall_radius**2
        self.slots_seen = None

    # By name: the field model's slot names, which this case was never told
    # except through the state.
    def metric(self, state):
        g = state.geom
        if self.slots_seen is None:
            self.slots_seen = len(g)
        return g["Vprime"], g["dVprime_dx"], g["gradx2"], g["B"]

    def D(self, B):
        return self.D0 * (self.Bref / B) ** self.gamma

    def SigmaFn(self, i, state, x, t):
        Vp, dVp, g2, B = self.metric(state)
        N, q = state.u[0], state.q[0]
        return self.D(B) * g2 * (q - N * dVp / Vp)

    def dSigmaFn_du(self, i, state, x, t):
        Vp, dVp, g2, B = self.metric(state)
        return np.array([-self.D(B) * g2 * dVp / Vp])

    def dSigmaFn_dq(self, i, state, x, t):
        Vp, dVp, g2, B = self.metric(state)
        return np.array([self.D(B) * g2])

    def dSigmaFn_dGeometry(self, i, state, x, t):
        Vp, dVp, g2, B = self.metric(state)
        N, q = state.u[0], state.q[0]
        D = self.D(B)
        sigma = D * g2 * (q - N * dVp / Vp)
        return np.array([
            D * g2 * N * dVp / Vp**2,
            -D * g2 * N / Vp,
            D * (q - N * dVp / Vp),
            -self.gamma * sigma / B,
        ])

    def profile(self, x):
        return math.exp(-x * x / (self.width * self.width))

    def Sources(self, i, state, x, t):
        Vp = self.metric(state)[0]
        return Vp * self.S0 * self.profile(x)

    def dSources_dGeometry(self, i, state, x, t):
        return np.array([self.S0 * self.profile(x), 0.0, 0.0, 0.0])

    def AuxG(self, i, state, x, t):
        Vp = self.metric(state)[0]
        return state.phi[0] - self.T * state.u[0] / Vp

    def AuxGPrime(self, i, out, state, x, t):
        Vp = self.metric(state)[0]
        out.u[0] = -self.T / Vp
        out.phi[0] = 1.0

    def dAuxG_dGeometry(self, i, state, x, t):
        Vp = self.metric(state)[0]
        return np.array([self.T * state.u[0] / Vp**2, 0.0, 0.0, 0.0])

    def dSources_dPhi(self, i, state, x, t):
        return np.zeros(1)

    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(1)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return self.wallA * self.edge

    def InitialValue(self, i, x):
        return self.wallA * (self.edge + (self.centre - self.edge) * (1.0 - x * x))

    def InitialDerivative(self, i, x):
        return self.wallA * (self.centre - self.edge) * (-2.0 * x)

    def InitialAuxValue(self, i, x):
        return self.T * (self.edge + (self.centre - self.edge) * (1.0 - x * x))


def jax_theta_pinch(**kwargs):
    """The same case on manta.jax, its geometry derivatives by autodiff.

    Built in a function so the module imports without JAX; the caller needs
    float64 enabled.
    """
    from typing import NamedTuple

    import jax.numpy as jnp

    from manta.jax import JAXTransportSystem

    class Params(NamedTuple):
        D0: float
        T: float

    class JAXThetaPinch(JAXTransportSystem):
        readsGeometry = True
        geometryDerivatives = manta.GeometryDerivatives.Analytic
        regrid = manta.Regrid.InPlace

        def __init__(self, D0=0.5, T=0.3):
            super().__init__(
                manta.SystemSpec(
                    variables=list(PythonThetaPinch.variables),
                    aux=list(PythonThetaPinch.aux),
                )
            )
            self.params = Params(D0=D0, T=T)
            self.gamma, self.S0, self.width = 2.0, 1.0, 0.3
            self.edge, self.centre, self.wallA = 0.1, 1.0, math.pi

        def sigma(self, index, state, x, t, params):
            Vp, dVp, g2, B = (state.Geometry[s] for s in range(4))
            D = params.D0 * (1.0 / B) ** self.gamma
            return D * g2 * (state.Derivative[0] - state.Variable[0] * dVp / Vp)

        def source(self, index, state, x, t, params):
            return state.Geometry[0] * self.S0 * jnp.exp(-x * x / (self.width * self.width))

        def aux(self, index, state, x, t, params):
            return state.Aux[0] - params.T * state.Variable[0] / state.Geometry[0]

        def LowerBoundary(self, index, t):
            return 0.0

        def UpperBoundary(self, index, t):
            return self.wallA * self.edge

        def InitialValue(self, index, x):
            return self.wallA * (self.edge + (self.centre - self.edge) * (1.0 - x * x))

        def InitialAuxValue(self, index, x):
            return 0.3 * (self.edge + (self.centre - self.edge) * (1.0 - x * x))

        # An objective reading the geometry: the density weighted by the field,
        # so dG/dD0 reaches G through B as well as through N.
        def g(self, state, x, params):
            return state.Geometry[3] * state.Variable[0]

        def createAdjointProblem(self):
            from manta.jax import JAXAdjointProblem

            return JAXAdjointProblem(self, self.g)

    return JAXThetaPinch(**kwargs)
