import jax
import manta as MaNTA
from typing import NamedTuple

from functools import partial

from manta.jax import VectorizedTransportSystem, JAXAdjointProblem, FFIRunner

import jax.numpy as jnp

import equinox as eqx


class NonlinearDiffusionParams(NamedTuple):
    SourceCentre: float
    D: float
    T_s: float
    a: float
    SourceWidth: float

    @classmethod
    def make(cls, SourceCentre, D, a ) -> "NonlinearDiffusionParams":
        return cls(
            SourceCentre=SourceCentre,
            D=D,
            T_s=50.0,
            a=a,
            SourceWidth=0.02,
        )


class JAXAuxTest(VectorizedTransportSystem):
    def __init__(self, params):
        super().__init__(MaNTA.numbered_spec(1, nAux = 1, lower=MaNTA.Neumann))

        self.params = params

        # %%
        config = {
            "OutputFilename": "output",
            "Polynomial_degree": 5,
            "Grid_size": 10,
            "Lower_boundary": -1.0,
            "Upper_boundary": 1.0,
            "Relative_tolerance": 0.01,
            "t_final": 1.0,
            "delta_t": 0.5,
            "solveAdjoint": True,
            "restart": False,
            "SteadyStateTolerance": 1e-3,
        }

        self.points = MaNTA.getNodes(
            config["Lower_boundary"],
            config["Upper_boundary"],
            config["Grid_size"],
            config["Polynomial_degree"],
        )

        self.adjointProblem = JAXAdjointProblem(self, self.g)

        self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np)
        self.runner.configure(config)

    def run(self, tFinal=None):
        if tFinal is not None:
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()

    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state.Variable[0]
        return 0.5 * u * u * params.D

    def sigma(self, index, state, x, t, params: NonlinearDiffusionParams):

        u = state.Variable[0]
        q = state.Derivative[0]
        return params.D * (u**params.a) * q

    def aux(self, index, state, x, t, params):
        a = state.Aux[0]
        u = state.Variable[0]
        return a - params.D * u * u

    def source(self, index, state, x, t, params: NonlinearDiffusionParams):
        y = x - params.SourceCentre
        u = state.Variable[0]
        a = state.Aux[0]
        return params.T_s * jnp.exp(-y * y / params.SourceWidth) + a - params.D * u * u

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3

    def InitialValue(self, index, x):
        return 0.3

    def InitialAuxValue(self, index, x):
        u0 = self.InitialValue(index, x)
        return self.params.D * u0 * u0


    def createAdjointProblem(self):
        return self.adjointProblem


def runMaNTA(params):
    transportSystem = JAXAuxTest(params)

    transportSystem.run(5.0)
    G, G_p = transportSystem.getAdjointGradients()
    uout = transportSystem.runner.Get_profile(0)
    return G, G_p


@jax.custom_jvp
def fun(params):
    G, G_p = runMaNTA(params)
    return G[0]


@fun.defjvp
def fun_jvp(primals, tangents):

    (params,) = primals
    (params_dot,) = tangents

    G, G_p = runMaNTA(params)
    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)

    return G[0], jnp.dot(G_p[0], params_dot_flatten)


params_new = NonlinearDiffusionParams.make(0.1, 0.1, 1.0)

print(jax.grad(fun)(params_new))
