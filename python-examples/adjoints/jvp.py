import jax
import manta as MaNTA
from typing import NamedTuple

from functools import partial

from manta.jax import VectorizedTransportSystem, JAXAdjointProblem, FFIRunner

import jax.numpy as jnp

import equinox as eqx

class LinearDiffusionParams(NamedTuple):
    Centre: float
    InitialWidth: float
    InitialHeight: float
    kappa: float
    
class JAXLinearDiffusion(VectorizedTransportSystem):
    def __init__(self, params):
        super().__init__(MaNTA.numbered_spec(1))

        self.params = params

        # %%
        config = {
            "OutputFilename": "output",
            "Polynomial_degree": 5,
            "Grid_size": 10,
            "Lower_boundary": -1.0,
            "Upper_boundary":  1.0,
            "Relative_tolerance" : 0.01,
            "tFinal": 1.0,
            "delta_t": 0.5,
            "solveAdjoint": True, 
            "restart" : False,
            "SteadyStateTolerance": 1e-3
        }

        self.points = MaNTA.getNodes(config["Lower_boundary"], config["Upper_boundary"], config["Grid_size"], config["Polynomial_degree"])

        self.adjointProblem = JAXAdjointProblem(self, self.g)
        
        self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np)
        self.runner.configure(config)

    def run(self, tFinal = None):
        if (tFinal is not None):
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()


    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u
 
    def setParams(self, params):
        self.params = params

    def sigma( self, index, state, x, t, params ):
        tprime = state.Derivative
        out = params.kappa * tprime[index]
        return out
    
    def source( self, index, state, x, t, params ):
        return 10.0 * (1 - params.Centre)
  
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0
    
    def InitialValue( self, index, x ):
        alpha = 1 / 0.02
        y = (x - self.params.Centre)
        return self.params.InitialHeight * jnp.exp(-alpha * y * y)
    
    def createAdjointProblem(self):
        return self.adjointProblem


def runMaNTA(params):
    transportSystem = JAXLinearDiffusion(params)

    transportSystem.run(tFinal = 5.0)
    G, G_p = transportSystem.getAdjointGradients()
    uout = transportSystem.runner.Get_profile(0)
    return G, G_p

@jax.custom_jvp
def fun(params):
    G, G_p = runMaNTA(params)
    return G[0]

@fun.defjvp
def fun_jvp(primals, tangents):

    params, = primals
    params_dot, = tangents

    G, G_p = runMaNTA(params)
    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)

    return G[0], jnp.dot(G_p[0], params_dot_flatten)

params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 2.0)

print(jax.grad(fun)(params_new))
