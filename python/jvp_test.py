from typing import NamedTuple

from functools import partial

import MaNTA

from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp
import jax
import equinox as eqx



class LinearDiffusionParams(NamedTuple):
    Centre: float
    InitialWidth: float
    InitialHeight: float
    kappa: float
    
class JAXLinearDiffusion(VectorizedTransportSystem):
    def __init__(self, params):
        super().__init__()

        self.nVars = 1

        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.params = params
        self.runner = MaNTA.Runner(self)

        # %%
        config = {
            "OutputFilename": "out",
            "Polynomial_degree": 5,
            "Grid_size": 10,
            "Lower_boundary": -1.0,
            "Upper_boundary":  1.0,
            "Relative_tolerance" : 0.01,
            "tFinal": 1.0,
            "delta_t": 0.5,
            "solveAdjoint": True, 
            "SteadyStateTolerance": 1e-3
        }

        self.runner.configure(config)

        self.points = self.runner.getPoints()

        self.adjointProblem = JAXAdjointProblem(self, self.g)
        self.runner.setAdjointProblem(self.adjointProblem)

        self.adjointoutput = [
            jax.ShapeDtypeStruct((self.adjointProblem.ng,), jnp.float32),
            jax.ShapeDtypeStruct((self.adjointProblem.ng, self.adjointProblem.np), jnp.float32)
        ]     

    def run(self, tFinal = None, params=None):
        if (params is not None):
            self.setParams(params)
        
        if (tFinal is not None):
            sFinal = jax.pure_callback(self.runner.run, [], tFinal)
       
        sFinal = jax.pure_callback(self.runner.run_ss, [])

        return sFinal

    def runAdjointSolve(self, params = None):
        if (params is not None):
            self.run(params=params)

        G, G_p = jax.pure_callback(self.runner.runAdjointSolve, self.adjointoutput) 
        return G, G_p

    def g(self, state, x, params):
        u = state["Variable"][0]
        return 0.5 * u * u
 
    def setParams(self, params):
        self.params = params

    def sigma( self, index, state, x, t, params ):
        tprime = state["Derivative"]
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
        adjointProblem = JAXAdjointProblem(self, self.g)
        return adjointProblem

params = LinearDiffusionParams(0.1, 0.1, 2.0, 1.0)
ld = JAXLinearDiffusion(params)

@jax.custom_jvp
def fun(params):
    G, _ = ld.runAdjointSolve(params=params)
    return G[0]

@fun.defjvp
def fun_jvp(primals, tangents):

    params, = primals
    params_dot, = tangents

    G, G_p = ld.runAdjointSolve(params=params)
    Gravel, _ = jax.flatten_util.ravel_pytree(G_p)
    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)

    return G[0], jnp.dot(Gravel, params_dot_flatten)

params_new = LinearDiffusionParams(0.1, 0.1, 2.0, 1.0)

grad = eqx.filter_jit(jax.grad(fun))
print(grad(params_new))






