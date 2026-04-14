import jax

import MaNTA

from typing import NamedTuple

from functools import partial



from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem


import jax.numpy as jnp

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

        self.runner.configure(config)

        self.points = MaNTA.getNodes(config["Lower_boundary"], config["Upper_boundary"], config["Grid_size"], config["Polynomial_degree"])

        self.adjointProblem = JAXAdjointProblem(self, self.g)
        self.runner.setAdjointProblem(self.adjointProblem)

        self.adjointoutput = [
            jax.ShapeDtypeStruct((self.adjointProblem.ng,), jnp.float32),
            jax.ShapeDtypeStruct((self.adjointProblem.ng, self.adjointProblem.np), jnp.float32)
        ]     
        jax.ffi.register_ffi_target("run_ffi", self.runner.run_ffi(), platform="cpu")
        # self.call_run = jax.ffi.ffi_call("run_ffi", [], vmap_method="broadcast_all")


    def run(self, tFinal = None, params=None):
        if (params is not None):
            self.setParams(params)
        
        if (tFinal is not None):
            
            # sFinal = io_callback(self.runner.run, [], tFinal, ordered=True)
            self.runner.run(tFinal)
            # self.call_run(tFinal)

        else:
            self.runner.run_ss()


    def runAdjointSolve(self, params = None):
        if (params is not None):
            self.run(params=params)

        # G, G_p = io_callback(self.runner.runAdjointSolve, self.adjointoutput, ordered=True) 
        G, G_p = self.runner.runAdjointSolve()
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

params = LinearDiffusionParams(0.1, 0.1, 0.1, 3.0)
ld = JAXLinearDiffusion(params)
ld.run()
@jax.custom_jvp
def fun(params):
    G, G_p = ld.runAdjointSolve(params=params)
    print(G_p)
    return G[0]

@fun.defjvp
def fun_jvp(primals, tangents):

    params, = primals
    params_dot, = tangents

    G, G_p = ld.runAdjointSolve(params=params)
    Gravel, _ = jax.flatten_util.ravel_pytree(G_p)
    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)

    dot = jax.vmap(lambda g, g_p: jnp.dot(g, g_p), in_axes=(0, None))(Gravel, params_dot_flatten)


    return G[0], jnp.sum(dot)

params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 2.0)

print(fun(params_new))

g1 = jax.grad(fun)
#g2 = eqx.filter_jit(jax.grad(fun))
print(g1(params_new))

params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 1.0)
print(g1(params_new))
#print(gprint(g1(params_new))2(params_new))






