from FFIRunner import FFIRunner
import jax
import os
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
    
flux_function = jax.jit(lambda q : q)
f2_ = jax.jit(lambda q : flux_function(q))



class JAXLinearDiffusion(VectorizedTransportSystem):
    def __init__(self, params, restart = False):
        super().__init__()

        self.nVars = 1

        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.params = params

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
            "restart" : restart,
            "SteadyStateTolerance": 1e-3
        }
        self.points = MaNTA.getNodes(config["Lower_boundary"], config["Upper_boundary"], config["Grid_size"], config["Polynomial_degree"])
        


        self.adjointProblem = JAXAdjointProblem(self, self.g)

        self.runner = FFIRunner(self, self.points, self.adjointProblem.ng, self.adjointProblem.np)

        self.runner.configure(config)



        # self.runner.setAdjointProblem(self.adjointProblem)
    


    def run(self, tFinal = None):
    
        if (tFinal is not None):
            self.runner.Run(tFinal)
            # self.call_run(tFinal)

        else:
            self.runner.run_ss()


    def runAdjointSolve(self):

        # G, G_p = io_callback(self.runner.runAdjointSolve, self.adjointoutput, ordered=True) 
        G, G_p = self.runner.Run_adjoint_solve()
        return G, G_p

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u

    def sigma( self, index, state, x, t, params ):
        tprime = state.Derivative
        out = params.kappa * f2_(tprime[index])
        return out
    
    @eqx.filter_jit
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

params = LinearDiffusionParams(0.1, 0.1, 0.1, 3.0)
ld = JAXLinearDiffusion(params)
ld.run()

print(ld.runner.get_profile(0, ld.points))

out = ld.runAdjointSolve()
print(out)

# u = ld.runner.get_profile(0)
# print(u)



# @jax.custom_jvp
# def fun(params):
#     ld = JAXLinearDiffusion(params, restart = False)
#     ld.run()
#     G, G_p = ld.runAdjointSolve()
#     return G[0]

# @fun.defjvp
# def fun_jvp(primals, tangents):

#     params, = primals
#     params_dot, = tangents

#     ld = JAXLinearDiffusion(params, restart = False)
#     ld.run()
#     G, G_p = ld.runAdjointSolve()

#     params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)
#     out = jnp.dot(G_p.flatten(), params_dot_flatten)
#     #dot = jax.vmap(lambda g, g_p: jnp.dot(g, g_p), in_axes=(0, None))(G_p.flatten(), params_dot_flatten)


#     return G[0], out

# params_new = LinearDiffusionParams(0.1, 0.1, 0.1, 5.0)

# g1 = jax.grad(fun)
# # # # # #g2 = eqx.filter_jit(jax.grad(fun))
# print(g1(params_new))

# # params_new = LinearDiffusionParams(0.1, 0.1, 0.1, 3.8)

# # print(g1(params_new))

# # params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 1.0)
# # print(g1(params_new))










# runner = jax.ffi.ffi_call("runner", None)()
# print(runner)