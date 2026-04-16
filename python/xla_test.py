import jax
import MaNTA
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# jax.ffi.register_ffi_type(
#     "runner", MaNTA.runner_type(), platform="cpu")
for name, target in MaNTA.runner_ffi_ops().items():
    jax.ffi.register_ffi_target(name, target)

from typing import NamedTuple

from functools import partial

from jax.experimental import io_callback

from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem


import jax.numpy as jnp

import equinox as eqx

class FFI_Runner:
    def __init__(self, runner, points, np, ng):
        self.runner = runner
        self.points = points
        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,), jnp.float64),
            jax.ShapeDtypeStruct((ng, np), jnp.float64)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),), jnp.float64)
    def run(self, tFinal):
        jax.ffi.ffi_call("run_ffi", [], has_side_effect=True)(jnp.float64(tFinal), obj=self.runner.get_address())
    def run_ss(self):
        jax.ffi.ffi_call("run_ss_ffi", [], has_side_effect=True)(obj=self.runner.get_address())
    def run_adjoint_solve(self):
        return jax.ffi.ffi_call("run_adjoint_solve_ffi", self.adjoint_output, has_side_effect=True)(obj=self.runner.get_address())
    def get_profile(self, var):
        return jax.ffi.ffi_call("get_solution_ffi", self.sol_output)(var, self.points, obj=self.runner.get_address())

#jax.ffi.register_ffi_target("runner_ffi_ops", MaNTA.runner_ffi_ops(), platform="cpu")

# jax.ffi.register_ffi_target("runner", MaNTA.Runner.handler(), platform="cpu")
# for name, target in MaNTA.Runner.ffi_ops().items():
#     jax.ffi.register_ffi_target(name, target, platform="cpu")

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
           
        self.runner_ffi = FFI_Runner(self.runner, self.points, self.adjointProblem.np, self.adjointProblem.ng)


    def run(self, tFinal = None, params=None):
        if (params is not None):
            self.setParams(params)
        
        if (tFinal is not None):
            self.runner_ffi.run(tFinal)
            # self.call_run(tFinal)

        else:
            self.runner_ffi.run_ss()


    def runAdjointSolve(self, params = None):
        if (params is not None):
            self.run(params=params)

        # G, G_p = io_callback(self.runner.runAdjointSolve, self.adjointoutput, ordered=True) 
        G, G_p = self.runner_ffi.run_adjoint_solve()
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
fjit = jax.jit(ld.run)
fjit(5.0)

u = ld.runner_ffi.get_profile(0)
print(u)

G, G_p = ld.runAdjointSolve()
print(G)
print(G_p)

@jax.custom_jvp
def fun(params):
    G, G_p = ld.runAdjointSolve(params=params)
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

g1 = eqx.filter_jit(jax.grad(fun))
# # # #g2 = eqx.filter_jit(jax.grad(fun))
print(g1(params_new))

# params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 1.0)
# print(g1(params_new))
#print(gprint(g1(params_new))2(params_new))










# runner = jax.ffi.ffi_call("runner", None)()
# print(runner)