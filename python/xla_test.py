import jax
import os
import MaNTA
# jax.config.update('jax_enable_x64', True)

ffi_ops = dict.fromkeys(["get_solution", "run_adjoint_solve", "run", "run_ss"])
def register_ffi_cpu(ops_dict):
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops().items(), list(ops_dict.keys())):
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="cpu")

def register_ffi_gpu(ops_dict):
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops_cuda().items(), list(ops_dict.keys())):
        print(dict_entry)
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="CUDA")
  

jax.lax.platform_dependent(ffi_ops, cpu=register_ffi_cpu, cuda=register_ffi_gpu)
print(ffi_ops)

from typing import NamedTuple

from functools import partial
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem


import jax.numpy as jnp

import equinox as eqx

class FFI_Runner:
    def __init__(self, runner, points, np, ng):
        self.runner = runner
        self.points = points
        
        self.dtype = jnp.float32
        #jax.lax.platform_dependent(self.dtype, cpu=output_dtype("cpu"),cuda=output_dtype("gpu"))

        print(self.dtype)
        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,), self.dtype),
            jax.ShapeDtypeStruct((ng, np), self.dtype)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),), self.dtype)
    def run(self, tFinal):
        return jax.ffi.ffi_call(ffi_ops["run"], [], has_side_effect=True)(self.dtype(tFinal), obj=self.runner.get_address())
    def run_ss(self):
        return jax.ffi.ffi_call(ffi_ops["run_ss"], [], has_side_effect=True)(obj=self.runner.get_address())
    def run_adjoint_solve(self):
        return jax.ffi.ffi_call(ffi_ops["run_adjoint_solve"], self.adjoint_output, has_side_effect=True)(obj=self.runner.get_address())
    def get_profile(self, var):
        return jax.ffi.ffi_call(ffi_ops["get_solution"], self.sol_output)(var, self.points, obj=self.runner.get_address())

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
fjit()

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

    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)
    out = jnp.dot(G_p.flatten(), params_dot_flatten)
    #dot = jax.vmap(lambda g, g_p: jnp.dot(g, g_p), in_axes=(0, None))(G_p.flatten(), params_dot_flatten)


    return G[0], out

params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 2.0)

g1 = eqx.filter_jit(jax.grad(fun))
# # # #g2 = eqx.filter_jit(jax.grad(fun))
print(g1(params_new))

# params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 1.0)
# print(g1(params_new))
#print(gprint(g1(params_new))2(params_new))










# runner = jax.ffi.ffi_call("runner", None)()
# print(runner)