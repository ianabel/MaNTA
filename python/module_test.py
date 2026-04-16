from FFIRunner import FFIRunner
from TransportModule import TransportModule, AdjointModule
import MaNTA
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

ld_config = {
    "nVars": 1,
    "nAux": 0,
    "isLowerDirichlet": True, 
    "isUpperDirichlet": True,
}

solver_config = {
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

class LinearDiffusionParams(NamedTuple):
    Centre: float
    InitialWidth: float
    InitialHeight: float
    kappa: float
    
class LinearDiffusion(TransportModule):
    def __init__(self, params):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = ld_config["nVars"]
        self.nAux = ld_config["nAux"]
        self.isUpperDirichlet = ld_config["isUpperDirichlet"]
        self.isLowerDirichlet = ld_config["isLowerDirichlet"]
        self.params = params
    
    def sigma( self, index, state, x, t, params ):
        tprime = state["Derivative"]
        out = params.kappa * tprime[index]
        return out
    
    def source( self, index, state, x, t, params ):
        return 10.0 * (1 - params.Centre)
    
    def g(self, state, x, params):
        u = state["Variable"][0]
        return 0.5 * u * u
    
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0
    
    @eqx.filter_jit
    def InitialValue( self, index, x ):
        alpha = 1 / 0.02
        y = (x - self.params.Centre)
        return self.params.InitialHeight * jnp.exp(-alpha * y * y)

params = LinearDiffusionParams(0.1, 0.1, 0.1, 3.0)
ld = LinearDiffusion(params)
points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
ap = AdjointModule.from_transport_system(ld)
runner = FFIRunner(ld, points, ap.ng, ap.np)

runner.configure(solver_config)

runner.setAdjointProblem(ap)

runner.run_ss()

runner.run_adjoint_solve()

# @jax.custom_jvp
# def fun(params):

#     ld = LinearDiffusion(params)
#     runner.setTransportSystem(ld)
#     ap = AdjointModule.from_transport_system(ld)
#     runner.setAdjointProblem(ap)
#     G, _ = runner.run_adjoint_solve()
#     return G[0]

# @fun.defjvp
# def fun_jvp(primals, tangents):

#     params, = primals
#     params_dot, = tangents

#     G = fun(params)
#     _, G_p = runner.run_adjoint_solve()

#     params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)
#     out = jnp.dot(G_p.flatten(), params_dot_flatten)
#     #dot = jax.vmap(lambda g, g_p: jnp.dot(g, g_p), in_axes=(0, None))(G_p.flatten(), params_dot_flatten)


#     return G[0], out


# params_new = LinearDiffusionParams(0.1, 0.1, 0.0, 2.0)

# g1 = jax.grad(fun)

# print(g1(params_new))


