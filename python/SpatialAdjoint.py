# %%
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import sys

import MaNTA

# %%
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp

import jax.scipy.special as sci
import jax.scipy.integrate as integrate

from functools import partial

import interpax

from jax.flatten_util import ravel_pytree

jax.config.update('jax_enable_x64', True)

# jax.ffi.register_ffi_type(
#     "runner", MaNTA.runner_type(), platform="cpu")
for name, target in MaNTA.runner_ffi_ops().items():
    jax.ffi.register_ffi_target(name, target)

class FFI_Runner:
    def __init__(self, runner, points, np, ng):
        self.runner = runner
        self.points = points
        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,), jnp.float64),
            jax.ShapeDtypeStruct((ng * len(self.points), np), jnp.float64)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),), jnp.float64)
    def run(self, tFinal):
        jax.ffi.ffi_call("run_ffi", [], has_side_effect=True)(jnp.float64(tFinal), obj=self.runner.get_address())
    def run_ss(self):
        jax.ffi.ffi_call("run_ss_ffi", [], has_side_effect=True)(obj=self.runner.get_address())
    def runAdjointSolve(self):
        out = jax.ffi.ffi_call("run_adjoint_solve_ffi", self.adjoint_output, has_side_effect=True)(obj=self.runner.get_address())
        return out
    def get_profile(self, var):
        out = jax.ffi.ffi_call("get_solution_ffi", self.sol_output)(var, self.points, obj=self.runner.get_address())
        return out

vmap_axes_adj = (None, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None,  0)

class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g):
        MaNTA.AdjointProblem.__init__(self)
        self.params = transport_system.params
        self.g = g


        self.ng = 1
        self.np = len(transport_system.params)
        # print(self.np_cell)
        self.npoints = len(transport_system.points)
        # self.np = self.np_cell * self.npoints
        self.np_boundary = 0
        self.spatialParameters = True
        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}



    def setParams(self, params):
        self.params = params

    def gFn(self, i, states, positions):
        x = jnp.array(positions)
        out =  jax.vmap(self.g, in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, 0))(states, x, self.params)
        return out

    #@partial(jax.jit, static_argnums=(0,1))
    def dgFndp(self, gIndex, states, positions):
        x = jnp.array(positions)
        dgdp = jax.vmap(jax.grad(self.g, argnums=2), in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, 0))(states, x, self.params)
        g, _ = ravel_pytree(dgdp)
        g = jnp.reshape(g, (self.np - self.np_boundary, len(positions)))

        out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
        return out.transpose()

    @partial(jax.jit, static_argnums=(0,))
    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g, argnums=0), in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, 0))(states, x, self.params)  
        out["Scalars"] = []
        return out

    #@partial(jax.jit, static_argnums=(0,))
    def dSigma(self, i, states, positions):
        x = jnp.array(positions)
        grad = jax.vmap(jax.grad(self.sigma, argnums=4), in_axes=(vmap_axes_adj))(i, states, x, 0.0, self.params)  
        grad_flattened, _ = jax.flatten_util.ravel_pytree(grad)
        grad_flattened = jnp.expand_dims(grad_flattened, 1)
        out = jnp.reshape(grad_flattened, (self.np, self.npoints ))
        return out
    
    
    @partial(jax.jit, static_argnums=(0,))
    def dSources(self, i, states, positions):
        x = jnp.array(positions)
        grad = jax.vmap(jax.grad(self.source, argnums=4), in_axes=(vmap_axes_adj))(i, states, x, 0.0, self.params)  
        grad_flattened, _ = jax.flatten_util.ravel_pytree(grad)
        grad_flattened = jnp.expand_dims(grad_flattened, 1)
        out = jnp.reshape(grad_flattened, (self.np, self.npoints ))
        return out

    @partial(jax.jit, static_argnums=(0,))
    def dgFn_dphi(self, i, state, x):
        return jax.grad(self.g, argnums=0)(state, x, self.params)["Aux"]
   
    def dAux_dp(self, index, pIndex, state, x):
        return self.daux_dp(index, state, x, 0.0, self.params )[pIndex]
    
    def computeUpperBoundarySensitivity(self, i, pIndex):
        if (i, pIndex) in self.UpperBoundarySensitivities:
            return True
        else:
            return False
        
    def computeLowerBoundarySensitivity(self, i, pIndex):
        if (i, pIndex) in self.LowerBoundarySensitivities:
            return True
        else:
            return False
    
    def getName(self, pIndex):
        if pIndex < len(self.params):
            return list(self.params._fields)[pIndex]
        else:
            return "BoundaryCondition"+str(pIndex)
        
    def addUpperBoundarySensitivity(self, i):
        self.UpperBoundarySensitivities[(i,self.np)] = True
        self.np += 1
        self.np_boundary += 1

    def addLowerBoundarySensitivity(self, i):
        self.LowerBoundarySensitivities[(i,self.np)] = True
        self.np += 1
        self.np_boundary += 1
    
   

# %%
from typing import NamedTuple
from VectorizedTransportSystem import VectorizedTransportSystem

vmap_axes = ({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, 0)

class JAXNonlinearDiffusion(VectorizedTransportSystem):
    def __init__(self, kappa):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False
        
        self.T_s = 50.0
        self.SourceWidth = 0.02
        self.SourceCentre = 0.3

        config = {
            "OutputFilename": "out",
            "Polynomial_degree": 4,
            "Grid_size": 20,
            "tau": 1.0, 
            "Lower_boundary": 0.0,
            "Upper_boundary": 1.0,
            "Relative_tolerance": 0.01,
            "delta_t": 1.0,
            "restart": False,
            "solveAdjoint": True, 
        }

        self.points = MaNTA.getNodes(config["Lower_boundary"], config["Upper_boundary"], config["Grid_size"], config["Polynomial_degree"])

        self.D = lambda x, kappa : (x + 1) * kappa

        self.params = {
            "D": self.D(self.points, kappa),
            "T_s": self.T_s * jnp.ones(len(self.points)),
            "a": 2.0 * jnp.ones(len(self.points)),
            "SourceWidth": self.SourceWidth * jnp.ones(len(self.points)),
            "SourceCentre": self.SourceCentre * jnp.ones(len(self.points))
        }
        self.runner = MaNTA.Runner(self)

        self.runner.configure(config)

        self.adjointProblem = JAXAdjointProblem(self, self.g)
        self.runner.setAdjointProblem(self.adjointProblem)

        self.runner_ffi = FFI_Runner(self.runner, self.points, self.adjointProblem.np, self.adjointProblem.ng)
        # This object will be passed to sigma and source functions
    
    def run(self, tFinal = None, kappa = None):
        if (kappa is not None):
            self.params["D"] = self.D(self.points, kappa)
        
        if (tFinal is not None):
            self.runner_ffi.run(tFinal)
            #sFinal = self.runner.run(tFinal)
        else: 
            self.runner_ffi.run_ss()


    def runAdjointSolve(self, kappa = None):
        if (kappa is not None):
            self.params["D"] = self.D(self.points, kappa)
        G, G_p = self.runner_ffi.runAdjointSolve()
        return G, G_p

    def g(self, state, x, params):
        u = state["Variable"][0]
        return 0.5 * u * u

         
    def SigmaFn( self, index, state, x, t ):
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return self.D(x, 4.0)*(u ** 2.0) * q

    def Sources(self, index, state, x, t):
        y = x - self.SourceCentre
        return self.T_s*jnp.exp(-y*y/self.SourceWidth)

    def SigmaFn_v( self, index, states, positions, t):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p, params : self.sigma(index, s, p, t, params), in_axes=(vmap_axes))(states, x, self.params)

    #@partial(jax.jit, static_argnums=(0,))
    def Sources_v( self, index, states, positions, t ):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p, params : self.source(index, s, p, t, params), in_axes=(vmap_axes))(states, x, self.params)
        
    #@partial(jax.jit, static_argnums=(0,))
    def dSigma(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p, params: jax.grad(self.sigma, argnums=1)(index, s, p, t, params), in_axes=(vmap_axes))(states, x, self.params)
        out["Scalars"] = []
        return out
    
    #@partial(jax.jit, static_argnums=(0,))
    def dSources(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p, params: jax.grad(self.source, argnums=1)(index, s, p, t, params), in_axes=(vmap_axes))(states, x, self.params)
        out["Scalars"] = []
        return out
    
    def sigma( self, index, state, x, t, params ):
        
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return params["D"]*(u ** params["a"]) * q

    def source( self, index, state, x, t, params ):
        y = x - params["SourceCentre"]
        return params["T_s"]*jnp.exp(-y*y/params["SourceWidth"])


    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3
    
    def InitialValue(self, index, x):
        return 0.3
    
    def createAdjointProblem(self):
        pass

# %%
nl = JAXNonlinearDiffusion(4.0)
nl.run(tFinal = 5.0)

# %%
import numpy as np
from netCDF4 import Dataset

data = Dataset("./out.nc")

Vars = data.groups
Grid = jnp.array(np.array(data.groups["Grid"].variables["CellBoundaries"]))
t = jnp.array(np.array(data.variables["t"]))
x = jnp.array(np.array(data.variables["x"]))
u = jnp.array(np.array(Vars["Var0"].variables["u"]))
data.close()

# %%
# fig,ax = plt.subplots()
# ax.plot(x,u[-1,:])

# %%
G, G_p = nl.runAdjointSolve()

print(G_p)

u_interp = interpax.interp1d(nl.points, x, u[-1,:], method='cubic')

g_approx = lambda u : jnp.trapezoid(0.5 * u * u * nl.params["D"], nl.points)

print(g_approx(u_interp))
