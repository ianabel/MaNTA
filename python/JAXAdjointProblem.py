import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
import MaNTA
from functools import partial

from jax.flatten_util import ravel_pytree

vmap_axes = (None, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None,  None)

class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g):
        MaNTA.AdjointProblem.__init__(self)
        self.params = transport_system.params
        self.g = g

        self.ng = 1

        self.np = len(transport_system.params)
        self.np_boundary = 0
        self.spatialParameters=True

        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

    def setParams(self, params):
        self.params = params

    def gFn(self, i, state, x):
        return self.g(state, x, self.params)

    #@partial(jax.jit, static_argnums=(0,1))
    def dgFndp(self, gIndex, states, positions):
        x = jnp.array(positions)
        dgdp = jax.vmap(jax.grad(self.g, argnums=2), in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))(states, x, self.params)
        g, _ = ravel_pytree(dgdp)
        g = jnp.reshape(g, (self.np - self.np_boundary, len(positions)))

        out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
        return out

    @partial(jax.jit, static_argnums=(0,))
    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g, argnums=0), in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))(states, x, self.params)  
        out["Scalars"] = []
        return out

    #@partial(jax.jit, static_argnums=(0,))
    def dSigma(self, i, states, positions):
        x = jnp.array(positions)
        out = jax.vmap(jax.grad(self.sigma, argnums=4), in_axes=(vmap_axes))(i, states, x, 0.0, self.params)  
        return out
    
    
    @partial(jax.jit, static_argnums=(0,))
    def dSources(self, i, states, positions):
        x = jnp.array(positions)
        out = jax.vmap(jax.grad(self.source, argnums=4), in_axes=(vmap_axes))(i, states, x, 0.0, self.params)  
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
    
   