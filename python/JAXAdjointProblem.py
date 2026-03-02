import jax
import jax.numpy as jnp
import MaNTA
from functools import partial

vmap_axes = (None, {"Variable": 1, "Derivative": 1, "Flux": 1, "Aux": 1, "Scalars": None}, 0, None,  None)

class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g):
        MaNTA.AdjointProblem.__init__(self)
        self.params = transport_system.params
        self.g = g

        self.np = len(transport_system.params)
        self.np_boundary = 0

        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

    def gFn(self, i, state, x):
        return self.g(state, x, self.params)

    @partial(jax.jit, static_argnums=(0,1))
    def dgFndp(self, i, state, x):
        if ( i < self.np-1 ):
            return jax.grad(self.g, argnums=2)(state, x, self.params)[i]
        else:
            return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g, argnums=0), in_axes=({"Variable": 1, "Derivative": 1, "Flux": 1, "Aux": 1, "Scalars": None}, 0, None))(states, x, self.params)  
        out["Scalars"] = []
        return out

    @partial(jax.jit, static_argnums=(0,))
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
    
   