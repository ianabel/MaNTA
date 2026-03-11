# %%
import MaNTA
import jax 
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import shard_map

# %%
import os


# %%
P = PartitionSpec

devices = jax.devices()
print(devices)
mesh = Mesh(devices, ('ax',))
# Create an array of random values:
# x = jax.random.normal(jax.random.key(0), (8192, 8192))

# and use jax.device_put to distribute it across devices:
# y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
# jax.debug.visualize_array_sharding(y)

# %%
from JAXAdjointProblem import JAXAdjointProblem
from typing import NamedTuple, Any
from functools import partial
"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""
vmap_axes = ({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0)
state_shard_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars": P(None)}
shard_map_specs = (state_shard_specs, P('ax',))
out_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars":P( None)}
# Need PyTree structure for class paramters to be able to compute adjoints

class NonlinearDiffusionParams(NamedTuple):
    SourceCentre: float
    D: float
    T_s: float
    a: float
    SourceWidth: float
   
    @classmethod
    def make(cls, config: MaNTA.TomlValue) -> 'NonlinearDiffusionParams':
        return cls(
             SourceCentre = config["SourceCentre"],
             D = config["D"],
             T_s = 50.0,
             a = config["a"],
             SourceWidth = 0.02
        )

class JAXNonlinearDiffusion(MaNTA.TransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid = None):
        super().__init__()
        self.nVars = 1
        self.nAux = 0
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        # This object will be passed to sigma and source functions
        self.params = NonlinearDiffusionParams.make(config)
        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))
        self.dSourcedvar = jax.jit(jax.grad(self.Sources, argnums=1))

    def SigmaFn( self, index, state, x, t ):
        return self.sigma(index, state, x, t, self.params)

    def Sources(self, index, state, x, t):
        return self.source(index, state, x, t, self.params)
    """
    Vectorized flux function 

    Parameters
    ----------
    Index: int
        Index of the channel to calculate fluxes for
    States: dict
        Global state made up of (nPoints x nVars) matrix with keys "Variable", "Derivative", "Flux", "Aux", "Scalars"
    Positions: vector
        Array of positions 
    t: float
        Time to calculate fluxes

    Returns
    -------
    Array of fluxes at each position

    """
    @partial(jax.jit, static_argnums=(0,))
    def SigmaFn_v( self, index, states, positions, t):
        x_s = jnp.array(positions) #jax.device_put(jnp.array(positions),sharding)
        sigma_vmap = jax.vmap(lambda s, p : self.sigma(index, s, p, t, self.params), in_axes=(vmap_axes))
        out = shard_map(sigma_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=P('ax',))(states, x_s)
        return out

    @partial(jax.jit, static_argnums=(0,))
    def Sources_v( self, index, states, positions, t ):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        source_vmap = jax.vmap(lambda s, p : self.source(index, s, p, t, self.params), in_axes=(vmap_axes))
        out = shard_map(source_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=P('ax',))(states, x_s)
        return out
        
    @partial(jax.jit, static_argnums=(0,))
    def dSigma(self, index, states, positions, t):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        g_vmap = jax.vmap(lambda s, p: jax.grad(self.sigma, argnums=1)(index, s, p, t, self.params), in_axes=(vmap_axes))
        out = shard_map(g_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=out_specs)(states,x_s)
        out["Scalars"] = []
        return out
    
    @partial(jax.jit, static_argnums=(0,))
    def dSources(self, index, states, positions, t):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        g_vmap = jax.vmap(lambda s, p: jax.grad(self.source, argnums=1)(index, s, p, t, self.params), in_axes=(vmap_axes))
        out = shard_map(g_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=out_specs)(states,x_s)
        out["Scalars"] = []
        return out

    
    """
    Sigma and source, and auxilliary functions to be overloaded in derived classes

    Parameters
    ----------
    index : int
        Variable index
    state : dict
        Dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar" arrays
    x : float
        Spatial location
    t : float
        Time
    params : NamedTuple
        Transport system parameters, passed for JAX PyTree compatibility
    Returns
    -------
    float
        Computed sigma or source term
    """
    def aux( self, index, state, x, t, params: NamedTuple):
        pass

    def dSigmaFn_dq( self, index, state, x, t):
        pass
    
    def dSigmaFn_du( self, index, state, x, t):
        pass
    
    def dSigma_dPhi( self, index, state, x, t):
        pass
        
    def dSources_du( self, index, state, x, t ):
        pass

    def dSources_dq( self, index, state, x, t ):
        pass

    def dSources_dsigma( self, index, state, x, t ):
        pass
    
    def dSources_dPhi( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Aux"]
    
    def AuxG( self, index, state, x, t):
        return self.aux(index, state, x, t, self.params)
    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state["Variable"][0]
        return 0.5 * u * u
    
    def sigma( self, index, state, x, t, params: NonlinearDiffusionParams ):
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return params.D*(u ** params.a) * q

    def source( self, index, state, x, t, params: NonlinearDiffusionParams ):
        y = x - params.SourceCentre
        return params.T_s*jnp.exp(-y*y/params.SourceWidth)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3
    
    def InitialValue(self, index, x):
        return 0.3

    
    def InitialDerivative( self, index, x ):
        return self.dInitialValue(index,x)
    
    def createAdjointProblem(self):
        adjointProblem = JAXAdjointProblem(self, self.g)
        adjointProblem.addUpperBoundarySensitivity(0)
        return adjointProblem
    

# %%
nl_config = {
    "D": 10.0,
    "SourceCentre": 0.3,
    "a" : 0.0,
}

config = {
    "OutputFilename": "gpu_test",
    "Polynomial_degree": 3,
    "Grid_size": 20,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.001,
    "Absolute_tolerance": [0.0001],
    "delta_t": 0.25,
    "SteadyStateTolerance": 1.0,
}

nl = JAXNonlinearDiffusion(nl_config)
runner = MaNTA.Runner(nl)
runner.configure(config)
runner.run_ss()

# %%


# %%


# %%



