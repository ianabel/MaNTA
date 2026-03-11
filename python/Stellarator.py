import MaNTA
import jax
import os

if os.environ["JAX_COMPILATION_CACHE_DIR"] is not None:
    print("Using cache directory: " + os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
#explain cache misses
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import shard_map
from jax.tree_util import tree_map
import equinox as eqx

# %%
P = PartitionSpec

devices = jax.devices()
print(devices)
mesh = Mesh(devices, ('ax',),axis_types=(jax.sharding.AxisType.Auto,))

from functools import partial

from yancc_wrapper import yancc_wrapper 

from typing import NamedTuple

def getStateAtIndex(states, i):
    out = {
        "Variable": states["Variable"][i,:],
        "Derivative": states["Derivative"][i,:],
        "Flux": states["Flux"][i,:],
        "Aux": states["Aux"][i,:],
        "Scalars":states["Scalars"]
    }
    return out

class StellaratorParams(NamedTuple):
    SourceCenter: float
    SourceHeight: float
    SourceWidth: float
    EdgeTemperature: float
    EdgeDensity: float
    n0: float

    @classmethod
    def from_config(cls, config: MaNTA.TomlValue):
        return cls(
            SourceCenter=config["SourceCenter"],
            SourceHeight=config["SourceHeight"],
            SourceWidth=config["SourceWidth"],
            EdgeTemperature=config["EdgeTemperature"],
            EdgeDensity=config["EdgeDensity"],
            n0=config["n0"]
        )

# Magic tuple to make vmap work
vmap_axes = ({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0)
vmap_axes_wfield = (None, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None},0, None, 0, 0, None)
state_shard_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars": P(None)}
shard_map_specs = (state_shard_specs, P('ax',))
shard_map_specs_wfield = (state_shard_specs, P(None),P(None),P(None),P(None))

out_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars":P( None)}
data_sharding = NamedSharding(mesh, P("ax",))
"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""
class StellaratorTransport(MaNTA.TransportSystem): 
    def __init__(self, config , grid: MaNTA.Grid = None):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1

        ### Remember to set boundary conditions ####
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        self.params = StellaratorParams.from_config(config)
        self.yancc_wrapper = yancc_wrapper(self.Density, 1e20, 1e3)

        self.fmap = []

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 1.5 * self.params.EdgeTemperature * self.Density(0.9)

    def SigmaFn( self, index, state, x, t ):
        field, vprime = self.yancc_wrapper.compute_field(x)
        flux = self.sigma(index, state, x, t, field, vprime, self.params)
        return flux
    
    def Sources(self, index, state, x, t):
        return self.source(index, state, x, t, self.params)
    
    def SigmaFn_v( self, index, states, positions, t):
        x = jnp.array(positions)
        (field, vprime) = eqx.filter_shard(self.yancc_wrapper.compute_fields(x), data_sharding)
        x_s = jax.device_put(x,data_sharding)
        states_s = jax.device_put(states, data_sharding)
        sigmavmap = jax.vmap(self.sigma, in_axes=(vmap_axes_wfield))
        out = sigmavmap(index, states_s, x_s, t, field, vprime, self.params)
        return out
    
    @partial(jax.jit, static_argnums=(0,1))
    def Sources_v( self, index, states, positions, t ):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        source_vmap = jax.vmap(lambda s, p : self.source(index, s, p, t, self.params), in_axes=(vmap_axes))
        out = shard_map(source_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=P('ax',))(states, x_s)
        return out
    
    def dSigma(self, index, states, positions, t):
        x = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        (field, vprime) = eqx.filter_shard(self.yancc_wrapper.compute_fields(x), data_sharding)
        x_s = jax.device_put(x,data_sharding)
        states_s = jax.device_put(states, data_sharding)
        fgrad = jax.grad(self.sigma,argnums=1)
        g_vmap = jax.vmap(fgrad, in_axes=(vmap_axes_wfield))
        out = g_vmap(index, states_s, x_s, t, field, vprime, self.params)
        out["Scalars"] = []
        return out

    @partial(jax.jit, static_argnums=(0,1))
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

    def sigma( self, index, state, x, t, field, vprime, params ):
        return -self.yancc_wrapper.flux(state, x, field, vprime) 

    def source( self, index, state, x, t, params: NamedTuple ):
        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    def g(self, state, x, params):
        u = state["Variable"][0]
        return u

    #@partial(jax.jit, static_argnums=(0,1))
    def dSigmaFn_dq( self, index, state, x, t):
        pass
    
    #@partial(jax.jit, static_argnums=(0,1))
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
    
    @partial(jax.jit, static_argnums=(0,1))
    def dSources_dPhi( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Aux"]
    
    @partial(jax.jit, static_argnums=(0,1))
    def InitialValue( self, index, x ):
        return 1.5 * self.params.EdgeTemperature * self.Density(x)
    
    @partial(jax.jit, static_argnums=(0,1))
    def InitialDerivative( self, index, x ):
        return jax.grad(self.InitialValue, argnums=1)(index, x)
    
    def InitialAuxValue(self, index, x):
        return 0.0

    # Constant density function
    def Density(self, x):
        return (self.params.n0 - self.params.EdgeDensity) * (1 - x*x) + self.params.EdgeDensity
    
    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """
    def createAdjointProblem(self):
        field, vprime = self.yancc_wrapper.compute_field(0.9)
        ap = StellaratorAdjointProblem(self, self.g, field, vprime)
        return ap

class StellaratorAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g, boundary_field, boundary_vprime):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g

        # only need to compute gradient of flux on the boundary 

        # pvals = outer flux surface

        # if xloc == outer, 
        self.field = boundary_field
        self.vprime = boundary_vprime

        flat, _ =  jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
        print(len(flat))
        self.np = len(flat)-1
        self.np_boundary = 0

        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.params = transport_system.params


        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

    def gFn(self, i, state, x):
        return self.g(state, x, self.params)

    def dgFndp(self, i, state, x):
        # g, _ = jax.flatten_util.ravel_pytree(eqx.filter_grad(lambda params: self.g(state, x, params))(self.field))
        # out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
        return jnp.zeros((self.np,))
        
    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g, argnums=0), in_axes=({"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))(states, x, self.params)  
        out["Scalars"] = []
        return out

    def dSigma(self, i, states, positions):

        boundary_state = getStateAtIndex(states, -1)
        out = eqx.filter_grad(lambda field: self.sigma(i, boundary_state, positions[-1], 0, field, self.vprime, self.params))(self.field)  
        out_flattened, self.unravel = jax.flatten_util.ravel_pytree(out)
        out_padded = jnp.zeros((len(out_flattened), len(positions)))
        out_padded = out_padded.at[:,-1].set(out_flattened)
        print(out_padded.shape)

        return out_padded
    
    def dSources(self, i, states, positions):
        #  boundary_state = getStateAtIndex(states, -1)
        # out = eqx.filter_grad(lambda field: self.source(i, boundary_state, positions[-1], 0, field, self.vprime, self.params))(self.field)  
        # out_flattened, _ = jax.flatten_util.ravel_pytree(out)
        # out_padded = jnp.zeros((len(out_flattened), len(positions)))
        # out_padded = out_padded.at[:,-1].set(out_flattened)
        # print(out_padded.shape)

        return jnp.zeros((self.np, len(positions)))

    def dgFn_dphi(self, i, state, x):
        pass
        #return jax.grad(self.g, argnums=0)(state, x, self.params)["Aux"]
   
    def dAux_dp(self, index, pIndex, state, x):
        pass
        #return self.daux_dp(index, state, x, 0.0, self.params )[pIndex]
    
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
    
   
