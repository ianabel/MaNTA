import MaNTA

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"
from FFIRunner import FFIRunner
import jax
if "JAX_COMPILATION_CACHE_DIR" in os.environ:
    print("Using cache directory: " + os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
#explain cache misses
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import shard_map
from jax.tree_util import tree_map
from jax.experimental import io_callback
import equinox as eqx

# %%
P = PartitionSpec

devices = jax.devices()
print(devices)
mesh = Mesh(devices, ('ax',),axis_types=(jax.sharding.AxisType.Auto,))

from functools import partial

from yancc_wrapper import yancc_data, flux



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
vmap_axes_sources = (None, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None},0, None, None)
state_shard_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars": P(None)}
shard_map_specs = (None, state_shard_specs, P('ax',), None, None)
shard_map_specs_wfield = (state_shard_specs, P(None),P(None),P(None),P(None))

out_specs = {"Variable": P('ax',), "Derivative":  P('ax',), "Flux":  P('ax',), "Aux":  P('ax',), "Scalars":P( None)}
data_sharding = NamedSharding(mesh, P("ax",))

ops = MaNTA.runner_ffi_ops_cuda()


name = "run_ss_ffi_cuda"
jax.ffi.register_ffi_target(name, ops[name], platform="CUDA")



"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""
class StellaratorTransport(MaNTA.TransportSystem): 
    def __init__(self, config, yancc_wrapper : yancc_data):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nAux = 0

        ### Remember to set boundary conditions ####
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False
        solver_config = config["Solver"]
        st_config = config["Stellarator"]
        self.points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
        self.params = StellaratorParams.from_config(st_config)

        self.xL = solver_config["Lower_boundary"]
        self.xR = solver_config["Upper_boundary"]
        self.yancc_wrapper = yancc_wrapper
        e = 1.6e-19
        self.pnorm = 1e20 * e * 1e3

        # Hold the MaNTA runner object in the class

        # %%

        self.field, self.vprime = self.yancc_wrapper.get_fields()
        self.field_shard, self.vprime_shard = eqx.filter_shard((self.field, self.vprime), data_sharding)

        g = [self.StoredEnergy]

        self.nc_flux = jax.jit(flux)
        # self.nc_flux = eqx.filter_jit(flux).trace()
        
        self.adjointProblem = StellaratorAdjointProblem(self, g, self.yancc_wrapper, len(self.points))
        # self.runner = MaNTA.Runner(self)
        self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np, spatialParameters=True)
        # self.runner = MaNTA.Runner(self)
        io_callback(lambda : self.runner.configure(solver_config), [], ordered=True)

        print("Successfully created StellaratorTransport object")

    def run(self, tFinal = None):
        if (tFinal is not None):
            self.runner.Run(tFinal)
        else:
            # eqx.filter_pure_callback(self.runner.run_ss, [], result_shape_dtypes=[])
            # io_callback(self.runner.Run_ss, [], ordered=True)()
            jax.debug.callback(self.runner.run_ss)
            # self.runner.Run_ss()


    def runAdjointSolve(self):
        # if (field is not None):
        #     yancc_wrapper = yancc_data.from_other(field, grid, other=self.yancc_wrapper)
        #     self.run(yancc_wrapper=yancc_wrapper)

        G, G_p = self.runner.Run_adjoint_solve()
        return G, G_p

    def getPressure(self):
        ui = self.runner.get_profile(0)

        return 2./3. * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 1.5 * self.params.EdgeTemperature * self.yancc_wrapper.Density(self.xR)
    
    def SigmaFn_v( self, index, states, positions, t):

        x = jnp.array(positions)

        x_s = jax.device_put(x,data_sharding)
        states_s = jax.device_put(states, data_sharding)
        
        sigma_vmap = jax.vmap(self.sigma, in_axes=(vmap_axes_wfield))
        out = sigma_vmap(index, states_s, x_s, t, self.field_shard, self.vprime_shard, self.params)
       
        return out
    
    @eqx.filter_jit
    def Sources_v( self, index, states, positions, t ):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        source_vmap = jax.vmap(self.source, in_axes=(vmap_axes_sources))
        out = shard_map(source_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=P('ax',))(index, states, x_s, t, self.params)
        return out
    
    def dSigma(self, index, states, positions, t):
        x = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        x_s = jax.device_put(x,data_sharding)
        states_s = jax.device_put(states, data_sharding)

        dsigma_vmap = jax.vmap(jax.grad(self.sigma,argnums=1), in_axes=(vmap_axes_wfield))
        out = dsigma_vmap(index, states_s, x_s, t, self.field_shard, self.vprime_shard, self.params)

        out["Scalars"] = []
        return out

    @eqx.filter_jit
    def dSources(self, index, states, positions, t):
        x_s = jnp.array(positions)#jax.device_put(jnp.array(positions),sharding)
        grad = jax.grad(self.source, argnums=1)
        g_vmap = jax.vmap(grad, in_axes=(vmap_axes_sources))
        out = shard_map(g_vmap, mesh=mesh, in_specs=shard_map_specs,out_specs=out_specs)(index, states, x_s, t, self.params)
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
        return -self.nc_flux(state, x, field, vprime, self.yancc_wrapper)

    def source( self, index, state, x, t, params: NamedTuple ):
        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    def StoredEnergy(self, field, state, x, params):
        u = state["Variable"][0]
        return u * self.pnorm
    
    @partial(jax.jit, static_argnums=(0,))
    def dSources_dPhi( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Aux"]
    
    @partial(jax.jit, static_argnums=(0,))
    def InitialValue( self, index, x ):
        return 1.5 * self.params.EdgeTemperature * self.yancc_wrapper.Density(x)
    
    @partial(jax.jit, static_argnums=(0,))
    def InitialDerivative( self, index, x ):
        return jax.grad(self.InitialValue, argnums=1)(index, x)
        
    def aux( self, index, state, x, t, params: NamedTuple):
        pass

    def InitialAuxValue(self, index, x):
        pass

    def AuxG( self, index, state, x, t):
        return self.aux(index, state, x, t, self.params)

    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """
    def createAdjointProblem(self):
        return self.adjointProblem

class StellaratorAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g, yancc_data, npoints):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g
        self.ng = len(self.g) # g functions passed in as an array
        self.field, self.vprime = yancc_data.get_fields()
  
        boundary_field = yancc_data.fields_unstacked[-1]

        flat, _ =  jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
        self.npoints = npoints
        self.np_cell = len(flat)-1
        self.np = (self.np_cell)
        self.np_boundary = 0

        self.spatialParameters = True
        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.params = transport_system.params

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

        self.adjointoutput = [
            jax.ShapeDtypeStruct((self.ng,), jnp.float32),
            jax.ShapeDtypeStruct((self.ng * self.npoints, self.np), jnp.float32)
        ]     

    def setField(self, field, vprime):
        self.field = field
        self.vprime = vprime

    def unravel(self, grad):
        self.unravel(grad)
    
    def gFn(self, i, states, positions):
        x = jnp.array(positions)
        out =  jax.vmap(self.g[i], in_axes=(0, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))(self.field, states, x, self.params)
        return out

    def dgFndp(self, i, states, positions):

        # filter_grad needs the differentiated parameter to be first so we have to use a lambda to swap the argument order
        fgrad = eqx.filter_grad(self.g[i]) #eqx.filter_grad(lambda field: self.g[i](state, x, field, self.params))
        x = jnp.array(positions)
        fgrad_vmap = eqx.filter_vmap(fgrad, in_axes=(0,{"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))
        grad_out = fgrad_vmap(self.field, states, x, self.params)
        grad, self.unravel = jax.flatten_util.ravel_pytree(eqx.filter(grad_out, eqx.is_array))
        grad = jnp.expand_dims(grad,1)
        out = jnp.reshape(grad, (self.npoints, self.np_cell ))
        print(out.shape)
        # out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
        return out
        
    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g[i], argnums=1), in_axes=(0, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, None))(self.field, states, x,  self.params)  
        out["Scalars"] = []
        return out

    def dSigma(self, i, states, positions):
        
        field = eqx.filter_shard(self.field, data_sharding)
        vprime = eqx.filter_shard(self.vprime, data_sharding)
        x_s = jax.device_put(jnp.array(positions),data_sharding)
        states_s = jax.device_put(states, data_sharding)
        fgrad = eqx.filter_grad(lambda field, states, x, vprime: self.sigma(i, states, x, 0, field, vprime, self.params))
        out_vmap = jax.vmap(fgrad, in_axes=(0, {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}, 0, 0))
        grad, _ = jax.flatten_util.ravel_pytree(out_vmap(field, states_s, x_s, vprime))
        grad = jnp.expand_dims(grad,1)
        out = jnp.reshape(grad, (self.np_cell, self.npoints ))
        return out
    
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
    
# %%

   
