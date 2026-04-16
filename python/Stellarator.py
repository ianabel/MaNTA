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

from jax.experimental import io_callback

# %%
P = PartitionSpec

devices = jax.devices()
print(devices)
mesh = Mesh(devices, ('ax',),axis_types=(jax.sharding.AxisType.Auto,))

from functools import partial

from yancc_wrapper import yancc_data, flux

from typing import NamedTuple

for name, target in MaNTA.runner_ffi_ops().items():
    jax.ffi.register_ffi_target(name, target)


def getStateAtIndex(states, i):
    out = {
        "Variable": states["Variable"][i,:],
        "Derivative": states["Derivative"][i,:],
        "Flux": states["Flux"][i,:],
        "Aux": states["Aux"][i,:],
        "Scalars":states["Scalars"]
    }
    return out

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
    def run_adjoint_solve(self):
        return jax.ffi.ffi_call("run_adjoint_solve_ffi", self.adjoint_output, has_side_effect=True)(obj=self.runner.get_address())
    def get_profile(self, var):
        return jax.ffi.ffi_call("get_solution_ffi", self.sol_output)(var, self.points, obj=self.runner.get_address())

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

"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""
class StellaratorTransport(MaNTA.TransportSystem): 
    def __init__(self, config , eq = None, grid = None):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1

        ### Remember to set boundary conditions ####
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False
        solver_config = config["Solver"]
        st_config = config["Stellarator"]
        self.points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
        self.params = StellaratorParams.from_config(st_config)
        self.yancc_wrapper = yancc_data.from_eq(Volume=self.points, Density=self.Density, eq=eq, grid=grid)
        e = 1.6e-19
        self.pnorm = 1e20 * e * 1e3

        # Hold the MaNTA runner object in the class
        self.runner = MaNTA.Runner(self)

        # %%

        self.field, self.vprime = self.yancc_wrapper.get_fields()
        self.field_shard, self.vprime_shard = eqx.filter_shard((self.field, self.vprime), data_sharding)

        self.runner.configure(solver_config)

        g = [self.StoredEnergy]

        self.adjointProblem = StellaratorAdjointProblem(self, g, self.yancc_wrapper, len(self.points))
        self.runner.setAdjointProblem(self.adjointProblem)

        self.runner_ffi = FFI_Runner(self.runner, self.points, self.adjointProblem.np, self.adjointProblem.ng)

        print("Successfully created StellaratorTransport object")

    def run(self, tFinal = None, yancc_wrapper = None):
        if (yancc_wrapper is not None):
            self.yancc_wrapper = yancc_wrapper
            self.field, self.vprime = self.yancc_wrapper.get_fields()
            self.field_shard, self.vprime_shard = eqx.filter_shard((self.field, self.vprime), data_sharding)
            self.adjointProblem.setField(self.field, self.vprime)

        if (tFinal is not None):
            self.runner_ffi.run(tFinal)
        else: 
            self.runner_ffi.run_ss()


    def runAdjointSolve(self, field = None, grid = None):
        if (field is not None):
            yancc_wrapper = yancc_data.from_other(field, grid, other=self.yancc_wrapper)
            self.run(yancc_wrapper=yancc_wrapper)

        G, G_p = self.runner_ffi.run_adjoint_solve()
        return G, G_p

    def getPressure(self):
        ui = io_callback(self.runner.getSolution, [jax.ShapeDtypeStruct((len(self.points),), jnp.float32)], 0, self.points)
        print(ui)
        return 2./3. * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 1.5 * self.params.EdgeTemperature * self.Density(1.0)

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
        return -flux(state, x, field, vprime, self.yancc_wrapper) 

    def source( self, index, state, x, t, params: NamedTuple ):
        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    def StoredEnergy(self, field, state, x, params):
        u = state["Variable"][0]
        return u * self.pnorm
    
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
        pass

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
        pass

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
        out = jnp.reshape(grad, (self.np_cell, self.npoints ))
        print(out.shape)
        # out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
        return out.transpose()
        
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

   
