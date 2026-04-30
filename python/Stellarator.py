import MaNTA

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from FFIRunner import FFIRunner
import jax
# jax.config.update("jax_enable_compilation_cache", False)
# jax.config.update('jax_cpu_enable_async_dispatch', False)
import equinox as eqx
# jax.config.update("jax_log_compiles" ,True)
if "JAX_COMPILATION_CACHE_DIR" in os.environ:
    print("Using cache directory: " + os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
#explain cache misses
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.tree_util import tree_map
from jax.experimental import io_callback

P = PartitionSpec
devices = jax.devices()
print(devices)
mesh = Mesh(devices, ('axis',),axis_types=(jax.sharding.AxisType.Auto,))
data_sharding = NamedSharding(mesh, P("axis",))
static_sharding = NamedSharding(mesh, P())

from yancc.species import LocalMaxwellian
from yancc.solve import solve_dke

from State import State

def MaNTA_Decorator(func):
    def wrapper(self, index, states, positions, *args):
        states_ = State.from_manta(states)
        positions_ = jnp.array(positions)

        def _wrap_shard(self, *args):
            
            args_s = tuple(jax.device_put(arg, data_sharding) if not jnp.isscalar(arg) else jax.device_put(arg, static_sharding) for arg in args)
            return func(self, *args_s)
        
        res = _wrap_shard(self, index, states_, positions_, *args)

        if (isinstance(res, State)):
            return res.to_manta()
        else: 
            return res
    return wrapper

from functools import partial

from yancc_wrapper import yancc_data
import yancc

from typing import NamedTuple



def put_on_gpu(tree):

    def map_fn(leaf):
        return leaf
        # if not jnp.isscalar(leaf):
        #     if (jnp.mod(leaf.shape[0], len(devices))==0):
        #         return jax.device_put(leaf, data_sharding)
        #     else:
        #         return jax.device_put(leaf, static_sharding)
        #     # return eqx.filter_shard(leaf, data_sharding)
        #     # condition = jnp.mod(leaf.shape[0], len(devices)) == 0
        #     # # lpad = (len(self.points) - leaf.shape[0],0)
        #     # # print(lpad)
        #     # return jax.lax.cond(condition, lambda leaf : jax.device_put(leaf, data_sharding), lambda leaf : jax.device_put(leaf, static_sharding), leaf)
        # else:
        #     return jax.device_put(leaf, static_sharding)
    return jax.tree.map(map_fn, tree)


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
vmap_axes = (State.vmap_axes(), 0)
vmap_axes_wfield = (None, State.vmap_axes(),0, None, 0, 0, None)
vmap_axes_sources = (None, State.vmap_axes() ,0, None, None)

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

        self.params = put_on_gpu(StellaratorParams.from_config(st_config))


        self.xL = solver_config["Lower_boundary"]
        self.xR = solver_config["Upper_boundary"]
        self.yancc_wrapper = put_on_gpu(yancc_wrapper)#jax.device_put(yancc_wrapper, data_sharding)

        e = 1.6e-19
        self.pnorm = 1e20 * e * 1e3

        # Hold the MaNTA runner object in the class

        # %%

        self.field, self.vprime = self.yancc_wrapper.get_fields()
        self.field_shard = put_on_gpu(self.field)
        self.vprime_shard = jax.device_put(self.vprime, data_sharding) #put_on_gpu(self.field)
        # self.field_shard, self.vprime_shard = eqx.filter_shard((self.field, self.vprime), data_sharding)

        g = [self.StoredEnergy]

     
        # self.nc_flux = eqx.filter_jit(flux).trace()
        
        self.adjointProblem = StellaratorAdjointProblem(self, g, self.yancc_wrapper, len(self.points))

        self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np, spatialParameters=True)
        self.adjoint_output = [
            jax.ShapeDtypeStruct((1,),jnp.float64),
            {"G_p": jax.ShapeDtypeStruct((1 * len(self.points), self.adjointProblem.np), jnp.float64)}
        ]  
        print("configuring")
        self.runner.configure(solver_config)

        print("Successfully created StellaratorTransport object")

    def run(self, tFinal = None):
        # if (tFinal is not None):
        #     self.runner.run(tFinal)
        # else:
            # eqx.filter_pure_callback(self.runner.run_ss, [], result_shape_dtypes=[])
        jax.debug.callback(self.runner.run_ss,ordered=True)

        G, G_p = io_callback(self.runner.runAdjointSolve, self.adjoint_output, ordered=True)
        
        return jnp.float32(G), jnp.float32(G_p["G_p"])  
            # self.runner.Run_ss()
            # io_callback(self.runner.run_ss, [], ordered=True)

        # return self.runAdjointSolve()
            # io_callback(self.runner.run_ss, [], ordered=True)
            # self.runner.run_ss()
            # self.runner.run_ss()
            # self.runner.Run_ss
    def runAdjointSolve(self):
        # if (field is not None):
        #     yancc_wrapper = yancc_data.from_other(field, grid, other=self.yancc_wrapper)
        #     self.run(yancc_wrapper=yancc_wrapper)

        G, G_p = io_callback(self.runner.runAdjointSolve, self.adjoint_output, ordered=True) #self.runner.Run_adjoint_solve()
        return jnp.float32(G), jnp.float32(G_p["G_p"])  

    def getPressure(self, points = None):

        ui = jnp.float32(self.runner.get_profile(0))
        return 2./3. * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 1.5 * self.params.EdgeTemperature * self.Density(self.xR)
    
    @MaNTA_Decorator
    def SigmaFn_v( self, index, states: State, positions, t ):
        sigma_vmap = eqx.filter_vmap(self.sigma, in_axes=(vmap_axes_wfield))
        out = sigma_vmap(index , states, positions, t, self.field_shard, self.vprime_shard, self.params)
        return out
    
    @MaNTA_Decorator
    def Sources_v( self, index, states: State, positions, t ):
        source_vmap = eqx.filter_vmap(self.source, in_axes=(vmap_axes_sources))
        return source_vmap(index, states, positions, t, self.params)
    
    @MaNTA_Decorator
    def dSigma(self, index, states: State, positions, t):
        dsigma_vmap = eqx.filter_vmap(jax.grad(self.sigma,argnums=1), in_axes=(vmap_axes_wfield))
        out = dsigma_vmap(index, states, positions, t, self.field_shard, self.vprime_shard, self.params)
        return out

    @MaNTA_Decorator
    def dSources(self, index, states: State, positions, t):
        grad = jax.grad(self.source, argnums=1)
        g_vmap = eqx.filter_vmap(grad, in_axes=(vmap_axes_sources))
        out = g_vmap(index, states, positions, t, self.params)
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
        put = lambda x : jax.device_put(x, static_sharding)
        n, nprime = put(jax.value_and_grad(self.Density)(x))

        p_i = 2. / 3. * state.Variable[0]
        p_i_prime = 2. / 3. * state.Derivative[0]
        
        dndrho = nprime * vprime
        Erho = put(jnp.array(0.0))
        Ti = p_i / n
        dTidrho = (p_i_prime* vprime - Ti*dndrho) / n
       
        species = [
        LocalMaxwellian(
            # can just give mass and charge in units of proton mass and elementary charge
            yancc.species.Species(1,1), 
            temperature=Ti * self.yancc_wrapper.Tnorm, 
            density=n * self.yancc_wrapper.nNorm, 
            dTdrho=dTidrho * self.yancc_wrapper.Tnorm, 
            dndrho=dndrho * self.yancc_wrapper.nNorm),
        ]
        
        _, _, fluxes, _  = eqx.filter_jit(solve_dke)(field, self.yancc_wrapper.pitchgrid, self.yancc_wrapper.speedgrid, species, Erho)

        fout = fluxes['<heat_flux>'][0] * vprime / (self.yancc_wrapper.FluxNorm)
        return -fout

    def source( self, index, state, x, t, params: NamedTuple ):

        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    def StoredEnergy(self, field, state, x, params):
        u = state.Variable[0]
        return u * self.pnorm
    
    def Density(self, x):
        return (self.params.n0 - self.params.EdgeDensity) * (1 - x * x) + self.params.EdgeDensity
    
    @partial(jax.jit, static_argnums=(0,))
    def dSources_dPhi( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Aux"]
    
    @partial(jax.jit, static_argnums=(0,))
    def InitialValue( self, index, x ):
        return 1.5 * self.params.EdgeTemperature * self.Density(x)
    
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
    def __init__(self, transport_system: MaNTA.TransportSystem, g, yancc_data : yancc_data, npoints):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g
        self.ng = len(self.g) # g functions passed in as an array
        self.field, self.vprime = yancc_data.get_fields()

        # self.field_shard, self.vprime_shard = eqx.filter_shard((self.field, self.vprime), data_sharding)
        self.field_shard = put_on_gpu(self.field)
        self.vprime_shard = jax.device_put(self.vprime, data_sharding) 
  
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
    
    @MaNTA_Decorator
    def gFn(self, i, states, positions):

        out =  jax.vmap(self.g[i], in_axes=(0, State.vmap_axes(), 0, None))(self.field, states, positions, self.params)
        return out

    @MaNTA_Decorator
    def dgFndp(self, i, states, positions):

        # filter_grad needs the differentiated parameter to be first so we have to use a lambda to swap the argument order
        fgrad = eqx.filter_grad(self.g[i]) #eqx.filter_grad(lambda field: self.g[i](state, x, field, self.params))
        fgrad_vmap = eqx.filter_vmap(fgrad, in_axes=(0, State.vmap_axes(), 0, None))
        grad_out = fgrad_vmap(self.field, states, positions, self.params)
        grad, _ = jax.flatten_util.ravel_pytree(eqx.filter(grad_out, eqx.is_array))
        grad = jnp.expand_dims(grad,1)
        out = jnp.reshape(grad, (self.npoints, self.np_cell ))

        return out
    
    @MaNTA_Decorator
    def dg(self, i, states, positions):
        out = jax.vmap(jax.grad(self.g[i], argnums=1), in_axes=(0, State.vmap_axes(), 0, None))(self.field, states, positions,  self.params)  
        return out

    @MaNTA_Decorator
    def dSigma(self, i, states, positions):
        fgrad = eqx.filter_grad(lambda field, states, x, vprime: self.sigma(i, states, x, 0, field, vprime, self.params))
        out_vmap = jax.vmap(fgrad, in_axes=(0, State.vmap_axes(), 0, 0))
        grad, _ = jax.flatten_util.ravel_pytree(out_vmap(self.field_shard, states, positions, self.vprime_shard))
        grad = jnp.expand_dims(grad,1)
        out = jnp.reshape(grad, (self.np_cell, self.npoints ))
        return out
    
    def dSources(self, i, states, positions):

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

   
