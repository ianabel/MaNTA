from functools import partial

from scipy.constants import elementary_charge
from FFIRunner import FFIRunner
from typing import NamedTuple
import yancc
from yancc_wrapper2 import yancc_data
from State import State, Physics_Decorator
from StellaratorState import StellaratorState, StellaratorParams, StellaratorDecorator
from yancc.solve import solve_dke
from yancc.species import LocalMaxwellian, Electron, Hydrogen
from desc.backend import tree_unstack
import interpax
from desc.batching import vmap_chunked
from jax.experimental import io_callback
from jax.tree_util import tree_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
import enum
import MaNTA
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# jax.config.update("jax_enable_compilation_cache", False)
# jax.config.update('jax_cpu_enable_async_dispatch', False)
# jax.config.update("jax_log_compiles" ,True)
if "JAX_COMPILATION_CACHE_DIR" in os.environ:
    print("Using cache directory: " + os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
# explain cache misses


P = PartitionSpec
devices = jax.devices()
print(devices)
mesh = Mesh(devices, ("axis",), axis_types=(jax.sharding.AxisType.Auto,))
data_sharding = NamedSharding(
    mesh,
    P(
        "axis",
    ),
)
static_sharding = NamedSharding(mesh, P())


def MaNTA_Decorator(func):
    """
    Converts from MaNTA to jax and vice versa, also performs sharding on inputs
    """

    def wrapper(self, index, states, positions, *args):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        positions_ = jnp.array(positions)

        # Empty arrays causes issues with jax.lax.map, other operations, so we remove them and then add them back after
        def _wrap_shard(self, *args):
            args_s = tuple(
                jax.device_put(arg, data_sharding)
                if not jnp.isscalar(arg)
                else jax.device_put(arg, static_sharding)
                for arg in args
            )
            return func(self, *args_s)

        result = _wrap_shard(self, index, states_, positions_, *args)

        if isinstance(result, State):
            # Recombine with empty arrays and convert back to MaNTA format
            return eqx.combine(result, empty).to_manta()
        else:
            return result

    return wrapper


def shard_inputs(func):
    def wrapper(self, states, positions, *args):
        # print(states)

        def _wrap_shard(self, *args):
            args_s = tuple(
                jax.device_put(arg, data_sharding)
                if not jnp.isscalar(arg)
                else jax.device_put(arg, static_sharding)
                for arg in args
            )
            return func(self, *args_s)

        result = _wrap_shard(self, states, positions, *args)
        return result

    return wrapper


def put_on_gpu(tree):
    #    def map_fn(leaf):
    #        if not jnp.isscalar(leaf):
    #            if (jnp.mod(leaf.shape[0], len(devices))==0):
    #                return jax.device_put(leaf, data_sharding)
    #            else:
    #                return jax.device_put(leaf, static_sharding)
    #        else:
    #            return jax.device_put(leaf, static_sharding)
    #    return jax.tree.map(map_fn, tree)
    return tree


# Magic tuple to make vmap work
vmap_axes = (State.vmap_axes(), 0)
vmap_axes_wfield = (State.vmap_axes(), 0, None, 0, 0, 0, None)
vmap_axes_sources = (None, State.vmap_axes(), 0, None, 0, None)
"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""


class StellaratorTransport(MaNTA.TransportSystem):
    def __init__(self, config, yancc_wrapper: yancc_data):
        MaNTA.TransportSystem.__init__(self)

        ### Remember to set boundary conditions ####
        self.isUpperDirichlet = True
        self.isLowerDirichlet = False
        solver_config = config["Solver"]
        st_config = config["Stellarator"]
        self.points = yancc_wrapper.rho
        # self.points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

        self.params = StellaratorParams(**st_config)

        if self.params.evolveDensity:
            self.nVars = 3
            self.nAux = 1
        else:
            self.nVars = 1
            self.nAux = 0

        self.xL = solver_config["Lower_boundary"]
        self.xR = solver_config["Upper_boundary"]
        # jax.device_put(yancc_wrapper, data_sharding)
        self.yancc_wrapper = yancc_wrapper

        e = 1.6e-19
        self.pnorm = 1e20 * e * 1e3
        self.field, self.vp, self.vpp = self.yancc_wrapper.get_fields()
        (self.field_shard, self.vp_shard, self.vpp_shard) = eqx.filter_shard(
            (self.field, self.vp, self.vpp), data_sharding
        )
        self.vp_interp = interpax.Akima1DInterpolator(self.points, self.vp, check=False)
        # g = [self.StoredEnergy]

        # self.adjointProblem = StellaratorAdjointProblem(
        #     self, g, self.yancc_wrapper, len(self.points)
        # )
        #
        np = 0
        self.runner = FFIRunner(self, self.points, 1, np, spatialParameters=True)

        print("configuring")
        self.runner.configure(solver_config)
        # io_callback(lambda : self.runner.configure(solver_config), [], ordered = True)
        print("Successfully created StellaratorTransport object")

    def run(self, tFinal=None):
        if tFinal is not None:
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()

    def G(self):
        return self.runner.Get_G()

    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def getPressure(self, points=None):
        ui = self.runner.Get_profile(0) / self.vp
        return 2.0 / 3.0 * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0  # self.uL

    def UpperBoundary(self, index, t):
        return self.InitialValue(index, self.xR)

    @Physics_Decorator
    @shard_inputs
    def ComputePhysics(self, states, positions, t):
        dke_data = eqx.filter_vmap(self.compute_dke_sol, in_axes=vmap_axes_wfield)(
            states,
            positions,
            t,
            self.field_shard,
            self.vp_shard,
            self.vpp_shard,
            self.params,
        )
        sources = eqx.filter_vmap(self.compute_sources, in_axes=vmap_axes_wfield)(
            states,
            positions,
            t,
            self.field_shard,
            self.vp_shard,
            self.vpp_shard,
            self.params,
        )

        return [dke_data[0], sources, dke_data[1]]

    @Physics_Decorator
    @shard_inputs
    def ComputePhysicsDerivatives(self, states, positions, t):
        ddke_data = eqx.filter_vmap(
            eqx.filter_jacrev(self.compute_dke_sol), in_axes=vmap_axes_wfield
        )(
            states,
            positions,
            t,
            self.field_shard,
            self.vp_shard,
            self.vpp_shard,
            self.params,
        )
        dsources = eqx.filter_vmap(
            eqx.filter_jacrev(self.compute_sources), in_axes=vmap_axes_wfield
        )(
            states,
            positions,
            t,
            self.field_shard,
            self.vp_shard,
            self.vpp_shard,
            self.params,
        )

        return [ddke_data[0], dsources, ddke_data[1]]

    """
    Computes physics information for stellarator model

    Parameters
    ----------
    index : int
        Variable index
    state : eqx.Module
        Object containing state information
    x : float
        Spatial location
    t : float
        Time
    field: yancc.Field
        Magnetic field object
    vp: float
        V'
    vpp: float
        V''
    params : NamedTuple
        Transport system parameters, passed for JAX PyTree compatibility
    Returns
    -------
    float
        Computed sigma or source term
    """

    @partial(jax.jit, static_argnums=(0,))
    @StellaratorDecorator
    def compute_dke_sol(
        self, state: StellaratorState, x, t, field, vp, vpp, params: StellaratorParams
    ):

        def constant_density(state, x, t, field, vp, vpp, params):
            Erho = jnp.array(0.0)

            species = [
                LocalMaxwellian(
                    Hydrogen,
                    temperature=state.Ti * self.yancc_wrapper.Tnorm_eV,
                    density=state.n * self.yancc_wrapper.nNorm,
                    dTdrho=state.dTidrho * self.yancc_wrapper.Tnorm_eV,
                    dndrho=state.dndrho * self.yancc_wrapper.nNorm,
                ),
            ]

            sol, info = solve_dke(
                put_on_gpu(field),
                self.yancc_wrapper.pitchgrid,
                self.yancc_wrapper.speedgrid,
                species,
                put_on_gpu(Erho),
            )
            flux = (
                -sol.get("<heat_flux>")[0]
                * vp
                / (
                    self.yancc_wrapper.Tnorm
                    * self.yancc_wrapper.nNorm
                    / self.yancc_wrapper.tnorm
                )
            )

            return [flux], []

        def ambipolar(state, x, t, field, vp, vpp, params):
            species = [
                LocalMaxwellian(
                    # can just give mass and charge in units of proton mass and elementary charge
                    Hydrogen,
                    temperature=state.Ti * self.yancc_wrapper.Tnorm_eV,
                    density=state.n * self.yancc_wrapper.nNorm,
                    dTdrho=state.dTidrho * self.yancc_wrapper.Tnorm_eV,
                    dndrho=state.dndrho * self.yancc_wrapper.nNorm,
                ),
                LocalMaxwellian(
                    # can just give mass and charge in units of proton mass and elementary charge
                    Electron,
                    temperature=state.Te * self.yancc_wrapper.Tnorm_eV,
                    density=state.n * self.yancc_wrapper.nNorm,
                    dTdrho=state.dTedrho * self.yancc_wrapper.Tnorm_eV,
                    dndrho=state.dndrho * self.yancc_wrapper.nNorm,
                ),
            ]

            sol, info = jax.jit(solve_dke)(
                put_on_gpu(field),
                self.yancc_wrapper.pitchgrid,
                self.yancc_wrapper.speedgrid,
                species,
                state.Er,
            )

            particle_flux = (
                -sol.get("<particle_flux>")[0]
                * vp
                / (self.yancc_wrapper.nNorm / self.yancc_wrapper.tnorm)
            )

            heat_flux = (
                -sol.get("<heat_flux>")
                * vp
                / (
                    self.yancc_wrapper.nNorm
                    * self.yancc_wrapper.Tnorm
                    / self.yancc_wrapper.tnorm
                )
            )

            aux_g_out = sol.get("J_rho") / (
                self.yancc_wrapper.nNorm * elementary_charge / self.yancc_wrapper.tnorm
            )

            return [particle_flux, heat_flux[0], heat_flux[1]], [aux_g_out]

        if self.params.evolveDensity:
            return ambipolar(state, x, t, field, vp, vpp, params)
        else:
            return constant_density(state, x, t, field, vp, vpp, params)

    @partial(jax.jit, static_argnums=(0,))
    @StellaratorDecorator
    def compute_sources(
        self, state: StellaratorState, x, t, field, vp, vpp, params: StellaratorParams
    ):

        if self.params.evolveDensity:
            sn = self.Sn(state, x, t, vp, params)
            spi = self.Spi(state, x, t, vp, params)
            spe = self.Spe(state, x, t, vp, params)
            return [sn, spi, spe]

        else:
            return [self.Spi(state, x, t, vp, params)]

    def Sn(self, state, x, t, vp, params):
        return (
            vp
            * params.ParticleSourceHeight
            * jnp.exp(
                -((x * x - params.ParticleSourceCenter) ** 2)
                / (2 * params.ParticleSourceWidth**2)
            )
        )

    def Spi(self, state, x, t, vp, params):
        return vp * (
            params.HeatSourceHeight
            * jnp.exp(
                -((x * x - params.HeatSourceCenter) ** 2)
                / (2 * params.HeatSourceWidth**2)
            )
            + self.CollisionalEnergyExchange(state)
        )

    def Spe(self, state, x, t, vp, params):
        return vp * (
            params.HeatSourceHeight
            * jnp.exp(
                -((x * x - params.HeatSourceCenter) ** 2)
                / (2 * params.HeatSourceWidth**2)
            )
            - self.CollisionalEnergyExchange(state)
        )

    def CollisionalEnergyExchange(self, state):

        pDiff = 10.0 * (state.pe - state.pi)
        return pDiff

    def StoredEnergy(self, field, state, x, params):
        u = state.Variable[0]
        return u

    @eqx.filter_jit
    def InitialValue(self, index, x):
        def constant_density(index, x):
            n = StellaratorState.initial_profile(
                x, self.params.EdgeDensity, self.params.n0
            )

            return 1.5 * self.params.EdgeTemperature * n * self.vp_interp(x)

        def ambipolar(index, x):
            def n0(x):
                return self.params.EdgeDensity * self.vp_interp(x)

            def ui0(x):
                return (
                    3.0
                    / 2.0
                    * self.params.EdgeDensity
                    * self.params.EdgeTemperature
                    * self.vp_interp(x)
                )

            def ue0(x):
                return (
                    3.0
                    / 2.0
                    * self.params.EdgeDensity
                    * self.params.EdgeTemperature
                    * self.vp_interp(x)
                )

            return jax.lax.switch(index, [n0, ui0, ue0], x)

        return jax.lax.cond(
            self.params.evolveDensity, ambipolar, constant_density, index, x
        )

    @eqx.filter_jit
    def InitialDerivative(self, index, x):
        return jax.grad(self.InitialValue, argnums=1)(index, x)

    def InitialAuxValue(self, index, x):
        return 0.0

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
    def __init__(
        self,
        transport_system: MaNTA.TransportSystem,
        g,
        yancc_data: yancc_data,
        npoints,
    ):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g
        self.ng = len(self.g)  # g functions passed in as an array
        self.field, self.vp, self.vpp = yancc_data.get_fields()
        (self.field_shard, self.vp_shard, self.vpp_shard) = eqx.filter_shard(
            (self.field, self.vp, self.vpp), data_sharding
        )
        boundary_field = yancc_data.fields_unstacked[-1]

        flat, _ = jax.flatten_util.ravel_pytree(
            (eqx.filter(boundary_field, eqx.is_array))
        )
        self.npoints = npoints
        # add 1 for vp and 1 for vpp, which we also take gradients with respect to
        self.np_cell = len(flat) - 1 + 1 + 1
        self.np = self.np_cell
        self.np_boundary = 0

        self.spatialParameters = True
        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.params = transport_system.params

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

    @MaNTA_Decorator
    def gFn(self, i, states, positions):
        out = jax.vmap(self.g[i], in_axes=(0, State.vmap_axes(), 0, None))(
            self.field, states, positions, self.params
        )
        return out

    @MaNTA_Decorator
    def dgFndp(self, i, states, positions):

        fgrad = eqx.filter_grad(self.g[i])
        fgrad_vmap = eqx.filter_vmap(fgrad, in_axes=(0, State.vmap_axes(), 0, None))
        grad_out = fgrad_vmap(self.field, states, positions, self.params)
        grad_unstack = tree_unstack(grad_out)
        grad_unraveled = jnp.stack(
            [jax.flatten_util.ravel_pytree(g)[0] for g in grad_unstack], axis=0
        )
        grad_w_vprime = jnp.pad(grad_unraveled, ((0, 0), (0, 2)), mode="constant")

        return grad_w_vprime

    @MaNTA_Decorator
    def dg(self, i, states, positions):
        out = jax.vmap(
            jax.grad(self.g[i], argnums=1), in_axes=(0, State.vmap_axes(), 0, None)
        )(self.field, states, positions, self.params)
        return out

    @Physics_Decorator
    @shard_inputs
    def ComputePhysicsDerivatives(self, states, positions):
        pass

    @MaNTA_Decorator
    def dSigma(self, i, states, positions):
        tree_in = (self.field_shard, self.vp_shard, self.vpp_shard)

        # set up lambda to take in (field, vp, vpp) as a single object
        def f_in(tree, states, x):
            return self.sigma(i, states, x, 0, tree[0], tree[1], tree[2], self.params)

        # compute gradient
        fgrad = eqx.filter_grad(f_in, has_aux=True)
        grad_out, _ = jax.vmap(fgrad, in_axes=(0, State.vmap_axes(), 0))(
            tree_in, states, positions
        )
        # Reshaping (this is where I think we may have issues)
        #   Meant to be a matrix of (nPoints x len(field))
        grad_unstack = tree_unstack(grad_out)
        grad_unraveled = jnp.stack(
            [jax.flatten_util.ravel_pytree(g)[0] for g in grad_unstack], axis=0
        )
        return grad_unraveled.transpose()

    @MaNTA_Decorator
    def dSources(self, i, states, positions):
        tree_in = (self.vp_shard,)

        def f_in(tree, states, x):
            return self.source(i, states, x, 0, tree[0], self.params)

        fgrad = eqx.filter_grad(f_in)
        grad_out = jax.vmap(fgrad, in_axes=(0, State.vmap_axes(), 0))(
            tree_in, states, positions
        )[0]
        grad_padded = jnp.pad(
            grad_out[:, jnp.newaxis], ((0, 0), (self.np - 2, 1)), mode="constant"
        )

        return grad_padded.transpose()

    def dgFn_dphi(self, i, state, x):
        pass
        # return jax.grad(self.g, argnums=0)(state, x, self.params)["Aux"]

    def dAux_dp(self, index, pIndex, state, x):
        pass
        # return self.daux_dp(index, state, x, 0.0, self.params )[pIndex]

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
            return "BoundaryCondition" + str(pIndex)

    def addUpperBoundarySensitivity(self, i):
        self.UpperBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1

    def addLowerBoundarySensitivity(self, i):
        self.LowerBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1


# %%
