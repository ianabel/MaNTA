"""Stellarator transport with density, ion energy and electron energy coupled.

The third generation of `stellarator.py` / `stellarator2.py`, and the one that
adds an ambipolar radial electric field: with `evolveDensity` on there are three
variables and one auxiliary, and the drift-kinetic solve is done at an `Er` the
aux row pins. With it off there is one variable, the ion energy, and the case
reduces to something close to `stellarator2.py`.

It is a separate module rather than a new revision of `stellarator2.py` because
its parameters are not the same ones -- `stellarator_state.StellaratorParams`
splits the single source into a particle and a heat source -- so a config
written for one does not construct the other. `stellarator2.py` still has four
consumers configured its way (`objective2.py`, `desc_optimize.py`,
`scan_eq.py`, `desc_optimize_bfgs.py` and the notebook), and none of this can be
run here to check a conversion. `scan_eq_ambipolar.py` and
`stellarator_example.py` are this module's drivers.

Ported from `Stellarator2.py` on `origin/optimize-mode` by three-way merge
against the common ancestor, so it carries that branch's physics *and* main's
interface migration. What the port itself changed: `nVars`, `nAux` and the
boundary flags are no longer assigned in the constructor -- they are read-only
now, derived from the `SystemSpec` that `buildSpec` builds from the config. See
`buildSpec` for what that spec is and why the variable order matters.

Not run. `desc`, `yancc` and `interpax` are not installable in the environment
this repository is tested in; see the README.
"""

import os

# These two used to be set by FFIRunner at import. They moved here when that
# module became library code inside the package: process-wide policy set as a
# side effect of importing a library applies to every caller, including ones
# that wanted the opposite.
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["EQX_ON_ERROR"] = "off"

import jax

# Globally silence all jax.debug.print statements
jax.debug.print = (
    None  # lambda *args, **kwargs: None if args or kwargs else lambda *a, **kw: None
)
from functools import partial
from scipy.constants import elementary_charge
import manta as MaNTA
from manta.jax import FFIRunner
from typing import NamedTuple
import yancc
from yancc_wrapper2 import yancc_data, compute_dke_sol
from manta.jax import State, Physics_Decorator
from stellarator_state import (
    StellaratorState,
    StellaratorDecorator,
    Channel,
)
from config import StellaratorParams, StellaratorConfig
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
from partial import HashablePartial

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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

        def _wrap_shard(self, *args):
            if self.params.config.useSharding:
                args_s = tuple(
                    jax.device_put(arg, data_sharding)
                    if not jnp.isscalar(arg)
                    else jax.device_put(arg, static_sharding)
                    for arg in args
                )
                return func(self, *args_s)
            else:
                return func(self, *args)

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
vmap_axes_wfield = (State.vmap_axes(), 0, None, 0, 0, 0, None, None, None, None)
vmap_axes_sources = (State.vmap_axes(), 0, None, 0, 0, 0, None)
"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""


def buildSpec(params: StellaratorParams) -> MaNTA.SystemSpec:

    bcs = {"lower": MaNTA.Neumann, "upper": MaNTA.Dirichlet}

    if not params.config.evolveDensity:
        return MaNTA.SystemSpec(
            variables=[MaNTA.Field("IonEnergy", "ion energy density", "n0 T0", **bcs)]
        )

    return MaNTA.SystemSpec(
        variables=[
            MaNTA.Field("Density", "particle density", "n0", **bcs),
            MaNTA.Field("IonEnergy", "ion energy density", "n0 T0", **bcs),
            MaNTA.Field("ElectronEnergy", "electron energy density", "n0 T0", **bcs),
        ],
        aux=[
            MaNTA.Aux(
                "Er",
                "radial electric field enforcing ambipolarity",
                "T0/e",
            )
        ],
    )


class StellaratorTransport(MaNTA.TransportSystem):
    def __init__(self, config, yancc_wrapper: yancc_data):
        solver_config = config["Solver"]
        st_config = config["Stellarator"]

        # Built and passed rather than assigned: see buildSpec. params is a
        # local until the base class exists, because until then there is no
        # C++ object behind self to hang attributes on.

        config = StellaratorConfig(**st_config)
        self.params = StellaratorParams(config)
        if "Superconvergent" in solver_config and solver_config["Superconvergent"]:
            raise RuntimeError(
                "Superconvergent is not compatible with this physics case"
            )
        MaNTA.TransportSystem.__init__(self, buildSpec(self.params))
        self.batch_size = (
            len(devices)
            if self.params.config.useBatching and self.params.config.useSharding
            else 0
        )
        if self.params.config.useSharding:
            print(f"Using batch size {self.batch_size}")

        self.xL = solver_config["Lower_boundary"]
        self.xR = solver_config["Upper_boundary"]
        # jax.device_put(yancc_wrapper, data_sharding)
        self.yancc_wrapper = yancc_wrapper
        self.points = yancc_wrapper.rho

        self.pnorm = self.params.constants.T0 * self.params.constants.n0
        self.field, self.vp, self.vpp = self.yancc_wrapper.get_fields()
        self.vp_interp = interpax.Akima1DInterpolator(self.points, self.vp, check=False)
        g = [self.StoredEnergy]

        self.adjointProblem = StellaratorAdjointProblem(
            self, g, self.yancc_wrapper, len(self.points)
        )

        self.runner = FFIRunner(
            self, self.points, 1, self.adjointProblem.np, spatialParameters=True
        )

        print("configuring")
        self.runner.configure(solver_config)
        # io_callback(lambda : self.runner.configure(solver_config), [], ordered = True)
        print("Successfully created StellaratorTransport object")

    def run(self, tFinal=None):
        if tFinal is not None:
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()
            self._report_objective_estimate()

    def objectiveEstimate(self):
        """G, its first-order correction to the fixed point, and a bound.

        A steady solve stops when ||F|| is small, not when G is, so a scan
        differencing G between two configurations is reading the answer moving
        *plus* each solve stopping short. `uncertainty` is a bound on the
        second, and is the number that says whether a difference between two
        points means anything.

        Empty for a time-marching run, and for a run configured without
        solveAdjoint -- there is no objective to estimate.
        """
        return self.runner.objectiveEstimate()

    def _report_objective_estimate(self):
        estimate = self.objectiveEstimate()
        if not estimate:
            return
        for i, (value, corrected, uncertainty) in enumerate(
            zip(estimate["value"], estimate["corrected"], estimate["uncertainty"])
        ):
            print(
                f"  G[{i}] = {value:.6e}, corrected {corrected:.6e}, "
                f"uncertainty {uncertainty:.2e}"
            )

    def G(self):
        return self.runner.Get_G()

    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def getPressure(self, points=None):
        ui = self.runner.Get_profile(0) / self.vp
        return 2.0 / 3.0 * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return self.InitialValue(index, self.xR)

    @Physics_Decorator
    @shard_inputs
    def ComputePhysics(self, states, positions, t):
        dke_data = eqx.filter_jit(
            eqx.filter_vmap(compute_dke_sol, in_axes=vmap_axes_wfield)
        )(
            states,
            positions,
            t,
            self.field,
            self.vp,
            self.vpp,
            self.yancc_wrapper.pitchgrid,
            self.yancc_wrapper.speedgrid,
            self.params,
            evolveDensity=self.params.config.evolveDensity,
        )
        sources = eqx.filter_jit(
            eqx.filter_vmap(self.compute_sources, in_axes=vmap_axes_sources)
        )(
            states,
            positions,
            t,
            self.field,
            self.vp,
            self.vpp,
            self.params,
        )

        return [dke_data[0], sources, dke_data[1]]

    @Physics_Decorator
    @shard_inputs
    def ComputePhysicsDerivatives(self, states, positions, t):

        if self.params.config.useBatching:
            # map only takes one argument, so pack everything into a tuple
            def dke_grad(args):
                (states, x, field, vp, vpp) = args
                return eqx.filter_jacrev(compute_dke_sol)(
                    states,
                    x,
                    t,
                    field,
                    vp,
                    vpp,
                    self.yancc_wrapper.pitchgrid,
                    self.yancc_wrapper.speedgrid,
                    self.params,
                    self.params.config.evolveDensity,
                )

            ddke_data = eqx.filter_jit(jax.lax.map)(
                dke_grad,
                (states, positions, self.field, self.vp, self.vpp),
                batch_size=self.batch_size,
            )
        else:
            ddke_data = eqx.filter_jit(
                eqx.filter_vmap(
                    eqx.filter_jacrev(compute_dke_sol), in_axes=vmap_axes_wfield
                )
            )(
                states,
                positions,
                t,
                self.field,
                self.vp,
                self.vpp,
                self.yancc_wrapper.pitchgrid,
                self.yancc_wrapper.speedgrid,
                self.params,
                self.params.config.evolveDensity,
            )
        dsources = eqx.filter_jit(
            eqx.filter_vmap(
                eqx.filter_jacrev(self.compute_sources), in_axes=vmap_axes_sources
            )
        )(
            states,
            positions,
            t,
            self.field,
            self.vp,
            self.vpp,
            self.params,
        )

        return [ddke_data[0], dsources, ddke_data[1]]

    @StellaratorDecorator
    def compute_sources(
        self, state: StellaratorState, x, t, field, vp, vpp, params: StellaratorParams
    ):

        if self.params.config.evolveDensity:
            sn = self.Sn(state, x, t, vp, params)
            spi = self.Spi(state, x, t, vp, params)
            spe = self.Spe(state, x, t, vp, params)
            return [sn, spi, spe]

        else:
            return [self.Spi(state, x, t, vp, params)]

    def Sn(self, state: StellaratorState, x, t, vp, params: StellaratorParams):
        return (
            vp
            * params.config.ParticleSourceHeight
            * jnp.exp(
                -((x - params.config.ParticleSourceCenter) ** 2)
                / (2 * params.config.ParticleSourceWidth**2)
            )
        )

    def Spi(self, state: StellaratorState, x, t, vp, params: StellaratorParams):
        return vp * (
            params.config.NBIPower
            * jnp.exp(
                -((x - params.config.NBICenter) ** 2) / (2 * params.config.NBIWidth**2)
            )
            + self.CollisionalEnergyExchange(state, params)
        )

    def Spe(self, state: StellaratorState, x, t, vp, params: StellaratorParams):
        return vp * (
            params.config.ECHPower
            * jnp.exp(
                -((x - params.config.ECHCenter) ** 2) / (2 * params.config.ECHWidth**2)
            )
            - self.CollisionalEnergyExchange(state, params)
        )

    def CollisionalEnergyExchange(
        self, state: StellaratorState, params: StellaratorParams
    ):
        return params.constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)

    def StoredEnergy(self, field, state, x, params: StellaratorParams):
        if self.params.config.evolveDensity:
            return (
                state.Variable[Channel.IonEnergy]
                + state.Variable[Channel.ElectronEnergy]
            )
        else:
            return state.Variable[0]

    @partial(jax.jit, static_argnums=(0,))
    def InitialValue(self, index, x):
        def constant_density(index, x):
            n = StellaratorState.initial_profile(
                x, self.params.config.EdgeDensity, self.params.config.n0
            )

            return 1.5 * self.params.config.EdgeTemperature * n * self.vp_interp(x)

        def ambipolar(index, x):
            def n0(x):
                return self.params.config.EdgeDensity * self.vp_interp(x)

            def ui0(x):
                return (
                    3.0
                    / 2.0
                    * self.params.config.EdgeDensity
                    * self.params.config.EdgeTemperature
                    * self.vp_interp(x)
                )

            def ue0(x):
                return (
                    3.0
                    / 2.0
                    * self.params.config.EdgeDensity
                    * self.params.config.EdgeTemperature
                    * self.vp_interp(x)
                )

            return jax.lax.switch(index, [n0, ui0, ue0], x)

        return jax.lax.cond(
            self.params.config.evolveDensity, ambipolar, constant_density, index, x
        )

    @partial(jax.jit, static_argnums=(0,))
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
        transport_system: StellaratorTransport,
        g,
        yancc_data: yancc_data,
        npoints,
    ):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g
        self.ng = len(self.g)  # g functions passed in as an array
        self.field, self.vp, self.vpp = yancc_data.get_fields()
        (self.field, self.vp, self.vpp) = eqx.filter_shard(
            (self.field, self.vp, self.vpp), data_sharding
        )
        self.yancc_wrapper = yancc_data
        boundary_field = yancc_data.fields_unstacked[-1]

        flat, _ = jax.flatten_util.ravel_pytree(
            (eqx.filter(boundary_field, eqx.is_array))
        )
        self.npoints = npoints
        # add 1 for vp and 1 for vpp, which we also take gradients with respect to
        # -2 is for NFP, B0 which we don't get gradients of
        self.np_cell = len(flat) - 2 + 1 + 1
        self.np = self.np_cell
        self.np_boundary = 0

        self.spatialParameters = True
        self.compute_sources = transport_system.compute_sources
        self.batch_size = transport_system.batch_size

        self.params = transport_system.params
        if self.params.config.evolveDensity:
            self.nVars = 3
            self.nAux = 1
        else:
            self.nVars = 1
            self.nAux = 0

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

        return grad_w_vprime.transpose()

    @MaNTA_Decorator
    def dg(self, i, states, positions):
        out = jax.vmap(
            jax.grad(self.g[i], argnums=1), in_axes=(0, State.vmap_axes(), 0, None)
        )(self.field, states, positions, self.params)
        return out

    @Physics_Decorator
    @shard_inputs
    def ComputePhysicsDerivatives(self, states, positions):

        tree_in = (self.field, self.vp, self.vpp)

        def dke_grad(tree, states, x):

            def dke_sol(tree, states, x):
                return compute_dke_sol(
                    states,
                    x,
                    0,
                    tree[0],
                    tree[1],
                    tree[2],
                    self.yancc_wrapper.pitchgrid,
                    self.yancc_wrapper.speedgrid,
                    self.params,
                    self.params.config.evolveDensity,
                )

            return eqx.filter_jacrev(dke_sol)(tree, states, x)

        ddke_data = eqx.filter_jit(
            eqx.filter_vmap(dke_grad, in_axes=(0, State.vmap_axes(), 0))
        )(tree_in, states, positions)

        def sources(tree, states, x):
            return self.compute_sources(
                states, x, 0, tree[0], tree[1], tree[2], self.params
            )

        dsources = eqx.filter_jit(
            eqx.filter_vmap(
                eqx.filter_jacrev(sources), in_axes=(0, State.vmap_axes(), 0)
            )
        )(tree_in, states, positions)

        def unravel(grad_out):
            grad_unstack = tree_unstack(grad_out)
            grad_unraveled = jnp.stack(
                [jax.flatten_util.ravel_pytree(g)[0] for g in grad_unstack], axis=0
            )
            return grad_unraveled.transpose()

        dsigma_out = []
        dsources_out = []
        daux_out = []
        for i in range(0, self.nVars):
            dsigma_out.append(unravel(ddke_data[0][i]))
            dsources_out.append(unravel(dsources[i]))

        for i in range(0, self.nAux):
            daux_out.append(unravel(ddke_data[1][i]))

        return [dsigma_out, dsources_out, daux_out]

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

    def addUpperBoundarySensitivity(self, i):
        self.UpperBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1

    def addLowerBoundarySensitivity(self, i):
        self.LowerBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1
