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

# check if we're running on multiple nodes
if "SLURM_NNODES" in os.environ:
    if int(os.environ["SLURM_NNODES"]) > 1:
        multinode = True
        jax.distributed.initialize()

        # Always print this to verify correct setup
        print(
            f"[{jax.process_index()}/{jax.process_count()}] devices:",
            jax.devices(),
            flush=True,
        )
        print(
            f"[{jax.process_index()}/{jax.process_count()}] local devices:",
            jax.local_devices(),
            flush=True,
        )
        print(
            f"I will be running a calculation among {jax.process_count()} tasks, "
            f"using a total of {len(jax.devices())} devices "
            f"({len(jax.local_devices())} per slurm task). "
            f"If this does not match your expected number of total devices, "
            f"something is misconfigured",
            flush=True,
        )
    else:
        multinode = False
else:
    multinode = False

from functools import partial
from typing import Callable
import manta as MaNTA
from manta.jax import FFIRunner
from yancc_wrapper import yancc_data

# import a simple flux model for testing purposes
# STELLARATOR_TEST_FLUX=physical swaps in a mock written in SI units instead,
# which goes through the same normalisation as the real solve; see
# check_normalization.py.
if "TEST_STELLARATOR" in os.environ:
    if os.environ.get("STELLARATOR_TEST_FLUX") == "physical":
        from dke_sol import physical_test_flux as compute_dke_sol
        from dke_sol import physical_test_flux_field_jac as dke_field_jac
    else:
        from dke_sol import test_flux as compute_dke_sol
        from dke_sol import test_flux_field_jac as dke_field_jac

    print("Running stellarator model in test context")
else:
    from dke_sol import compute_dke_sol, dke_field_jac

from itg_wrapper import ITGFlux, SumFlux

from manta.jax import State, Physics_Decorator, vmap_batched
from stellarator_state import (
    StellaratorState,
    StellaratorDecorator,
    Channel,
)
from config import StellaratorParams, StellaratorConfig, normalization_args
from desc.backend import tree_unstack
import interpax
from jax.sharding import PartitionSpec, NamedSharding
from jax.experimental import multihost_utils
import jax.numpy as jnp
import equinox as eqx

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
mesh = jax.make_mesh(
    (jax.device_count(),), ("axis",), axis_types=(jax.sharding.AxisType.Auto,)
)
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
        states_ = _pad_state(states_, self.pad_width)

        positions_ = jnp.pad(jnp.array(positions), (0, self.pad_width), mode="edge")

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


class FieldJac(eqx.Module):
    """`dke_field_jac` for any flux with `compute_dke_sol`'s signature.

    The jacobian over `(field, vp, vpp)` of the flux the case actually
    evaluates. With the ITG term on, that is `SumFlux`, which reads V' and V''
    as well as the neoclassical solve does; differentiating `compute_dke_sol`
    alone would drop that half. A module, not a closure, so `vmap_batched`'s
    `filter_jit` traces the ITG arrays rather than recompiling per run.
    """

    flux: Callable

    def __call__(
        self, states, positions, t, field, vp, vpp, pitchgrid, speedgrid, params,
        evolveDensity=False,
    ):
        def flux_of(tree_in):
            _field, _vp, _vpp = tree_in
            return self.flux(
                states, positions, t, _field, _vp, _vpp, pitchgrid, speedgrid,
                params, evolveDensity,
            )

        return eqx.filter_jacrev(flux_of)((field, vp, vpp))


def itg_geometry_jac(itg: ITGFlux, geometry, states, x, vp, vpp, params, evolveDensity):
    """d(ITG flux)/d(signals) at one point: the adjoint's ITG columns."""
    return eqx.filter_jacrev(
        lambda g: itg.local(g, states, x, 0.0, vp, vpp, params, evolveDensity)
    )(geometry)


vmap_axes_itg_geometry = (None, 0, State.vmap_axes(), 0, 0, 0, None, None)

# Magic tuple to make vmap work
vmap_axes = (State.vmap_axes(), 0)
vmap_axes_wfield = (State.vmap_axes(), 0, None, 0, 0, 0, None, None, None, None)
vmap_axes_sources = (State.vmap_axes(), 0, None, 0, 0, 0, None)
shard_map_specs = (
    P(
        "axis",
    ),
    P(
        "axis",
    ),
    P(),
    P(
        "axis",
    ),
    P(
        "axis",
    ),
    P(
        "axis",
    ),
    P(),
    P(),
    P(),
    P(),
)
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


# The two batched entry points this case serves. Everything else MaNTA calls --
# InitialValue, the boundary values -- is pointwise and stays on the calling
# thread.
BATCHED = (
    MaNTA.EvaluationEntry.ComputePhysics,
    MaNTA.EvaluationEntry.ComputePhysicsDerivatives,
)


def _pad_state(tree, pad_width):
    """Pads all array leaves of a PyTree along axis 0 by repeating the last entry."""
    dynamic, static = eqx.partition(tree, eqx.is_array)
    padded = jax.tree_util.tree_map(
        lambda leaf: jnp.pad(leaf, ((0, pad_width), (0, 0)), mode="edge"), dynamic
    )
    return eqx.combine(padded, static)


def _unpad_state(tree, length):
    """Slices all array leaves of a PyTree back to `length` along axis 0."""
    dynamic, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(
        jax.tree_util.tree_map(lambda leaf: leaf[:length, ...], dynamic), static
    )


class StellaratorTransport(MaNTA.TransportSystem):
    regrid = MaNTA.Regrid.InPlace

    def __init__(self, config, yancc_wrapper: yancc_data, itg: ITGFlux = None):
        """`itg`, an `itg_wrapper.ITGFlux`, adds the NN ITG heat flux to the
        neoclassical one; it is required when the config sets `useITGProxy`
        and refused otherwise, so the config says what physics a run had."""
        solver_config = config["Solver"]
        st_config = config["Stellarator"]
        normalization = config.get("Normalization", {})
        # what the adjoint differentiates: the name of one of the objective
        # methods below (StoredEnergy, ITGHeatFlux, ...)
        objective = config.get("Objective", "StoredEnergy")

        # Built and passed rather than assigned: see buildSpec. params is a
        # local until the base class exists, because until then there is no
        # C++ object behind self to hang attributes on.

        self.xL = solver_config["LowerBoundary"]
        self.xR = solver_config["UpperBoundary"]

        config = StellaratorConfig(rhoUpper=self.xR, **st_config)

        # The reference values everything is normalised to; see
        # config.DEFAULT_NORMALIZATION. Changing one changes what the
        # normalised entries of config["Stellarator"] mean, so a physically
        # identical run needs config.rescale_config as well.
        self.params = StellaratorParams(
            config, **normalization_args(normalization)
        )

        if self.params.config.useITGProxy and itg is None:
            raise ValueError("useITGProxy is set but no ITGFlux was passed")
        if itg is not None and not self.params.config.useITGProxy:
            raise ValueError("an ITGFlux was passed but useITGProxy is not set")
        self.itg = itg
        # what ComputePhysics evaluates: an eqx.Module when the ITG term is on,
        # so its arrays are traced, not baked in
        self.flux_fn = compute_dke_sol if itg is None else SumFlux(compute_dke_sol, itg)
        # if "Superconvergent" in solver_config and solver_config["Superconvergent"]:
        #     raise RuntimeError(
        #         "Superconvergent is not compatible with this physics case"
        #     )

        MaNTA.TransportSystem.__init__(self, buildSpec(self.params))
        self.batch_size = self.params.config.batch_size

        self.yancc_wrapper = yancc_wrapper
        self.points = MaNTA.getNodes(
            solver_config["GridPoints"],
            solver_config["PolynomialDegree"],
        )

        self.doTimeMarch = (
            True if solver_config.get("SteadyStateSolver") == "TimeMarch" else False
        )

        self.nchunks = self.params.config.nchunks
        self.max_batch = self.nchunks * self.batch_size

        self.pnorm = self.params.constants.T0 * self.params.constants.n0
        # self.field, self.vp, self.vpp = self.yancc_wrapper.get_fields()

        self.field, self.vp, self.vpp = self.yancc_wrapper.get_fields_at_rho(
            self.points
        )
        self.vp_interp = interpax.Akima1DInterpolator(self.points, self.vp, check=False)
        objectives = {
            "StoredEnergy": self.StoredEnergy,
            "FusionPower": self.FusionPower,
            "tauE": self.tauE,
            "TripleProduct": self.TripleProduct,
            "ITGHeatFlux": self.ITGHeatFlux,
        }
        if objective not in objectives:
            raise ValueError(
                f"unknown Objective {objective!r}; expected one of {sorted(objectives)}"
            )
        if objective == "ITGHeatFlux" and self.itg is None:
            raise ValueError("Objective = ITGHeatFlux needs the ITG term (useITGProxy)")
        g = [objectives[objective]]

        self.levels = []  # (cells, k) of each plan MaNTA handed over
        ### for adaptive grid problems ###

        # batch size M -> (rounds, padded Params). Set by prepareEvaluation from
        # MaNTA's plan; see there.
        self.layout = {}

        self.rounds = 0  # rounds the machine ran

        if self.params.config.adaptiveGrid:
            solver_config["tauScaling"] = "Diffusive"
            solver_config["tauUpdate"] = "ContinuationStep"
            solver_config["MeshAdaptation"] = True
            solver_config["MeshAdaptationThreshold"] = 1.25
            solver_config["DegreeTolerance"] = 1e-6
            solver_config["PhysicsParallelism"] = self.batch_size
            solver_config["MaxPhysicsBatch"] = self.max_batch
            solver_config.pop("GridPoints", None)

        self.adjointProblem = StellaratorAdjointProblem(
            self, g, self.yancc_wrapper, self.points
        )

        self.runner = FFIRunner(
            self, self.points, 1, self.adjointProblem.np, spatialParameters=True
        )

        self.runner.configure(solver_config)

    def reconfigure(self, solver_config):
        self.runner.configure(solver_config)
        return True

    def run(self, tFinal=None):
        # what getAdjointGradients hands the gradient read; only the sliced
        # steady solve below produces one
        self.steady_outcome = None
        if tFinal is not None:
            self.runner.Run(tFinal)
            return 0
        else:
            if self.params.config.adaptiveGrid or self.doTimeMarch:
                self.runner.Run_ss()
                return 1
            else:
                outcome = self.runner.start_steady()

                jax.debug.print(
                    "MaNTA returned steady-state outcome {val}", val=outcome
                )

                with jax.default_device(jax.devices("cpu")[0]):

                    # each branch returns the outcome, so the value leaving
                    # the cond exists only once finish_steady (where the
                    # adjoint is solved) or abandon_steady has run: that is
                    # what orders a later gradient read after them
                    def true_fn():
                        self.runner.finish_steady()
                        return outcome

                    def false_fn():
                        self.runner.abandon_steady()
                        return outcome

                    self.steady_outcome = jax.lax.cond(
                        jnp.equal(outcome, MaNTA.SteadyOutcome.Converged),
                        true_fn,
                        false_fn,
                    )
                    ec = jnp.equal(
                        self.steady_outcome, MaNTA.SteadyOutcome.Converged
                    )
                    jax.debug.print("ec: {val}", val=ec)

                    # self._report_objective_estimate()
                    return ec

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
        G, G_p = self.runner.Get_adjoint_gradients(
            getattr(self, "steady_outcome", None)
        )
        return G, G_p

    def getPressure(self, points=None):
        if self.params.config.evolveDensity:
            ui = self.runner.Get_profile(Channel.IonEnergy) / self.vp
        else:
            ui = self.runner.Get_profile(0) / self.vp
        return 2.0 / 3.0 * ui * self.pnorm

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return self.InitialValue(index, self.xR)

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
        return vp * (
            params.config.ParticleSourceHeight
            * jnp.exp(
                -((x - params.config.ParticleSourceCenter) ** 2)
                / (2 * params.config.ParticleSourceWidth**2)
            )
            - params.constants.FusionRate(state.n, state.Ti)
            / params.constants.DensityEquationNormalization()
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
            + params.config.FusionFactor
            * params.constants.AlphaHeating(state.n, state.Ti)
            / params.constants.HeatEquationNormalization()
        )

    def CollisionalEnergyExchange(
        self, state: StellaratorState, params: StellaratorParams
    ):
        return params.constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)

    def StoredEnergy(self, field, vp, vpp, geometry, state, x, params: StellaratorParams):

        if self.params.config.evolveDensity:
            return (
                state.Variable[Channel.IonEnergy]
                + state.Variable[Channel.ElectronEnergy]
            )
        else:
            return state.Variable[0]

    def FusionPower(self, field, vp, vpp, geometry, state, x, params: StellaratorParams):
        if self.params.config.evolveDensity:
            n = state.Variable[Channel.Density]
            pi = 2.0 / 3.0 * state.Variable[Channel.IonEnergy]
            Ti = pi / n
        else:
            n = StellaratorState.initial_profile(
                x,
                params.config.EdgeDensity,
                params.config.n0,
                params.config.rhoUpper,
                params.config.alpha,
            )
            Ti = 2.0 / 3.0 * state.Variable[0] / n

        return (
            params.constants.AlphaHeating(n, Ti)
            / params.constants.HeatEquationNormalization()
        )

    def tauE(self, field, vp, vpp, geometry, state, x, params: StellaratorParams):
        args = (field, vp, vpp, geometry, state, x, params)
        return self.StoredEnergy(*args) / self.FusionPower(*args)

    def TripleProduct(self, field, vp, vpp, geometry, state, x, params):
        if self.params.config.evolveDensity:
            n = state.Variable[Channel.Density]
            pi = 2.0 / 3.0 * state.Variable[Channel.IonEnergy]
            Ti = pi / n
        else:
            n = StellaratorState.initial_profile(
                x,
                params.config.EdgeDensity,
                params.config.n0,
                params.config.rhoUpper,
                params.config.alpha,
            )
            Ti = 2.0 / 3.0 * state.Variable[0] / n

        return n * Ti * self.tauE(field, vp, vpp, geometry, state, x, params)

    def ITGHeatFlux(self, field, vp, vpp, geometry, state, x, params):
        """V' <q_ITG . grad rho>, the NN ITG ion heat flux, in the units MaNTA
        transports it in and with the sign of an outward flux: G is then its
        integral over rho. Reads `geometry` (the ITG signals at this node), V'
        and V'' explicitly, so the adjoint's explicit term carries all three.
        """
        flux, _ = self.itg.local(
            geometry, state, x, 0.0, vp, vpp, params, params.config.evolveDensity
        )
        channel = Channel.IonEnergy if params.config.evolveDensity else 0
        # local's sign follows compute_dke_sol's, where the flux enters negated
        return -flux[channel]

    @partial(jax.jit, static_argnums=(0,))
    def InitialValue(self, index, x):
        return self._initial_value(index, x, self.vp_interp)

    def upper_boundary_values(self, Vp):
        """The Dirichlet data at x_R, as a function of V' on the field's rho.

        `UpperBoundary` is `InitialValue(i, xR)`, and the evolved variables are
        V' times a profile, so the edge data move with V'. The adjoint returns
        dG/d(g_D) for each variable (the boundary sensitivities); this is the
        g_D(V') to chain it through. It repeats the two interpolations V' goes
        through on its way to `vp_interp`: onto the MaNTA nodes
        (`get_fields_at_rho`), then the Akima interpolant evaluated at x_R.
        """
        vp = interpax.interp1d(self.points, self.yancc_wrapper.rho, Vp, extrap=True)
        vp_interp = interpax.Akima1DInterpolator(self.points, vp, check=False)
        return jnp.stack(
            [self._initial_value(i, self.xR, vp_interp) for i in range(self.nVars)]
        )

    def _initial_value(self, index, x, vp_interp):
        def constant_density(index, x):
            n = StellaratorState.initial_profile(
                x,
                self.params.config.EdgeDensity,
                self.params.config.n0,
                self.params.config.rhoUpper,
                self.params.config.alpha,
            )

            return 1.5 * self.params.config.EdgeTemperature * n * vp_interp(x)

        def ambipolar(index, x):
            def n0(x):
                return StellaratorState.initial_profile(
                    x,
                    self.params.config.EdgeDensity,
                    self.params.config.n0,
                    self.params.config.rhoUpper,
                    self.params.config.alpha,
                ) * vp_interp(x)

            def ui0(x):
                return (
                    3.0
                    / 2.0
                    * n0(x)
                    * StellaratorState.initial_profile(
                        x,
                        self.params.config.EdgeTemperature,
                        self.params.config.T0,
                        self.params.config.rhoUpper,
                        self.params.config.alpha,
                    )
                )

            def ue0(x):
                return (
                    3.0
                    / 2.0
                    * n0(x)
                    * StellaratorState.initial_profile(
                        x,
                        self.params.config.EdgeTemperature,
                        self.params.config.T0,
                        self.params.config.rhoUpper,
                        self.params.config.alpha,
                    )
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

    @Physics_Decorator
    def ComputePhysics(self, states, positions, t):

        dke_data = self._evaluate(
            self.flux_fn,
            states,
            positions,
            t,
            vmap_axes_wfield,
            self.yancc_wrapper.pitchgrid,
            self.yancc_wrapper.speedgrid,
            self.params,
            self.params.config.evolveDensity,
        )

        sources = self._evaluate(
            self.compute_sources, states, positions, t, vmap_axes_sources, self.params
        )

        return [dke_data[0], sources, dke_data[1]]

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, states, positions, t):

        ddke_data = self._evaluate(
            eqx.filter_jacrev(self.flux_fn),
            states,
            positions,
            t,
            vmap_axes_wfield,
            self.yancc_wrapper.pitchgrid,
            self.yancc_wrapper.speedgrid,
            self.params,
            self.params.config.evolveDensity,
        )

        dsources = self._evaluate(
            eqx.filter_jacrev(self.compute_sources),
            states,
            positions,
            t,
            vmap_axes_sources,
            self.params,
        )

        return [ddke_data[0], dsources, ddke_data[1]]

    def prepareEvaluation(self, plan):
        # called before physics are evaluated at a new set of points, generates field data at these points
        self.layout = {}
        points_of = {}

        for site in plan.sites:
            if site.entry not in BATCHED:
                continue
            points = jnp.asarray(site.points).ravel()
            m = len(points)
            if m in points_of:
                if not jnp.array_equal(points_of[m], points):
                    raise RuntimeError(
                        f"two batched sites of {m} points at different points; "
                        "Params cannot be keyed by batch size on this plan"
                    )
                continue
            if m > self.max_batch:
                raise RuntimeError(
                    f"the plan announces a batch of {m} points, past the {self.max_batch} "
                    f"this case holds at once ({self.batch_size / len(jax.devices())} rounds of {self.batch_size})"
                )
            points_of[m] = points

            rounds = -(-m // self.batch_size)
            pad_width = rounds * self.batch_size - m

            # don't pad the field itself, just evaluate at the padded points
            fields_interp, vp_interp, vpp_interp = self.yancc_wrapper.get_fields_at_rho(
                jnp.pad(points, (0, pad_width), mode="edge")
            )
            self.layout[m] = (
                rounds,
                fields_interp,
                vp_interp,
                vpp_interp,
            )

        self.levels.append((plan.grid.getNCells(), plan.k))

    def _evaluate(self, func, _states, _positions, t, vmap_axes, *args):
        """
        Evaluate physics, add padding
        """
        m = len(_positions)
        rounds = self.nchunks
        if m not in self.layout:
            raise RuntimeError(
                f"a batch of {m} points that the evaluation plan did not announce"
            )

        rounds, fields, vp, vpp = self.layout[m]
        if self.params.config.useBatching:
            pad = rounds * self.batch_size - m
            states = _pad_state(_states, pad)
            positions = jnp.pad(_positions, (0, pad), mode="edge")
        else:
            rounds = 0
            ndevices = len(jax.devices())
            pad = 0
            if m % ndevices != 0:
                pad += ndevices - m % ndevices

            states = _pad_state(_states, pad)
            positions = jnp.pad(_positions, (0, pad), mode="edge")

        out = vmap_batched(
            func,
            vmap_axes,
            chunk_size=self.batch_size,
            nchunks=rounds,
            sharding=data_sharding,
        )(states, positions, t, fields, vp, vpp, *args)

        self.rounds += rounds
        return _unpad_state(out, m)

    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    MaNTA.AdjointProblem
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
        points,
    ):
        MaNTA.AdjointProblem.__init__(self)

        self.g = g
        self.ng = len(self.g)  # g functions passed in as an array
        self.yancc_wrapper = yancc_data
        boundary_field = yancc_data.fields_unstacked[-1]

        flat, _ = jax.flatten_util.ravel_pytree(
            (eqx.filter(boundary_field, eqx.is_array))
        )

        ###

        # perform padding of arrays
        npoints = len(points)

        self.batch_size = transport_system.batch_size
        rounds = -(-npoints // self.batch_size)
        self.pad_width = rounds * self.batch_size - npoints
        self.field, self.vp, self.vpp = self.yancc_wrapper.get_fields_at_rho(
            jnp.pad(points, (0, self.pad_width), mode="edge")
        )

        ###

        self.npoints = npoints
        # add 1 for vp and 1 for vpp, which we also take gradients with respect to
        # -2 is for NFP and source, integer leaves which we don't get gradients of
        self.np_field = len(flat) - 2 + 1 + 1

        # The ITG geometry at each node -- its signals -- is a spatial
        # parameter block of its own after V'': n_itg columns, one node's
        # signals ravelled. objective.py contracts these columns with the
        # tangent of the same quantity.
        self.itg = transport_system.itg
        self.flux_fn = transport_system.flux_fn
        if self.itg is not None:
            self.itg_geometry = self.itg.geometry_at(
                jnp.pad(points, (0, self.pad_width), mode="edge")
            )
            node0 = jax.tree.map(lambda leaf: leaf[0], self.itg_geometry)
            self.n_itg = jax.flatten_util.ravel_pytree(node0)[0].size
        else:
            self.itg_geometry = None
            self.n_itg = 0

        self.np_cell = self.np_field + self.n_itg
        self.np = self.np_cell
        self.np_boundary = 0

        self.spatialParameters = True
        self.compute_sources = transport_system.compute_sources

        self.params = transport_system.params
        if self.params.config.evolveDensity:
            self.nVars = 3
            self.nAux = 1
        else:
            self.nVars = 1
            self.nAux = 0

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

        # The Dirichlet edge data depend on V' (see upper_boundary_values), so
        # G also depends on them. One parameter per variable, appended after
        # the np_cell spatial ones: columns np_cell .. np_cell + nVars - 1.
        for i in range(self.nVars):
            self.addUpperBoundarySensitivity(i)

    # each g is g(field, vp, vpp, geometry, state, x, params), geometry the ITG
    # signals at the node (None without the ITG term)
    _g_axes = (0, 0, 0, 0, State.vmap_axes(), 0, None)

    def _g_explicit(self):
        """What g reads besides the state: the parameters its explicit
        derivative is taken with respect to, in G_p's column order."""
        return (self.field, self.vp, self.vpp, self.itg_geometry)

    @MaNTA_Decorator
    def gFn(self, i, states, positions):
        out = _unpad_state(
            jax.vmap(self.g[i], in_axes=self._g_axes)(
                *self._g_explicit(), states, positions, self.params
            ),
            self.npoints,
        )
        return out

    @MaNTA_Decorator
    def dgFndp(self, i, states, positions):

        g = self.g[i]

        def g_of(explicit, state, x, params):
            return g(*explicit, state, x, params)

        fgrad_vmap = eqx.filter_vmap(
            eqx.filter_grad(g_of), in_axes=((0, 0, 0, 0), State.vmap_axes(), 0, None)
        )
        grad_out = fgrad_vmap(self._g_explicit(), states, positions, self.params)

        # ravelled per node as (field, V', V'', ITG geometry): the order of
        # G_p's columns, so no reordering is needed
        grad_unstack = tree_unstack(grad_out)[: self.npoints]
        grad_unraveled = jnp.stack(
            [jax.flatten_util.ravel_pytree(g)[0] for g in grad_unstack], axis=0
        )
        assert grad_unraveled.shape[1] == self.np_cell, (
            grad_unraveled.shape,
            self.np_cell,
        )
        # unweighted: PyAdjointProblem::explicitParameterDerivative applies
        # the quadrature weight w_i of each node itself
        # zero columns for the boundary sensitivities: g does not read them
        grad_w_vprime = jnp.pad(
            grad_unraveled,
            ((0, 0), (0, self.np_boundary)),
            mode="constant",
        )

        return grad_w_vprime.transpose()

    @MaNTA_Decorator
    def dg(self, i, states, positions):
        out = jax.vmap(jax.grad(self.g[i], argnums=4), in_axes=self._g_axes)(
            *self._g_explicit(), states, positions, self.params
        )
        return _unpad_state(out, self.npoints)

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, _states, _positions):

        positions = jnp.pad(_positions, (0, self.pad_width), mode="edge")
        states = _pad_state(_states, self.pad_width)

        rounds = -(-self.npoints // self.batch_size)
        nchunks = 0 if self.batch_size == 0 else rounds
        # dke_field_jac itself without the ITG term, so the traced function --
        # and with it JAX's persistent-cache key -- is what it always was
        ddke_data = vmap_batched(
            dke_field_jac if self.itg is None else FieldJac(self.flux_fn),
            vmap_axes=vmap_axes_wfield,
            chunk_size=self.batch_size,
            nchunks=nchunks,
            sharding=data_sharding,
        )(
            states,
            positions,
            0,
            self.field,
            self.vp,
            self.vpp,
            self.yancc_wrapper.pitchgrid,
            self.yancc_wrapper.speedgrid,
            self.params,
            self.params.config.evolveDensity,
        )

        tree_in = (self.field, self.vp, self.vpp)

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

            grad_unstack = tree_unstack(grad_out)[: self.npoints]
            grad_unraveled = jnp.stack(
                [jax.flatten_util.ravel_pytree(g)[0] for g in grad_unstack], axis=0
            )
            return grad_unraveled.transpose()

        if self.itg is not None:
            ditg = vmap_batched(
                itg_geometry_jac,
                vmap_axes=vmap_axes_itg_geometry,
                chunk_size=self.batch_size,
                nchunks=nchunks,
                sharding=data_sharding,
            )(
                self.itg,
                self.itg_geometry,
                states,
                positions,
                self.vp,
                self.vpp,
                self.params,
                self.params.config.evolveDensity,
            )

        def with_itg(block, itg_block):
            # rows: the field's np_field, then the ITG geometry's n_itg
            if self.itg is None:
                return block
            if itg_block is None:  # the sources do not read the ITG geometry
                return jnp.concatenate(
                    [block, jnp.zeros((self.n_itg, block.shape[1]))], axis=0
                )
            return jnp.concatenate([block, unravel(itg_block)], axis=0)

        dsigma_out = []
        dsources_out = []
        daux_out = []
        for i in range(0, self.nVars):
            dsigma_out.append(
                with_itg(unravel(ddke_data[0][i]), ditg[0][i] if self.itg is not None else None)
            )
            dsources_out.append(with_itg(unravel(dsources[i]), None))

        for i in range(0, self.nAux):
            daux_out.append(
                with_itg(unravel(ddke_data[1][i]), ditg[1][i] if self.itg is not None else None)
            )

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
