"""The nc toy problem with its flux spread over every JAX device on the host.

    -d/dx[ 2 x D u^n u' ] = H exp(-x^2 / W),   sigma(0) = 0,   u(1) = u_b

has a wall layer at x = 1, which MeshAdaptation grades towards and the degree
loop then resolves.

The machine. One *round* is one jitted, vmapped call on WIDTH points, sharded
across the devices: WIDTH = (number of devices) x EVALS_PER_DEVICE. MaNTA hands
the case each batch of M points in one call; the case pads it to a whole number
of rounds and runs ceil(M / WIDTH) of them one after another (manta.jax's
vmap_batched), so every round has the one compiled shape whatever M is.

Two configuration keys describe that machine to MaNTA, and they answer
different questions:

  * PhysicsParallelism = WIDTH is what a batch *costs*: ceil(M / WIDTH) rounds.
    The adaptation controllers fill each level they choose -- more cells for the
    graded mesh, a higher degree in the degree loop -- up to the rounds that
    level already pays for, since the padding would be evaluated anyway.

  * MaxPhysicsBatch = MAX_ROUNDS x WIDTH is what a batch may *be*. The case holds
    a whole batch on the devices at once -- its padded state and parameters, and
    every round's results, which are concatenated there before they come back --
    so the device memory one call needs grows with M. MaNTA promises never to
    hand over a recurring batch (per residual, per Jacobian build, per
    continuation step) larger than this: the degree loop stops at the highest
    degree within it, no fill passes it, and a configured level past it is
    refused before the case is asked for a single point.

The first is advice about cost and the run is correct without it; the second is
a limit of the case, so it is set on every run.

The same machine runs the problem twice: once with MaNTA told nothing about the
cost (PhysicsParallelism = 1) and once told WIDTH. Both are capped.

    python nc_multigpu.py

On a host with one device, XLA_FLAGS=--xla_force_host_platform_device_count=8
gives the CPU eight to shard over.
"""

import math
import time

import jax

# Off by default, so JAX computes the physics in float32: double precision can
# cost a lot of throughput on a GPU. The price is that the flux the solver
# differences and converges to 1e-8 carries single-precision round-off, which
# can slow Newton or stall a tight SteadyStateTolerance. Uncomment to run the
# physics in double precision, like the rest of the solve. A driver may set it;
# manta.jax itself never does.
# jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import ArrayLike, Float  # noqa: E402
from scipy.integrate import quad  # noqa: E402

import manta  # noqa: E402
from manta.jax import Physics_Decorator, State, vmap_batched  # noqa: E402

# from jax.sharding import PartitionSpec, NamedSharding
#
# if jax.default_backend() != "gpu":
#     raise RuntimeError("This example is intended to be run on multiple gpus")
#
# P = PartitionSpec
# devices = jax.devices()
# print(devices)
# mesh = jax.make_mesh(
#     (jax.device_count(),), ("axis",), axis_types=(jax.sharding.AxisType.Auto,)
# )
# data_sharding = NamedSharding(
#     mesh,
#     P(
#         "axis",
#     ),
# )
data_sharding = None


# works at n0 = 0.5
PARAMS = dict(n0=2.0, H0=0.1, W0=0.2, D0=0.01, u_b=0.2)

# The two batched entry points this case serves. Everything else MaNTA calls --
# InitialValue, the boundary values -- is pointwise and stays on the calling
# thread.
BATCHED = (
    manta.EvaluationEntry.ComputePhysics,
    manta.EvaluationEntry.ComputePhysicsDerivatives,
)

# Points each device evaluates in one round, and the most rounds one call may
# take. WIDTH is a multiple of the device count, which sharding needs.
EVALS_PER_DEVICE = 16
MAX_ROUNDS = 4
WIDTH = len(jax.devices()) * EVALS_PER_DEVICE  # -> PhysicsParallelism
MAX_BATCH = MAX_ROUNDS * WIDTH  # -> MaxPhysicsBatch


def source(x, H, W):
    return H * jnp.exp(-x * x / W)


class Params(eqx.Module):
    n: Float[ArrayLike, "..."]
    S: Float[ArrayLike, "..."]
    D: Float[ArrayLike, "..."]
    u_b: Float

    @classmethod
    def make(cls, points, n0, H0, W0, D0, u_b):
        np = len(points)
        n = n0 * jnp.ones((np,))
        S = jax.vmap(source, in_axes=(0, None, None))(points, H0, W0)
        D = D0 * jnp.ones((np,))
        return cls(n=n, S=S, D=D, u_b=u_b)

    @staticmethod
    def vmap_axes():
        return Params(0, 0, 0, None)


def _compute_physics(state, x, t, p: Params):
    # prefer defining fluxes and sources outside of the class because it keeps them "pure"
    def sigma(index, state, x, t, p: Params):
        return 2 * x * p.D * state.Variable[index] ** p.n * state.Derivative[index]

    return [[sigma(0, state, x, t, p)], [p.S], []]


# Built once rather than per call, so that the jitted round vmap_batched wraps
# around it is the same function every time and compiles once.
_compute_physics_derivatives = eqx.filter_jacrev(_compute_physics)

_VMAP_AXES = (State.vmap_axes(), 0, None, Params.vmap_axes())


class ParallelNC(manta.TransportSystem):
    """The nc flux, evaluated WIDTH points a round across the host's devices.

    `regrid = InPlace`: this case keeps nothing that depends on the mesh except
    the per-batch layout below, which prepareEvaluation rebuilds from every new
    plan. That is exactly what the adaptive drivers need to move one instance
    from level to level, so it can say so.
    """

    regrid = manta.Regrid.InPlace

    def __init__(self):
        # Zero flux on the axis -- sigma(0) = 0 -- rather than u'(0) = 0, which the
        # steady state does not satisfy.
        super().__init__(manta.numbered_spec(1, lower=manta.Mixed(d=1.0)))

        # batch size M -> (rounds, padded Params). Set by prepareEvaluation from
        # MaNTA's plan; see there.
        self.layout = {}

        self.levels = []  # (cells, k) of each plan MaNTA handed over
        self.rounds = 0  # rounds the machine ran
        self.points = 0  # points it was asked for

    # --- the boundary and initial values, pointwise --------------------------------

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return PARAMS["u_b"]

    def InitialValue(self, index, x):
        return PARAMS["u_b"]

    def InitialDerivative(self, i: int, x: float) -> float:
        return 0.0

    # --- where this case changes the plan -----------------------------------------

    def prepareEvaluation(self, plan):
        """Lay out every batch the plan announces as whole rounds, with its Params.

        MaNTA calls this before its first evaluation of a run, and again only
        when the plan changes -- a graded mesh from MeshAdaptation, a new degree
        from the degree loop.

        Params holds one entry per point, and the kernel pairs entry i with the
        state at position i of the batch -- it never looks at x to find it. So a
        Params is right only for the exact point set it was built from, and that
        set has to be the plan's, site by site, not one derived here from the
        grid and k. MeshAdaptation turns on Superconvergent, and then a level
        evaluates at two sets:

          * Residual, Jacobian and InitialCondition: the k+2 star nodes of each
            cell -- the initial sigma and du/dt are solved from the residual's
            own rows, so they sample where it samples;
          * TauFaces: both faces of each cell, for the Diffusive tau.

        A batch of M points takes ceil(M / WIDTH) rounds, and its Params are
        padded to that many rounds' worth here, once per plan, rather than on
        each of the thousands of calls. MaxPhysicsBatch means MaNTA plans no
        recurring batch past MAX_ROUNDS rounds; the check below is for the
        once-per-run sites the key does not cover, which on this case are never
        larger than the residual's.

        The batches are told apart by their size, which is what ComputePhysics
        can see cheaply. Two sites of one size at different points (k = 1
        without Superconvergent makes the nodes and the faces both 2 a cell)
        would need another key, so that is refused here rather than read with
        the wrong Params.
        """
        self.layout = {}
        points_of = {}
        for site in plan.sites:
            if site.entry not in BATCHED:
                continue
            points = np.asarray(site.points).ravel()
            m = len(points)
            if m in points_of:
                if not np.array_equal(points_of[m], points):
                    raise RuntimeError(
                        f"two batched sites of {m} points at different points; "
                        "Params cannot be keyed by batch size on this plan"
                    )
                continue
            if m > MAX_BATCH:
                raise RuntimeError(
                    f"the plan announces a batch of {m} points, past the {MAX_BATCH} "
                    f"this case holds at once ({MAX_ROUNDS} rounds of {WIDTH})"
                )
            points_of[m] = points
            rounds = -(-m // WIDTH)  # ceil(M / WIDTH)
            self.layout[m] = (
                rounds,
                self._pad_tree(Params.make(jnp.asarray(points), **PARAMS), rounds * WIDTH - m),
            )

        self.levels.append((plan.grid.getNCells(), plan.k))

    @staticmethod
    def _pad_tree(tree, pad_width):
        """Pads all array leaves of a PyTree along axis 0 by repeating the last entry."""
        dynamic, static = eqx.partition(tree, eqx.is_array)
        padded = jax.tree_util.tree_map(lambda leaf: jnp.pad(leaf, (0, pad_width), mode="edge"), dynamic)
        return eqx.combine(padded, static)

    @staticmethod
    def _unpad_tree(tree, length):
        """Slices all array leaves of a PyTree back to `length` along axis 0."""
        dynamic, static = eqx.partition(tree, eqx.is_array)
        return eqx.combine(jax.tree_util.tree_map(lambda leaf: leaf[:length, ...], dynamic), static)

    # --- where the layout is carried out -------------------------------------------

    def _evaluate(self, func, states, positions, t):
        """One batch from MaNTA as `rounds` jitted calls of WIDTH points each.

        The batch is padded to whole rounds by repeating its last point, so every
        round has the compiled shape; the padding is dropped on the way back. A
        batch the plan did not announce is refused, not run.
        """
        m = len(positions)
        if m not in self.layout:
            raise RuntimeError(f"a batch of {m} points that the evaluation plan did not announce")
        rounds, params = self.layout[m]
        pad = rounds * WIDTH - m

        out = vmap_batched(
            func, _VMAP_AXES, chunk_size=WIDTH, nchunks=rounds, sharding=data_sharding
        )(
            self._pad_tree(states, pad),
            jnp.pad(positions, (0, pad), mode="edge"),
            t,
            params,
        )
        self.rounds += rounds
        self.points += m
        return self._unpad_tree(out, m)

    @Physics_Decorator
    def ComputePhysics(self, states, positions, t):
        return self._evaluate(_compute_physics, states, positions, t)

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, states, positions, t):
        return self._evaluate(_compute_physics_derivatives, states, positions, t)


def u_exact(x):
    """The steady state, from integrating the flux balance once."""
    n, H, W, D, u_b = (PARAMS[k] for k in ("n0", "H0", "W0", "D0", "u_b"))
    flux = lambda s: H * math.sqrt(math.pi * W) / 2 * math.erf(s / math.sqrt(W))  # noqa: E731
    f = lambda s: flux(s) / s if s > 0 else H  # noqa: E731
    integral = np.array([quad(f, xi, 1.0, epsabs=1e-14, epsrel=1e-13)[0] for xi in x])
    return (u_b ** (n + 1) + (n + 1) / (2 * D) * integral) ** (1 / (n + 1))


def run(told):
    case = ParallelNC()
    runner = manta.Runner(case)
    runner.configure(
        {
            "OutputFilename": "nc_multigpu",
            "WriteOutput": False,
            "PolynomialDegree": 4,
            "GridSize": 5,
            "LowerBoundary": 0.0,
            "UpperBoundary": 1.0,
            "tau": 1.0,
            "tauScaling": "Diffusive",
            "tauUpdate": "ContinuationStep",
            # p -> h -> p: a uniform sample, a graded mesh, then the degree loop.
            # MaxPolynomialDegree is left at its default: on this machine the
            # degree loop's real ceiling is the highest k whose N (k + 2) points
            # fit MaxPhysicsBatch, and the log says when that is what stopped it.
            "MeshAdaptation": True,
            "DegreeTolerance": 1e-3,
            # What a batch costs. The machine is WIDTH wide either way; this is
            # whether MaNTA knows, and so whether it fills the levels it chooses.
            "PhysicsParallelism": WIDTH if told else 1,
            # What a batch may be: a limit of the case, so set on both runs.
            "MaxPhysicsBatch": MAX_BATCH,
            "Relative_tolerance": 1e-6,
            "Absolute_tolerance": [1e-6],
            "initialTimestep": 1e-3,
            "MinStepSize": 1e-12,
            "SteadyStateSolver": "PseudoTransient",
            "SteadyStateTolerance": 1e-8,
            "delta_t": 1.0,
        }
    )
    start = time.perf_counter()
    runner.run_ss()
    seconds = time.perf_counter() - start

    x = np.unique(
        np.concatenate(
            [np.linspace(0, 1, 401)[1:-1] + 1 / 800, 1 - np.geomspace(1e-6, 0.05, 200)]
        )
    )
    x = x[(x > 0) & (x < 1)]
    u = np.asarray(runner.getSolution(0, list(x))).ravel()
    ue = u_exact(x)
    return dict(
        levels=case.levels,
        rounds=case.rounds,
        occupancy=case.points / (case.rounds * WIDTH),
        linf=float(np.max(np.abs(u - ue)) / np.max(ue)),
        seconds=seconds,
    )


if __name__ == "__main__":
    print(
        f"{len(jax.devices())} device(s): a round is {WIDTH} points, and a call may "
        f"carry at most {MAX_ROUNDS} rounds ({MAX_BATCH} points).\n"
    )
    print(
        f"{'MaNTA told':<11} {'levels (cells, k)':<34} {'rounds':>7} {'occupied':>9} "
        f"{'L_inf':>9} {'wall':>7}"
    )
    for told in (False, True):
        r = run(told)
        levels = " ".join(f"({c},{k})" for c, k in r["levels"])
        print(
            f"{'N = ' + str(WIDTH) if told else 'nothing':<11} {levels:<34} "
            f"{r['rounds']:7d} {r['occupancy']:8.0%} {r['linf']:9.2e} {r['seconds']:6.1f}s"
        )
