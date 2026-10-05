"""The nc toy problem on a machine that evaluates the flux N points at a time.

    -d/dx[ 2 x D u^n u' ] = H exp(-x^2 / W),   sigma(0) = 0,   u(1) = u_b

has a wall layer at x = 1 about 5e-4 wide at n = 2.5, which MeshAdaptation grades
towards and the degree loop then resolves.

The machine here is N worker threads. MaNTA hands the case each batch of M points
in one call; the case redistributes it across the N workers as N vectorised calls
of ceil(M / N) points each, so a batch takes ceil(M / N) point-times on the
longest share -- the cost PhysicsParallelism = N describes. The redistribution is
the case's own plan, and it is made from MaNTA's: see prepareEvaluation.

The same machine runs the problem twice: once with MaNTA told nothing
(PhysicsParallelism = 1), once told N, which lets each adaptive level be filled
to the rounds it already costs.

    python nc_parallel.py [N]        # N defaults to 64
"""

import math
import sys
import time

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, ArrayLike
from scipy.integrate import quad

import manta
from manta.jax import State, Physics_Decorator, vmap_batched

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


MAX_EVALS_PER_GPU = 4
N_BATCHES = 2
MAX_WIDTH = N_BATCHES * len(jax.devices()) * MAX_EVALS_PER_GPU


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


class ParallelNC(manta.TransportSystem):
    """The nc flux on an N-thread machine.

    `regrid = InPlace`: this case keeps nothing that depends on the mesh except
    the distribution below, which prepareEvaluation rebuilds from every new plan.
    That is exactly what the adaptive drivers need to move one instance from
    level to level, so it can say so.
    """

    regrid = manta.Regrid.InPlace

    def __init__(self, width):
        # Zero flux on the axis -- sigma(0) = 0 -- rather than u'(0) = 0, which the
        # steady state does not satisfy.
        super().__init__(manta.numbered_spec(1, lower=manta.Mixed(d=1.0)))
        self.width = width
        self.points = []

        # batch size M -> share, the points each worker evaluates per call. Set
        # by prepareEvaluation from MaNTA's plan; see there.
        self.share = {}

        self.levels = []  # (cells, k) of each plan MaNTA handed over

    # --- the physics, written once for one point; the parent vmaps it -----------
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
        """
        This should take the new grid and recompute the parameters with padding up to
        the max width, set by the maximum allowable number of points
        """

        cell_boundaries = plan.grid.cellBoundaries()
        k = plan.k
        self.points = manta.getNodes(cell_boundaries, k)
        self.pad_width = 0
        if len(self.points) < MAX_WIDTH:
            self.pad_width = MAX_WIDTH - len(self.points)

        self.params = self._pad_tree(Params.make(self.points, **PARAMS), self.pad_width)

        self.levels.append((plan.grid.getNCells(), plan.k))

    @staticmethod
    def _pad_tree(tree, pad_width):
        """Pads all leaves of a PyTree along the first axis (axis=0)."""
        dynamic, static = eqx.partition(tree, eqx.is_array)

        def pad_leaf(leaf):
            # Construct pad width for axis 0, and no padding for other dimensions
            return jnp.pad(leaf, (0, pad_width), mode="edge")

        return eqx.combine(jax.tree_util.tree_map(pad_leaf, dynamic), static)

    @staticmethod
    def _unpad_tree(tree, current_batch_size):
        """Slices all leaves of a PyTree back to the target length along axis=0."""

        dynamic, static = eqx.partition(tree, eqx.is_array)

        def unpad_leaf(leaf):
            return leaf[:current_batch_size, ...]

        return eqx.combine(jax.tree_util.tree_map(unpad_leaf, tree), static)

    @Physics_Decorator
    def ComputePhysics(self, states, positions, t):
        pad_width = MAX_WIDTH - len(positions)
        x_padded = jnp.pad(positions, pad_width=(0, pad_width), mode="edge")
        out_padded = vmap_batched(
            _compute_physics,
            (State.vmap_axes(), 0, None, Params.vmap_axes()),
            chunk_size=len(jax.devices()) * MAX_EVALS_PER_GPU,
            nchunks=N_BATCHES,
            sharding=data_sharding,
        )(
            self._pad_tree(states, pad_width),
            x_padded,
            t,
            self.params,
        )

        return self._unpad_tree(out_padded, len(positions))

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, states, positions, t):
        pad_width = MAX_WIDTH - len(positions)
        x_padded = jnp.pad(positions, pad_width=(0, pad_width), mode="edge")
        out_padded = vmap_batched(
            eqx.filter_jacrev(_compute_physics),
            (State.vmap_axes(), 0, None, Params.vmap_axes()),
            chunk_size=len(jax.devices()) * MAX_EVALS_PER_GPU,
            nchunks=N_BATCHES,
            sharding=data_sharding,
        )(
            self._pad_tree(states, pad_width),
            x_padded,
            t,
            self.params,
        )

        return self._unpad_tree(out_padded, len(positions))


def u_exact(x):
    """The steady state, from integrating the flux balance once."""
    n, H, W, D, u_b = (PARAMS[k] for k in ("n0", "H0", "W0", "D0", "u_b"))
    flux = lambda s: H * math.sqrt(math.pi * W) / 2 * math.erf(s / math.sqrt(W))  # noqa: E731
    f = lambda s: flux(s) / s if s > 0 else H  # noqa: E731
    integral = np.array([quad(f, xi, 1.0, epsabs=1e-14, epsrel=1e-13)[0] for xi in x])
    return (u_b ** (n + 1) + (n + 1) / (2 * D) * integral) ** (1 / (n + 1))


def run(width, told):
    case = ParallelNC(width)
    runner = manta.Runner(case)
    runner.configure(
        {
            "OutputFilename": "nc_parallel",
            "WriteOutput": False,
            "PolynomialDegree": 4,
            "GridSize": 5,
            "LowerBoundary": 0.0,
            "UpperBoundary": 1.0,
            "tau": 1.0,
            "tauScaling": "Diffusive",
            "tauUpdate": "ContinuationStep",
            "MeshAdaptation": True,
            "DegreeTolerance": 1e-3,
            "MaxPolynomialDegree": 5,
            # The machine is `width` wide either way; this is whether MaNTA knows.
            "PhysicsParallelism": 16,
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
        linf=float(np.max(np.abs(u - ue)) / np.max(ue)),
        seconds=seconds,
    )


if __name__ == "__main__":
    width = MAX_WIDTH
    print(f"{width} workers; a batch of M points takes ceil(M/{width}) point-times.\n")
    print(f"{'MaNTA told':<11} {'levels (cells, k)':<34} {'L_inf':>9} {'wall':>7}")
    for told in (False, True):
        r = run(width, told)
        levels = " ".join(f"({c},{k})" for c, k in r["levels"])
        print(
            f"{'N = ' + str(width) if told else 'nothing':<11} {levels:<34} "
            f"{r['linf']:9.2e} {r['seconds']:6.1f}s"
        )
