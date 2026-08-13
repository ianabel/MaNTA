from typing import NamedTuple

from functools import partial

import manta as MaNTA
from manta.jax import VectorizedTransportSystem, JAXAdjointProblem
import jax.numpy as jnp
import numpy as np
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import equinox as eqx

# Float64, before any array is built. JAX defaults to float32, which caps a
# gradient check at about six digits and makes the finite-difference reference
# the least accurate thing in the comparison rather than the most. The solver
# tolerances below are tightened to match -- there is no point differencing an
# objective to 1e-10 if the steady state it comes from is only converged to 1e-2.
jax.config.update("jax_enable_x64", True)


class NonlinearDiffusionParams(NamedTuple):
    T_s: Float[ArrayLike, "..."]
    D: Float[ArrayLike, "..."]
    a: Float[ArrayLike, "..."]
    SourceWidth: Float[ArrayLike, "..."]
    SourceCentre: Float[ArrayLike, "..."]

    @classmethod
    def make(cls, config) -> "NonlinearDiffusionParams":
        return cls(
            T_s=50.0,
            SourceCentre=config["SourceCentre"],
            D=config["D"],
            a=config["a"],
            SourceWidth=0.02,
        )

    @classmethod
    def makeSpatial(cls, config) -> "NonlinearDiffusionParams":
        ones = jnp.ones_like(config["T_s"])
        return cls(
            T_s=config["T_s"],
            SourceCentre=config["SourceCentre"] * ones,
            D=config["D"] * ones,
            a=config["a"] * ones,
            SourceWidth=0.02 * ones,
        )


class JAXNonlinearDiffusion(VectorizedTransportSystem):
    def __init__(self, config):
        super().__init__(
            MaNTA.numbered_spec(1, lower=MaNTA.Neumann), spatialParameters=True
        )

        solver_config = config["solver"]

        self.points = MaNTA.getNodes(
            solver_config["Lower_boundary"],
            solver_config["Upper_boundary"],
            solver_config["Grid_size"],
            solver_config["Polynomial_degree"],
        )

        self.params = NonlinearDiffusionParams.makeSpatial(config["ts"])
        self.adjointProblem = JAXAdjointProblem(self, self.g, spatialParameters=True)
        self.runner = MaNTA.Runner(self)

        self.runner.configure(solver_config)

        # This object will be passed to sigma and source functions

    def run(self, tFinal=None):
        if tFinal is not None:
            sFinal = self.runner.run(tFinal)
        else:
            sFinal = self.runner.run_ss()

        return sFinal

    def getAdjointGradients(self):
        G, G_p = self.runner.getAdjointGradients()
        return G, G_p["G_p"]

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u

    def sigma(self, index, state, x, t, params):

        u = state.Variable[0]
        q = state.Derivative[0]
        return params.D * (u**params.a) * q

    def source(self, index, state, x, t, params):
        return params.T_s

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3

    def InitialValue(self, index, x):
        return 0.3

    def createAdjointProblem(self):
        return self.adjointProblem


def runMaNTA(func, solver_config):

    ts_config = {"SourceCentre": 0.3, "D": 2.0, "a": 0.0, "T_s": func}
    config = {"solver": solver_config, "ts": ts_config}
    transportSystem = JAXNonlinearDiffusion(config)

    transportSystem.run()
    return transportSystem.getAdjointGradients()


solver_config = {
    "OutputFilename": "out",
    "Polynomial_degree": 3,
    "Grid_size": 4,
    "tau": 1.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    # Tight, so the finite-difference reference is limited by its step size
    # rather than by how well the steady state was resolved. MinStepSize has to
    # come down with them: the 1e-7 default is one IDA hits while still at
    # t = 0 at these tolerances, and it reports that as a repeated error-test
    # failure rather than as anything about the step floor. 1e-8 / 1e-10 is a
    # step too far on this 4-cell grid -- IDASolve gives up with IDA_ERR_FAIL.
    "Relative_tolerance": 1e-7,
    "Absolute_tolerance": [1e-9],
    "MinStepSize": 1e-12,
    # run_ss() arms steady-state termination whatever the config says, but
    # falls back to 1e-3, and that -- not Relative_tolerance -- is then the
    # floor on how well G is determined. Left at the default it swamps any
    # finite difference worth taking.
    "SteadyStateTolerance": 1e-9,
    "delta_t": 1.0,
    "restart": False,
    "solveAdjoint": True,
}

points = MaNTA.getNodes(
    solver_config["Lower_boundary"],
    solver_config["Upper_boundary"],
    solver_config["Grid_size"],
    solver_config["Polynomial_degree"],
)

f = partial(runMaNTA, solver_config=solver_config)
T = lambda x: 50.0 * jnp.sin(2 * jnp.pi * x) ** 2
G, G_p_adj = f(T(points))
T_p = T(points)

G_p_fd = []


def fd_jvp(tangent):
    # Central. The one-sided difference this used to take had an O(dT)
    # truncation error, and at dT = 0.1 * ||T_p|| -- about 13, on a parameter of
    # norm ~129 -- that error was most of the 2% the check then reported: a
    # measurement of the step size rather than of the gradient. Central
    # differencing makes it O(dT^2), and float64 leaves room to shrink the step.
    #
    # 1e-2 of ||T_p|| is the optimum and not much of a compromise: the agreement
    # is ~7e-7 there, against ~2e-6 at 1e-3 and ~1e-2 at 1e-6, where the
    # steady state's own round-off takes over. Both ends of that curve are the
    # measurement rather than the gradient, which is the state a finite-difference
    # check should be in.
    dT = 1e-2 * jnp.linalg.norm(T_p)

    direction = tangent / jnp.linalg.norm(tangent)
    G_plus = f(T_p + dT * direction)[0]
    G_minus = f(T_p - dT * direction)[0]
    return (G_plus - G_minus) / (2.0 * dT)


def adj_jvp(tangent):
    return jnp.dot(G_p_adj[:, 0], tangent)


# for i in range(0,len(points)):
#     x = points[i]
#     dT = 0.001 + 0.1* T(x)
#     T_pert = T_p
#     T_pert = T_pert.at[i].set(T_p[i] + dT)
#     G_1 = f(T_pert)[0]
#     G_2 = f(T_p)[0]

#     print(G_1)
#     print(G_2)
#     G_p_fd.append((G_1-G_2) / (dT))
# G_d_fd_arr = jnp.array(G_p_fd)
# fig, ax = plt.subplots()
# ax.plot(points, G_p_fd, 'bo', label="finite differences")
# ax.plot(points, G_p_adj[:,0], 'rx', label="adjoints")
# ax.legend()
# fig, ax = plt.subplots()
# ax.semilogy(points,jnp.abs(jnp.array(G_p_fd)- G_p_adj[:,0])**2)
# plt.show()

import matplotlib.pyplot as plt

nTangents = 50
rng_key = jax.random.PRNGKey(5)


def make_random_tangent(primal):
    global rng_key
    rng_key, key = jax.random.split(rng_key)

    # key = jax.random.key(69)
    def map_fn(leaf):
        if jnp.isscalar(leaf) or jnp.isdtype(leaf, "integral"):
            return leaf
        else:
            v = jax.random.normal(key=key, shape=leaf.shape, dtype=leaf.dtype)
            return v / jnp.linalg.norm(v)

    tangent_field = jax.tree.map(map_fn, primal)
    return tangent_field


fig, ax = plt.subplots()
err = 0.0
for i in range(0, nTangents):
    t = make_random_tangent(T_p)
    adj = adj_jvp(t)
    fd = fd_jvp(t)
    err += jnp.abs(adj - fd)
    ax.plot(i, adj, "bo")
    ax.plot(i, fd, "rx")
err /= nTangents

print(f"Average error for {nTangents} iterations: {err}")
plt.show()
