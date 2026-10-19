"""Global scalars in manta.jax, from one pure function of the parameters.

A JAX case writes `sigma`, `source` and `aux` as functions of `params` and the
base classes derive the rest. `scalar(index, states, states_dot, integrator, t,
params)` does the same for a global scalar constraint: from it come ScalarG,
ScalarGPrime and, in JAXAdjointProblem, dScalarG_dp. dSources_dScalars comes from
`source`, and dg's Scalars entry -- dG/dmu, integrated -- from the objective.

The oracle is test_adjoint_combinations.py's hand-written case, whose gradient
is checked there against finite differences: the same two-scalar problem,

    -d_x( kappa d_x u ) = S + mu + x nu,
    mu - int u dx - int c dx = 0,    nu - 2 mu - int x^2 u dx = 0,

with the single JAX objective G = int [ u^2/2 + (mu + 3 nu) u ] dx, the sum of
that file's two. Agreement to round-off pins every derived hook at once, since
each one is on the path to G_p; a sign or a missing weight in any of them is an
O(1) disagreement.

Float64 throughout, through jax.enable_x64 rather than the global flag, which
would change every other JAX test in the session.
"""

from typing import NamedTuple

import numpy as np
import pytest

pytest.importorskip("equinox")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import manta as MaNTA  # noqa: E402
from manta.jax import (  # noqa: E402
    JAXAdjointProblem,
    JAXTransportSystem,
    ScalarG_Decorator,
    ScalarGPrime_Decorator,
    VectorizedTransportSystem,
)

from test_adjoint import cell_weights  # noqa: E402
from test_adjoint_combinations import (  # noqa: E402
    C0,
    K,
    KAPPA,
    NCELLS,
    S0,
    Combined,
    config,
    physics_points,
)


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


class Params(NamedTuple):
    kappa: float
    S: float
    c: float


def make_params(spatial, n):
    if spatial:
        return Params(KAPPA * jnp.ones(n), S0 * jnp.ones(n), C0 * jnp.ones(n))
    return Params(KAPPA, S0, C0)


class Physics:
    """The problem, as pure functions of the parameters, for either base."""

    def setup(self, points, params, spatial):
        self.params = params
        self.spatial = spatial
        edges = np.linspace(0.0, 1.0, NCELLS + 1)
        self.physics_weights = jnp.asarray(cell_weights(np.asarray(points), edges))
        # ScalarG's states are on the basis nodes, whatever the scheme.
        self.basis_points = jnp.asarray(MaNTA.getNodes(0.0, 1.0, NCELLS, K))
        self.adjoint = JAXAdjointProblem(self, self.g, spatialParameters=spatial)

    def sigma(self, index, state, x, t, params):
        return params.kappa * state.Derivative[index]

    def source(self, index, state, x, t, params):
        return params.S + state.Scalars[0] + x * state.Scalars[1]

    def aux(self, index, state, x, t, params):
        return 0.0

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u + (state.Scalars[0] + 3.0 * state.Scalars[1]) * u

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0

    def InitialValue(self, index, x):
        return 0.0 * x

    def InitialAuxValue(self, index, x):
        return 0.0

    def InitialScalarValue(self, s):
        return 0.0

    def InitialScalarDerivative(self, s, states, states_dot, weights):
        return 0.0

    def createAdjointProblem(self):
        return self.adjoint


class WithScalar(Physics):
    def scalar(self, index, states, states_dot, integrator, t, params):
        u = states.Variable[:, 0]
        mu, nu = states.Scalars[0], states.Scalars[1]
        # int c dx over the physics nodes the field is defined on; for one
        # number, c times the domain's length.
        c_integral = self.physics_weights @ params.c if self.spatial else params.c
        if index == 0:
            return mu - integrator(u) - c_integral
        return nu - 2.0 * mu - integrator(self.basis_points**2 * u)


class VectorizedScalars(WithScalar, VectorizedTransportSystem):
    def __init__(self, points, spatial):
        VectorizedTransportSystem.__init__(
            self, MaNTA.numbered_spec(1, nScalars=2), spatialParameters=spatial
        )
        self.setup(points, make_params(spatial, len(points)), spatial)


class PointwiseScalars(WithScalar, JAXTransportSystem):
    def __init__(self, points, spatial):
        assert not spatial, "the pointwise base reads params whole"
        JAXTransportSystem.__init__(self, MaNTA.numbered_spec(1, nScalars=2))
        self.setup(points, make_params(spatial, len(points)), spatial)


def run(system, tmp_path, superconvergent):
    runner = MaNTA.Runner(system)
    runner.configure(config(tmp_path, superconvergent))
    runner.run_ss()
    return runner


def oracle(spatial, superconvergent, tmp_path):
    """The hand-written case's G and G_p, summed over its two objectives."""
    points = physics_points(superconvergent)
    n = len(points)
    fields = (KAPPA * np.ones(n), S0 * np.ones(n), C0 * np.ones(n))
    G, gradients = run(
        Combined(False, True, spatial, points, *fields), tmp_path / "oracle", superconvergent
    ).getAdjointGradients()
    return float(np.sum(G)), np.asarray(gradients["G_p"]).reshape(2, -1, 3).sum(axis=0)


CASES = [
    (VectorizedScalars, False, False),
    (VectorizedScalars, False, True),
    (VectorizedScalars, True, False),
    (VectorizedScalars, True, True),
    (PointwiseScalars, False, False),
    (PointwiseScalars, False, True),
]


@pytest.mark.parametrize(
    "case, spatial, superconvergent",
    CASES,
    ids=[
        f"{c.__name__}{'-spatial' if sp else ''}{'-star' if sc else ''}" for c, sp, sc in CASES
    ],
)
def test_the_derived_scalar_hooks_give_the_hand_written_gradient(
    tmp_path, case, spatial, superconvergent
):
    points = physics_points(superconvergent)
    G, gradients = run(case(points, spatial), tmp_path / "jax", superconvergent).getAdjointGradients()
    G_p = np.asarray(gradients["G_p"])

    G_ref, G_p_ref = oracle(spatial, superconvergent, tmp_path)
    assert float(np.asarray(G)[0]) == pytest.approx(G_ref, rel=1e-12)
    assert G_p.shape == G_p_ref.shape
    np.testing.assert_allclose(G_p, G_p_ref, rtol=0.0, atol=1e-10 * np.abs(G_p_ref).max())


class HandWritten(Physics, VectorizedTransportSystem):
    """A case written before `scalar` existed: ScalarG and ScalarGPrime by hand,
    reading self.params, which is what the layer has to go on supporting."""

    def __init__(self, points):
        VectorizedTransportSystem.__init__(self, MaNTA.numbered_spec(1, nScalars=2))
        self.setup(points, make_params(False, len(points)), False)

    def _G(self, index, states, integrator):
        u = states.Variable[:, 0]
        mu, nu = states.Scalars[0], states.Scalars[1]
        if index == 0:
            return mu - integrator(u) - self.params.c
        return nu - 2.0 * mu - integrator(self.basis_points**2 * u)

    @ScalarG_Decorator
    def ScalarG(self, index, states, states_dot, integrator, t):
        return self._G(index, states, integrator)

    @ScalarGPrime_Decorator
    def ScalarGPrime(self, states, states_dot, integrator, t):
        grads = [jax.grad(self._G, argnums=1)(s, states, integrator) for s in range(2)]
        zeros = [jax.tree_util.tree_map(jnp.zeros_like, d) for d in grads]
        return [grads, zeros]


def test_a_case_writing_its_own_ScalarG_runs_as_before(tmp_path):
    """The forward solve is unchanged; only its adjoint lacks dScalarG_dp, and
    says how to supply it rather than assuming the constraints ignore p."""
    points = physics_points(False)
    system = HandWritten(points)
    runner = MaNTA.Runner(system)
    cfg = config(tmp_path, False)
    cfg["solveAdjoint"] = False
    runner.configure(cfg)
    runner.run_ss()

    reference = run(VectorizedScalars(points, False), tmp_path / "ref", False)
    x = np.linspace(0.05, 0.95, 7)
    np.testing.assert_allclose(
        runner.getSolution(0, x), reference.getSolution(0, x), rtol=1e-12, atol=1e-14
    )

    with_adjoint = MaNTA.Runner(HandWritten(points))
    with_adjoint.configure(config(tmp_path / "adjoint", False))
    with pytest.raises(Exception, match=r"scalar\(index, states"):
        with_adjoint.run_ss()


def test_a_scalar_case_with_neither_form_is_refused_at_construction():
    class Neither(Physics, VectorizedTransportSystem):
        def __init__(self):
            VectorizedTransportSystem.__init__(self, MaNTA.numbered_spec(1, nScalars=1))

    with pytest.raises(TypeError, match="neither scalar"):
        Neither()


class ParameterCoupled(VectorizedScalars):
    """S(x) multiplies the first scalar's share of the source, so dS/dmu reads
    the parameter at its own node -- which the pointwise dSources_dScalars has
    to find by position when the parameters are spatial."""

    def source(self, index, state, x, t, params):
        return params.S * (1.0 + 0.5 * state.Scalars[0]) + x * state.Scalars[1]


def parameter_coupled_G(points, S, tmp_path):
    system = ParameterCoupled(points, True)
    system.params = system.params._replace(S=jnp.asarray(S))
    system.adjoint.setParams(system.params)
    G, gradients = run(system, tmp_path, False).getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p"])


def test_a_spatial_scalar_coupling_reads_each_nodes_parameters(tmp_path):
    """dG/dS_j against finite differences, for a non-uniform S(x).

    With S uniform, or a coupling that does not read it, every node's
    parameters are the same and a wrong lookup is exact. G is not quadratic in
    S here, but its curvature is small: the worst disagreement is 7.5e-12 of the
    largest entry, against 1e-9.
    """
    points = physics_points(False)
    S = S0 * (1.0 + 0.3 * np.sin(5.0 * points))
    _, G_p = parameter_coupled_G(points, S, tmp_path / "base")
    adjoint = G_p[:, 1]
    scale = np.abs(adjoint).max()

    perCell = len(points) // NCELLS
    h = 1e-3 * S0
    for j in (perCell - 1, perCell, len(points) // 2):
        plus, minus = S.copy(), S.copy()
        plus[j] += h
        minus[j] -= h
        fd = (
            parameter_coupled_G(points, plus, tmp_path / f"p{j}")[0]
            - parameter_coupled_G(points, minus, tmp_path / f"m{j}")[0]
        ) / (2.0 * h)
        assert abs(adjoint[j] - fd) < 1e-9 * scale, (
            f"node {j}: adjoint {adjoint[j]}, finite difference {fd}, largest {scale}"
        )
