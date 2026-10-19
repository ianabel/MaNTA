"""The adjoint on demand: Runner.solve_adjoint and the three reads beside it.

solve_adjoint solves J^T z = dG/dy at the current state from the last Jacobian
build, without calling the physics; adjoint_state, objective_gradient and
objective_sensitivity read the answer in nodal layouts. The per-node check of
the sensitivity against finite differences is in
Tests/UnitTests/AdjointOnDemandTests.cpp, where the steady solves are cheap
enough to rerun once per node; what is checked here is what only the Python
surface can get wrong -- the layouts, the lifecycle, and that a Python case is
not called.

The case is test_adjoint's ParametricDiffusion: -kappa u'' = S on [0, 1] with
u = 0 at both ends, and G = int u^2 / 2 dx. Its adjoint solves -kappa z'' = u,

    z = S (x^4 - 2 x^3 + x) / (24 kappa^2).

A quartic, but the discrete adjoint does not reproduce it at PolynomialDegree = 4
the way the forward solve reproduces the quadratic u, and that is not a defect.
G is the interpolatory quadrature sum_m w_m u_m^2 / 2, so its exact gradient is
w_m u_m node by node -- diag(w), not the mass matrix the continuous Riesz
representer would give -- and the two differ by an alternating, mean-free
pattern within each cell. z inherits it: 2.4e-4 of max |z| here, against the
round-off a consistent right-hand side would leave.
"""

import netCDF4
import numpy as np
import pytest

import manta as MaNTA

from test_adjoint import (
    KAPPA0,
    NP,
    SOURCE0,
    ParametricDiffusion,
    adjoint_config,
    exact_dG,
)


def exact_z(x, kappa=KAPPA0, source=SOURCE0):
    return source * (x**4 - 2 * x**3 + x) / (24 * kappa**2)


def exact_dz(x, kappa=KAPPA0, source=SOURCE0):
    return source * (4 * x**3 - 6 * x**2 + 1) / (24 * kappa**2)


def steady(tmp_path, system=None, **overrides):
    system = system or ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    runner = MaNTA.Runner(system)
    cfg = adjoint_config(tmp_path, SteadyStateSolver="Newton", SteadyStateTolerance=1e-11)
    cfg.update(overrides)
    runner.configure(cfg)
    runner.run_ss()
    return runner


class CountedDiffusion(ParametricDiffusion):
    """ParametricDiffusion, counting every physics hook the solver calls."""

    HOOKS = (
        "SigmaFn",
        "Sources",
        "dSigmaFn_du",
        "dSigmaFn_dq",
        "dSources_du",
        "dSources_dq",
        "dSources_dsigma",
    )

    def __init__(self, p):
        super().__init__(p)
        self.calls = 0

    def SigmaFn(self, i, state, x, t):
        self.calls += 1
        return super().SigmaFn(i, state, x, t)

    def Sources(self, i, state, x, t):
        self.calls += 1
        return super().Sources(i, state, x, t)

    def dSigmaFn_du(self, i, state, x, t):
        self.calls += 1
        return super().dSigmaFn_du(i, state, x, t)

    def dSigmaFn_dq(self, i, state, x, t):
        self.calls += 1
        return super().dSigmaFn_dq(i, state, x, t)

    def dSources_du(self, i, state, x, t):
        self.calls += 1
        return super().dSources_du(i, state, x, t)

    def dSources_dq(self, i, state, x, t):
        self.calls += 1
        return super().dSources_dq(i, state, x, t)

    def dSources_dsigma(self, i, state, x, t):
        self.calls += 1
        return super().dSources_dsigma(i, state, x, t)


def test_the_adjoint_state_is_the_closed_form(tmp_path):
    """z on the nodes, in the layout the docs give, against the quartic above.

    The sigma rows' adjoint is z' -- the flux enters the residual as the
    derivative's dual -- which pins the sign convention of the second block.
    """
    runner = steady(tmp_path)
    runner.solve_adjoint()
    z = runner.adjoint_state(0)

    nCells, k = 6, 4
    x = np.asarray(z["x"])
    assert x.shape == (nCells * (k + 1),)
    for key in ("sigma", "q", "u"):
        assert np.asarray(z[key]).shape == (nCells * (k + 1), 1), key
    assert np.asarray(z["aux"]).shape == (nCells * (k + 1), 0)
    assert np.asarray(z["lambda"]).shape == (nCells + 1, 1)
    assert np.asarray(z["x_faces"]).shape == (nCells + 1,)
    assert np.asarray(z["field"]).shape == (0,)
    assert z["operator_at_state"]

    scale = np.max(np.abs(exact_z(x)))
    np.testing.assert_allclose(np.asarray(z["u"])[:, 0], exact_z(x), rtol=0, atol=1e-3 * scale)
    scale = np.max(np.abs(exact_dz(x)))
    np.testing.assert_allclose(np.asarray(z["sigma"])[:, 0], exact_dz(x), rtol=0, atol=1e-3 * scale)


def test_the_end_of_run_adjoint_is_readable_without_another_solve(tmp_path):
    """solveAdjoint leaves its states behind, and solve_adjoint gives the same."""
    runner = steady(tmp_path)
    before = runner.adjoint_state(0)
    runner.solve_adjoint(0)
    after = runner.adjoint_state(0)
    for key in ("sigma", "q", "u", "lambda"):
        np.testing.assert_array_equal(np.asarray(after[key]), np.asarray(before[key]))


def test_the_sensitivities_add_up_to_the_parameter_gradient(tmp_path):
    """sigma_hat = kappa q and S is constant, so by the chain rule

        dG/dkappa = sum_m w_flux_m q_m,      dG/dS = sum_m w_sources_m

    -- the same contraction computeAdjointGradients makes with dF/dp, reached
    from the other side. An aggregate: the per-node check is the C++ one.
    """
    runner = steady(tmp_path)
    runner.solve_adjoint()
    w = runner.objective_sensitivity(0)
    x = np.asarray(w["x"])
    flux = np.asarray(w["flux"])[:, 0]
    sources = np.asarray(w["sources"])[:, 0]
    assert flux.shape == x.shape == sources.shape
    assert w["operator_at_state"]

    q = np.asarray(runner.getDerivative(0, list(x)))
    _, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"]).reshape(-1)
    assert G_p.shape == (NP,)

    from_w = np.array([np.dot(flux, q), np.sum(sources)])
    np.testing.assert_allclose(from_w, G_p, rtol=1e-10)
    np.testing.assert_allclose(from_w, exact_dG(KAPPA0, SOURCE0), rtol=1e-6)


def test_the_objective_gradient_is_the_right_hand_side(tmp_path):
    """G_y for G = int u^2/2 is w_m u_m on the u rows, so G_y . u = 2 G."""
    runner = steady(tmp_path)
    gy = runner.objective_gradient(0)
    u = np.asarray(runner.getSolution(0, None))
    G = float(np.asarray(runner.G())[0])
    np.testing.assert_allclose(np.dot(np.asarray(gy["u"])[:, 0], u), 2 * G, rtol=1e-12)
    for key in ("sigma", "q", "lambda"):
        assert not np.any(np.asarray(gy[key])), key


def test_an_adjoint_solve_asks_the_case_for_nothing(tmp_path):
    system = CountedDiffusion(np.array([KAPPA0, SOURCE0]))
    runner = steady(tmp_path, system)
    assert system.calls > 0
    system.calls = 0

    runner.solve_adjoint()
    runner.adjoint_state(0)
    runner.objective_sensitivity(0)
    runner.objective_gradient(0)
    assert system.calls == 0, f"{system.calls} physics calls from the adjoint"


def test_without_solveAdjoint_the_objective_is_built_on_demand(tmp_path):
    """solve_adjoint needs the objective, not the gradient machinery.

    The run estimated nothing, so the last build is the final Newton iteration's
    and the operator is one iterate behind -- which it says. This problem is
    linear, so that operator is the same matrix and z agrees regardless.
    """
    reference = steady(tmp_path / "with")
    reference.solve_adjoint()
    expected = np.asarray(reference.adjoint_state(0)["u"])

    runner = steady(tmp_path / "without", solveAdjoint=False)
    runner.solve_adjoint()
    z = runner.adjoint_state(0)
    assert not z["operator_at_state"]
    np.testing.assert_allclose(np.asarray(z["u"]), expected, rtol=1e-9, atol=1e-14)


def test_between_slices(tmp_path):
    """The current state between slices is the last slice's.

    A slice that estimated its objective built the Jacobian there, so the
    operator is exact; the answer is withdrawn once the state moves on.
    """
    runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
    runner.configure(
        adjoint_config(
            tmp_path,
            SteadyStateSolver="PseudoTransient",
            SteadyStateTolerance=1e-11,
            PseudoTransientInitialStep=1e-4,
            MaxContinuationSteps=2,
        )
    )
    outcome = runner.start_steady(True)
    assert outcome == MaNTA.SteadyOutcome.OutOfSteps
    runner.solve_adjoint()
    assert runner.adjoint_state(0)["operator_at_state"]

    outcome = runner.continue_steady(False)
    with pytest.raises(RuntimeError, match="solve_adjoint"):
        runner.adjoint_state(0)
    runner.solve_adjoint()
    assert not runner.adjoint_state(0)["operator_at_state"]
    runner.abandon_steady()


def test_solve_adjoint_needs_a_run_first(tmp_path):
    runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
    runner.configure(adjoint_config(tmp_path))
    with pytest.raises(RuntimeError, match="No Jacobian has been built"):
        runner.solve_adjoint()
    with pytest.raises(RuntimeError, match="solve_adjoint"):
        runner.adjoint_state(0)


def test_an_objective_out_of_range_is_refused(tmp_path):
    runner = steady(tmp_path)
    with pytest.raises(IndexError):
        runner.solve_adjoint(1)
    with pytest.raises(IndexError):
        runner.objective_gradient(1)


def test_write_adjoints_puts_z_beside_u(tmp_path, monkeypatch):
    """WriteAdjoints: G and dG/dp at the root, z in each variable's group."""
    monkeypatch.chdir(tmp_path)
    runner = steady(tmp_path, WriteOutput=True, WriteAdjoints=True, OutputFilename="written")
    G, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"]).reshape(-1)

    with netCDF4.Dataset(tmp_path / "written.nc") as nc:
        assert float(nc["G0"][...]) == pytest.approx(float(np.asarray(G)[0]), rel=1e-14)
        assert float(nc["G0_p"]["kappa"][...]) == pytest.approx(G_p[0], rel=1e-14)
        assert float(nc["G0_p"]["source"][...]) == pytest.approx(G_p[1], rel=1e-14)
        x = np.asarray(nc["x"][:])
        group = nc["Var0"]
        z_u = np.asarray(group["z0_u"][:])
        z_sigma = np.asarray(group["z0_sigma"][:])
        assert "z0_q" in group.variables
    assert z_u.shape == x.shape
    np.testing.assert_allclose(z_u, exact_z(x), rtol=0, atol=1e-3 * np.max(np.abs(exact_z(x))))
    np.testing.assert_allclose(
        z_sigma, exact_dz(x), rtol=0, atol=1e-3 * np.max(np.abs(exact_dz(x)))
    )


def test_write_adjoints_needs_solve_adjoint(tmp_path):
    runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
    with pytest.raises(ValueError, match="WriteAdjoints needs solveAdjoint"):
        runner.configure(adjoint_config(tmp_path, solveAdjoint=False, WriteAdjoints=True))
