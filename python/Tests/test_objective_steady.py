"""SteadyStateTest = "Objective" from a Runner.configure dict.

The solve stops when |z^T F|, the adjoint estimate of how far each objective is
from its value at the fixed point, is inside the algebraic share of
ObjectiveTolerance. The C++ half -- the estimate against an adjoint solve, the
threshold walked from both sides, the sliced resume, the stall -- is
Tests/UnitTests/ObjectiveSteadyTests.cpp. What is only reachable here is the
dict surface, the statistics a driver reads back, and that the test asks a
Python case for nothing beyond Newton's own sweeps.

The case is nonlinear diffusion, -d_x((1 + u^2) d_x u) = 1 on [0, 1] with
u(0) = 1 and u(1) = 0, so that Newton takes several iterations, and the
objective is G = int u dx.
"""

import numpy as np
import pytest

import manta as MaNTA


class NonlinearDiffusion(MaNTA.TransportSystem):
    variables = [MaNTA.Field("u", "", "", lower=MaNTA.Dirichlet, upper=MaNTA.Dirichlet)]

    def __init__(self):
        super().__init__()
        self.flux_calls = 0
        self.derivative_calls = 0

    def SigmaFn(self, i, state, x, t):
        self.flux_calls += 1
        u = state.u[0]
        return (1.0 + u * u) * state.q[0]

    def Sources(self, i, state, x, t):
        return 1.0

    def dSigmaFn_du(self, i, state, x, t):
        return np.array([2.0 * state.u[0] * state.q[0]])

    def dSigmaFn_dq(self, i, state, x, t):
        self.derivative_calls += 1
        u = state.u[0]
        return np.array([1.0 + u * u])

    def LowerBoundary(self, i, t):
        return 1.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 1.0 - x

    def InitialDerivative(self, i, x):
        return -1.0

    def createAdjointProblem(self):
        return StoredU()


class StoredU(MaNTA.AdjointProblem):
    """G = int u dx, with one parameter that nothing depends on."""

    def __init__(self):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = False

    def gFn(self, g, states, x):
        return np.asarray(states["Variable"])[:, 0].copy()

    def dgFndp(self, g, states, x):
        return np.zeros((1, len(x)))

    def dg(self, g, states, x):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": np.ones_like(V),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(x), 0)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, x):
        return np.zeros((1, len(x)))

    def dSources(self, i, states, x):
        return np.zeros((1, len(x)))

    def getName(self, p):
        return "none"


NCELLS, K = 6, 2


def config(tmp_path, **overrides):
    cfg = {
        "PolynomialDegree": K,
        "GridSize": NCELLS,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "delta_t": 1.0,
        "Relative_tolerance": 1e-6,
        "Absolute_tolerance": 1e-8,
        "SteadyStateSolver": "Newton",
        "SteadyStateSolve": True,
        "solveAdjoint": True,
        "EstimateObjectiveOnFinish": False,
        "OutputFilename": str(tmp_path / "objective_steady"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


def solve(tmp_path, **overrides):
    case = NonlinearDiffusion()
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path, **overrides))
    case.flux_calls = case.derivative_calls = 0
    runner.run_ss()
    return runner, case


def test_the_error_in_G_is_inside_its_share(tmp_path):
    """Against the discrete fixed point, at three tolerances."""
    reference, _ = solve(tmp_path, SteadyStateStepTolerance=1e-6)
    G_ref = reference.G()[0]
    for tol in (1e-2, 1e-4, 1e-6):
        runner, _ = solve(tmp_path, SteadyStateTest="Objective", ObjectiveTolerance=tol)
        stats = runner.steadyStats()
        assert stats["outcome"] == MaNTA.SteadyOutcome.Converged
        assert abs(runner.G()[0] - G_ref) <= 0.5 * tol * abs(G_ref)
        assert stats["objective_test"] <= 1.0
        assert stats["objective_error"].shape == (1,)
        np.testing.assert_array_less(stats["objective_error"], stats["objective_tolerance"])


def test_the_adjoint_solves_ask_the_case_for_nothing(tmp_path):
    """Newton's sweeps and no others: one flux sweep for the initial sigma, then
    one flux and one derivative sweep per iteration, and no adjoint solve. The
    one derivative sweep more is the end-of-run gradient's, solveAdjoint's, at
    the final state where no build was made."""
    runner, case = solve(tmp_path, SteadyStateTest="Objective", ObjectiveTolerance=1e-8)
    stats = runner.steadyStats()
    nodes = NCELLS * (K + 1)
    n = stats["newton_iterations"]
    assert n > 2
    assert stats["adjoint_solves"] == 0
    assert case.flux_calls == nodes * (1 + n)
    assert case.derivative_calls == nodes * (n + 1)


def test_newton_step_reports_no_objective_test(tmp_path):
    runner, _ = solve(tmp_path)
    stats = runner.steadyStats()
    assert stats["adjoint_solves"] == 0
    assert np.isnan(stats["objective_test"])
    assert stats["objective_error"].shape == (0,)


def test_a_tolerance_below_round_off_stalls(tmp_path):
    runner, _ = solve(tmp_path, SteadyStateTest="Objective", ObjectiveTolerance=1e-30)
    stats = runner.steadyStats()
    assert stats["outcome"] == MaNTA.SteadyOutcome.Stalled
    assert stats["objective_test"] > 1.0


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"SteadyStateTest": "Objective", "solveAdjoint": False}, "solveAdjoint"),
        ({"ObjectiveTolerance": 1e-3}, "ObjectiveTolerance"),
        ({"SteadyStateTest": "Objective", "ObjectiveBudget": [0.7, 0.7]}, "ObjectiveBudget"),
    ],
)
def test_misconfiguration_is_refused(tmp_path, overrides, needle):
    runner = MaNTA.Runner(NonlinearDiffusion())
    with pytest.raises((ValueError, RuntimeError), match=needle):
        runner.configure(config(tmp_path, **overrides))


def test_a_tolerance_per_objective_must_match_the_objectives(tmp_path):
    """One objective, two tolerances: refused when the solve starts, the first
    point at which the objective count is known."""
    runner = MaNTA.Runner(NonlinearDiffusion())
    runner.configure(config(tmp_path, SteadyStateTest="Objective", ObjectiveTolerance=[1e-3, 1e-4]))
    with pytest.raises(Exception, match="one each"):
        runner.run_ss()
