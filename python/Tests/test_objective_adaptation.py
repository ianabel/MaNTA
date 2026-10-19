"""AdaptationTarget = "Objective" reached from Python.

The control quantity, the mesh decision and the drivers are covered in
Tests/UnitTests/ObjectiveAdaptationTests.cpp. What is checked here is the surface
a study uses: the keys in a Runner.configure dict, the answer against a closed
form through Runner.G(), the refusals as RuntimeError, and that naming the default
target changes nothing.
"""

import math

import numpy as np
import pytest

import manta as MaNTA

from test_runner import LinearDiffusion, base_config


class Integral(MaNTA.AdjointProblem):
    """G = int u dx, or int u^2 dx, with one parameter nothing depends on."""

    def __init__(self, square=False):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = False
        self.square = square

    def gFn(self, g, states, x):
        u = np.asarray(states["Variable"])[:, 0]
        return u * u if self.square else u

    def dgFndp(self, g, states, x):
        return np.zeros((1, len(x)))

    def dg(self, g, states, x):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        dV = 2.0 * V if self.square else np.ones_like(V)
        return {"Variable": dV, "Derivative": zeros, "Flux": zeros,
                "Aux": np.zeros((len(x), 0)), "Scalars": np.zeros(0)}

    def dSigma(self, i, states, x):
        return np.zeros((1, len(x)))

    def dSources(self, i, states, x):
        return np.zeros((1, len(x)))

    def getName(self, p):
        return "none"


class NonlinearSine(MaNTA.TransportSystem):
    """-d/dx[(1 + u^2) u'] = S with u = sin(pi x) on [0, 1]; G = int u^2 = 1/2."""

    regrid = MaNTA.Regrid.InPlace
    EXACT_G = 0.5

    def __init__(self):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        u = state.u[0]
        return (1.0 + u * u) * state.q[0]

    def Sources(self, i, state, x, t):
        s, c = math.sin(math.pi * x), math.cos(math.pi * x)
        return math.pi**2 * s * (1.0 + s * s) - 2.0 * math.pi**2 * s * c * c

    def dSigmaFn_du(self, i, state, x, t):
        return np.array([2.0 * state.u[0] * state.q[0]])

    def dSigmaFn_dq(self, i, state, x, t):
        u = state.u[0]
        return np.array([1.0 + u * u])

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.5 * x * (1.0 - x)

    def InitialDerivative(self, i, x):
        return 0.5 - x

    def createAdjointProblem(self):
        return Integral(square=True)


class AxisSingular(LinearDiffusion):
    """-u'' = (4/9) x^(-2/3), u = x - x^(4/3): singular at the axis; G = int u = 1/14.

    The same problem test_mesh_adaptation.py grades by its modal decay.
    """

    EXACT_G = 1.0 / 14.0

    def Sources(self, i, state, x, t):
        return (4.0 / 9.0) * x ** (-2.0 / 3.0)

    def createAdjointProblem(self):
        return Integral()


def config(tmp_path, **overrides):
    cfg = base_config(tmp_path)
    cfg.update({
        "PolynomialDegree": 1,
        "GridSize": 4,
        "delta_t": 1.0,
        "SteadyStateSolve": True,
        "SteadyStateSolver": "Newton",
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": 1e-10,
        "MinStepSize": 1e-12,
        "MaxPolynomialDegree": 12,
        "solveAdjoint": True,
        "DegreeAdaptation": True,
        "AdaptationTarget": "Objective",
    })
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("tol", [1e-4, 1e-7])
def test_the_degree_loop_meets_the_objective_tolerance(tmp_path, tol):
    """Against the closed form, through the G a study reads."""
    runner = MaNTA.Runner(NonlinearSine())
    runner.configure(config(tmp_path, ObjectiveTolerance=tol))
    runner.run_ss()

    G = float(np.asarray(runner.G())[0])
    assert abs(G - NonlinearSine.EXACT_G) <= tol * NonlinearSine.EXACT_G

    # And the estimate that stopped it is within the discretisation share.
    e = runner.objectiveEstimate()
    assert abs(e["discretisation"][0]) <= 0.5 * tol * abs(e["corrected"][0])
    assert e["value"][0] == G


def test_the_mesh_is_graded_where_the_objective_error_is(tmp_path):
    """p -> h -> p to the objective, on a singular problem the sample does not resolve."""
    runner = MaNTA.Runner(AxisSingular())
    runner.configure(config(tmp_path, PolynomialDegree=3, GridSize=10, DegreeAdaptation=False,
                            MeshAdaptation=True, ObjectiveTolerance=1e-5))
    runner.run_ss()

    widths = np.diff(np.asarray(runner.getCellBoundaries()))
    assert len(widths) == 10, "the cell budget moved"
    assert widths[0] < 1e-4 and widths[0] == widths.min(), f"not graded at the axis: {widths}"

    G = float(np.asarray(runner.G())[0])
    assert abs(G - AxisSingular.EXACT_G) <= 1e-5 * AxisSingular.EXACT_G


def test_naming_the_solution_target_changes_nothing(tmp_path):
    """The default, named: the same solves, bit for bit."""
    def solve(cfg):
        runner = MaNTA.Runner(NonlinearSine())
        runner.configure(cfg)
        runner.run_ss()
        return np.asarray(runner.getSolution(0, list(np.linspace(0.0, 1.0, 17))))

    plain = config(tmp_path, DegreeTolerance=1e-8)
    del plain["AdaptationTarget"]
    named = dict(plain, AdaptationTarget="Solution")
    assert np.array_equal(solve(plain), solve(named))


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"solveAdjoint": False}, "solveAdjoint"),
        ({"DegreeAdaptation": False}, "neither DegreeAdaptation"),
        ({"DegreeTolerance": 1e-6}, "DegreeTolerance"),
        ({"AdaptationTarget": "Output"}, "AdaptationTarget"),
    ],
)
def test_what_it_cannot_serve_is_refused(tmp_path, overrides, needle):
    runner = MaNTA.Runner(NonlinearSine())
    with pytest.raises(RuntimeError, match=needle):
        runner.configure(config(tmp_path, **overrides))
