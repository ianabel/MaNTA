"""EstimateDiscretisationError, as Runner.objectiveEstimate reports it.

The estimate itself is measured against closed forms in
Tests/UnitTests/DiscretisationEstimateTests.cpp, across degrees, meshes,
objectives and both schemes. What is checked here is the Python surface: that a
Python case reaches it, the keys and their shapes, the per-cell split, and the
configuration it is refused without.

The case is nonlinear diffusion with a manufactured u = sin(pi x) on [0, 1],

    -d/dx[ (1 + u^2) u' ] = S,     u(0) = u(1) = 0,

and the objective G = int u^2 dx = 1/2 exactly.
"""

import numpy as np
import pytest

import manta as MaNTA

EXACT_G = 0.5


class ManufacturedNonlinear(MaNTA.TransportSystem):
    def __init__(self):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        u = state.u[0]
        return (1.0 + u * u) * state.q[0]

    def Sources(self, i, state, x, t):
        s, c = np.sin(np.pi * x), np.cos(np.pi * x)
        return np.pi**2 * s * (1.0 + s * s) - 2.0 * np.pi**2 * s * c * c

    def dSigmaFn_du(self, i, state, x, t):
        return np.array([2.0 * state.u[0] * state.q[0]])

    def dSigmaFn_dq(self, i, state, x, t):
        u = state.u[0]
        return np.array([1.0 + u * u])

    def dSources_du(self, i, state, x, t):
        return np.zeros(1)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(1)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(1)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return np.sin(np.pi * x)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def createAdjointProblem(self):
        return SquareIntegral()


class SquareIntegral(MaNTA.AdjointProblem):
    """G = int u^2 dx, with one parameter nothing depends on."""

    def __init__(self):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = False

    def gFn(self, g, states, x):
        u = np.asarray(states["Variable"])[:, 0]
        return u * u

    def dgFndp(self, g, states, x):
        return np.zeros((1, len(x)))

    def dg(self, g, states, x):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {"Variable": 2.0 * V, "Derivative": zeros, "Flux": zeros,
                "Aux": np.zeros((len(x), 0)), "Scalars": np.zeros(0)}

    def dSigma(self, i, states, x):
        return np.zeros((1, len(x)))

    def dSources(self, i, states, x):
        return np.zeros((1, len(x)))

    def getName(self, p):
        return "none"


def config(tmp_path, **overrides):
    cfg = {
        "PolynomialDegree": 2,
        "GridSize": 8,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "delta_t": 1.0,
        "Relative_tolerance": 1e-10,
        "Absolute_tolerance": [1e-12],
        "SteadyStateSolve": True,
        "SteadyStateSolver": "Newton",
        "solveAdjoint": True,
        "EstimateDiscretisationError": True,
        "OutputFilename": str(tmp_path / "discretisation_estimate"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


def estimate(tmp_path, **overrides):
    runner = MaNTA.Runner(ManufacturedNonlinear())
    runner.configure(config(tmp_path, **overrides))
    runner.run_ss()
    return runner.objectiveEstimate()


@pytest.mark.parametrize("superconvergent", [False, True])
def test_the_estimate_tracks_the_error(tmp_path, superconvergent):
    """Its effectivity against the closed form, and G put back where it is."""
    e = estimate(tmp_path, Superconvergent=superconvergent)
    error = e["corrected"][0] - EXACT_G
    effectivity = e["discretisation"][0] / error
    assert 0.9 < effectivity < 1.2
    # So the continuum estimate is far closer to G than the run's own value.
    assert e["continuum"][0] == pytest.approx(e["corrected"][0] - e["discretisation"][0])
    assert abs(e["continuum"][0] - EXACT_G) < 0.2 * abs(error)


def test_the_per_cell_indicators_sum_to_the_total(tmp_path):
    e = estimate(tmp_path)
    per_cell = np.asarray(e["discretisation_per_cell"])
    assert per_cell.shape == (1, 8)
    assert np.asarray(e["x_faces"]) == pytest.approx(np.linspace(0.0, 1.0, 9))
    assert per_cell.sum() == pytest.approx(e["discretisation"][0], rel=1e-12)


def test_without_the_key_there_is_no_discretisation_entry(tmp_path):
    e = estimate(tmp_path, EstimateDiscretisationError=False)
    assert "value" in e
    for key in ("discretisation", "discretisation_per_cell", "x_faces", "continuum"):
        assert key not in e


@pytest.mark.parametrize(
    "overrides",
    [
        {"solveAdjoint": False},
        {"EstimateObjectiveOnFinish": False},
        {"SteadyStateSolver": "TimeMarch"},
    ],
)
def test_it_is_refused_without_what_it_needs(tmp_path, overrides):
    runner = MaNTA.Runner(ManufacturedNonlinear())
    with pytest.raises(ValueError, match="EstimateDiscretisationError"):
        runner.configure(config(tmp_path, **overrides))
