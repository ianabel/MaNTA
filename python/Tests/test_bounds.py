"""Bounds on a case's variables, declared from Python: manta.Field(bound=...).

The C++ side is Tests/UnitTests/VariableBoundTests.cpp. What is checked here is
that the declaration reaches the solver through the binding and is honoured:
the physics of a Python case is never handed a value that breaches its bound,
on a steady solve whose unbounded first damped step goes far negative with a
finite residual.
"""

import math

import numpy as np
import pytest

import manta as MaNTA

from test_runner import base_config

S0, C = 100.0, 0.1
U_STAR = math.atanh(C)


class SaturatingSink(MaNTA.TransportSystem):
    """S = S0 (C - tanh u) between Dirichlet ends at atanh(C).

    The steady state is u = atanh(C) everywhere. From a bump of height 2,
    where tanh is flat, a damped step at dt = 1 throws the middle of the
    profile far below zero, where the residual is finite.
    """

    def __init__(self, bound):
        super().__init__(variables=[MaNTA.Field("n", "density", bound=bound)])
        self.least = math.inf

    def SigmaFn(self, i, state, x, t):
        self.least = min(self.least, state.u[0])
        return state.q[0]

    def Sources(self, i, state, x, t):
        self.least = min(self.least, state.u[0])
        return S0 * (C - math.tanh(state.u[0]))

    def dSigmaFn_dq(self, i, state, x, t):
        return np.ones(1)

    def dSources_du(self, i, state, x, t):
        return np.array([-S0 / math.cosh(state.u[0]) ** 2])

    def LowerBoundary(self, i, t):
        return U_STAR

    def UpperBoundary(self, i, t):
        return U_STAR

    def InitialValue(self, i, x):
        return U_STAR + 8.0 * x * (1.0 - x)

    def InitialDerivative(self, i, x):
        return 8.0 * (1.0 - 2.0 * x)


def solve(tmp_path, bound):
    case = SaturatingSink(bound)
    runner = MaNTA.Runner(case)
    runner.configure(
        base_config(
            tmp_path,
            GridSize=6,
            SteadyStateSolver="PseudoTransient",
            SteadyStateTolerance=1.0e-3,
            PseudoTransientInitialStep=1.0,
            AbsoluteTolerance=[1.0e-8],
            RelativeTolerance=1.0e-6,
            MinStepSize=1.0e-12,
        )
    )
    outcome = runner.start_steady(False)
    u = np.array(runner.getSolution(0, [0.25, 0.5, 0.75]))
    runner.finish_steady()
    return outcome, case.least, u


def test_a_field_carries_its_bound():
    f = MaNTA.Field("n", bound=MaNTA.Bound.NonNegative, bound_epsilon=1e-6)
    assert f.bound == MaNTA.Bound.NonNegative
    assert f.bound_epsilon == 1e-6
    g = MaNTA.Field("T")
    assert g.bound == MaNTA.Bound.Uncontrolled
    assert g.bound_epsilon is None


def test_a_margin_without_a_bound_is_refused():
    with pytest.raises(ValueError, match="margin"):
        MaNTA.SystemSpec([MaNTA.Field("n", bound_epsilon=1e-3)]).validate()


@pytest.mark.parametrize("bound", [MaNTA.Bound.Positive, MaNTA.Bound.NonNegative])
def test_a_python_case_is_never_handed_a_breach(tmp_path, bound):
    free_outcome, free_least, _ = solve(tmp_path / "free", MaNTA.Bound.Uncontrolled)
    assert free_least < -1.0, "the unbounded solve never went negative; this tests nothing"

    outcome, least, u = solve(tmp_path / "bounded", bound)
    assert outcome == MaNTA.SteadyOutcome.Converged
    if bound == MaNTA.Bound.Positive:
        assert least > 0.0
    else:
        assert least >= 0.0
    np.testing.assert_allclose(u, U_STAR, atol=1e-5)
