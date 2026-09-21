"""Sources that read ``state.udot``, from Python.

The C++ side is covered in ``Tests/UnitTests/TimeDerivativeSourceTests.cpp``,
including the two mutations that show the tests bite. What is Python's own is
the path from a class attribute to the solver: ``manta.Field(...,
source_reads_time_derivatives=True)`` has to reach the spec, ``state.udot`` has
to be filled by the time ``Sources`` is called, and ``dSources_dudot`` has to be
found by the trampoline's optional-override probe.

The manufactured problem is the same shape as the C++ one. Two variables solve
the same equation with the same exact solution; variable 1's source carries
``C * udot[0]`` with a compensating ``- C * du_exact/dt`` subtracted, so the
exact solution is unchanged and a run that never filled ``udot`` would be
integrating a source short by ``C du/dt``. That is what makes the comparison
against ``C = 0`` an assertion rather than a formality: measured, dropping the
term takes the error from 7e-5 to 2e-2.
"""

import numpy as np
import pytest

import manta


C = 0.4
XS = [0.125, 0.25, 0.5, 0.75, 0.875]


def exact(x, t):
    return np.sin(np.pi * x) * (1.0 + t)


def diffusion_source(x, t):
    """u_t - u_xx for u = sin(pi x)(1 + t)."""
    return np.sin(np.pi * x) * (1.0 + np.pi**2 * (1.0 + t))


class Coupled(manta.TransportSystem):
    """Variable 1's source reads du_0/dt; variable 0's reads nothing.

    The asymmetry is deliberate. Only variable 1 declares the flag, so a run in
    which ``udot`` were filled for everything, or the flag honoured per system
    rather than per variable, would still pass a test that declared both.
    """

    variables = [
        manta.Field("n", "first", ""),
        manta.Field("T", "second", "", source_reads_time_derivatives=True),
    ]

    def __init__(self, coupling=C):
        super().__init__()
        self.coupling = coupling
        self.udot_seen = []

    def SigmaFn(self, i, s, x, t):
        return s.q[i]

    def Sources(self, i, s, x, t):
        if i == 0:
            return diffusion_source(x, t)
        self.udot_seen.append(s.udot[0])
        return (diffusion_source(x, t)
                - self.coupling * np.sin(np.pi * x)
                + self.coupling * s.udot[0])

    def dSigmaFn_dq(self, i, s, x, t):
        v = np.zeros(2)
        v[i] = 1.0
        return v

    def dSources_dudot(self, i, s, x, t):
        v = np.zeros(2)
        if i == 1:
            v[0] = self.coupling
        return v

    def InitialValue(self, i, x):
        return exact(x, 0.0)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0


def solved(system, t_final, tmp_path, stem):
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": t_final,
        "Relative_tolerance": 1e-10,
        "Absolute_tolerance": [1e-11],
        "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / stem),
        "WriteOutput": False,
    })
    runner.run(t_final)
    return np.asarray(runner.getSolution(1, list(XS))).reshape(-1)


def test_the_flag_reaches_the_spec():
    """A class attribute, not a runtime setting: it describes the equations."""
    case = Coupled()
    assert case.spec.variables[1].source_reads_time_derivatives
    assert not case.spec.variables[0].source_reads_time_derivatives


def test_the_default_is_off():
    """Every existing case keeps its behaviour by declaring nothing."""
    assert not manta.Field("u").source_reads_time_derivatives


def test_udot_is_filled_before_sources_is_called(tmp_path):
    """The value has to be there, and it has to be the time derivative.

    At the exact solution ``du/dt = sin(pi x)``, which is O(1) on this domain, so
    a ``udot`` that were never filled would show up here as a column of zeros
    rather than as a small error.
    """
    case = Coupled()
    solved(case, 0.05, tmp_path, "udot_seen")

    seen = np.abs(np.asarray(case.udot_seen))
    assert seen.size > 0
    assert seen.max() > 0.1


def test_the_coupled_run_matches_the_closed_form(tmp_path):
    """The assertion the coupling is built to make fail if udot were dropped.

    Both problems have the same exact solution, so their errors should be
    comparable; without the term the coupled one is short by ``C du/dt`` and the
    error is two orders larger.
    """
    t_final = 0.1

    coupled = solved(Coupled(C), t_final, tmp_path, "udot_coupled")
    uncoupled = solved(Coupled(0.0), t_final, tmp_path, "udot_uncoupled")
    reference = exact(np.asarray(XS), t_final)

    err_coupled = np.max(np.abs(coupled - reference))
    err_uncoupled = np.max(np.abs(uncoupled - reference))

    assert err_coupled < 1e-4
    assert err_coupled < 10.0 * err_uncoupled


def test_a_case_that_declares_nothing_sees_zero(tmp_path):
    """udot is not filled at all unless a variable asks, and reads as zero."""

    class Silent(manta.TransportSystem):
        variables = [manta.Field("u", "the diffused quantity", "")]

        def __init__(self):
            super().__init__()
            self.udot_seen = []

        def SigmaFn(self, i, s, x, t):
            return s.q[i]

        def Sources(self, i, s, x, t):
            self.udot_seen.append(s.udot[0])
            return diffusion_source(x, t)

        def dSigmaFn_dq(self, i, s, x, t):
            return np.array([1.0])

        def InitialValue(self, i, x):
            return exact(x, 0.0)

        def InitialDerivative(self, i, x):
            return np.pi * np.cos(np.pi * x)

        def LowerBoundary(self, i, t):
            return 0.0

        def UpperBoundary(self, i, t):
            return 0.0

    case = Silent()
    runner = manta.Runner(case)
    runner.configure({
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.05,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-9],
        "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / "udot_silent"),
        "WriteOutput": False,
    })
    runner.run(0.05)

    seen = np.asarray(case.udot_seen)
    assert seen.size > 0
    assert np.all(seen == 0.0)
