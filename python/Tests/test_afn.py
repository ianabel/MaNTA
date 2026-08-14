"""aFn from Python: the coefficient a_i(x) on du_i/dt.

The equation MaNTA integrates is

    a_i(x) d_t u_i - d_x[ sigma_hat_i ] = S_i

and `aFn` is the one physics hook with a default rather than a pure virtual --
the base returns 1.0. It had no Python binding at all until this file's
counterpart in Python.cpp, so a Python case could not write anything but
a_i = 1 even though the C++ solver has supported it throughout (`ADTestProblem`
is the only case in the tree that overrides it).

Two things need testing, and the second is the one that matters:

  * a case that says nothing still gets a_i = 1, because the trampoline uses
    PYBIND11_OVERRIDE and falls back to the base rather than throwing the way
    `override_for` does for a required hook;
  * an override actually reaches the solver. A binding that is present but not
    consulted would pass any test that only calls `sys.aFn(...)` directly, so
    these check the *effect* on a solve.

The C++ side is covered in Tests/UnitTests/AFnTests.cpp, including that the
residual and the Jacobian agree for a non-unit, position-dependent a_i -- the
case where a wrong answer is invisible and only Newton slows.
"""

import numpy as np
import pytest

import manta


# ------------------------------------------------------------------ fixtures --

class Decay(manta.TransportSystem):
    """a_i d_t u = d_xx u on [0, 1], u = 0 at both ends, u(x, 0) = sin(pi x).

    Closed form: u = sin(pi x) exp(-pi^2 t / a). So a_i rescales time and does
    nothing else, which is what makes the checks below exact rather than
    approximate: two runs sharing a mesh share a discrete eigenvalue, so the
    spatial error cancels between them.
    """

    variables = [manta.Field("u", "the diffused quantity", "")]

    def __init__(self, a=None):
        super().__init__()
        self.a = a          # None -> do not override aFn at all
        self.a_calls = 0

    # Defined unconditionally, but returns the base value when self.a is None so
    # that the "says nothing" case is still exercised through the same class.
    def aFn(self, i, x):
        self.a_calls += 1
        if self.a is None:
            return 1.0
        return self.a

    def SigmaFn(self, i, s, x, t):
        return s.q[i]

    def Sources(self, i, s, x, t):
        return 0.0

    def dSigmaFn_dq(self, i, s, x, t):
        return np.array([1.0])

    def InitialValue(self, i, x):
        return np.sin(np.pi * x)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0


class Silent(manta.TransportSystem):
    """A case that never mentions aFn -- what most cases look like.

    Deliberately *not* derived from Decay: subclassing it and trying to remove the
    method does not work, since the inherited one is still found, and the test
    would then be checking Decay.aFn rather than the fallback.
    """

    variables = [manta.Field("u", "the diffused quantity", "")]

    def SigmaFn(self, i, s, x, t):
        return s.q[i]

    def Sources(self, i, s, x, t):
        return 0.0

    def dSigmaFn_dq(self, i, s, x, t):
        return np.array([1.0])

    def InitialValue(self, i, x):
        return np.sin(np.pi * x)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0


XS = [0.125, 0.25, 0.5, 0.75, 0.875]


def solved(system, t_final, tmp_path, stem):
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 4,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": t_final,
        "Relative_tolerance": 1e-11,
        "Absolute_tolerance": [1e-12],
        "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / stem),
        "WriteOutput": False,
    })
    runner.run(t_final)
    return np.asarray(runner.getSolution(0, list(XS))).reshape(-1)


# --------------------------------------------------------------- the tests --

def test_a_case_that_says_nothing_gets_one():
    """The fallback, and it must be a *value* rather than a throw.

    A required hook with no override raises from `override_for`; aFn has a default,
    so PYBIND11_OVERRIDE finds nothing in the subclass and calls the base, which
    returns 1.0. `aFn` is not in Silent's own dictionary, only in the bound base.
    """
    assert "aFn" not in Silent.__dict__
    system = Silent()
    assert system.aFn(0, 0.3) == 1.0
    assert system.aFn(0, 7.5) == 1.0


def test_a_case_that_says_nothing_solves_as_if_a_is_one(tmp_path):
    """The fallback where it matters: in a run, not just in a direct call."""
    T = 0.02
    silent = solved(Silent(), T, tmp_path, "afn_silent")
    unity = solved(Decay(a=1.0), T, tmp_path, "afn_unity")
    assert np.allclose(silent, unity, atol=1e-10), (silent, unity)


def test_an_override_is_actually_called_by_the_solver(tmp_path):
    """A binding that existed but was never consulted would pass a direct call."""
    system = Decay(a=2.0)
    solved(system, 0.01, tmp_path, "afn_called")
    assert system.a_calls > 0, "the solver never called aFn"


def test_a_constant_coefficient_rescales_time(tmp_path):
    """The load-bearing check: a = A must be the a = 1 problem at t/A.

    Both runs discretise the same eigenfunction on the same mesh, so they share a
    discrete eigenvalue and the spatial error cancels exactly. What is left is the
    effect of A.
    """
    T, A = 0.02, 4.0
    one = solved(Decay(a=1.0), T, tmp_path, "afn_py_1")
    four = solved(Decay(a=A), A * T, tmp_path, "afn_py_4")

    assert np.max(np.abs(one)) > 1e-6, one
    assert np.allclose(one, four, atol=1e-8), (one, four)


def test_the_decay_rate_matches_the_closed_form(tmp_path):
    """Against an absolute reference, so a pair of runs wrong alike cannot pass.

    Also pins the *direction*: a larger a_i decays more slowly, so dividing by
    a_i the wrong way round fails here even though the rescaling test above would
    still pass.
    """
    T, A = 0.02, 3.0
    got = solved(Decay(a=A), T, tmp_path, "afn_py_form")
    want = np.sin(np.pi * np.array(XS)) * np.exp(-np.pi**2 * T / A)
    assert np.allclose(got, want, atol=1e-6), (got, want)

    quick = solved(Decay(a=1.0), T, tmp_path, "afn_py_quick")
    assert np.max(np.abs(quick)) < np.max(np.abs(got))


def test_a_position_dependent_coefficient_is_accepted(tmp_path):
    """a_i(x), not just a constant -- the shape ADTestProblem uses.

    No closed form here, so this checks the run completes and that the answer
    differs from the constant-coefficient one, i.e. that x is really being used.
    A weight of 1 + x on [0, 1] averages 1.5, so the decay should sit between the
    a = 1 and a = 2 answers rather than matching either.
    """
    T = 0.02

    class Weighted(Decay):
        def aFn(self, i, x):
            self.a_calls += 1
            return 1.0 + x

    got = solved(Weighted(), T, tmp_path, "afn_py_x")
    fast = solved(Decay(a=1.0), T, tmp_path, "afn_py_x1")
    slow = solved(Decay(a=2.0), T, tmp_path, "afn_py_x2")

    peak, lo, hi = np.max(np.abs(got)), np.max(np.abs(fast)), np.max(np.abs(slow))
    assert lo < peak < hi, (lo, peak, hi)


def test_the_hook_receives_the_variable_index(tmp_path):
    """Two variables with different coefficients, so an index slip is visible."""

    class TwoVar(manta.TransportSystem):
        variables = [manta.Field("u0", "", ""), manta.Field("u1", "", "")]

        def __init__(self):
            super().__init__()
            self.seen = set()

        def aFn(self, i, x):
            self.seen.add(int(i))
            return 1.0 if i == 0 else 4.0

        def SigmaFn(self, i, s, x, t):
            return s.q[i]

        def Sources(self, i, s, x, t):
            return 0.0

        def dSigmaFn_dq(self, i, s, x, t):
            out = np.zeros(2)
            out[i] = 1.0
            return out

        def InitialValue(self, i, x):
            return np.sin(np.pi * x)

        def InitialDerivative(self, i, x):
            return np.pi * np.cos(np.pi * x)

        def LowerBoundary(self, i, t):
            return 0.0

        def UpperBoundary(self, i, t):
            return 0.0

    T = 0.02
    system = TwoVar()
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 4, "Grid_size": 8,
        "Lower_boundary": 0.0, "Upper_boundary": 1.0,
        "delta_t": T, "Relative_tolerance": 1e-11,
        "Absolute_tolerance": [1e-12], "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / "afn_py_2var"), "WriteOutput": False,
    })
    runner.run(T)

    assert system.seen == {0, 1}, system.seen
    u0 = np.asarray(runner.getSolution(0, list(XS))).reshape(-1)
    u1 = np.asarray(runner.getSolution(1, list(XS))).reshape(-1)

    # Same equation, same initial condition, different a_i: u1 must have decayed
    # less. Equal peaks would mean one coefficient was used for both.
    assert np.max(np.abs(u1)) > np.max(np.abs(u0)), (u0, u1)
