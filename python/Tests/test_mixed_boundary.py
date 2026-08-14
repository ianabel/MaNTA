"""Mixed (Robin) boundary conditions, from Python.

The assembly is covered in C++ (Tests/UnitTests/MixedBoundaryTests.cpp). What is
Python-specific, and what this file covers:

  * a case declares a Mixed end through `manta.Mixed(a=, b=, d=)`, and the run
    reaches the closed-form answer that condition implies;
  * `manta.Neumann` still works wherever a condition is now wanted, which is the
    implicit conversion every example and both python-physics systems rely on;
  * a Mixed end with only a `u` term is refused, with a message naming Dirichlet;
  * an adjoint *boundary parameter* on a non-Dirichlet end is refused rather than
    silently given a Dirichlet-shaped derivative. F_p has no lambda rows, and a
    Mixed or Neumann datum enters through L_global in the trace row, so there is
    nothing correct to put there yet.
"""

import numpy as np
import pytest

import manta


# ------------------------------------------------------------------ fixtures --

class LinearRelax(manta.TransportSystem):
    """Steady linear diffusion, sigma_hat = q, no source -- so u is linear.

    A linear u lies in P_k and satisfies the discrete mixed row exactly, so the
    run reproduces the closed form rather than converging to it at an order.
    """

    variables = [manta.Field("u", "the diffused quantity", "")]

    def __init__(self, lower, upper, c_lower, c_upper):
        type(self).variables = [manta.Field("u", "the diffused quantity", "",
                                            lower=lower, upper=upper)]
        super().__init__()
        self.c_lower = c_lower
        self.c_upper = c_upper

    def SigmaFn(self, i, s, x, t):
        return s.q[i]

    def Sources(self, i, s, x, t):
        return 0.0

    def dSigmaFn_dq(self, i, s, x, t):
        return np.array([1.0])

    def InitialValue(self, i, x):
        return 1.0 + np.sin(np.pi * x)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return self.c_lower

    def UpperBoundary(self, i, t):
        return self.c_upper


def relaxed(system, tmp_path, xs, stem="mixed_py"):
    """Relax to steady state; return u at xs."""
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 3,
        "Grid_size": 6,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 10.0,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-9],
        "MinStepSize": 1e-12,
        # The algebraic rows are otherwise in IDA's local error test, which at
        # these tolerances is the wall docs/running.rst describes rather than
        # anything to do with the boundary condition.
        "SuppressAlgebraicError": True,
        "OutputFilename": str(tmp_path / stem),
        "WriteOutput": False,
    })
    runner.run(10.0)          # ~100 diffusion times on [0, 1]
    return np.asarray(runner.getSolution(0, list(xs))).reshape(-1)


XS = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------- the tests --

def test_a_mixed_lower_end_reaches_its_closed_form(tmp_path):
    """a = 2, b = -1, c = -1 below; u(1) = 1 above. Steady u = x.

    A + B = 1 and 2A - B = -1 give B = 1, A = 0. The signs of a and b are
    opposite deliberately: at the lower end that is the dissipative choice, and
    the same signs there give an anti-dissipative condition and a run that
    diverges.
    """
    system = LinearRelax(manta.Mixed(a=2.0, b=-1.0), manta.Dirichlet, -1.0, 1.0)
    got = relaxed(system, tmp_path, XS)

    assert np.allclose(got, np.array(XS), atol=1e-8), got


def test_a_mixed_upper_end_reaches_its_closed_form(tmp_path):
    """The mirror: a = 2, b = 1, c = 3 above; u(0) = 1 below. Steady u = 1 + x/3.

    Both ends, because `a` carries the outward normal -- an implementation using
    one sign at both would pass the test above and fail this one.
    """
    system = LinearRelax(manta.Dirichlet, manta.Mixed(a=2.0, b=1.0), 1.0, 3.0)
    got = relaxed(system, tmp_path, XS)

    want = 1.0 + np.array(XS) / 3.0
    assert np.allclose(got, want, atol=1e-8), got


def test_the_d_coefficient_reads_the_stored_sigma(tmp_path):
    """d = 1, c = 0.5 below; u(1) = 1 above.

    The stored sigma is -sigma_hat, and here sigma_hat = q, so sigma = -q.
    sigma(0) = -B = 0.5 gives B = -0.5 and A = 1.5, i.e. u = 1.5 - 0.5x. Read
    against sigma_hat instead and the sign flips to u = 0.5 + 0.5x, so this
    discriminates rather than merely passing.
    """
    system = LinearRelax(manta.Mixed(d=1.0), manta.Dirichlet, 0.5, 1.0)
    got = relaxed(system, tmp_path, XS)

    want = 1.5 - 0.5 * np.array(XS)
    assert np.allclose(got, want, atol=1e-8), got


def test_mixed_with_b_one_matches_neumann(tmp_path):
    """Mixed(b=1) and Neumann with the same value relax to the same solution.

    The equivalence at the level of an answer. The unit-level one became
    tautological when the Neumann assembly was expressed through the mixed path,
    so this is the version that still compares two declarations rather than two
    spellings of one code path.
    """
    neumann = LinearRelax(manta.Neumann, manta.Dirichlet, -0.4, 1.0)
    mixed = LinearRelax(manta.Mixed(b=1.0), manta.Dirichlet, -0.4, 1.0)

    n = relaxed(neumann, tmp_path, XS, stem="eq_n")
    m = relaxed(mixed, tmp_path, XS, stem="eq_m")

    assert np.array_equal(n, m), (n, m)
    assert np.allclose(m, 1.4 - 0.4 * np.array(XS), atol=1e-8), m


def test_a_mixed_end_constraining_only_u_is_refused():
    """b = d = 0 is a weakly imposed Dirichlet, not the Dirichlet kind."""
    with pytest.raises(ValueError, match="Dirichlet"):
        LinearRelax(manta.Mixed(a=1.0), manta.Dirichlet, 0.0, 1.0)


# ----------------------------------------------------- the adjoint refusal --

class TrivialAdjoint(manta.AdjointProblem):
    """G = int u dx with one parameter, declared a *lower* boundary sensitivity.

    The point is not the gradient but that asking for it is refused: the lower
    end here is not Dirichlet, and computeAdjointGradients has no term for a
    datum that enters through L_global.
    """

    def __init__(self):
        manta.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 1
        self.spatialParameters = False

    def gFn(self, gIndex, states, positions):
        return np.asarray(states["Variable"])[:, 0]

    def dgFndp(self, gIndex, states, positions):
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": np.ones_like(V),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), 0)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def dSources(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def getName(self, pIndex):
        return "boundary_datum"

    def computeLowerBoundarySensitivity(self, i, pIndex):
        return True

    def computeUpperBoundarySensitivity(self, i, pIndex):
        return False


class AdjointOnMixed(LinearRelax):
    def __init__(self):
        super().__init__(manta.Mixed(a=2.0, b=-1.0), manta.Dirichlet, -1.0, 1.0)

    def createAdjointProblem(self):
        return TrivialAdjoint()


def test_a_boundary_sensitivity_on_a_mixed_end_is_refused(tmp_path):
    """Refused loudly rather than answered wrongly.

    F_p is 3*nVars*(k+1) + nAux*(k+1) tall -- no lambda rows -- while a Mixed or
    Neumann datum reaches the residual through L_global in the trace row. The
    Dirichlet-shaped term that is there would return a plausible wrong gradient
    with a perfectly good G, which is the failure mode CLAUDE.md records for the
    dSigma/dPhi block.
    """
    system = AdjointOnMixed()
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 2,
        "Grid_size": 4,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.5,
        "Relative_tolerance": 1e-6,
        "Absolute_tolerance": [1e-8],
        "MinStepSize": 1e-12,
        "SuppressAlgebraicError": True,
        "solveAdjoint": True,
        "OutputFilename": str(tmp_path / "adjoint_mixed"),
        "WriteOutput": False,
    })

    with pytest.raises(RuntimeError, match="not Dirichlet"):
        runner.run(0.5)
