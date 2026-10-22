"""Adjoint sensitivities to Dirichlet boundary data, at both ends.

`addLowerBoundarySensitivity` / `addUpperBoundarySensitivity` append one
parameter per (variable, end), and `accumulateAdjointGradients` gives it the
derivative of the residual with respect to the datum g_D. That datum enters the
residual twice, through `updateBoundaryConditions`'s RF_cellwise:

    q rows:  -< g_D, v n >         (n = -1 below, +1 above)
    u rows:  -< tau g_D, w >

and both terms have to be in F_p. Nothing covered this before: the upper end was
missing the tau term and the lower end had the q term's sign flipped as well,
and both converged runs reported a plausible G alongside the wrong dG/dg_D.

The physics is the JAX linear-diffusion example (python/Tests/JAXLinearDiffusion
and python-examples/jax-linear-diffusion) with its boundary values made
parameters. On [-1, 1] the steady state of -kappa u'' = S with u(-1) = g_L,
u(1) = g_R is the quadratic

    u = -S x^2 / (2 kappa) + (g_R - g_L) x / 2 + (g_L + g_R) / 2 + S / (2 kappa)

so at PolynomialDegree = 4 the solution and G = int 0.5 u^2 dx are both exact,
and the closed form is a reference rather than an approximation. The two data
are deliberately unequal: with g_L = g_R the problem is symmetric and
dG/dg_L = dG/dg_R, which would hide a sign error at one end.
"""

import numpy as np
import pytest

import manta as MaNTA

jax = pytest.importorskip("jax")

from JAXLinearDiffusion import JAXLinearDiffusion  # noqa: E402
from manta.jax import JAXAdjointProblem  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _float64():
    """The tolerances below are far under float32 round-off, and JAX computes in
    float32 unless told otherwise. In float32 ||F|| bottoms out near 1.2e-7, the
    Newton correction against Absolute_tolerance = 1e-10 never gets under the
    NewtonStep test's 0.33, and the steady solve halves its way to the
    continuation-step limit (a transient run fails in IDA's corrector instead).
    Scoped to this module and restored, since it is process-wide."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


KAPPA = 2.0
SOURCE = 10.0  # JAXLinearDiffusion.source
G_LOWER, G_UPPER = 0.3, -0.2
TAU = 0.5


class DirichletData(JAXLinearDiffusion):
    """The example, with its two Dirichlet values as parameters."""

    def __init__(self, g_lower, g_upper):
        super().__init__({"Centre": 0.0, "kappa": KAPPA}, None)
        self.g_lower, self.g_upper = g_lower, g_upper

    def LowerBoundary(self, index, t):
        return self.g_lower

    def UpperBoundary(self, index, t):
        return self.g_upper

    def InitialValue(self, index, x):
        # the line between the two data rather than the example's narrow
        # Gaussian, which is ~0 at x = +-1 and so jumps to the data at both
        # ends -- IDA_ERR_FAIL on the first step at this resolution
        return self.g_lower * (1 - x) / 2 + self.g_upper * (1 + x) / 2

    def createAdjointProblem(self):
        adjoint = JAXAdjointProblem(self, self.g)
        adjoint.addLowerBoundarySensitivity(0)  # boundary column 0
        adjoint.addUpperBoundarySensitivity(0)  # boundary column 1
        return adjoint


def _gauss(f):
    x, w = np.polynomial.legendre.leggauss(8)  # exact to degree 15 on [-1, 1]
    return float(np.sum(w * f(x)))


def exact(g_lower, g_upper):
    """G and (dG/dg_L, dG/dg_R) for the steady state."""

    def u(x):
        return (
            -SOURCE * x**2 / (2 * KAPPA)
            + (g_upper - g_lower) * x / 2
            + (g_lower + g_upper) / 2
            + SOURCE / (2 * KAPPA)
        )

    G = _gauss(lambda x: 0.5 * u(x) ** 2)
    dG = np.array(
        [
            _gauss(lambda x: u(x) * (1 - x) / 2),  # du/dg_L = (1 - x) / 2
            _gauss(lambda x: u(x) * (1 + x) / 2),  # du/dg_R = (1 + x) / 2
        ]
    )
    return G, dG


def solve(g_lower, g_upper, tmp_path):
    runner = MaNTA.Runner(DirichletData(g_lower, g_upper))
    runner.configure(
        {
            "PolynomialDegree": 4,
            "GridSize": 6,
            "LowerBoundary": -1.0,
            "UpperBoundary": 1.0,
            "tau": TAU,
            "delta_t": 0.5,
            "Relative_tolerance": 1e-8,
            "Absolute_tolerance": [1e-10],
            "MinStepSize": 1e-12,
            # linear, so the undamped Newton solve converges in two iterations
            "SteadyStateSolver": "Newton",
            "SteadyStateTolerance": 1e-11,
            "solveAdjoint": True,
            "OutputFilename": str(tmp_path / "boundary_sensitivity"),
            "WriteOutput": False,
        }
    )
    runner.run_ss()
    G, gradients = runner.getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p_boundary"]).reshape(-1)


def test_the_objective_matches_the_closed_form(tmp_path):
    G, _ = solve(G_LOWER, G_UPPER, tmp_path)
    assert G == pytest.approx(exact(G_LOWER, G_UPPER)[0], rel=1e-8)


def test_boundary_sensitivities_match_the_closed_form(tmp_path):
    _, dG = solve(G_LOWER, G_UPPER, tmp_path)
    reference = exact(G_LOWER, G_UPPER)[1]
    assert dG.shape == (2,), dG.shape
    np.testing.assert_allclose(
        dG, reference, rtol=1e-8, err_msg="columns are (lower, upper)"
    )


def test_boundary_sensitivities_match_finite_differences(tmp_path):
    """Independent of the closed form: G is quadratic in the data, so a central
    difference is exact up to round-off and the solver tolerance."""
    _, dG = solve(G_LOWER, G_UPPER, tmp_path)
    h = 1e-4
    fd = np.array(
        [
            solve(G_LOWER + h, G_UPPER, tmp_path)[0] - solve(G_LOWER - h, G_UPPER, tmp_path)[0],
            solve(G_LOWER, G_UPPER + h, tmp_path)[0] - solve(G_LOWER, G_UPPER - h, tmp_path)[0],
        ]
    ) / (2 * h)
    np.testing.assert_allclose(dG, fd, rtol=1e-6, err_msg="columns are (lower, upper)")
