"""Degree adaptation reached from Python (PyRunner + DegreeAdaptation.cpp).

The C++ side is covered by Tests/UnitTests/DegreeAdaptationTests.cpp. What is
only reachable here is the *ownership* change: with DegreeAdaptation armed,
run() and run_ss() replace the SystemSolver that configure() built, once per
polynomial degree. PyRunner has always documented the opposite -- configure()
builds the solver, run() only runs it -- so what these check is that everything
hanging off that pointer still works afterwards.
"""

import math

import numpy as np
import pytest

import manta as MaNTA

from test_runner import LinearDiffusion, base_config


class SineSource(LinearDiffusion):
    """Steady state u = sin(pi x) / pi^2, which no polynomial space holds.

    MaNTA integrates a d_t u - d_x[sigma_hat] = S with sigma_hat = kappa q, so
    the steady state solves -kappa u'' = S. With S = sin(pi x), kappa = 1 and
    zero Dirichlet ends that is u = sin(pi x) / pi^2 exactly.

    The parent's constant source gives u = x(1-x)/2 instead -- a quadratic,
    which k >= 2 represents exactly, so nothing would ever need raising.
    """

    def Sources(self, i, state, x, t):
        return math.sin(math.pi * x)

    @staticmethod
    def exact(x):
        return np.sin(np.pi * x) / (np.pi**2)


def adaptive_config(tmp_path, **overrides):
    cfg = base_config(tmp_path)
    cfg.update(
        {
            "Polynomial_degree": 1,
            "Grid_size": 6,
            "SteadyStateSolver": "Newton",
            "SteadyStateTolerance": 1.0e-11,
            "Absolute_tolerance": 1.0e-10,
            "MinStepSize": 1.0e-12,
            "DegreeAdaptation": True,
            "DegreeTolerance": 1.0e-9,
            "MaxPolynomialDegree": 12,
        }
    )
    cfg.update(overrides)
    return cfg


def test_degree_adaptation_beats_the_degree_it_started_from(tmp_path):
    """The point of the feature: a better answer than the configured degree gives.

    Compared against the exact steady state, not against the estimate, so this
    cannot pass by the indicator agreeing with itself.
    """
    points = [0.15, 0.35, 0.5, 0.65, 0.85]
    exact = SineSource.exact(np.array(points))

    fixed = MaNTA.Runner(SineSource())
    fixed.configure(adaptive_config(tmp_path, DegreeAdaptation=False))
    fixed.run_ss()
    fixed_err = np.max(np.abs(fixed.getSolution(0, points) - exact))

    adaptive = MaNTA.Runner(SineSource())
    adaptive.configure(adaptive_config(tmp_path))
    adaptive.run_ss()
    adaptive_err = np.max(np.abs(adaptive.getSolution(0, points) - exact))

    assert adaptive_err < fixed_err / 1e4, (
        f"adaptive reached {adaptive_err:.3e} against a fixed k=1's "
        f"{fixed_err:.3e}; the loop is not buying much"
    )
    assert adaptive_err < 1e-8


def test_the_runner_still_works_after_its_solver_was_replaced(tmp_path):
    """The ownership change, which is what makes this surface different.

    run_ss() destroys the solver configure() built and puts a different one in
    its place. Everything the Python API reads goes through that pointer --
    getSolution, getDerivative, getPostprocessedSolution -- so each is exercised
    here rather than assumed. They all return by value, which is what stops a
    caller holding a reference into a solver that has been replaced.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(adaptive_config(tmp_path))
    runner.run_ss()

    points = [0.2, 0.5, 0.8]
    u = runner.getSolution(0, points)
    q = runner.getDerivative(0, points)
    u_star = runner.getPostprocessedSolution(0, points)

    assert len(u) == len(points)

    # q is d_x u, and the exact derivative is cos(pi x)/pi.
    np.testing.assert_allclose(u, SineSource.exact(np.array(points)), atol=1e-8)
    np.testing.assert_allclose(q, np.cos(np.pi * np.array(points)) / np.pi, atol=1e-7)

    # u* is a different field from u, but on a converged solution the two agree
    # to better than the tolerance the loop was asked for.
    np.testing.assert_allclose(u_star, u, atol=1e-8)


def test_a_solution_the_space_already_holds_is_not_refined(tmp_path):
    """The parent's steady state is x(1-x)/2, exact at k = 2.

    Starting there, the loop must stop at once. Checking it converged to the
    right answer at the degree it started from is the observable half; the
    C++ test checks the "after 1 solve" line directly.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(adaptive_config(tmp_path, Polynomial_degree=2))
    runner.run_ss()

    points = np.array([0.15, 0.4, 0.75])
    expected = points * (1.0 - points) / 2.0
    np.testing.assert_allclose(runner.getSolution(0, points), expected, atol=1e-9)


def test_a_second_run_after_an_adaptive_one_does_not_resume_from_it(tmp_path):
    """`restarting` is sticky, and the loop uses it to carry state between levels.

    Left armed, the next run on the same TransportSystem would build its initial
    condition from the second-to-last level instead of from InitialValue --
    silently, and only on the second run. The driver clears it; this is what
    says so from outside.
    """
    problem = SineSource()
    runner = MaNTA.Runner(problem)

    runner.configure(adaptive_config(tmp_path))
    runner.run_ss()
    first = runner.getSolution(0, [0.25, 0.5, 0.75])

    # A fresh configuration, no restart asked for anywhere.
    runner.configure(adaptive_config(tmp_path))
    runner.run_ss()
    second = runner.getSolution(0, [0.25, 0.5, 0.75])

    np.testing.assert_allclose(first, second, atol=1e-12)


def test_a_transient_run_is_refused_rather_than_silently_made_steady(tmp_path):
    """run() means "integrate the transient"; adaptation is steady-only.

    Refusing rather than quietly doing a steady solve, because run() already
    does the opposite for a config carrying SteadyStateTolerance -- it clears
    the flag and warns, on the grounds that the caller asked for the path and
    not the endpoint. Silently contradicting that would be worse than an error.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(adaptive_config(tmp_path))
    with pytest.raises(RuntimeError, match="run_ss"):
        runner.run(0.1)


def test_adaptation_without_steady_termination_is_refused(tmp_path):
    """The hole that let a transient through, from the dict surface.

    SteadyStateSolver defaults to "PseudoTransient", but the mode is only
    consulted once steady-state termination is *armed*, and arming happens
    through the presence of SteadyStateTolerance. A configuration that simply
    omits it names a steady mode and time-marches anyway.

    On this surface run_ss() supplies its own fallback tolerance, so the
    configuration alone cannot tell -- which is why runAdaptiveDegree checks the
    built solver rather than the config. Reached here through run(), the one
    route that leaves it unarmed.
    """
    cfg = adaptive_config(tmp_path)
    del cfg["SteadyStateTolerance"]

    runner = MaNTA.Runner(SineSource())
    runner.configure(cfg)

    # run() refuses first, being the blunter of the two checks.
    with pytest.raises(RuntimeError):
        runner.run(0.1)

    # ...and run_ss() arms its own tolerance, so it works -- but to that
    # tolerance, the 1e-3 fallback, and not to the 1e-9 DegreeTolerance the
    # config names. The two measure different things: DegreeTolerance is how well
    # *resolved* the answer must be, SteadyStateTolerance how far the solve is
    # driven towards the fixed point, and no amount of the first makes up for a
    # loose second.
    #
    # Measured 3.7e-7 out, against the 1e-8 the same problem reaches when a
    # tolerance is named. Not a large gap -- ||F|| = 1e-3 is a tighter statement
    # about u than it looks on this problem -- but a real one, and the guard
    # below is what keeps this case honest about which of the two it is testing.
    runner.run_ss()
    got = runner.getSolution(0, [0.5])[0]
    exact = SineSource.exact(np.array([0.5]))[0]
    assert got == pytest.approx(exact, abs=1e-6)
    assert abs(got - exact) > 1e-8, (
        "run_ss() without SteadyStateTolerance now converges as tightly as a "
        "named one; if the fallback changed, say so here rather than loosening "
        "the assertion above"
    )


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"Superconvergent": False}, "Superconvergent"),
        ({"SteadyStateSolver": "TimeMarch"}, "TimeMarch"),
        ({"DegreeAdaptationBase": 2.0}, "DegreeAdaptationBase"),
        ({"DegreeTolerance": 0.0}, "DegreeTolerance"),
        ({"MaxPolynomialDegree": 0}, "MaxPolynomialDegree"),
    ],
)
def test_inconsistent_degree_adaptation_configs_are_refused(tmp_path, override, fragment):
    """Caught in the configuration rather than part-way through a run.

    Superconvergent = false is the interesting one: adaptation turns the flag on
    for you, so an explicit false is a contradiction rather than a preference,
    and silently overriding a key the user wrote would be worse than refusing
    it.

    configure() raises RuntimeError, not ValueError: loadSolverConfig throws
    std::invalid_argument and PyRunner re-wraps it, which is the behaviour this
    surface has always had.
    """
    runner = MaNTA.Runner(SineSource())
    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(adaptive_config(tmp_path, **override))
    assert fragment in str(excinfo.value)
