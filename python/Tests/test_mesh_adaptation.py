"""The p -> h -> p driver reached from Python (PyRunner + MeshAdaptation.cpp).

What the C++ suite covers and this cannot: the grading *decision*, driven by
assigning functions whose smoothness is known by construction, which is a sharper
test than any solve. See Tests/UnitTests/MeshAdaptationTests.cpp.

What only this can cover: the ownership change. With MeshAdaptation armed, the
driver replaces both the SystemSolver *and the Grid* that configure() built -- the
grid because the adapted mesh has to outlive the solver and every later
getSolution call. PyRunner has always documented configure() as the thing that
builds those, so what these check is that everything hanging off both pointers
still works afterwards.
"""

import math

import numpy as np
import pytest

import manta as MaNTA

from test_runner import LinearDiffusion, base_config


class SineSource(LinearDiffusion):
    """Steady state u = sin(pi x) / pi^2 -- smooth, and in no polynomial space.

    Smooth is the point: this is the *negative* control for the grading decision.
    A rule that grades a problem with no singularity is useless however well it
    localises one that has it, and that is the half MESH-REFINEMENT.md section 7
    never tested.
    """

    def Sources(self, i, state, x, t):
        return math.sin(math.pi * x)

    @staticmethod
    def exact(x):
        return np.sin(np.pi * x) / (np.pi**2)


class AxisSingular(LinearDiffusion):
    """Steady state u = x - x^(4/3): an x^(4/3) singularity at the lower end.

    The *positive* control, and it had to be built rather than borrowed. Shestakov's
    problem is where every grading measurement in MESH-REFINEMENT.md came from, and
    the driver cannot run on it: the sequence needs a continuation mode, and neither
    PseudoTransient nor Newton converges on that degenerate `D0 q^3/u^2` flux -- it
    fails with KINSol -7 even at 10 cells, which is why its own config pins
    TimeMarch. So no benchmark in the tree exercises this path.

    This one does, and is linear so that Newton reaches it in a single step. MaNTA
    integrates a d_t u - d_x[sigma_hat] = S with sigma_hat = kappa q, so the steady
    state solves -kappa u'' = S; with kappa = 1 and S = (4/9) x^(-2/3),

        u'' = -(4/9) x^(-2/3)  =>  u = A x + B - x^(4/3)

    and zero Dirichlet ends give A = 1, B = 0. The second derivative diverges like
    x^(-2/3) at the axis, which is exactly the regularity Shestakov has there.

    The source is singular at x = 0 too, and that is safe rather than lucky: the
    nodes are Chebyshev points of the first kind and so strictly interior, and no
    quadrature this uses evaluates at a cell boundary.
    """

    def Sources(self, i, state, x, t):
        return (4.0 / 9.0) * x ** (-2.0 / 3.0)

    @staticmethod
    def exact(x):
        return x - x ** (4.0 / 3.0)


def mesh_config(tmp_path, **overrides):
    cfg = base_config(tmp_path)
    cfg.update(
        {
            "PolynomialDegree": 4,
            "GridSize": 10,
            "SteadyStateSolver": "Newton",
            "SteadyStateTolerance": 1.0e-11,
            "Absolute_tolerance": 1.0e-10,
            "MinStepSize": 1.0e-12,
            "MeshAdaptation": True,
            "DegreeTolerance": 1.0e-9,
            "MaxPolynomialDegree": 12,
        }
    )
    cfg.update(overrides)
    return cfg


def test_the_sequence_runs_and_leaves_a_smooth_problem_uniform(tmp_path):
    """End to end on a problem that wants no grading.

    Two things asserted, and the second is the one worth having: the run produces
    a good answer, *and* it decided not to grade. A driver that graded everything
    would also produce a good answer here, so the answer alone proves nothing.

    Read from getCellBoundaries rather than from the log. The driver's progress
    lines go through C++ std::println, which is buffered and not flushed when
    pytest reads the descriptor, so capfd sees an empty string -- an assertion on
    it passes or fails for reasons unrelated to the behaviour.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(mesh_config(tmp_path))
    runner.run_ss()

    widths = np.diff(np.asarray(runner.getCellBoundaries()))
    assert len(widths) == 10, "the cell budget moved"
    assert np.allclose(widths, 0.1), f"the mesh was graded on a smooth problem: {widths}"

    x = np.linspace(0.0, 1.0, 41)
    u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
    worst = np.max(np.abs(u - SineSource.exact(x)))
    assert worst < 1e-7, f"worst error {worst:.3e} after the full sequence"


def test_the_sequence_grades_an_axis_singularity_and_beats_p_alone(tmp_path):
    """The positive path, end to end, and the argument for the whole sequence.

    Three runs on the same 10 cells, so the DOF budget is the only thing held
    fixed and the mesh is the only thing that differs:

        uniform, no adaptation      9.43e-03
        p only (degree adaptation)  4.04e-03      2.3x
        p -> h -> p                 4.26e-05    221x

    **p alone buys 2.3x and then stops**, which is the regularity cap: raising the
    degree cannot resolve an x^(4/3) singularity, and MESH-REFINEMENT.md section 6
    measured the same wall on Shestakov (19x from k = 2 to 12, and then nothing).
    Grading at the same budget buys 95x *on top of* that. Both halves of the
    sequence are load-bearing and the h half is the larger one.

    The decision itself: lower end decay rate 1.20 against an interior median of
    7.83 and an upper end of 9.63, so 6.51x rougher at the axis against a threshold
    of 2.0 -- three times the margin it needed.
    """
    x = np.linspace(0.0, 1.0, 201)
    exact = AxisSingular.exact(x)

    def relative_l1(**extra):
        runner = MaNTA.Runner(AxisSingular())
        runner.configure(mesh_config(tmp_path, **extra))
        runner.run_ss()
        u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
        widths = np.diff(np.asarray(runner.getCellBoundaries()))
        return np.sum(np.abs(u - exact)) / np.sum(np.abs(exact)), widths

    p_only, p_widths = relative_l1(MeshAdaptation=False, DegreeAdaptation=True,
                                   Superconvergent=True)
    php, php_widths = relative_l1()

    # p alone leaves the mesh alone, by construction.
    assert np.allclose(p_widths, 0.1)

    # ...and the sequence grades it, at the same cell count.
    assert len(php_widths) == 10, "the cell budget moved"
    assert php_widths.min() / php_widths.max() < 1e-3, (
        f"the mesh was not graded: widths {php_widths}"
    )
    assert php_widths[0] == php_widths.min(), "graded at the wrong end"

    # The measurement that justifies the h stage existing at all. Held to 10x
    # against a measured 95x, so this fails on a regression rather than on noise.
    assert php < p_only / 10.0, (
        f"p -> h -> p gave {php:.3e} against p alone at {p_only:.3e}; the grading "
        "stage is not paying"
    )


def test_the_solution_survives_the_driver_replacing_the_grid(tmp_path):
    """The ownership check, and the reason this file exists.

    The driver may replace `grid` as well as `system`, and getSolution evaluates
    the element polynomials over whichever grid the surviving solver holds. If the
    Runner dropped the adapted mesh while keeping a solver that points into it,
    this reads freed memory -- which would not necessarily crash, so the assertion
    is on the values rather than on merely surviving the call.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(mesh_config(tmp_path))
    runner.run_ss()

    x = np.linspace(0.0, 1.0, 41)
    first = np.asarray(runner.getSolution(0, list(x))).reshape(-1)

    # Read again, and read the derivative too, which walks the same cell list.
    second = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
    q = np.asarray(runner.getDerivative(0, list(x))).reshape(-1)

    assert np.array_equal(first, second)
    assert np.all(np.isfinite(q))

    # q = du/dx = cos(pi x)/pi for this steady state, which is a real check that
    # the grid the solver is using is the one the coefficients belong to: reading a
    # correct u through a wrong mesh is possible, reading a correct derivative is
    # much less so.
    assert np.max(np.abs(q - np.cos(np.pi * x) / np.pi)) < 1e-6


def test_a_second_configure_and_run_starts_from_a_uniform_mesh_again(tmp_path):
    """configure() must reset the mesh, not inherit the last run's graded one.

    PyRunner supports repeated configure/run cycles -- that is what the
    optimisation drivers use -- and the driver replacing `grid` puts a new way to
    break it. A second configure() rebuilds the grid from GridSize, so the sequence
    starts over rather than compounding.
    """
    runner = MaNTA.Runner(SineSource())
    for _ in range(2):
        runner.configure(mesh_config(tmp_path))
        runner.run_ss()

        x = np.linspace(0.0, 1.0, 41)
        u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
        assert np.max(np.abs(u - SineSource.exact(x))) < 1e-7


def test_a_degree_below_three_is_refused(tmp_path):
    """The load-bearing refusal: at k = 2 the grading verdict is reversed."""
    runner = MaNTA.Runner(SineSource())
    with pytest.raises(RuntimeError, match="PolynomialDegree >= 3"):
        runner.configure(mesh_config(tmp_path, PolynomialDegree=2))


def test_deciding_the_mesh_twice_is_refused(tmp_path):
    runner = MaNTA.Runner(SineSource())
    with pytest.raises(RuntimeError, match="GradedGridBoundary"):
        runner.configure(mesh_config(tmp_path, GradedGridBoundary=True))

    with pytest.raises(RuntimeError, match="GridPoints"):
        runner.configure(mesh_config(tmp_path, GridPoints=[0.0, 0.5, 1.0]))


def test_mesh_adaptation_is_refused_by_run(tmp_path):
    """run() means "integrate the transient", which the sequence is not.

    Inherited from DegreeAdaptation's rule, and refused rather than quietly turned
    into a steady solve for the same reason: each stage would take the previous
    stage's final state as its initial condition and integrate the interval again.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(mesh_config(tmp_path))
    with pytest.raises(RuntimeError, match="steady solves"):
        runner.run(0.1)


def test_mesh_adaptation_implies_the_degree_loop(tmp_path):
    """It is the whole sequence, so the last stage runs without being asked for.

    Asserted through the *answer* rather than through a config field: starting at
    k = 3 on 10 cells, SineSource's error is around 7e-06, and reaching 1e-9 needs
    the degree to have been raised. So an accurate answer here is only reachable if
    the last stage ran.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(mesh_config(tmp_path, PolynomialDegree=3))
    runner.run_ss()

    x = np.linspace(0.0, 1.0, 41)
    u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
    worst = np.max(np.abs(u - SineSource.exact(x)))

    # A plain k = 3 solve on this mesh gives about 5e-07 in this norm; the loop
    # takes it to 1e-12. Two orders of margin either side of the assertion.
    assert worst < 1e-9, f"worst error {worst:.3e}; the degree loop did not run"
