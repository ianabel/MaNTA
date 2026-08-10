"""Tests for MaNTA.Runner (PyRunner.cpp) and the PyTransportSystem trampoline.

PyRunner is the API the optimisation drivers use -- configure(dict) / run() /
getSolution() -- and it had no test coverage at all. It is also the only route
that supports repeated configure/run cycles in one process.

The transport system here is deliberately plain Python (no JAX): it overrides
only the scalar virtuals, which exercises PyTransportSystem's fallback path
where the vectorised methods are absent and the C++ base loops over the scalar
ones.
"""

import math
import os

import numpy as np
import pytest

import manta as MaNTA


class LinearDiffusion(MaNTA.TransportSystem):
    """d_t u = d_x( kappa d_x u ) + S, with Dirichlet ends.

    sigma = kappa * q, so the derivatives are constant and easy to state
    exactly -- which is the point: any error is in the plumbing, not here.
    """

    def __init__(self, kappa=1.0, source=1.0, spec=None):
        # spec is an argument so the two-variable subclass below can widen it.
        # That used to be `self.nVars = 2` after construction.
        MaNTA.TransportSystem.__init__(self, spec or MaNTA.numbered_spec(1))
        self.kappa = kappa
        self.source = source

    # --- required scalar interface -------------------------------------
    def SigmaFn(self, i, state, x, t):
        return self.kappa * state["Derivative"][i]

    def Sources(self, i, state, x, t):
        return self.source

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self.kappa)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    # --- boundaries and initial data -----------------------------------
    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


def base_config(tmp_path, **overrides):
    cfg = {
        "Polynomial_degree": 2,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.05,
        "OutputFilename": str(tmp_path / "runner_test"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------ validation --


def test_missing_required_parameters_are_all_reported(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())

    # Polynomial_degree, Grid_size, delta_t and OutputFilename are required.
    with pytest.raises(RuntimeError) as excinfo:
        runner.configure({})

    message = str(excinfo.value)
    for key in ("Polynomial_degree", "Grid_size", "delta_t", "OutputFilename"):
        assert key in message, f"{key} missing from error: {message}"


def test_one_missing_required_parameter_is_named(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    cfg = base_config(tmp_path)
    del cfg["Grid_size"]

    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(cfg)
    assert "Grid_size" in str(excinfo.value)


def test_boundaries_are_required_without_grid_points(tmp_path):
    """Lower_boundary/Upper_boundary are needed unless Grid_points is given.

    They are absent from the `params` table, so the up-front required-parameter
    check cannot see them. Omitting them used to surface as "Failed to retrieve
    default value for key: Lower_boundary; possible type mismatch" -- an error
    naming a cause that had nothing to do with the problem.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    cfg = base_config(tmp_path)
    del cfg["Lower_boundary"]
    del cfg["Upper_boundary"]

    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(cfg)
    message = str(excinfo.value)
    assert "Lower_boundary" in message
    assert "type mismatch" not in message


def test_wrong_parameter_type_is_rejected(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    cfg = base_config(tmp_path, Grid_size="not an integer")

    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(cfg)
    assert "Grid_size" in str(excinfo.value)


def test_configure_accepts_the_minimal_required_set(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path))


@pytest.mark.parametrize(
    "key,value",
    [
        ("tau", 2.0),
        ("tZero", 0.0),
        ("Relative_tolerance", 1e-4),
        ("Absolute_tolerance", [1e-4]),
        ("MinStepSize", 1e-8),
        ("OutputPoints", 51),
        ("solveAdjoint", False),
        ("SteadyStateTolerance", 1e-2),
        ("WriteOutput", False),
        ("zeroFlux", False),
        ("initialTimestep", 0.0),
        ("High_Grid_Boundary", False),
        ("Lower_Boundary_Fraction", 0.25),
        ("Upper_Boundary_Fraction", 0.25),
        ("restart", False),
    ],
)
def test_every_optional_parameter_is_accepted(tmp_path, key, value):
    """Walk the declarative parameter table in PyRunner.cpp.

    Each optional key must be settable without disturbing the others; a typo in
    the table (wrong variant type) shows up here as a cast failure.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path, **{key: value}))


def test_high_grid_boundary_builds_a_clustered_grid(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(
        base_config(
            tmp_path,
            Grid_size=9,
            High_Grid_Boundary=True,
            Lower_Boundary_Fraction=0.2,
            Upper_Boundary_Fraction=0.2,
        )
    )
    runner.run(0.05)


def test_explicit_grid_points_are_honoured(tmp_path):
    points = [0.0, 0.1, 0.3, 0.6, 1.0]
    runner = MaNTA.Runner(LinearDiffusion())
    cfg = base_config(tmp_path, Grid_size=len(points) - 1, Grid_points=points)
    del cfg["Lower_boundary"]
    del cfg["Upper_boundary"]
    runner.configure(cfg)
    runner.run(0.05)

    # getSolution on the default grid should return one value per nodal point.
    u = runner.getSolution(0, None)
    assert len(u) > 0


# ------------------------------------------------------------------ runs --


def test_run_advances_and_getSolution_returns_the_profile(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion(kappa=1.0, source=1.0))
    runner.configure(base_config(tmp_path))
    runner.run(0.2)

    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    u = np.asarray(runner.getSolution(0, xs))
    assert u.shape == (len(xs),)
    assert np.all(np.isfinite(u))

    # Dirichlet zero at both ends. HDG imposes Dirichlet data weakly, on the
    # trace variable lambda -- the DG field u only approaches it at the
    # discretisation rate, so this is a discretisation-level check, not an
    # exact one.
    assert abs(u[0]) < 1e-3
    assert abs(u[-1]) < 1e-3

    # A positive source with zero Dirichlet ends drives u positive inside.
    assert u[2] > 0.0


def test_solution_converges_towards_the_steady_state(tmp_path):
    """-kappa u'' = S with u(0) = u(1) = 0 has u = S x (1-x) / (2 kappa)."""
    kappa, source = 1.0, 1.0
    runner = MaNTA.Runner(LinearDiffusion(kappa=kappa, source=source))
    runner.configure(base_config(tmp_path, Polynomial_degree=3, Grid_size=16))
    runner.run(3.0)  # long enough to be near steady state

    xs = [0.125 * i for i in range(1, 8)]
    u = np.asarray(runner.getSolution(0, xs))
    exact = np.array([source * x * (1.0 - x) / (2.0 * kappa) for x in xs])

    assert np.allclose(u, exact, atol=1e-3), f"got {u}, want {exact}"


def test_run_ss_reaches_the_same_steady_state(tmp_path):
    kappa, source = 1.0, 1.0
    runner = MaNTA.Runner(LinearDiffusion(kappa=kappa, source=source))
    runner.configure(
        base_config(tmp_path, Polynomial_degree=3, Grid_size=16, SteadyStateTolerance=1e-4)
    )
    runner.run_ss()

    xs = [0.25, 0.5, 0.75]
    u = np.asarray(runner.getSolution(0, xs))
    exact = np.array([source * x * (1.0 - x) / (2.0 * kappa) for x in xs])

    # run_ss stops on a weighted dlambda/dt criterion, not on an accuracy
    # target, so it exits while still a few percent from the exact steady
    # state. Assert the shape is right and it is close, not that it converged
    # to the same tolerance as the long integration above.
    assert np.allclose(u, exact, rtol=0.05), f"got {u}, want {exact}"
    assert u[1] > u[0] and u[1] > u[2], "profile should peak in the middle"


def test_reconfigure_and_rerun_in_one_process(tmp_path):
    """The optimisation drivers reuse a Runner across configurations.

    This is the path that the un-invalidated Integrator caches used to corrupt
    (see PyIntegratorTests.cpp) -- a second solve on a different grid must give
    the same physics, not the first grid's quadrature.
    """
    kappa, source = 1.0, 1.0
    exact = lambda x: source * x * (1.0 - x) / (2.0 * kappa)
    xs = [0.25, 0.5, 0.75]

    runner = MaNTA.Runner(LinearDiffusion(kappa=kappa, source=source))

    for grid_size, degree in [(8, 2), (17, 3), (11, 4)]:
        runner.configure(
            base_config(tmp_path, Grid_size=grid_size, Polynomial_degree=degree)
        )
        runner.run(3.0)
        u = np.asarray(runner.getSolution(0, xs))
        want = np.array([exact(x) for x in xs])
        assert np.allclose(u, want, atol=2e-3), (
            f"grid={grid_size} k={degree}: got {u}, want {want}"
        )


def test_configure_requires_a_transport_system():
    # Constructing with None must not segfault; configure has an explicit guard.
    with pytest.raises((TypeError, RuntimeError)):
        MaNTA.Runner(None).configure({})


def test_get_adjoint_gradients_without_an_adjoint_problem_raises(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path))
    runner.run(0.05)

    with pytest.raises(RuntimeError):
        runner.getAdjointGradients()


# ------------------------------------------------------------- getSolution --


def test_get_solution_without_points_uses_the_output_grid(tmp_path):
    """The no-argument overload samples at the solver's own quadrature nodes.

    A separate branch of PyRunner::getSolution, and the one an optimisation
    driver uses when it wants the full state rather than named probes.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path))
    runner.run(0.5)

    everywhere = np.asarray(runner.getSolution(0, None))

    # Grid_size cells at Polynomial_degree k give nCells * (k + 1) nodes.
    assert everywhere.shape == (8 * (2 + 1),)
    assert np.all(np.isfinite(everywhere))

    # Sampling the same nodes explicitly must give the same numbers. The nodes
    # are Chebyshev-Gauss, so read them from MaNTA rather than assuming.
    nodes = list(np.asarray(MaNTA.getNodes(0.0, 1.0, 8, 2)))
    at_nodes = np.asarray(runner.getSolution(0, nodes))
    assert np.allclose(everywhere, at_nodes)


@pytest.mark.parametrize("point", [-0.001, 1.001, 5.0])
def test_get_solution_rejects_points_outside_the_domain(tmp_path, point):
    """Off-grid evaluation would silently extrapolate the local polynomial."""
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path))
    runner.run(0.1)

    with pytest.raises(IndexError):
        runner.getSolution(0, [point])


def test_get_solution_accepts_the_exact_boundaries(tmp_path):
    """The bounds check is inclusive, so the endpoints themselves are valid."""
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path))
    runner.run(0.1)

    ends = np.asarray(runner.getSolution(0, [0.0, 1.0]))
    assert np.all(np.isfinite(ends))
    # HDG imposes Dirichlet data weakly on the trace, so u only meets it to
    # discretisation order -- hence a loose bound rather than an exact zero.
    assert np.all(np.abs(ends) < 1e-3), ends


# ------------------------------------------------------------ run / run_ss --


def test_run_before_configure_is_refused():
    runner = MaNTA.Runner(LinearDiffusion())
    with pytest.raises(RuntimeError, match="must be configured"):
        runner.run(1.0)


def test_run_ss_before_configure_is_refused():
    runner = MaNTA.Runner(LinearDiffusion())
    with pytest.raises(RuntimeError, match="must be configured"):
        runner.run_ss()


def test_run_after_run_ss_clears_the_steady_state_termination(tmp_path):
    """run_ss latches TerminateOnSteadyState; a later run() must undo it.

    Otherwise the second call would stop early on the steady-state criterion
    instead of integrating to the requested time -- silently returning a
    different answer than asked for. PyRunner::run warns and resets.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path, SteadyStateTolerance=1e-2))

    runner.run_ss()
    u_ss = np.asarray(runner.getSolution(0, [0.5]))

    # Now a plain run on the same Runner: it must complete rather than
    # terminating immediately on the still-satisfied steady-state criterion.
    runner.run(1.0)
    u_after = np.asarray(runner.getSolution(0, [0.5]))

    assert np.all(np.isfinite(u_after))
    # Already at steady state, so the value should barely move -- the point is
    # that the call returned a solution at all.
    assert u_after[0] == pytest.approx(u_ss[0], rel=5e-2)


# ------------------------------------------------------------- restarting --


def test_output_filename_keeps_only_the_basename(tmp_path):
    """Pins current behaviour: the directory part of OutputFilename is dropped.

    PyRunner passes OutputFilename to `setInputFile`, and Solver.cpp does
    `baseName = inputFilePath.stem()` -- so `/some/where/run1` writes `run1.nc`
    and `run1.restart.nc` into the *current* directory, not into
    `/some/where/`. That is reasonable for the standalone binary, where the
    argument is a config file and the output is meant to land beside you, but
    surprising for a parameter named OutputFilename: two drivers running in
    different directories with the same basename will overwrite each other.

    Recorded rather than changed, because the standalone solver's output naming
    depends on it.
    """
    name = "outputpath_probe"
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(
        base_config(tmp_path, OutputFilename=str(tmp_path / name), WriteOutput=True)
    )
    runner.run(0.05)

    assert not (tmp_path / f"{name}.nc").exists(), (
        "output appeared in the requested directory -- if this was fixed "
        "deliberately, update this test and the restart tests below"
    )
    assert os.path.exists(f"{name}.nc"), "output did not appear in the cwd either"

    for suffix in (".nc", ".dat", ".restart.nc"):
        _unlink(name + suffix)


def test_a_run_can_be_restarted_from_its_own_output():
    """Run to t1, restart, continue to t2; compare against a single run to t2.

    This is the contract StoreGridInfo / setRestartValues exist to support, and
    it is the only test of the restart branch of PyRunner::configure. It also
    depends on Grid::operator== surviving the netCDF round trip -- the defect
    that made a clustered grid compare unequal to itself was found this way.

    Output paths are cwd-relative (see the test above), so the files are named
    uniquely and cleaned up rather than placed under tmp_path.
    """
    ref_name, split_name = "restart_reference", "restart_split"
    try:
        reference_runner = MaNTA.Runner(LinearDiffusion())
        reference_runner.configure(
            _cwd_config(OutputFilename=ref_name, WriteOutput=True)
        )
        reference_runner.run(0.4)
        reference = np.asarray(reference_runner.getSolution(0, XS_RESTART))

        first = MaNTA.Runner(LinearDiffusion())
        first.configure(_cwd_config(OutputFilename=split_name, WriteOutput=True))
        first.run(0.2)

        second = MaNTA.Runner(LinearDiffusion())
        second.configure(
            _cwd_config(
                OutputFilename=split_name,
                WriteOutput=True,
                restart=True,
                RestartFile=split_name + ".restart.nc",
                tZero=0.2,
            )
        )
        second.run(0.4)
        restarted = np.asarray(second.getSolution(0, XS_RESTART))

        assert np.allclose(restarted, reference, rtol=1e-3, atol=1e-6), (
            f"continuous={reference}\nrestarted={restarted}"
        )
    finally:
        _cleanup(ref_name, split_name)


def test_a_restart_with_the_wrong_variable_count_is_rejected():
    """The DOF check in configure guards against restarting the wrong physics.

    Without it the restart data would be reinterpreted under a different
    layout: no error, just a nonsense initial state.
    """

    class TwoVariable(LinearDiffusion):
        def __init__(self):
            super().__init__(spec=MaNTA.numbered_spec(2))

    name = "restart_dofmismatch"
    try:
        first = MaNTA.Runner(LinearDiffusion())
        first.configure(_cwd_config(OutputFilename=name, WriteOutput=True))
        first.run(0.1)

        second = MaNTA.Runner(TwoVariable())
        with pytest.raises(ValueError, match="inconsistent with physics case"):
            second.configure(
                _cwd_config(
                    OutputFilename=name,
                    restart=True,
                    RestartFile=name + ".restart.nc",
                )
            )
    finally:
        _cleanup(name)


def test_a_missing_restart_file_is_reported_with_its_path(tmp_path):
    runner = MaNTA.Runner(LinearDiffusion())
    with pytest.raises(RuntimeError, match="Failed to open restart netCDF file"):
        runner.configure(
            base_config(
                tmp_path,
                restart=True,
                RestartFile=str(tmp_path / "absent.restart.nc"),
            )
        )


XS_RESTART = [0.125 * i for i in range(1, 8)]


def _cwd_config(**overrides):
    """base_config, but with output names left relative to the cwd."""
    cfg = {
        "Polynomial_degree": 2,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.05,
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup(*names):
    for name in names:
        for suffix in (".nc", ".dat", ".restart.nc", ".dydt.dat", ".res.dat"):
            _unlink(name + suffix)
