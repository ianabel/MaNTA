"""A steady solve driven in slices (PyRunner::start_steady / continue_steady).

The point of the feature is that a slice is free: a solve stopped by its
MaxContinuationSteps budget keeps the state it reached *and* the pseudo-time
step SER climbed to, so resuming costs no extra continuation steps. A second
solveSteadyState() would keep neither -- it re-enters at
PseudoTransientInitialStep -- and re-climbing the ramp is the whole solve rather
than a margin on it.

The C++ half is Tests/UnitTests/SolverLifecycleTests.cpp; what is only reachable
here is the lifecycle. A slice loop owns live SUNDIALS objects that nothing else
frees -- ~SystemSolver does not call destroySundials() -- so the teardown paths
carry as much weight as the arithmetic.
"""

import numpy as np
import pytest

import manta as MaNTA

from test_adjoint import ParametricDiffusion, adjoint_config, KAPPA0, SOURCE0
from test_degree_adaptation import SineSource, base_config

POINTS = [0.15, 0.35, 0.5, 0.65, 0.85]
SLICE = 3


def slice_config(tmp_path, **overrides):
    cfg = base_config(tmp_path)
    cfg.update(
        {
            "Polynomial_degree": 2,
            "Grid_size": 6,
            "SteadyStateSolver": "PseudoTransient",
            "SteadyStateTolerance": 1.0e-10,
            "Absolute_tolerance": 1.0e-10,
            "MinStepSize": 1.0e-12,
            # Small enough that the SER ramp is several slices long, which is
            # what makes re-climbing it distinguishable from resuming.
            "PseudoTransientInitialStep": 1.0e-4,
            "MaxContinuationSteps": SLICE,
        }
    )
    cfg.update(overrides)
    return cfg


def drive(runner, estimate=False):
    """Slice to convergence, returning (outcome, total steps, slices)."""
    outcome = runner.start_steady(estimate)
    total, slices = runner.steadyStats()["steps"], 1
    while outcome == MaNTA.SteadyOutcome.OutOfSteps and slices < 40:
        outcome = runner.continue_steady(estimate)
        total += runner.steadyStats()["steps"]
        slices += 1
    return outcome, total, slices


def test_slicing_costs_no_extra_continuation_steps(tmp_path):
    """The measurement the feature exists for, against an uninterrupted solve.

    Same answer, same number of continuation steps. Not "about the same": a
    resumed slice picks up mid-ramp, so there is no partial step to redo.
    """
    whole = MaNTA.Runner(SineSource())
    whole.configure(slice_config(tmp_path / "whole", MaxContinuationSteps=200))
    whole_outcome, whole_steps, whole_slices = drive(whole)
    whole_u = whole.getSolution(0, POINTS)
    whole.finish_steady()

    assert whole_outcome == MaNTA.SteadyOutcome.Converged
    assert whole_slices == 1, "the reference solve was itself sliced"
    assert whole_steps > 2 * SLICE, (
        f"the reference converged in {whole_steps} steps, too few for slices "
        f"of {SLICE} to be a meaningful interruption"
    )

    sliced = MaNTA.Runner(SineSource())
    sliced.configure(slice_config(tmp_path / "sliced"))
    outcome, steps, slices = drive(sliced)
    sliced_u = sliced.getSolution(0, POINTS)
    sliced.finish_steady()

    assert outcome == MaNTA.SteadyOutcome.Converged
    assert slices > 2, f"only {slices} slices; the budget is not biting"
    assert steps == whole_steps, (
        f"slicing cost {steps} continuation steps against {whole_steps} "
        f"uninterrupted -- the SER ramp is being re-climbed"
    )
    np.testing.assert_allclose(sliced_u, whole_u, rtol=0, atol=1e-12)


def test_the_pseudo_transient_step_climbs_across_slices(tmp_path):
    """dt is what a resumed slice picks up, so it has to be visible and rising.

    Distinguishes resuming from restarting directly, rather than inferring it
    from the step count: a restarted slice would report the same dt every time.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    outcome = runner.start_steady(False)
    steps = [runner.steadyStats()["pseudo_transient_step"]]
    while outcome == MaNTA.SteadyOutcome.OutOfSteps and len(steps) < 40:
        outcome = runner.continue_steady(False)
        steps.append(runner.steadyStats()["pseudo_transient_step"])
    runner.finish_steady()

    assert len(steps) > 2
    # Strictly rising while the budget is what stops each slice. The last pair
    # can be equal: the converging slice takes its step at the dt it inherited
    # and finishes rather than growing it again.
    assert all(b > a for a, b in zip(steps[:-1], steps[1:-1])), (
        f"dt did not rise across the interrupted slices: {steps}"
    )
    assert steps[-1] >= steps[-2], f"dt fell on the final slice: {steps}" 
    assert steps[-1] > 1e3 * steps[0], (
        f"dt only went {steps[0]:.3e} -> {steps[-1]:.3e}; that is not a ramp"
    )


def test_the_residual_falls_and_is_reported_per_slice(tmp_path):
    """||F|| is the only measure of progress a slicing driver has.

    It is the *steady* residual, not the damped one KINSol converged -- which
    any small enough dt makes small, and so would report progress that is not
    there.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    outcome = runner.start_steady(False)
    norms = [runner.steadyStats()["residual_norm"]]
    while outcome == MaNTA.SteadyOutcome.OutOfSteps and len(norms) < 40:
        outcome = runner.continue_steady(False)
        norms.append(runner.steadyStats()["residual_norm"])
    runner.finish_steady()

    assert all(np.isfinite(n) for n in norms), norms
    assert all(b <= a for a, b in zip(norms, norms[1:])), (
        f"||F|| did not fall monotonically across slices: {norms}"
    )
    assert norms[-1] < 1e-10


def test_the_state_between_slices_is_the_state_reached(tmp_path):
    """getSolution() reads yJac, which a slice has to refresh.

    Without that a driver looking between slices is handed the *initial
    condition* -- silently, because yJac is always a valid state, just not this
    one. Checked by requiring the mid-loop answer to be nearer the exact steady
    state than the initial condition is.
    """
    exact = SineSource.exact(np.array(POINTS))

    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))
    runner.start_steady(False)
    initial = runner.getSolution(0, POINTS)

    outcome = MaNTA.SteadyOutcome.OutOfSteps
    while outcome == MaNTA.SteadyOutcome.OutOfSteps:
        outcome = runner.continue_steady(False)
    mid = runner.getSolution(0, POINTS)
    runner.finish_steady()
    final = runner.getSolution(0, POINTS)

    assert np.max(np.abs(mid - exact)) < np.max(np.abs(initial - exact))
    np.testing.assert_allclose(mid, final, rtol=0, atol=0)


def test_out_of_steps_is_returned_and_a_failure_is_raised(tmp_path):
    """The two exits a driver has to tell apart without reading a message."""
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))
    assert runner.start_steady(False) == MaNTA.SteadyOutcome.OutOfSteps
    runner.abandon_steady()

    # A tolerance nothing reaches still ends in OutOfSteps, not a raise: the
    # budget is what stops it.
    stubborn = MaNTA.Runner(SineSource())
    stubborn.configure(slice_config(tmp_path / "stubborn", SteadyStateTolerance=1e-30))
    outcome = stubborn.start_steady(False)
    for _ in range(6):
        if outcome != MaNTA.SteadyOutcome.OutOfSteps:
            break
        outcome = stubborn.continue_steady(False)
    assert outcome == MaNTA.SteadyOutcome.OutOfSteps
    stubborn.abandon_steady()


class ExplodesPartWayThrough(SineSource):
    """A physics case that raises once the solve is under way.

    The realistic version of a failure that is not one of solveSteadyState's own
    exits: the exception comes from inside the residual, so it bypasses finish()
    entirely and no outcome is recorded for it.
    """

    def __init__(self, after):
        super().__init__()
        self._after = after
        self.calls = 0

    def SigmaFn(self, i, state, x, t):
        self.calls += 1
        if self.calls > self._after:
            raise ArithmeticError("physics case gave up")
        return SineSource.SigmaFn(self, i, state, x, t)


def test_a_failure_inside_a_slice_raises_rather_than_looping(tmp_path):
    """An exception that bypasses finish() must not be read as OutOfSteps.

    finish() is the only thing that sets the outcome, so without clearing it at
    the top of each solve a raising physics case leaves the *previous* slice's
    verdict standing. A driver classifying the exception by asking what the
    outcome was would be told OutOfSteps -- and a loop written on OutOfSteps
    would swallow the exception and spin forever. The timeout is the assertion
    that matters here as much as the exception type.
    """
    runner = MaNTA.Runner(ExplodesPartWayThrough(400))
    runner.configure(slice_config(tmp_path))

    outcome = runner.start_steady(False)
    with pytest.raises(RuntimeError, match="SigmaFn"):
        for _ in range(40):
            if outcome != MaNTA.SteadyOutcome.OutOfSteps:
                break
            outcome = runner.continue_steady(False)
        else:
            pytest.fail("the slice loop never terminated")

    # Torn down on the way out, so nothing is left holding SUNDIALS objects.
    with pytest.raises(RuntimeError, match="no sliced solve running"):
        runner.continue_steady(False)


def test_the_slice_methods_refuse_to_be_used_out_of_order(tmp_path):
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    with pytest.raises(RuntimeError, match="no sliced solve running"):
        runner.continue_steady(False)
    with pytest.raises(RuntimeError, match="no sliced solve running"):
        runner.finish_steady()

    runner.start_steady(False)
    with pytest.raises(RuntimeError, match="already running"):
        runner.start_steady(False)
    runner.abandon_steady()


def test_degree_adaptation_is_refused(tmp_path):
    """Adapting the degree replaces the solver; a slice loop holds the old one."""
    runner = MaNTA.Runner(SineSource())
    runner.configure(
        slice_config(
            tmp_path,
            DegreeAdaptation=True,
            DegreeTolerance=1e-9,
            MaxPolynomialDegree=6,
        )
    )
    with pytest.raises(RuntimeError, match="DegreeAdaptation cannot be combined"):
        runner.start_steady(False)


def test_reconfiguring_abandons_a_live_slice_loop(tmp_path):
    """configure() is an implicit abandon, and has to free what it drops.

    Nothing else frees a live loop's SUNDIALS objects, so the check that
    matters is that the runner still works afterwards -- a leak shows up as the
    next run misbehaving rather than as an error here.
    """
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))
    assert runner.start_steady(False) == MaNTA.SteadyOutcome.OutOfSteps

    runner.configure(slice_config(tmp_path / "again", MaxContinuationSteps=200))
    runner.run_ss()

    exact = SineSource.exact(np.array(POINTS))
    assert np.max(np.abs(runner.getSolution(0, POINTS) - exact)) < 1e-3

    # And the loop really is gone: continue_steady has nothing to resume.
    with pytest.raises(RuntimeError, match="no sliced solve running"):
        runner.continue_steady(False)


def test_the_context_manager_finishes_a_clean_exit(tmp_path):
    exact = SineSource.exact(np.array(POINTS))
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    seen = []
    with MaNTA.SteadySolve(runner, estimate=False) as solve:
        for outcome, stats in solve:
            seen.append((outcome, stats["residual_norm"]))

    assert len(seen) > 2
    assert seen[-1][0] == MaNTA.SteadyOutcome.Converged
    assert np.max(np.abs(runner.getSolution(0, POINTS) - exact)) < 1e-3


def test_the_context_manager_stops_early_on_request(tmp_path):
    """stop() ends the loop and still writes the partial answer."""
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    slices = 0
    with MaNTA.SteadySolve(runner, estimate=False) as solve:
        for outcome, stats in solve:
            slices += 1
            if stats["residual_norm"] < 1e-1:
                solve.stop()

    assert 1 < slices < 6, f"stopped after {slices} slices"
    exact = SineSource.exact(np.array(POINTS))
    # Stopped short, so it is *not* converged -- but it is a real partial state.
    err = np.max(np.abs(runner.getSolution(0, POINTS) - exact))
    assert 1e-4 < err < 1.0, err


def test_the_context_manager_unwinds_on_a_driver_exception(tmp_path):
    """A driver's own exception must not leave the solve half-open."""
    runner = MaNTA.Runner(SineSource())
    runner.configure(slice_config(tmp_path))

    with pytest.raises(ValueError, match="driver gave up"):
        with MaNTA.SteadySolve(runner, estimate=False) as solve:
            for _outcome, _stats in solve:
                raise ValueError("driver gave up")

    # Torn down, not merely stopped: nothing is left to continue or to finish.
    with pytest.raises(RuntimeError, match="no sliced solve running"):
        runner.continue_steady(False)

    # ...and the runner is reusable.
    runner.configure(slice_config(tmp_path / "after", MaxContinuationSteps=200))
    runner.run_ss()
    exact = SineSource.exact(np.array(POINTS))
    assert np.max(np.abs(runner.getSolution(0, POINTS) - exact)) < 1e-3


def test_suppressing_the_estimate_saves_the_work_it_costs(tmp_path):
    """`estimate` is a cost knob, and the cost is charged per slice.

    One residual, one Jacobian build and one solve each time. A driver reading
    the estimate only at the end should not pay for it on every slice, and
    without an AdjointProblem there is nothing to estimate and nothing to pay.
    """
    def total_cost(estimate):
        runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
        runner.configure(
            adjoint_config(
                tmp_path / f"est{estimate}",
                SteadyStateSolver="PseudoTransient",
                SteadyStateTolerance=1e-11,
                PseudoTransientInitialStep=1e-4,
                MaxContinuationSteps=SLICE,
            )
        )
        outcome = runner.start_steady(estimate)
        stats = runner.steadyStats()
        builds, slices = stats["jacobian_builds"], 1
        while outcome == MaNTA.SteadyOutcome.OutOfSteps and slices < 40:
            outcome = runner.continue_steady(estimate)
            builds += runner.steadyStats()["jacobian_builds"]
            slices += 1
        estimate_at_end = runner.objectiveEstimate()
        runner.finish_steady()
        return builds, slices, estimate_at_end

    on_builds, on_slices, on_estimate = total_cost(True)
    off_builds, off_slices, off_estimate = total_cost(False)

    assert on_slices == off_slices, "the flag changed the solve, not just the cost"
    assert on_builds > off_builds, (
        f"estimating cost {on_builds} Jacobian builds against {off_builds} "
        f"without -- the flag is not reaching estimateObjective()"
    )
    # One build per slice is exactly what it should cost.
    assert on_builds - off_builds == on_slices

    # And it is the estimate, not just work: armed, the last slice leaves one.
    assert on_estimate, "estimate=True left no objective estimate"
    assert not off_estimate, "estimate=False left one anyway"
