"""FFIRunner's steady-slice outcome, eagerly and under a trace.

Only an XLA_FFI build has the ops, so every test here skips without one.

A slice op returns the outcome as an int32. Eagerly FFIRunner turns it into a
manta.SteadyOutcome, which is what manta.SteadySolve branches on. Under a trace
-- jit, a custom_jvp -- the solve has not run, so the outcome is a traced scalar
and int() of it raises; it comes back as that scalar, and because SteadyOutcome
is an IntEnum it is tested against the members directly, which is how
python-physics/stellarator drives its solve.
"""

import pytest

import manta as MaNTA

from test_runner import LinearDiffusion, base_config

if not hasattr(MaNTA._manta, "runner_ffi_ops"):
    pytest.skip("FFIRunner needs an XLA_FFI build", allow_module_level=True)

jax = pytest.importorskip("jax")
jnp = jax.numpy
from manta.jax import FFIRunner  # noqa: E402


def steady_config(tmp_path):
    return base_config(tmp_path, SteadyStateSolver="Newton", SteadyStateTolerance=1e-10)


def make_runner(tmp_path):
    runner = FFIRunner(LinearDiffusion(), [0.5], 1, 0)
    runner.configure(steady_config(tmp_path))
    return runner


def test_an_eager_slice_returns_a_steady_outcome(tmp_path):
    runner = make_runner(tmp_path)
    outcome = runner.start_steady()
    assert outcome is MaNTA.SteadyOutcome.Converged
    runner.finish_steady()


def test_steady_solve_drives_an_ffi_runner(tmp_path):
    runner = make_runner(tmp_path)
    with MaNTA.SteadySolve(runner, estimate=False) as solve:
        outcomes = [outcome for outcome, _ in solve]
    assert outcomes == [MaNTA.SteadyOutcome.Converged]


def test_a_traced_slice_is_tested_against_the_members(tmp_path):
    # The stellarator driver's shape: the slice and the branch on its outcome
    # inside one traced function, the finish or the abandonment chosen by
    # lax.cond. int() would raise here; the comparison traces.
    runner = make_runner(tmp_path)

    def solve():
        outcome = runner.start_steady()
        assert isinstance(outcome, jax.core.Tracer)

        def finish():
            runner.finish_steady()
            return True

        def abandon():
            runner.abandon_steady()
            return False

        return jax.lax.cond(jnp.equal(outcome, MaNTA.SteadyOutcome.Converged), finish, abandon)

    with jax.default_device(jax.devices("cpu")[0]):
        assert bool(jax.jit(solve)())


def test_steady_outcome_is_an_int_enum():
    import enum

    assert issubclass(MaNTA.SteadyOutcome, enum.IntEnum)
    assert MaNTA.SteadyOutcome.Converged == 1
    assert bool(jax.jit(lambda x: jnp.equal(x, MaNTA.SteadyOutcome.OutOfSteps))(jnp.int32(2)))
