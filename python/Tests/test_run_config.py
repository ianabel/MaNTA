"""Tests for runManta's configuration handling, driven through MaNTA.run.

`runManta` (MaNTA.cpp) is the entry point of the standalone solver, and its
validation is the only thing between a mistyped config and either a crash or a
silently wrong run. It is also awkward to reach from C++ -- it takes a filename
and builds the whole solver -- but `MaNTA.run` exposes it directly to Python,
which makes every branch a two-line test.

`runManta` reports failures two different ways and both are part of the
contract:

  * `return 1` for "I could not start" -- missing config file, unknown
    TransportSystem. These surface in Python as a return value, not an
    exception, because the standalone binary uses them as its exit code.
  * `throw std::invalid_argument` for a malformed option, which pybind11
    translates to ValueError.
"""

import numpy as np
import pytest

import MaNTA

CASE_NAME = "UnitTestRunConfigCase"


class Diffusion(MaNTA.TransportSystem):
    def __init__(self, config, grid):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nScalars = 0
        self.nAux = 0
        self.isLowerDirichlet = True
        self.isUpperDirichlet = True

    def SigmaFn(self, i, state, x, t):
        return state["Derivative"][i]

    def Sources(self, i, state, x, t):
        return 1.0

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.ones(self.nVars)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


MaNTA.registerPhysicsCase(CASE_NAME, Diffusion)


BASE = {
    "TransportSystem": f'"{CASE_NAME}"',
    "Polynomial_degree": "2",
    "Grid_size": "6",
    "Lower_boundary": "0.0",
    "Upper_boundary": "1.0",
    "t_final": "0.1",
    "delta_t": "0.1",
    "OutputPoints": "11",
}


def write_config(tmp_path, name="cfg", drop=(), **overrides):
    """Build a [configuration] section from BASE with edits applied."""
    entries = dict(BASE)
    for key in drop:
        entries.pop(key)
    entries.update({k: str(v) for k, v in overrides.items()})

    body = "[configuration]\n" + "".join(f"{k} = {v}\n" for k, v in entries.items())
    path = tmp_path / f"{name}.conf"
    path.write_text(body)
    return str(path)


# ------------------------------------------------------------- happy paths --


def test_the_baseline_config_runs():
    """Anchor: everything below is a single deliberate deviation from this."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert MaNTA.run(write_config(pathlib.Path(d))) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        # TOML distinguishes 1 from 1.0. A config author writing an integer
        # means the number, and every float-valued option has an is_integer()
        # branch to accept it -- but Lower_boundary/Upper_boundary called
        # as_floating() inside that branch and threw toml::type_error. The
        # equivalent defect in getFloat/getFloatWithDefault was fixed earlier;
        # these two are open-coded in runManta and were missed.
        {"Lower_boundary": "0", "Upper_boundary": "1"},
        {"tau": "1"},
        {"t_final": "1", "delta_t": "1"},
        {"Relative_tolerance": "1"},
        {"MinStepSize": "1"},
    ],
    ids=["boundaries", "tau", "times", "rtol", "minstep"],
)
def test_integer_literals_are_accepted_for_float_options(tmp_path, overrides):
    assert MaNTA.run(write_config(tmp_path, **overrides)) == 0


@pytest.mark.parametrize(
    "atol",
    ["1.0e-4", "[1.0e-4]", "1"],
    ids=["scalar-float", "array", "scalar-int"],
)
def test_absolute_tolerance_accepts_scalar_and_array_forms(tmp_path, atol):
    """Absolute_tolerance is either one number or one per variable."""
    assert MaNTA.run(write_config(tmp_path, Absolute_tolerance=atol)) == 0


def test_a_wrong_length_absolute_tolerance_is_silently_ignored(tmp_path):
    """Pins current behaviour, which is a trap rather than a design.

    `getErrorWeights` reads

        double absTol = 1e-8;
        if (atol.size() == 1)          absTol = atol[0];
        else if (atol.size() == nVars) absTol = atol[v];

    -- so an Absolute_tolerance whose length is neither 1 nor nVars is
    discarded and replaced by a hard-coded 1e-8, with nothing said. `runManta`
    accepts the array happily (it only checks that the key appears once), so a
    user who writes two tolerances for a one-variable problem gets a far
    tighter solve than they asked for and no indication why.

    Demonstrated by making the consequence visible: at MinStepSize = 1e-7 the
    scalar form runs and the two-element form drives IDA into the step floor.
    Both configs are otherwise identical.

    If this ever starts failing because the mismatch is honoured or rejected,
    that is an improvement -- update the test, do not loosen it.
    """
    ok = write_config(tmp_path, "loose", Absolute_tolerance="1.0e-1", MinStepSize="1e-7")
    assert MaNTA.run(ok) == 0

    mismatched = write_config(
        tmp_path, "mismatched", Absolute_tolerance="[1.0e-1, 1.0e-1]", MinStepSize="1e-7"
    )
    with pytest.raises(RuntimeError, match="IDASolve could not complete"):
        MaNTA.run(mismatched)


def test_high_grid_boundary_is_accepted_with_enough_cells(tmp_path):
    cfg = write_config(
        tmp_path,
        Grid_size=12,
        High_Grid_Boundary="true",
        Lower_Boundary_Fraction="0.25",
        Upper_Boundary_Fraction="0.25",
    )
    assert MaNTA.run(cfg) == 0


def test_the_defaults_are_enough(tmp_path):
    """Only the un-defaulted options are genuinely required."""
    cfg = write_config(tmp_path, drop=("OutputPoints",))
    assert MaNTA.run(cfg) == 0


# ------------------------------------------------------ "cannot start" -> 1 --


def test_a_missing_config_file_returns_one(tmp_path):
    assert MaNTA.run(str(tmp_path / "nope.conf")) == 1


def test_an_unknown_transport_system_returns_one(tmp_path):
    assert MaNTA.run(write_config(tmp_path, TransportSystem='"NotARegisteredCase"')) == 1


def test_a_missing_restart_file_returns_one(tmp_path):
    cfg = write_config(
        tmp_path, restart="true", RestartFile='"no-such-file.restart.nc"'
    )
    assert MaNTA.run(cfg) == 1


# -------------------------------------------------- malformed -> ValueError --


def test_a_non_integer_polynomial_degree_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Polynomial_degree must be specified"):
        MaNTA.run(write_config(tmp_path, Polynomial_degree="2.5"))


def test_a_non_integer_grid_size_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Grid_size must be specified"):
        MaNTA.run(write_config(tmp_path, Grid_size="6.0"))


def test_a_small_grid_with_dense_boundaries_is_rejected(tmp_path):
    """The clustered grid divides nCells into three, so it needs at least four."""
    with pytest.raises(ValueError, match="Grid size must exceed 4 cells"):
        MaNTA.run(write_config(tmp_path, Grid_size=3, High_Grid_Boundary="true"))


@pytest.mark.parametrize("key", ["Lower_boundary", "Upper_boundary"])
def test_a_non_numeric_boundary_is_rejected(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key} specified incorrrectly"):
        MaNTA.run(write_config(tmp_path, **{key: '"not a number"'}))


def test_a_missing_transport_system_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="TransportSystem needs to specified"):
        MaNTA.run(write_config(tmp_path, drop=("TransportSystem",)))


@pytest.mark.parametrize("key", ["delta_t", "t_final"])
def test_a_missing_required_float_is_named(tmp_path, key):
    """getFloat has no default to fall back on, so it says which key is absent."""
    with pytest.raises(ValueError, match=f"{key} was not specified"):
        MaNTA.run(write_config(tmp_path, drop=(key,)))


@pytest.mark.parametrize("key", ["tau", "Relative_tolerance", "MinStepSize"])
def test_a_non_numeric_optional_float_is_named(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key} specified incorrrectly"):
        MaNTA.run(write_config(tmp_path, **{key: '"text"'}))


def test_a_non_integer_output_point_count_is_rejected(tmp_path):
    """getIntWithDefault takes integers only -- 301.0 is not one."""
    with pytest.raises(ValueError, match="OutputPoints specified incorrrectly"):
        MaNTA.run(write_config(tmp_path, OutputPoints="301.0"))


def test_a_missing_configuration_section_is_rejected(tmp_path):
    """Everything is read out of [configuration]; without it there is nothing."""
    path = tmp_path / "no-section.conf"
    path.write_text('TransportSystem = "whatever"\n')
    with pytest.raises(Exception):
        MaNTA.run(str(path))
