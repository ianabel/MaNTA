"""A field from Python: a Runner configured with a field or equilibrium model.

The geometry source is a registered model named in the dict --
``FieldModel`` or ``EquilibriumModel``, with its own table nested under its
name as a C++ case's is -- or an ``EquilibriumModel`` object handed over with
``Runner.set_equilibrium_model``. Either way it goes through
``geometrySourceFromConfig`` / ``geometrySourceFor`` (GeometrySource.hpp), the
functions runManta uses, so ``FieldCoupling`` resolves the same way on both
surfaces.

The load-bearing test is the first: one run description, at every tier, run
through a config file and through a dict, gives netCDF output identical bit for
bit -- the transport variable and every field slot. Exact equality on purpose,
as in test_cpp_cases.py: a tolerance is what would let a differently resolved
coupling through.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

import manta as MaNTA

from test_cpp_cases import remove_output, write_config


# The regression theta pinch, smaller and shorter: B on axis still falls well
# below the wall's, so the coupling is doing something.
BASE = {
    "PolynomialDegree": 3,
    "GridSize": 6,
    "LowerBoundary": 0.0,
    "UpperBoundary": 1.0,
    "t_final": 0.3,
    "delta_t": 0.1,
    "Relative_tolerance": 1.0e-6,
    "Absolute_tolerance": 1.0e-8,
    "MinStepSize": 1.0e-12,
    "ThetaPinchDiffusion": {"Temperature": 0.3, "D0": 0.5, "Gamma": 2.0},
}

FIELD_MODEL = {
    "FieldModel": "ThetaPinchEquilibrium",
    "ThetaPinchEquilibrium": {"PressureAux": 0},
}

NATIVE = {
    "EquilibriumModel": "ThetaPinchPressureBalance",
    "ThetaPinchPressureBalance": {"PressureAux": 0},
}


def tier(source, coupling):
    return dict(BASE, **source, FieldCoupling=coupling)


def read_groups(stem):
    """Every variable of the transport and field groups, by group/name."""
    out = {}
    with Dataset(stem + ".nc") as nc:
        for group in ("N", "ThetaPinch"):
            for name, v in nc.groups[group].variables.items():
                out[f"{group}/{name}"] = np.array(v[:])
    return out


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(tier(FIELD_MODEL, "coupled"), id="field-coupled"),
        pytest.param(tier(FIELD_MODEL, "reduced"), id="field-reduced"),
        pytest.param(tier(FIELD_MODEL, "consistent"), id="field-consistent"),
        pytest.param(tier(NATIVE, "reduced"), id="native-reduced"),
        pytest.param(tier(NATIVE, "consistent"), id="native-consistent"),
    ],
)
def test_a_dict_couples_a_field_exactly_as_a_config_file_does(config):
    case = "ThetaPinchDiffusion"
    from_toml, from_dict = "field_from_toml", "field_from_dict"
    try:
        write_config(from_toml + ".conf", case, dict(config, OutputFilename=from_toml))
        assert MaNTA.run(from_toml + ".conf") == 0

        runner = MaNTA.Runner(case)
        runner.configure(dict(config, OutputFilename=from_dict))
        runner.run()

        a, b = read_groups(from_toml), read_groups(from_dict)
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(a[key], b[key]), (
                f"{key}: the dict coupled this field differently from the config "
                f"file, worst difference {np.max(np.abs(a[key] - b[key])):.3e}"
            )
        # The field is far from vacuum, so the comparison above is of a
        # coupling and not of two copies of B = 1.
        assert np.min(b["ThetaPinch/B"]) < 0.97
    finally:
        remove_output(from_toml)
        remove_output(from_dict)


def test_a_coupled_restart_resumes_from_python():
    """The field unknowns ride in the restart file, and a Runner reads them.

    Coupled, so nField > 0: the restart file's Y carries psi after the scalars,
    which a Runner used to refuse outright.
    """
    case = "ThetaPinchDiffusion"
    config = tier(FIELD_MODEL, "coupled")
    try:
        whole = MaNTA.Runner(case)
        whole.configure(dict(config, OutputFilename="field_whole", t_final=0.6))
        whole.run()

        first = MaNTA.Runner(case)
        first.configure(dict(config, OutputFilename="field_half", t_final=0.3))
        first.run()

        rest = MaNTA.Runner(case)
        rest.configure(
            dict(
                config,
                OutputFilename="field_rest",
                restart=True,
                RestartFile="field_half.restart.nc",
                t_initial=0.3,
                t_final=0.6,
            )
        )
        rest.run()

        x = np.linspace(0.05, 0.95, 10)
        gap = np.max(np.abs(rest.getSolution(0, x) - whole.getSolution(0, x)))
        assert gap < 1e-5, f"restarted run is {gap:.3e} from the uninterrupted one"
    finally:
        for stem in ("field_whole", "field_half", "field_rest"):
            remove_output(stem)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"FieldCoupling": "reduced"}, "without a FieldModel or EquilibriumModel"),
        (dict(FIELD_MODEL, **NATIVE), "FieldModel"),
        (dict(NATIVE, FieldCoupling="coupled"), "coupled"),
        (dict(FIELD_MODEL, DegreeAdaptation=True, MaxPolynomialDegree=5), "adaptive drivers"),
        ({"FieldModel": "NoSuchModel"}, "NoSuchModel"),
    ],
)
def test_a_field_configuration_is_refused_by_name(extra, match):
    runner = MaNTA.Runner("ThetaPinchDiffusion")
    with pytest.raises(RuntimeError, match=match):
        runner.configure(dict(BASE, OutputFilename="field_refused", **extra))


# ------------------------------------- an equilibrium model written in Python --


from theta_pinch_balance import PythonPressureBalance  # noqa: E402


def native(coupling, **table):
    return dict(
        BASE,
        EquilibriumModel="ThetaPinchPressureBalance",
        ThetaPinchPressureBalance=dict(PressureAux=0, **table),
        FieldCoupling=coupling,
    )


def run_native(stem, coupling, **table):
    runner = MaNTA.Runner("ThetaPinchDiffusion")
    runner.configure(dict(native(coupling, **table), OutputFilename=stem))
    runner.run()
    return runner


def run_python(stem, coupling):
    runner = MaNTA.Runner("ThetaPinchDiffusion")
    model = PythonPressureBalance(np.linspace(0.0, 1.0, BASE["GridSize"] + 1))
    runner.set_equilibrium_model(model)
    runner.configure(dict(BASE, FieldCoupling=coupling, OutputFilename=stem))
    runner.run()
    return runner, model


# The Python port and the C++ model compute the same balance by the same rules,
# summed in a different order, so their equilibria agree to round-off; the
# integrator then sees round-off-different residuals. Measured worst relative
# gap over every output variable: 5.3e-11 at tier 0, 6.8e-11 at tier 1.
PORT_GAP = 1e-9


@pytest.mark.parametrize(
    "coupling,table",
    [
        pytest.param("consistent", {}, id="tier0"),
        # Tier 1 through the base class's differenced response on both sides:
        # the port declares no response of its own.
        pytest.param("reduced", {"DifferencedResponse": True}, id="tier1-differenced"),
    ],
)
def test_a_python_equilibrium_model_runs_the_native_models_balance(coupling, table):
    try:
        run_native("balance_native", coupling, **table)
        _, model = run_python("balance_python", coupling)
        assert model.solves > 0

        a, b = read_groups("balance_native"), read_groups("balance_python")
        assert a.keys() == b.keys()
        for key in a:
            scale = max(np.max(np.abs(a[key])), 1e-30)
            gap = np.max(np.abs(a[key] - b[key])) / scale
            assert gap < PORT_GAP, f"{key}: Python port is {gap:.3e} from the C++ model"
        assert np.min(b["ThetaPinch/B"]) < 0.97
    finally:
        remove_output("balance_native")
        remove_output("balance_python")


class RecordingBalance(PythonPressureBalance):
    """The port, registered by name, recording what reached it."""

    built = []

    def __init__(self, config, grid):
        super().__init__(grid.cellBoundaries(), pressure_aux=int(config["PressureAux"]))
        RecordingBalance.built.append(self)


MaNTA.registerEquilibriumModel("PythonPressureBalanceForTest", RecordingBalance)


def test_a_registered_python_equilibrium_model_is_named_like_a_native_one():
    """registerEquilibriumModel, then EquilibriumModel = name in the dict.

    The table arrives as the model's own, nested under its registered name as a
    C++ model's does. Also how manta.cli reaches a Python model from a config
    file.
    """
    RecordingBalance.built.clear()
    try:
        runner = MaNTA.Runner("ThetaPinchDiffusion")
        runner.configure(
            dict(
                BASE,
                EquilibriumModel="PythonPressureBalanceForTest",
                PythonPressureBalanceForTest={"PressureAux": 0},
                FieldCoupling="consistent",
                OutputFilename="balance_registered",
            )
        )
        runner.run()
        assert len(RecordingBalance.built) == 1
        assert RecordingBalance.built[0].solves > 0
    finally:
        remove_output("balance_registered")


def test_an_equilibrium_model_object_and_a_named_one_are_refused_together():
    runner = MaNTA.Runner("ThetaPinchDiffusion")
    runner.set_equilibrium_model(PythonPressureBalance(np.linspace(0.0, 1.0, 7)))
    with pytest.raises(RuntimeError, match="one source"):
        runner.configure(dict(native("consistent"), OutputFilename="balance_refused"))


def test_a_failing_python_solve_is_a_recoverable_step_failure():
    """An exception from Solve inside a residual cuts the step, as a hook's does.

    Raised on a few solves after the first, so the run has a step to cut; the
    run completes, and the failures were seen.
    """

    class Flaky(PythonPressureBalance):
        def __init__(self, boundaries):
            super().__init__(boundaries)
            self.failures = 0

        def Solve(self, states, points, weights, t, warm):
            if self.solves in (20, 21):
                self.solves += 1
                self.failures += 1
                raise RuntimeError("no equilibrium today")
            return super().Solve(states, points, weights, t, warm)

    model = Flaky(np.linspace(0.0, 1.0, BASE["GridSize"] + 1))
    try:
        runner = MaNTA.Runner("ThetaPinchDiffusion")
        runner.set_equilibrium_model(model)
        runner.configure(dict(BASE, FieldCoupling="consistent", OutputFilename="balance_flaky"))
        runner.run()
        assert model.failures == 2
    finally:
        remove_output("balance_flaky")


# ------------------------------------------ a Python case reading the geometry --


from theta_pinch_case import PythonThetaPinch  # noqa: E402

# The Python case writes by hand what the C++ case's autodiff derives, so the
# two agree to round-off per evaluation; the gap below is the integrator's
# answer to that. Measured worst relative gap: 2.0e-11 coupled, 2.4e-12
# reduced, 2.1e-11 at tier 0 on the native model.
CASE_GAP = 1e-9


@pytest.mark.parametrize(
    "source,coupling",
    [
        pytest.param(FIELD_MODEL, "coupled", id="field-coupled"),
        pytest.param(FIELD_MODEL, "reduced", id="field-reduced"),
        pytest.param(NATIVE, "consistent", id="native-consistent"),
    ],
)
def test_a_python_case_reads_the_geometry_as_the_cpp_case_does(source, coupling):
    """The same theta pinch, the transport case written in C++ and in Python.

    Coupled at tier 2, the Python case's dSigmaFn_dGeometry, dSources_dGeometry
    and dAuxG_dGeometry build the coupled Jacobian's A1 blocks; reduced, A_g.
    """
    config = tier(source, coupling)
    try:
        cpp = MaNTA.Runner("ThetaPinchDiffusion")
        cpp.configure(dict(config, OutputFilename="case_cpp"))
        cpp.run()

        case = PythonThetaPinch()
        py_runner = MaNTA.Runner(case)
        py_runner.configure(dict(config, OutputFilename="case_python"))
        py_runner.run()
        assert case.slots_seen == 4

        a, b = read_groups("case_cpp"), read_groups("case_python")
        for key in a:
            scale = max(np.max(np.abs(a[key])), 1e-30)
            gap = np.max(np.abs(a[key] - b[key])) / scale
            assert gap < CASE_GAP, f"{key}: Python case is {gap:.3e} from the C++ case"
    finally:
        remove_output("case_cpp")
        remove_output("case_python")


def test_a_run_can_be_python_on_both_sides_of_the_field():
    """A Python transport case and a Python equilibrium model, together."""
    try:
        cpp = MaNTA.Runner("ThetaPinchDiffusion")
        cpp.configure(dict(tier(NATIVE, "reduced"), OutputFilename="both_cpp"))
        cpp.run()

        runner = MaNTA.Runner(PythonThetaPinch())
        model = PythonPressureBalance(np.linspace(0.0, 1.0, BASE["GridSize"] + 1))
        runner.set_equilibrium_model(model)
        runner.configure(dict(BASE, FieldCoupling="reduced", OutputFilename="both_python"))
        runner.run()

        a, b = read_groups("both_cpp"), read_groups("both_python")
        # Tier 1 with the native model's exact response against the port's
        # differenced one. The response only steers Newton, so the answers
        # still agree to the solver's tolerances: measured 1.6e-10.
        for key in a:
            scale = max(np.max(np.abs(a[key])), 1e-30)
            gap = np.max(np.abs(a[key] - b[key])) / scale
            assert gap < 1e-8, f"{key}: {gap:.3e}"
    finally:
        remove_output("both_cpp")
        remove_output("both_python")


def test_a_geometry_slot_name_the_model_does_not_have_raises():
    class Misnamed(PythonThetaPinch):
        def metric(self, state):
            return state.geom["NotASlot"], 0.0, 1.0, 1.0

    runner = MaNTA.Runner(Misnamed())
    runner.configure(dict(tier(NATIVE, "consistent"), OutputFilename="misnamed"))
    try:
        with pytest.raises(Exception, match="NotASlot"):
            runner.run()
    finally:
        remove_output("misnamed")


@pytest.fixture
def _float64():
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "source,coupling",
    [
        pytest.param(FIELD_MODEL, "coupled", id="field-coupled"),
        pytest.param(NATIVE, "reduced", id="native-reduced"),
    ],
)
def test_a_jax_case_differentiates_through_the_geometry(_float64, source, coupling):
    """State.Geometry, and dSigmaFn_dGeometry and friends by jax.grad.

    The geometry derivatives come out of the same gradient as the others, so a
    coupled run on them is the C++ case's run to the integrator's answer to
    round-off.
    """
    pytest.importorskip("equinox")
    from theta_pinch_case import jax_theta_pinch

    config = tier(source, coupling)
    try:
        cpp = MaNTA.Runner("ThetaPinchDiffusion")
        cpp.configure(dict(config, OutputFilename="jax_cpp"))
        cpp.run()

        runner = MaNTA.Runner(jax_theta_pinch())
        runner.configure(dict(config, OutputFilename="jax_python"))
        runner.run()

        a, b = read_groups("jax_cpp"), read_groups("jax_python")
        for key in a:
            scale = max(np.max(np.abs(a[key])), 1e-30)
            gap = np.max(np.abs(a[key] - b[key])) / scale
            assert gap < CASE_GAP, f"{key}: JAX case is {gap:.3e} from the C++ case"
    finally:
        remove_output("jax_cpp")
        remove_output("jax_python")


def test_an_objective_reading_the_geometry_has_the_right_gradient(_float64):
    """dg's "Geometry" rows cross to the solver and reach dG/dp.

    G = Int B N over the steady theta pinch, coupled, against central
    differences in D0: B moves with D0 through the pressure, so a gradient
    missing dg/dB -- the rows the solver contracts with dGeometry/dpsi -- is
    off by that share. Measured: with State.to_manta dropping the "Geometry"
    rows the adjoint gives -0.70304 against -0.69713, 0.85% out.
    """
    pytest.importorskip("equinox")
    from theta_pinch_case import jax_theta_pinch

    steady = dict(
        tier(FIELD_MODEL, "coupled"),
        SteadyStateSolver="Newton",
        SteadyStateTolerance=1e-10,
        WriteOutput=False,
        OutputFilename="geometry_objective",
    )

    def solve(D0, adjoint):
        runner = MaNTA.Runner(jax_theta_pinch(D0=D0))
        runner.configure(dict(steady, solveAdjoint=adjoint))
        runner.run_ss()
        if adjoint:
            G, grads = runner.getAdjointGradients()
            return float(np.asarray(G)[0]), np.asarray(grads["G_p"]).reshape(-1)
        return float(runner.G()[0]), None

    D0, h = 0.5, 1e-4
    _, grad = solve(D0, True)
    Gp, _ = solve(D0 * (1 + h), False)
    Gm, _ = solve(D0 * (1 - h), False)
    fd = (Gp - Gm) / (2 * h * D0)
    assert grad[0] == pytest.approx(fd, rel=1e-5), f"adjoint {grad[0]:.10e}, differences {fd:.10e}"
