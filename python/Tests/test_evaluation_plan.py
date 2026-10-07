"""The evaluation plan and the regrid policy, reached from Python.

The C++ suite (Tests/UnitTests/EvaluationPlanTests.cpp) pins that the plan is
complete across the solver's configurations, and what a rebuild does to the
instances and the adjoint. What only this can cover is the trampoline and the
Runner: that a Python case's ``prepareEvaluation`` is called, and only on a new
plan; that the plan crosses with numpy-friendly fields; that ``regrid`` reaches
the spec; and what ``Runner.configure`` accepts and refuses -- for a case object,
which can be moved only in place, and for a named C++ case, which
``RebuildPhysicsOnRegrid`` may rebuild.
"""

import numpy as np
import pytest

import manta as MaNTA

from test_mesh_adaptation import AxisSingular, mesh_config
from test_trampolines import VectorisedSystem, config

Kind = MaNTA.EvaluationKind
Entry = MaNTA.EvaluationEntry


class AnnouncedVectorised(VectorisedSystem):
    """The trampoline suite's numpy case, asserting every batch was announced."""

    def __init__(self):
        super().__init__()
        self.plans = []
        self.calls_before_plan = []
        self.unannounced = []

    def prepareEvaluation(self, plan):
        self.calls_before_plan.append(
            self.compute_physics_calls + self.compute_derivative_calls
        )
        self.plans.append(plan)

    def _check(self, entry, positions):
        plan = self.plans[-1]
        if not plan.announces(entry, list(positions)):
            self.unannounced.append((entry, len(positions)))

    def ComputePhysics(self, states, positions, t):
        self._check(Entry.ComputePhysics, positions)
        return super().ComputePhysics(states, positions, t)

    def ComputePhysicsDerivatives(self, states, positions, t):
        self._check(Entry.ComputePhysicsDerivatives, positions)
        return super().ComputePhysicsDerivatives(states, positions, t)


def test_a_python_case_is_handed_the_plan_before_its_first_evaluation(tmp_path):
    case = AnnouncedVectorised()
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path, Superconvergent=True))
    runner.run(0.5)

    assert len(case.plans) == 1
    assert case.calls_before_plan == [0]
    assert case.compute_physics_calls > 0 and case.compute_derivative_calls > 0
    assert case.unannounced == [], f"calls outside the plan: {case.unannounced}"

    plan = case.plans[0]
    assert plan.k == 3 and plan.superconvergent and not plan.steady
    assert plan.grid.getNCells() == 8

    # Numpy arrays, cell-major: the k+2 star nodes per cell, for the residual
    # and the initial condition alike -- one set of cell points per level.
    residual = plan.points(Kind.Residual)
    assert isinstance(residual, np.ndarray) and residual.shape == (8 * 5,)
    assert np.array_equal(plan.points(Kind.InitialCondition), residual)
    assert np.all(np.diff(residual) > 0)
    assert plan.batchSizes(Entry.ComputePhysics) == [40]

    site = plan.sitesOf(Kind.Residual)[0]
    assert site.entry == Entry.ComputePhysics
    assert site.cadence == MaNTA.EvaluationCadence.PerResidual
    assert site.pointsPerCell == 5 and site.batchSize() == 40

    # And the case can read it back afterwards.
    assert case.evaluationPlan is not None
    assert np.array_equal(case.evaluationPlan.points(Kind.Residual), residual)


def test_the_diffusive_tau_faces_are_announced(tmp_path):
    """Both faces of every cell, one-sided, whenever tau is Diffusive."""
    case = AnnouncedVectorised()
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path, tauScaling="Diffusive"))
    runner.run(0.5)

    assert case.unannounced == []
    faces = case.plans[0].points(Kind.TauFaces)
    assert faces.shape == (16,)
    edges = np.asarray(case.plans[0].grid.cellBoundaries())
    assert np.array_equal(faces[0::2], edges[:-1])
    assert np.array_equal(faces[1::2], edges[1:])


def test_a_fixed_case_runs_and_reruns_on_an_unchanged_plan(tmp_path):
    """No adaptation and an equal plan: nothing to refuse, and nothing to say.

    A case object that declares nothing is Fixed, and that costs it nothing
    until something would evaluate it elsewhere: a run, a rerun, and a second
    configure() with the same mesh and degree all go ahead, and the case is told
    about the plan once.
    """
    case = AnnouncedVectorised()
    assert case.regridPolicy() == MaNTA.Regrid.Fixed
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path))
    runner.run(0.5)
    runner.run(0.5)
    runner.configure(config(tmp_path, Relative_tolerance=1e-4))
    runner.run(0.5)
    assert len(case.plans) == 1
    assert case.unannounced == []


def test_reconfiguring_a_fixed_case_object_onto_a_new_plan_is_refused(tmp_path):
    case = AnnouncedVectorised()
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path))
    runner.run(0.5)
    with pytest.raises(RuntimeError, match="RegridPolicy is Fixed"):
        runner.configure(config(tmp_path, GridSize=6))
    assert len(case.plans) == 1


def test_regrid_is_read_from_the_class_on_either_path():
    class FromAttributes(MaNTA.TransportSystem):
        variables = [MaNTA.Field("u")]
        regrid = MaNTA.Regrid.InPlace

        def __init__(self):
            super().__init__()

    class FromSpec(MaNTA.TransportSystem):
        regrid = MaNTA.Regrid.InPlace

        def __init__(self, spec):
            super().__init__(spec)

    spec = MaNTA.numbered_spec(1)
    assert FromAttributes().regridPolicy() == MaNTA.Regrid.InPlace
    assert FromSpec(spec).regridPolicy() == MaNTA.Regrid.InPlace
    assert spec.regrid == MaNTA.Regrid.Fixed, "the caller's spec was edited in place"
    assert MaNTA.TransportSystem(spec).regridPolicy() == MaNTA.Regrid.Fixed


# ------------------------------------------------- case objects that adapt --


class RecordingAxis(AxisSingular):
    """test_mesh_adaptation's axis singularity, recording the plans it is given.

    InPlace by inheritance: test_runner.LinearDiffusion declares it, honestly.
    """

    def __init__(self):
        super().__init__()
        self.plans = []

    def prepareEvaluation(self, plan):
        self.plans.append(plan)


class FixedAxis(RecordingAxis):
    regrid = MaNTA.Regrid.Fixed


ADAPTATION = [
    pytest.param({"MeshAdaptation": True, "DegreeTolerance": 1e-2}, id="mesh"),
    pytest.param(
        {"MeshAdaptation": False, "DegreeAdaptation": True, "DegreeTolerance": 1e-6,
         "MaxPolynomialDegree": 6},
        id="degree",
    ),
]


def same_plan(a, b):
    return a.k == b.k and np.array_equal(a.grid.cellBoundaries(), b.grid.cellBoundaries())


@pytest.mark.parametrize("mode", ADAPTATION)
def test_an_in_place_case_object_follows_adaptation(tmp_path, mode):
    case = RecordingAxis()
    assert case.regridPolicy() == MaNTA.Regrid.InPlace
    runner = MaNTA.Runner(case)
    runner.configure(mesh_config(tmp_path, **mode))
    runner.run_ss()

    assert len(case.plans) >= 2
    for before, after in zip(case.plans, case.plans[1:]):
        assert not same_plan(before, after), "a plan was delivered twice"
    final = case.plans[-1]
    assert np.array_equal(final.grid.cellBoundaries(), np.asarray(runner.getCellBoundaries()))
    if mode.get("MeshAdaptation"):
        assert not np.allclose(np.diff(case.plans[1].grid.cellBoundaries()), 0.1)
    else:
        assert final.k > 4


@pytest.mark.parametrize("mode", ADAPTATION)
def test_a_fixed_case_object_is_refused_at_configure(tmp_path, mode):
    case = FixedAxis()
    runner = MaNTA.Runner(case)
    with pytest.raises(RuntimeError, match="RegridPolicy is Fixed"):
        runner.configure(mesh_config(tmp_path, **mode))
    assert case.plans == []


def test_a_case_object_cannot_ask_to_be_rebuilt(tmp_path):
    """RebuildPhysicsOnRegrid rebuilds by name; an object has none."""
    runner = MaNTA.Runner(FixedAxis())
    with pytest.raises(RuntimeError, match="Regrid.InPlace"):
        runner.configure(mesh_config(tmp_path, RebuildPhysicsOnRegrid=True))


# ----------------------------------------------------- named C++ cases ----


def named_config(tmp_path, **overrides):
    cfg = {
        "PolynomialDegree": 4,
        "GridSize": 6,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "delta_t": 0.1,
        "OutputFilename": str(tmp_path / "named"),
        "WriteOutput": False,
        "SteadyStateSolver": "Newton",
        "SteadyStateTolerance": 1.0e-10,
        "Absolute_tolerance": 1.0e-10,
        "MinStepSize": 1.0e-12,
        "DegreeTolerance": 1.0e-6,
        "MaxPolynomialDegree": 6,
    }
    cfg.update(overrides)
    return cfg


NAMED_ADAPTATION = [
    pytest.param({"MeshAdaptation": True}, id="mesh"),
    pytest.param({"DegreeAdaptation": True}, id="degree"),
]


@pytest.mark.parametrize("mode", NAMED_ADAPTATION)
def test_a_named_in_place_case_adapts(tmp_path, mode):
    # LinearDiffusion reads nothing from its grid and declares InPlace.
    runner = MaNTA.Runner("LinearDiffusion")
    runner.configure(named_config(
        tmp_path, **mode,
        DiffusionProblem={"Kappa": 1.0, "Centre": 0.5, "InitialWidth": 0.2}))
    runner.run_ss()


LEGACY_NAME = "UnitTestEvaluationPlanLegacyCase"


class LegacyDiffusion(MaNTA.TransportSystem):
    """A registered case written as if before plans: Fixed, and it keeps its grid.

    -u'' = 1 with u = 1 at both ends, so u(0) = 1 exactly.
    """

    plans_seen = []

    def __init__(self, config, grid):
        super().__init__(MaNTA.numbered_spec(1))
        self.xR = grid.upperBoundary()
        self.plans = 0
        LegacyDiffusion.plans_seen.append(self)

    def prepareEvaluation(self, plan):
        self.plans += 1

    def SigmaFn(self, i, state, x, t):
        return state.q[0]

    def Sources(self, i, state, x, t):
        return 1.0

    def dSigmaFn_dq(self, i, state, x, t):
        return np.ones(1)

    def LowerBoundary(self, i, t):
        return 1.0

    def UpperBoundary(self, i, t):
        return 1.0

    def InitialValue(self, i, x):
        return 1.0

    def InitialDerivative(self, i, x):
        return 0.0


MaNTA.registerPhysicsCase(LEGACY_NAME, LegacyDiffusion)


@pytest.mark.parametrize("mode", NAMED_ADAPTATION)
def test_a_named_fixed_case_is_refused_without_the_key(tmp_path, mode):
    runner = MaNTA.Runner(LEGACY_NAME)
    with pytest.raises(RuntimeError, match="RebuildPhysicsOnRegrid"):
        runner.configure(named_config(tmp_path, **mode))


@pytest.mark.parametrize("mode", NAMED_ADAPTATION)
def test_a_named_fixed_case_is_rebuilt_under_the_key(tmp_path, mode):
    LegacyDiffusion.plans_seen.clear()
    runner = MaNTA.Runner(LEGACY_NAME)
    runner.configure(named_config(tmp_path, **mode, RebuildPhysicsOnRegrid=True))
    runner.run_ss()
    x = np.linspace(0.0, 1.0, 5)
    u = np.asarray(runner.getSolution(0, list(x))).reshape(-1)
    assert u == pytest.approx(1.0 + x * (1.0 - x) / 2.0, abs=1e-8)
    # Rebuilt rather than moved: no instance is handed a second plan.
    assert LegacyDiffusion.plans_seen
    assert all(case.plans <= 1 for case in LegacyDiffusion.plans_seen)


# ----------------------------------------------------------------- JAX ----


def test_a_jax_case_can_override_the_hook(tmp_path):
    pytest.importorskip("equinox")
    from JAXLinearDiffusion import JAXLinearDiffusion

    class PlannedJAX(JAXLinearDiffusion):
        def prepareEvaluation(self, plan):
            self.shapes = list(plan.batchSizes(Entry.ComputePhysics))

    case = PlannedJAX({"Centre": 0.0, "kappa": 2.0}, None)
    assert case.regridPolicy() == MaNTA.Regrid.Fixed
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path, GridSize=4, PolynomialDegree=2))
    runner.run(0.1)
    assert case.shapes == [12]
