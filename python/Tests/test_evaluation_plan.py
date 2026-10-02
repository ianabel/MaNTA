"""The evaluation plan and regrid hooks, reached from Python.

The C++ suite (Tests/UnitTests/EvaluationPlanTests.cpp) pins that the plan is
complete across the solver's configurations. What only this can cover is the
trampoline: that a Python case's ``prepareEvaluation`` and ``regrid`` are
called at all, that the plan crosses with numpy-friendly fields, that the
class attribute ``supports_regrid`` reaches the spec, and that a vectorised
case can check each batch it is handed against what it was told.
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

    # Numpy arrays, cell-major: the k+2 star nodes per cell for the residual,
    # the k+1 basis nodes for the initial condition.
    residual = plan.points(Kind.Residual)
    assert isinstance(residual, np.ndarray) and residual.shape == (8 * 5,)
    assert plan.points(Kind.InitialCondition).shape == (8 * 4,)
    assert np.all(np.diff(residual) > 0)
    assert sorted(plan.batchSizes(Entry.ComputePhysics)) == [32, 40]

    site = plan.sitesOf(Kind.Residual)[0]
    assert site.entry == Entry.ComputePhysics
    assert site.cadence == MaNTA.EvaluationCadence.PerResidual
    assert site.pointsPerCell == 5 and site.batchSize() == 40

    # And the case can read it back afterwards.
    assert case.evaluationPlan is not None
    assert np.array_equal(case.evaluationPlan.points(Kind.Residual), residual)


def test_a_case_that_defines_neither_hook_is_unaffected(tmp_path):
    case = VectorisedSystem()
    assert case.evaluationPlan is None
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path))
    runner.run(0.5)
    assert case.evaluationPlan is not None
    assert not case.supportsRegrid()


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


class RegriddableAxis(AxisSingular):
    supports_regrid = True

    def __init__(self):
        super().__init__(spec=MaNTA.numbered_spec(1))
        self.regrids = []

    def regrid(self, grid, k, plan):
        self.regrids.append((np.asarray(grid.cellBoundaries()), k, plan.grid.getNCells()))


def test_supports_regrid_is_read_from_the_class_on_either_path():
    class FromAttributes(MaNTA.TransportSystem):
        variables = [MaNTA.Field("u")]
        supports_regrid = True

        def __init__(self):
            super().__init__()

    spec = MaNTA.numbered_spec(1)
    assert FromAttributes().supportsRegrid()
    assert RegriddableAxis().supportsRegrid()
    assert not spec.supports_regrid, "the caller's spec was edited in place"
    assert not AxisSingular().supportsRegrid()


def test_mesh_adaptation_tells_a_python_case_about_the_graded_mesh(tmp_path):
    case = RegriddableAxis()
    runner = MaNTA.Runner(case)
    runner.configure(mesh_config(tmp_path, DegreeTolerance=1e-2))
    runner.run_ss()

    graded = np.asarray(runner.getCellBoundaries())
    assert not np.allclose(np.diff(graded), 0.1), "the driver did not grade"
    assert len(case.regrids) == 1
    boundaries, k, n_cells = case.regrids[0]
    assert np.array_equal(boundaries, graded)
    assert k == 4 and n_cells == 10


def test_a_jax_case_can_override_the_hook(tmp_path):
    pytest.importorskip("equinox")
    from JAXLinearDiffusion import JAXLinearDiffusion

    class PlannedJAX(JAXLinearDiffusion):
        def prepareEvaluation(self, plan):
            self.shapes = list(plan.batchSizes(Entry.ComputePhysics))

    case = PlannedJAX({"Centre": 0.0, "kappa": 2.0}, None)
    runner = MaNTA.Runner(case)
    runner.configure(config(tmp_path, GridSize=4, PolynomialDegree=2))
    runner.run(0.1)
    assert case.shapes == [12]
