"""PhysicsParallelism reached from Python: the key, the fill, and the warning.

The cost model and every controller are covered by
Tests/UnitTests/ParallelFillTests.cpp. What is checked here is that the dict
surface carries the key to them, read from what the case is handed -- its
evaluation plans -- and from the mesh, never from the progress log, which goes
through buffered C++ stdout that capfd does not reliably see.
"""

import numpy as np
import pytest

import manta as MaNTA

from test_degree_adaptation import SineSource, adaptive_config
from test_mesh_adaptation import AxisSingular, mesh_config


class RecordingSine(SineSource):
    def __init__(self):
        super().__init__()
        self.degrees = []

    def prepareEvaluation(self, plan):
        self.degrees.append(plan.k)


def test_the_degree_loop_fills_each_raise(tmp_path):
    # 6 cells under the superconvergent scheme are 6 (k + 2) points: 18 at
    # k = 1, and up to 60 -- k = 8 -- in the same round of 64.
    case = RecordingSine()
    runner = MaNTA.Runner(case)
    runner.configure(adaptive_config(tmp_path, PhysicsParallelism=64))
    runner.run_ss()
    assert case.degrees[:2] == [1, 8], case.degrees

    points = [0.15, 0.35, 0.5, 0.65, 0.85]
    err = np.max(np.abs(runner.getSolution(0, points) - SineSource.exact(np.array(points))))
    assert err < 1e-8


def test_a_graded_mesh_takes_the_cells_a_round_has_room_for(tmp_path):
    def faces(width):
        runner = MaNTA.Runner(AxisSingular())
        runner.configure(mesh_config(tmp_path, PhysicsParallelism=width))
        runner.run_ss()
        return np.asarray(runner.getCellBoundaries())

    unfilled, filled = faces(1), faces(256)
    assert len(filled) > len(unfilled), "the graded mesh was not filled"
    assert np.min(np.diff(filled)) == pytest.approx(np.min(np.diff(unfilled)), rel=1e-12)


def test_an_underused_configured_level_is_warned_about(tmp_path, capfd):
    runner = MaNTA.Runner(SineSource())
    runner.configure(adaptive_config(tmp_path, DegreeAdaptation=False, PhysicsParallelism=64))
    err = capfd.readouterr().err
    assert "PhysicsParallelism = 64" in err, err
    assert "PolynomialDegree up to" in err, err


def test_zero_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="PhysicsParallelism"):
        MaNTA.Runner(SineSource()).configure(adaptive_config(tmp_path, PhysicsParallelism=0))
