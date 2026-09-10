"""
End-to-end tests for stellarator model
"""

import os

os.environ["TEST_STELLARATOR"] = "true"
if "SLURM_NNODES" in os.environ:
    if int(os.environ["SLURM_NNODES"]) > 1:
        multinode = True
    else:
        multinode = False
else:
    multinode = False
from stellarator_multichannel import StellaratorTransport

os.environ.pop("TEST_STELLARATOR", None)
import desc
from yancc_wrapper2 import yancc_data
import numpy as np
import manta as MaNTA
import jax


def make_stellarator():

    st_config = {
        "ParticleSourceCenter": 0.0,
        "ParticleSourceHeight": 0.02,
        "ParticleSourceWidth": 0.25,
        "NBICenter": 0.0,
        "NBIPower": 0.15,
        "NBIWidth": 0.25,
        "ECHCenter": 0.0,
        "ECHPower": 0.15,
        "ECHWidth": 0.25,
        "EdgeTemperature": 0.2,
        "EdgeDensity": 0.3,
        "n0": 0.5,
        "evolveDensity": True,
        "useBatching": False,
    }
    # runner = MaNTA.Runner(st)

    rho_upper = 1.0
    rtol = 1e-2
    atol = 1e-4
    # nodes = [0.0,0.5, 0.75, 0.9, 1.0]
    npoints = 8
    degree = 3
    base = 2.5
    tau = 10.0
    nodes = 1 - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 1)
    nodes = np.concatenate(([0], nodes, [1]))
    # # %%
    solver_config = {
        "OutputFilename": "stellarator_gpu_test",
        "Polynomial_degree": degree,
        "Grid_points": nodes,
        "Grid_size": len(nodes) - 1,
        "tau": tau,
        "Lower_boundary": 0.0,
        "Upper_boundary": rho_upper,
        "Relative_tolerance": rtol,
        "Absolute_tolerance": [atol],
        "delta_t": 1.0,
        "initialTimestep": 1e-2,
        "MinStepSize": 1e-9,
        "SteadyStateTolerance": 2e-3,
        "AggressiveTimesteps": False,
        "WriteDatFile": True,
        "restart": False,
        "SteadyStateSolver": "Newton",
        "SteadyStateStepDiagnostics": True,
        "zeroFlux": True,
        "solveAdjoint": False,
        "MaxRejectedSteps": 1,
        "PseudoTransientSERRate": 2.0,
    }

    config = {
        "Stellarator": st_config,
        "Solver": solver_config,
    }

    points = MaNTA.getNodes(
        nodes,
        solver_config["Polynomial_degree"],
    )

    yancc_ntheta = 17
    yancc_nzeta = 25

    yancc_res = {"na": 43, "nx": 7}

    eq = desc.examples.get("W7-X")

    # Reduce the number of modes (not sure if this is a good thing to do)
    eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
    eq = eq.solve(x_scale="ess")[0]
    eq_init = eq.copy()
    yancc_wrapper = yancc_data.from_eq(
        points, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
    )

    return StellaratorTransport(config, yancc_wrapper=yancc_wrapper)


def test_sol():
    st = make_stellarator()
    st.run()


def test_objective():
    pass


def test_returning():
    st = make_stellarator()
    print(st.run())


def test_multinode():
    if multinode:
        st = make_stellarator()
        # with jax.log_compiles(True):
        st.run()


test_returning()
