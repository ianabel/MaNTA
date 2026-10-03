import os

# uncomment to run test flux
# os.environ["TEST_STELLARATOR"] = "true"
from stellarator_multichannel import StellaratorTransport
from objective import make_objective

# os.environ.pop("TEST_STELLARATOR", None)
from yancc_wrapper import yancc_data
import matplotlib.pyplot as plt

import desc
import jax.numpy as jnp
import numpy as np
import jax
import manta as MaNTA


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

fname = "stellarator_opt_amb"

fac = 1.0
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": fac * 1.6e-2,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": fac * 1.25,
    "NBIWidth": 0.25,
    "ECHCenter": 0.0,
    "ECHPower": fac * 8.76e-2,
    "ECHWidth": 0.26,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.2,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useBatching": True,
}


rho_upper = 1.0
rtol = 1e-2
atol = 1e-10
npoints = 8
degree = 3

base = 1.5
tau = 0.5
#
nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 4)

nodes = np.concatenate(([0, 0.05, 0.1], nodes, [0.99, rho_upper]))
print(nodes)
solver_config = {
    "OutputFilename": fname,
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "Grid_size": len(nodes) - 1,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 0.01,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-5,
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "SteadyStateDiagnostics": True,
    "SteadyStateStepDiagnostics": True,
    "MaxRejectedSteps": 10,
    "restart": False,
    "zeroFlux": True,
    "solveAdjoint": False,
    "PseudoTransientSERRate": 1.0,
    "PseudoTransientSERFloor": 2.0,
}


config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}


points = MaNTA.getNodes(
    nodes,
    solver_config["Polynomial_degree"],
)

yancc_rho = jnp.array(points)

yancc_rho = jnp.array(points)
yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}

eqs = desc.io.load("eq_random0_all_equilibria.h5")
eq = eqs[-1].copy()


# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(
    points, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)

st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
st.run()
