import os

# uncomment to run test flux
# os.environ["TEST_STELLARATOR"] = "true"
from stellarator_multichannel import StellaratorTransport
from objective import make_objective

# os.environ.pop("TEST_STELLARATOR", None)
from yancc_wrapper import yancc_data
import matplotlib.pyplot as plt

from desc.profiles import SplineProfile
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
    "ParticleSourceHeight": fac * 0.02,
    "ParticleSourceWidth": 0.4,
    "NBICenter": 0.0,
    "NBIPower": fac * 0.5,
    "NBIWidth": 0.1,
    "ECHCenter": 0.0,
    "ECHPower": fac * 25.0,
    "ECHWidth": 0.05,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useBatching": True,
    "adaptiveGrid": False,
    "nchunks": 3,
}


rho_upper = 1.0
rtol = 1e-2
atol = 1e-10
npoints = 8
degree = 5

base = 1.8
tau = 0.2
#
nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 4)

nodes = np.concatenate(([0, 0.01, 0.05, 0.2], nodes, [rho_upper]))
solver_config = {
    "OutputFilename": fname,
    "PolynomialDegree": degree,
    "GridPoints": nodes,
    "GridSize": npoints,
    "tau": tau,
    "LowerBoundary": 0.0,
    "UpperBoundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-2,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-4,
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "SteadyStateDiagnostics": True,
    "SteadyStateStepDiagnostics": True,
    # "SteadyStateSolver": "TimeMarch",
    # "ForceConsistentIC": True,
    "SteadyStateSolver": "PseudoTransient",
    "SuppressAlgebraicError": True,
    "NewtonJacobianReuse": 5,
    # "tauScaling": "Diffusive",
    # "tauUpdate": "Residual",
    "MaxRejectedSteps": 10,
    "restart": True,
    "zeroFlux": True,
    "solveAdjoint": False,
    "PseudoTransientSERRate": 1.0,
    "PseudoTransientSERFloor": 2.0,
}


config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}


def cheb_nodes(nCells):
    nodes = np.ndarray((nCells + 1,))
    for i in range(0, len(nodes)):
        nodes[nCells - i] = 0.5 * (1 + np.cos(i * np.pi / nCells))
    return nodes


points = MaNTA.getNodes(
    solver_config["GridPoints"],
    solver_config["PolynomialDegree"],
)


yancc_rho = jnp.array(points)
yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}


pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)


eq_name = "W7-X"
eq = desc.examples.get(eq_name)
eq = desc.compat.rescale(eq, L=("a", 1.7), B=("<B>", 5.86), copy=True, verbose=1)
# # eq = desc.examples.get("W7-X")
# # # # Reduce the number of modes (not sure if this is a good thing to do)
# # #
# eq = desc.compat.rescale(eq, L=("R0", 10), B=("B0", 5.0))
# # # #
# # #
# # # eq = desc.examples.get("reactor_QA")
eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
eq.solve(x_scale="ess")[0]


# eqs = desc.io.load("eq_random0_all_equilibria.h5")
# eq = eqs[-1].copy()
# eq = desc.examples.get("ARIES-CS")
points = MaNTA.getNodes(
    solver_config["GridPoints"],
    solver_config["PolynomialDegree"],
)


# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(
    yancc_rho, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)

st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
st.run()
