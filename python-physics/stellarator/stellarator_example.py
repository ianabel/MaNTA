from stellarator_multichannel import StellaratorTransport
from yancc_wrapper2 import yancc_data
import yancc
import jax.numpy as jnp
import numpy as np
import jax
import manta as MaNTA
import desc
from desc.plotting import plot_comparison
import matplotlib.pyplot as plt
from desc.profiles import SplineProfile
from desc.optimize._constraint_wrappers import ProximalProjection
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPsi,
    ForceBalance,
    LinearObjectiveFromUser,
    ObjectiveFunction,
    ObjectiveFromUser,
    RotationalTransform,
    Volume,
)
from desc.grid import Grid, LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium, EquilibriaFamily
import desc.io
from desc import set_device
from scipy.constants import mu_0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


st_config = {
    "ParticleSourceCenter": 0.1,
    "ParticleSourceHeight": 0.01,
    "ParticleSourceWidth": 0.1,
    "HeatSourceCenter": 0.1,
    "HeatSourceHeight": 0.1,
    "HeatSourceWidth": 0.1,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.2,
    "n0": 0.5,
    "evolveDensity": True,
}
# runner = MaNTA.Runner(st)

rho_upper = 1.0
rtol = 1e-2
atol = 1e-2
# nodes = [0.0,0.5, 0.75, 0.9, 1.0]
npoints = 5
degree = 3
base = 1.6
tau = 10.0
nodes = 1 - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 1)
nodes = np.concatenate(([0], nodes, [1]))
# # %%
solver_config = {
    "OutputFilename": "stellarator_w7x",
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    # "initialTimestep": 1e-3,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-2,
    "aggressiveTimesteps": True,
    "restart": True,
    "zeroFlux": True,
    "solveAdjoint": False,
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
yancc_ntheta = 17
yancc_nzeta = 31

yancc_res = {"na": 55, "nx": 5}
## to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eq = desc.examples.get("W7-X")

# Reduce the number of modes (not sure if this is a good thing to do)
eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
eq = eq.solve(x_scale="ess")[0]
eq_init = eq.copy()
yancc_wrapper = yancc_data.from_eq(
    points, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)
# with jax.log_compiles(True):
st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)

st.run()
