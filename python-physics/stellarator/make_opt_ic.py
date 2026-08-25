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

fname = "stellarator_opt_amb"

eq_name = "eq_amb"
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": 0.5,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": 2.0,
    "NBIWidth": 0.4,
    "ECHCenter": 0.0,
    "ECHPower": 2.0,
    "ECHWidth": 0.4,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.4,
    "n0": 0.5,
    "evolveDensity": True,
}

rho_upper = 1.0
rtol = 1e-2
atol = 1e-4
# nodes = [0.0,0.5, 0.75, 0.9, 1.0]
npoints = 4
degree = 4
base = 2.5
tau = 1.0
nodes = 1 - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 1)
nodes = np.concatenate(([0], nodes, [1]))
# # %%
solver_config = {
    "OutputFilename": "stellarator_opt_amb0",
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "Grid_size": len(nodes) - 1,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-4,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-3,
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "zeroFlux": True,
    "solveAdjoint": False,
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

yancc_rho = jnp.array(points)

yancc_rho = jnp.array(points)
yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 43, "nx": 7}

pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, 0.1],
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=4,
)
# create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# default is zero pressure and zero current
eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
# this is usually all you need to solve a fixed boundary equilibrium
eq = eq.solve(x_scale="ess")[0]
# print(pressure_rho)
eqs = EquilibriaFamily(eq)
# eq = desc.io.load("eq_self_consistent_pressure.h5")
# desc_pressure = eq.get_profile('p')
eq_init = eq.copy()

V0 = eq.compute("V")["V"]
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(
    points, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)
st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
st.run()
