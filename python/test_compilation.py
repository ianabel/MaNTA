# %%

import MaNTA
from Objective import make_objective
from Stellarator import StellaratorTransport

from yancc_wrapper import yancc_data

# %%
import desc
from desc import set_device
set_device("cpu")

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
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
from desc.profiles import SplineProfile
from desc.plotting import (
    plot_grid,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_qs_error,
    plot_boundaries,
    plot_boundary,
)
import equinox as eqx
import jax
import jax.numpy as jnp

st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 40.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.1,
    "EdgeDensity": 0.0,
    "n0": 0.5,
}

# # %%
solver_config = {
    "OutputFilename": "compilation_test",
    "Polynomial_degree": 3,
    "Grid_size": 3,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 0.9,
    "Relative_tolerance": 0.01,
    "delta_t": 0.001,
    "restart": False,
    "solveAdjoint": True, 
}

config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}

Density = lambda x : (st_config["n0"] - st_config["EdgeDensity"]) * (1 - x * x) + st_config["EdgeDensity"]

points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

yancc_rho = jnp.array(points)
yancc_ntheta = 13
yancc_nzeta = 25

# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eq = desc.examples.get("W7-X")
# surf = eq_est.get_surface_at(rho=1)
# eq = Equilibrium(M=4, N=4, Psi=0.087, surface=surf, pressure=desc_pressure)
# eq = eq.solve(x_scale="ess")[0]
# eq = eq.solve(x_scale="ess")[0]
# # store initial equilibrium for comparison later
eq_init = eq.copy()
# yancc_grid = desc.grid.LinearGrid(rho=yancc_rho, M=eq_init.M_grid, N = eq_init.N_grid, NFP=eq_init.NFP)
# points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(points, eq=eq_init)

V0 = eq.compute("V")["V"]



# %%

jax.config.update("jax_log_compiles", True)
jax.config.update("jax_explain_cache_misses", True)

st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
st.runner.Run_ss()
# Objective = make_objective(config)
# g = eqx.filter_grad(Objective, has_aux=True)
# fields = yancc_wrapper.fields
# grid = yancc_wrapper.grid

# Vprime = yancc_wrapper.Vprim
# print(g(fields, grid, Vprime))



# %%


# %%



