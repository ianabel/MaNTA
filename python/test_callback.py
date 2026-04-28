# %%

import MaNTA

from Stellarator import StellaratorTransport
from Objective import make_objective
from yancc_wrapper import yancc_data

# %%
# # %%
# st_config = {
#     "SourceCenter": 0.0,
#     "SourceHeight": 30.0,
#     "SourceWidth": 0.2,
#     "EdgeTemperature":0.5,
#     "EdgeDensity": 0.1,
#     "n0": 0.5,
# }
# # runner = MaNTA.Runner(st)

# # # %%
# solver_config = {
#     "OutputFilename": "stellarator2",
#     "Polynomial_degree": 3,
#     "Grid_size": 6,
#     "tau": 1.0, 
#     "Lower_boundary": 0.0,
#     "Upper_boundary": 1.0,
#     "Relative_tolerance": 0.01,
#     "delta_t": 0.01,
#     "restart": False,
#     "solveAdjoint": True, 
# }

# config = {
#     "Stellarator": st_config,
#     "Solver": solver_config,
# }

# points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
# Density = lambda x : (st_config["n0"] - st_config["EdgeDensity"]) * (1 - x*x) + st_config["EdgeDensity"]

# yancc_wrapper = yancc_data.from_eq(points, Density=Density)

# st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
# # field_ = st.field
# # grid_ = yancc_wrapper.grid
# vprim_ = st.vprime

# %%
import desc
from desc import set_device
set_device("gpu")

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
    "SourceHeight": 80.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.1,
    "EdgeDensity": 0.0,
    "n0": 0.5,
}

# # %%
solver_config = {
    "OutputFilename": "stellarator_opt_nb",
    "Polynomial_degree": 3,
    "Grid_size": 6,
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
yancc_ntheta = 17
yancc_nzeta = 33

# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eq_est = desc.examples.get("ESTELL")
surf = eq_est.get_surface_at(rho=1)
eq = Equilibrium(M=4, N=4, Psi=0.087, surface=surf, pressure=desc_pressure)
eq = eq.solve(x_scale="ess")[0]
# eq = eq.solve(x_scale="ess")[0]
# # store initial equilibrium for comparison later
eq_init = eq.copy()
# yancc_grid = desc.grid.LinearGrid(rho=yancc_rho, M=eq_init.M_grid, N = eq_init.N_grid, NFP=eq_init.NFP)
# points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(points, eq=eq_init, Density=Density)

V0 = eq.compute("V")["V"]

# st = StellaratorTransport(config, yancc_wrapper)
# st.run()

# st_config = {
#     "SourceCenter": 0.0,
#     "SourceHeight": 80.0,
#     "SourceWidth": 0.2,
#     "EdgeTemperature":0.1,
#     "EdgeDensity": 0.0,
#     "n0": 0.5,
# }
# # runner = MaNTA.Runner(st)

# # # %%
# solver_config = {
#     "OutputFilename": "stellarator_opt",
#     "Polynomial_degree": 3,
#     "Grid_size": 6,
#     "tau": 100.0, 
#     "Lower_boundary": 0.0,
#     "Upper_boundary": 0.9,
#     "Relative_tolerance": 0.01,
#     "delta_t": 0.001,
#     "restart": True,
#     "solveAdjoint": True, 
# }

# config = {
#     "Stellarator": st_config,
#     "Solver": solver_config,
# }




# %%
# import time
# import datetime

# tFinal = 1.0

# start = time.time()

# st.run()

# end = time.time()
# time_duration = datetime.timedelta(seconds=end-start)
# print("Elapsed time:")
# print(time_duration)


# %%
# import matplotlib.pyplot as plt
# u = st.getPressure()

# plt.plot(points, u)


# %%
# @eqx.filter_custom_jvp
# def Objective(fields, grid, yin):
#     yancc_wrapper = yancc_data.from_other(fields, grid, yin)

#     solver_config = {
#         "OutputFilename": "stellarator_opt",
#         "Polynomial_degree": 3,
#         "Grid_size": 6,
#         "tau": 100.0, 
#         "Lower_boundary": 0.0,
#         "Upper_boundary": 0.9,
#         "Relative_tolerance": 0.01,
#         "delta_t": 0.001,
#         "restart": True,
#         "solveAdjoint": True, 
#     }

#     config = {
#         "Stellarator": st_config,
#         "Solver": solver_config,
#     }
#     st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
#     st.run()
#     G, G_p = st.runAdjointSolve()

#     pi = jnp.array(st.getPressure())

#     return G[0], pi

# @Objective.def_jvp
# def Objective_jvp(primals, tangents):
#     fields, grid, yin = primals
#     field_dot,_,_ = tangents
#     yancc_wrapper = yancc_data.from_other(fields, grid, yin)

#     solver_config = {
#         "OutputFilename": "stellarator_opt",
#         "Polynomial_degree": 3,
#         "Grid_size": 6,
#         "tau": 100.0, 
#         "Lower_boundary": 0.0,
#         "Upper_boundary": 0.9,
#         "Relative_tolerance": 0.01,
#         "delta_t": 0.001,
#         "restart": True,
#         "solveAdjoint": True, 
#     }

#     config = {
#         "Stellarator": st_config,
#         "Solver": solver_config,
#     }
#     st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
#     st.run()
#     G, G_p = st.runAdjointSolve()
#     pi = jnp.array(st.getPressure())
    
#     field_dot_flatten,_ = jax.flatten_util.ravel_pytree(field_dot)

#     return (G[0], pi), (jnp.float32(jnp.dot(G_p.flatten(), field_dot_flatten)), None)

    

# %%

Objective = make_objective(config, vectorized=True)
g = eqx.filter_grad(Objective, has_aux=True)
fields = yancc_wrapper.fields
grid = yancc_wrapper.grid



print(g(fields, grid, yancc_wrapper))


# %%


# %%



