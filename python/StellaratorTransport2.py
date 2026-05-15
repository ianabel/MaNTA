
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"]= "FALSE"
from Stellarator2 import StellaratorTransport
import MaNTA
from Objective2 import (make_objective, make_objective_fd)
# from Stellarator import StellaratorTransport

from yancc_wrapper2 import yancc_data

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

# st_config = {
#     "SourceCenter": 0.2,
#     "SourceHeight": 350.0,
#     "SourceWidth": 0.4,
#     "EdgeTemperature":0.2,
#     "EdgeDensity": 0.0,
#     "n0": 0.25,
# }

st_config = {
    "SourceCenter": 0.2,
    "SourceHeight": 10.0,
    "SourceWidth": 0.4,
    "EdgeTemperature":0.2,
    "EdgeDensity": 0.0,
    "n0": 0.25,
}

# # %%
solver_config = {
    "OutputFilename": "stellarator_opt_fd_test",
    "Polynomial_degree": 3,
    "Grid_size": 4,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-4,
    "MinStepSize": 1e-8, 
    "SteadyStateTolerance": 1e-2,
    "restart": False,
    "solveAdjoint": True, 
    "zeroFlux": True,
}

config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}

points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

yancc_rho = jnp.array(points)
yancc_ntheta = 13
yancc_nzeta = 23

# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eq_est = desc.examples.get("W7-X")
# surf = eq_est.get_surface_at(rho=1)
# eq = Equilibrium(M=4, N=4, Psi=0.087, surface=surf, pressure=desc_pressure)
# eq = eq.solve(x_scale="ess")[0]
# eq = eq.solve(x_scale="ess")[0]
# # store initial equilibrium for comparison later
eq = eq_est.copy()
# yancc_grid = desc.grid.LinearGrid(rho=yancc_rho, M=eq_init.M_grid, N = eq_init.N_grid, NFP=eq_init.NFP)
# points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(points, eq=eq, nx=5, na=33, nz=yancc_nzeta, nt = yancc_ntheta)

V0 = eq_est.compute("V")["V"]

# st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)

# # pi = 2./3. * st.InitialValue(0, yancc_rho)/yancc_wrapper.Vp
# # Ti = pi/st.Density(yancc_rho)
# # print(Ti)

# st.run()

# %%
import yancc
manta_objective = make_objective(config)
manta_objective_fd = make_objective_fd(config)

yancc_desc_grid = yancc_wrapper.grid 

def objective_from_user_fun(grid, data):
  # note: don't change the signature to this function
    yancc_dat = {
        "B_sup_t": data["B^theta"],
        "B_sup_z": data["B^zeta"],
        "B_sub_t": data["B_theta"],
        "B_sub_z": data["B_zeta"],
        "Bmag": data["|B|"],
        "dBdt": data["|B|_t"],
        "dBdz": data["|B|_z"],
        "sqrtg": data["sqrt(g)"],
    }

    yancc_dat = {
        key: grid.meshgrid_reshape(val, "rtz") for key, val in yancc_dat.items()
    }

    yancc_dat["Psi"] = grid.compress(
        data["Psi"] / grid.nodes[:, 0] ** 2, surface_label="rho"
    )
    yancc_dat["a_minor"] = jnp.full(grid.num_rho, data["a"])
    yancc_dat["R_major"] = jnp.full(grid.num_rho, data["R0"])
    yancc_dat["iota"] = grid.compress(data["iota"], surface_label="rho")
    yancc_dat["rho"] = grid.compress(grid.nodes[:, 0], surface_label="rho")

    V = grid.compress(data['V(r)'])
    V_r = grid.compress(data['V_r(r)'])
    V_rr = grid.compress(data['V_rr(r)'])
    Vp = V_r/V[-1]
    Vpp = V_rr/V[-1]
    
    fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

    desc_pressure = grid.compress(data["p"], surface_label="rho")
    
    stored_energy, manta_pressure = manta_objective((fields, Vp, Vpp), grid)
    print("------------ STORED ENERGY ----------------")
    print(stored_energy)
    print("-------------------------------------------")
    
    pressure_error = desc_pressure - manta_pressure

    print("------------TOTAL PRESSURE ERROR-----------")
    print(pressure_error)
    print("-------------------------------------------")

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return jnp.append(pressure_error, 1 / stored_energy)

pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 0.0)
stored_energy_weight = 1.0
objective_from_user_weight = jnp.append(pressure_error_weight, stored_energy_weight)

objfun = ObjectiveFromUser(
    objective_from_user_fun,
    eq,
    target=0,
    weight=objective_from_user_weight,
    grid=yancc_desc_grid,
    deriv_mode="fwd")
    # need this assuming manta only has vjp, if using jvp switch to fwd
 
# objfun = ObjectiveFunction(objectives)
objfun.build(use_jit=False)
J1 = 0
with jax.default_device(jax.devices('cpu')[0]):
    J1 = objfun.jac_scaled(eq.params_dict)[0]

def objective_from_user_fun(grid, data):
  # note: don't change the signature to this function
    yancc_dat = {
        "B_sup_t": data["B^theta"],
        "B_sup_z": data["B^zeta"],
        "B_sub_t": data["B_theta"],
        "B_sub_z": data["B_zeta"],
        "Bmag": data["|B|"],
        "dBdt": data["|B|_t"],
        "dBdz": data["|B|_z"],
        "sqrtg": data["sqrt(g)"],
    }

    yancc_dat = {
        key: grid.meshgrid_reshape(val, "rtz") for key, val in yancc_dat.items()
    }

    yancc_dat["Psi"] = grid.compress(
        data["Psi"] / grid.nodes[:, 0] ** 2, surface_label="rho"
    )
    yancc_dat["a_minor"] = jnp.full(grid.num_rho, data["a"])
    yancc_dat["R_major"] = jnp.full(grid.num_rho, data["R0"])
    yancc_dat["iota"] = grid.compress(data["iota"], surface_label="rho")
    yancc_dat["rho"] = grid.compress(grid.nodes[:, 0], surface_label="rho")

    V = grid.compress(data['V(r)'])
    V_r = grid.compress(data['V_r(r)'])
    V_rr = grid.compress(data['V_rr(r)'])
    Vp = V_r/V[-1]
    Vpp = V_rr/V[-1]
    
    fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

    desc_pressure = grid.compress(data["p"], surface_label="rho")
    
    stored_energy, manta_pressure = manta_objective_fd((fields, Vp, Vpp), grid)
    print("------------ STORED ENERGY ----------------")
    print(stored_energy)
    print("-------------------------------------------")
    
    pressure_error = desc_pressure - manta_pressure

    print("------------TOTAL PRESSURE ERROR-----------")
    print(pressure_error)
    print("-------------------------------------------")

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return jnp.append(pressure_error, 1 / stored_energy)

objfun = ObjectiveFromUser(
    objective_from_user_fun,
    eq,
    target=0,
    weight=objective_from_user_weight,
    grid=yancc_desc_grid,
    deriv_mode="fwd")

objfun.build(use_jit=False)
J2 = 0
with jax.default_device(jax.devices('cpu')[0]):
    J2 = objfun.jac_scaled(eq.params_dict)[0]
# g = eqx.filter_grad(Objective, has_aux=True)
# g_fd = eqx.filter_grad(Objective_fd, has_aux=True)
# fields = yancc_wrapper.fields
# grid = yancc_wrapper.grid

# Vprime = yancc_wrapper.Vp
# Vpp = yancc_wrapper.Vpp

# g_out = g(fields, grid, Vprime, Vpp)
# g_fd_out = g_fd(fields, grid, Vprime, Vpp)
# # # print(g(fields, grid, Vprime))


# # %%
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

# ax.plot(g_out.Bmag_fsa, label="adjoints")
# ax.plot(g_fd_out.Bmag_fsa, label="finite_differences")

# ax.legend()

# fig, ax = plt.subplots()

# ax.plot(g_out.sqrtg, label="adjoints")
# ax.plot(g_fd_out.sqrtg, label="finite_differences")

# ax.legend()


# %%


# %%



