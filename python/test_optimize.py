from scipy.constants import mu_0
import MaNTA
from Objective import make_objective


import jax
import jax.numpy as jnp
import equinox as eqx
import yancc

from yancc_wrapper import yancc_data

import desc
from desc import set_device
set_device("gpu")
import desc.io
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



# %%
from Stellarator import StellaratorTransport

st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 250.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.1,
    "EdgeDensity": 0.0,
    "n0": 0.5,
}
# runner = MaNTA.Runner(st)

# # %%
solver_config = {
    "OutputFilename": "stellarator_opt2",
    "Polynomial_degree": 3,
    "Grid_size": 3,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 0.9,
    "Relative_tolerance": 0.01,
    "delta_t": 0.0001,
    "MinStepSize": 1e-8,
    "restart": False,
    "solveAdjoint": True, 
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
surf = eq_est.get_surface_at(rho=1)
eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
eq = eq.solve(x_scale="ess")[0]

# eq = desc.io.load("eq_self_consistent_pressure.h5")
# desc_pressure = eq.get_profile('p')
eq_init = eq.copy()

V0 = eq.compute("V")["V"]
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(points, eq=eq_init)

manta_objective = make_objective(config, vectorized=True)

# def manta_yancc_fun(fields, grid, Vprime):

#     stored_energy, pressure = Objective(fields, grid, Vprime) 

#     return stored_energy, pressure

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
    Vprime = V_r/V[-1]

    fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

    desc_pressure = grid.compress(data["p"], surface_label="rho")
    
    stored_energy, manta_pressure = manta_objective(fields, grid, Vprime)
    
    
    pressure_error = manta_pressure - desc_pressure

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return jnp.append(pressure_error, 1 / stored_energy)

    

# %%
import interpax


yancc_rho = yancc_wrapper.rho
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

# grid where desc needs to evaluate field for yancc/manta
yancc_desc_grid = LinearGrid(
    rho=yancc_rho, theta=yancc_ntheta, zeta=yancc_nzeta, NFP=eq.NFP
)


desc_data = eq.compute(["V(r)", "V_r(r)"], grid=yancc_desc_grid)

V = yancc_desc_grid.compress(desc_data['V(r)'])
Vn = V/V[-1] # normalize
Vn = interpax.CubicSpline(yancc_rho, Vn)

rho_from_normalized_volume = lambda Vnorm : desc.backend.root_scalar(lambda x: Vn(x) - Vnorm, jnp.sqrt(Vnorm))    

# %%
domain_boundary_rho = rho_from_normalized_volume(0.9)
print(domain_boundary_rho)
def pressure_constraint_fun(params):
    # function to fix dp/dr=0 at axis and p=0 at edge
    # can modify this for other BC (eg fix p at rho=0.8)
    p_l = params["p_l"]
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    p1 = desc_pressure(Grid(jnp.zeros((1, 3)).at[0, 0].set(domain_boundary_rho), jitable=True), p_l)
    return jnp.array([dp0, p1]).squeeze()



pressure_constraint_target = jnp.array([0.0, 0.0])

# initial optimization just to get self consistent pressure with fixed initial boundary
pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 0.01)
stored_energy_weight = 1000
objective_from_user_weight = jnp.append(pressure_error_weight, stored_energy_weight)

# objectives = [
obj = ObjectiveFromUser(
    objective_from_user_fun,
    eq,
    target=0,
    weight=objective_from_user_weight,
    grid=yancc_desc_grid,
    deriv_mode="fwd", 
    use_jit=False,# need this assuming manta only has vjp, if using jvp switch to fwd
)



obj.build()

# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)

J = obj.jac_scaled(eq.params_dict)
print(J)



# ]

# constraints = [
#     ForceBalance(eq=eq),  # J x B - grad(p) = 0
#     FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
#     FixPsi(eq=eq),  # fix total magnetic flux
#     FixBoundaryR(eq=eq),  # fix boundary shape
#     FixBoundaryZ(eq=eq),
#     LinearObjectiveFromUser(
#         pressure_constraint_fun, eq, target=pressure_constraint_target
#     ),
# ]

# objective = ObjectiveFunction(objectives)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)
# eq, info_out = eq.optimize(
#     objective=objective,
#     constraints=constraints,
#     optimizer="proximal-lsq-exact",
#     maxiter=50,
#     verbose=3,
#     ftol=0.01,
#     x_scale="ess",
#     copy=True,
#     options={
#         # pressure is O(1e4) so we use a larger trust region for the self consistency part
#         "initial_trust_radius": 1000.0,
#         "max_trust_radius": 1e5,
#     },
# )
# # save for later
# eq_self_consistent_pressure = eq.copy()
# eq_self_consistent_pressure.save("eq_self_consistent_pressure.h5")




