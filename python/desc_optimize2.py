# %%
from scipy.constants import mu_0

import MaNTA

from Stellarator import StellaratorTransport


import yancc

from yancc_wrapper import yancc_data

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

import equinox as eqx
import jax
jax.clear_caches()
import jax.numpy as jnp


# %%

st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 80.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.1,
    "EdgeDensity": 0.0,
    "n0": 0.5,
}
# runner = MaNTA.Runner(st)

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
    "restart": True,
    "solveAdjoint": False, 
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

eq_init = eq.copy()

yancc_wrapper = yancc_data.from_eq(points, eq=eq_init, Density=Density)

V0 = eq.compute("V")["V"]

st = StellaratorTransport(config, yancc_wrapper = yancc_wrapper)
st.run()


@eqx.filter_custom_jvp
def Objective(data, grid):
    yancc_wrapper = yancc_data.from_data(data, grid, Density=Density)

    solver_config = {
        "OutputFilename": "stellarator_opt_nb",
        "Polynomial_degree": 3,
        "Grid_size": 6,
        "tau": 100.0, 
        "Lower_boundary": 0.0,
        "Upper_boundary": 0.9,
        "Relative_tolerance": 0.01,
        "delta_t": 0.01,
        "restart": True,
        "solveAdjoint": True, 
    }

    config = {
        "Stellarator": st_config,
        "Solver": solver_config,
    }
    st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
    st.run()
    G, G_p = st.runAdjointSolve()

    pi = st.getPressure()

    return G[0], pi 

@Objective.def_jvp
def Objective_jvp(primals, tangents):
    data, grid = primals
    field_dot, _ = tangents

    yancc_wrapper = yancc_data.from_data(data, grid, Density=Density)

    solver_config = {
        "OutputFilename": "stellarator_opt_nb",
        "Polynomial_degree": 3,
        "Grid_size": 6,
        "tau": 100.0, 
        "Lower_boundary": 0.0,
        "Upper_boundary": 0.9,
        "Relative_tolerance": 0.01,
        "delta_t": 0.01,
        "restart": True,
        "solveAdjoint": True, 
    }

    config = {
        "Stellarator": st_config,
        "Solver": solver_config,
    }
    st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
    st.run()
    G, G_p = st.runAdjointSolve()
    pi = st.getPressure()
    field_dot_flatten, _ = jax.flatten_util.ravel_pytree(field_dot)

    return (G[0], pi), jnp.dot(G_p.flatten(), field_dot_flatten)


def manta_yancc_fun(grid, data):

    stored_energy, pressure = Objective(data, grid) 

    return stored_energy, pressure


def objective_from_user_fun(grid, data):

    stored_energy, manta_pressure = manta_yancc_fun(grid, data)
    
    desc_pressure = grid.compress(data["p"], surface_label="rho")
    pressure_error = manta_pressure - desc_pressure

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return jnp.append(pressure_error, 1 / stored_energy)

# %%

yancc_rho = yancc_wrapper.rho
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

# grid where desc needs to evaluate field for yancc/manta
yancc_desc_grid = LinearGrid(
    rho=yancc_rho, theta=yancc_ntheta, zeta=yancc_nzeta, NFP=eq.NFP
)

# grid where desc needs to evaluate field for yancc/manta
yancc_desc_grid = LinearGrid(
    rho=yancc_rho, theta=yancc_ntheta, zeta=yancc_nzeta, NFP=eq.NFP
)


def pressure_constraint_fun(params):
    # function to fix dp/dr=0 at axis and p=0 at edge
    # can modify this for other BC (eg fix p at rho=0.8)
    p_l = params["p_l"]
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    p1 = desc_pressure(Grid(jnp.zeros((1, 3)).at[0, 0].set(1.0), jitable=True), p_l)
    return jnp.array([dp0, p1]).squeeze()


pressure_constraint_target = jnp.array([0.0, 0.0])

# initial optimization just to get self consistent pressure with fixed initial boundary
pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 1)
stored_energy_weight = 0
objective_from_user_weight = jnp.append(pressure_error_weight, stored_energy_weight)

objectives = [
    ObjectiveFromUser(
        objective_from_user_fun,
        eq,
        target=0,
        weight=objective_from_user_weight,
        grid=yancc_desc_grid,
        deriv_mode="fwd",  # need this assuming manta only has vjp, if using jvp switch to fwd
    )
]

constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
    FixPsi(eq=eq),  # fix total magnetic flux
    FixBoundaryR(eq=eq),  # fix boundary shape
    FixBoundaryZ(eq=eq),
    LinearObjectiveFromUser(
        pressure_constraint_fun, eq, target=pressure_constraint_target
    ),
]

objective = ObjectiveFunction(objectives)

eq, info_out = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    maxiter=50,
    verbose=3,
    x_scale="ess",
    copy=True,
    options={
        # pressure is O(1e4) so we use a larger trust region for the self consistency part
        "initial_trust_radius": 1e3,
        "max_trust_radius": 1e5,
    },
)
# save for later
eq_self_consistent_pressure = eq.copy()


# main optimization, varying boundary to maximize stored energy


# other objectives are non-dimensionalized, so weights should account for that
# and handle relative weighting, this will likely need trial and error
pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 10)
stored_energy_weight = 1000
objective_from_user_weight = jnp.append(pressure_error_weight, stored_energy_weight)

objectives = [
    AspectRatio(eq=eq, target=6, weight=10),
    Volume(eq=eq, target=V0, weight=10),
    RotationalTransform(eq=eq, target=0.42, weight=10),
    ObjectiveFromUser(
        objective_from_user_fun,
        eq,
        target=0,
        weight=objective_from_user_weight,
        grid=yancc_desc_grid,
        deriv_mode="fwd",  # need this assuming manta only has vjp, if using jvp switch to fwd
    ),
]
constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
    FixPsi(eq=eq),  # fix total magnetic flux
    LinearObjectiveFromUser(
        pressure_constraint_fun, eq, target=pressure_constraint_target
    ),
]

objective = ObjectiveFunction(objectives)

eq, info_out = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    maxiter=50,
    verbose=3,
    x_scale="ess",
    copy=True,
)

eq_optimized = eq.copy()


# do a final pass with just the self consistency part to make sure profiles match
pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 1)
stored_energy_weight = 0
objective_from_user_weight = jnp.append(pressure_error_weight, stored_energy_weight)

objectives = [
    ObjectiveFromUser(
        objective_from_user_fun,
        eq,
        target=0,
        weight=objective_from_user_weight,
        grid=yancc_desc_grid,
        deriv_mode="fwd",  # need this assuming manta only has vjp, if using jvp switch to fwd
    )
]

constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
    FixPsi(eq=eq),  # fix total magnetic flux
    FixBoundaryR(eq=eq),  # fix boundary shape
    FixBoundaryZ(eq=eq),
    LinearObjectiveFromUser(
        pressure_constraint_fun, eq, target=pressure_constraint_target
    ),
]

objective = ObjectiveFunction(objectives)

eq, info_out = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    maxiter=50,
    verbose=3,
    x_scale="ess",
    copy=True,
    options={
        "initial_trust_radius": 1e3,
        "max_trust_radius": 1e5,
    },
)
eq_optimized_self_consistent = eq.copy()

# %%
eq_optimized_self_consistent.save("optimized_equilibrium.h5")

# %%


# %%



