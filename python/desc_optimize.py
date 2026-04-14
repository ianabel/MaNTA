from scipy.constants import mu_0

import MaNTA

from Stellarator import StellaratorTransport

st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 30.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.5,
    "EdgeDensity": 0.1,
    "n0": 0.5,
}

st = StellaratorTransport(st_config)

import equinox as eqx
import jax
import jax.numpy as jnp

@eqx.filter_custom_jvp
def Objective(field, grid):
    G, _ = st.runAdjointSolve(field=field, grid=grid)
    return G[0]

@Objective.def_jvp
def Objective_jvp(primals, tangents):
    field, grid = primals
    field_dot, _ = tangents
    G, G_p = st.runAdjointSolve(field = field, grid = grid)   

    field_dot_flatten = jax.flatten_util.ravel_pytree(field_dot)

    return G[0], jnp.dot(G_p['G_p'].flatten(), field_dot_flatten)


import yancc

import desc
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

points = st.points

def manta_yancc_fun(yancc_fields, grid):

    stored_energy = Objective(yancc_fields, grid)* st.pnorm
    pressure = st.getPressure()
    ## run manta here to get steady state profiles given python list of yancc field objects
    # returns stored energy as a scalar, and an array of the steady state pressure at the
    # radial points of yancc/desc grid defined below
    # if needed we can generalize this to return pressure at different points from where
    # fluxes are calculated
    # as a dummy placeholder for testing, we just use the beta*B^2 energy/pressure from yancc fields
    # beta = 1e-2
    # pressure_profile_at_yancc_desc_grid_rho_pts = jnp.array(
    #     [beta * f.Bmag ** 2 / (2 * mu_0) * (1 - f.rho) ** 2 for f in yancc_fields]
    # )
    # stored_energy = jnp.sum(pressure_profile_at_yancc_desc_grid_rho_pts)
    return stored_energy, pressure


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

    yancc_fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)
    #yancc_fields = desc.backend.tree_unstack(yancc_fields)

    stored_energy, manta_pressure = manta_yancc_fun(yancc_fields, grid)
    
    desc_pressure = grid.compress(data["p"], surface_label="rho")
    pressure_error = manta_pressure - desc_pressure

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return jnp.append(pressure_error, 1 / stored_energy)


# create initial surface. Aspect ratio ~ 6, circular cross section with slight
# axis torsion to make it nonplanar
surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.166, 0.1],
    Z_lmn=[-0.166, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=2,
)


# change the rho coordinates here to wherever manta needs to evaluate yancc fluxes
yancc_rho = jnp.array([0.2, 0.4, 0.6, 0.8])
yancc_ntheta = 15
yancc_nzeta = 45

# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

# create initial equilibrium. Psi chosen to give B ~ 1 T.
# M=N=4 is fine for quick testing, for actual optimization may want to increase to ~8-10
eq = Equilibrium(M=4, N=4, Psi=0.087, surface=surf, pressure=desc_pressure)
eq = eq.solve(x_scale="ess")[0]
# store initial equilibrium for comparison later
eq_init = eq.copy()
V0 = eq.compute("V")["V"]

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
        deriv_mode="fwd", 
        use_jit = False # need this assuming manta only has vjp, if using jvp switch to fwd
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
        deriv_mode="rev",  # need this assuming manta only has vjp, if using jvp switch to fwd
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
        deriv_mode="rev",  # need this assuming manta only has vjp, if using jvp switch to fwd
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
