"""Maximise the stored energy MaNTA reaches with the NN ITG heat flux on.

desc-optimize-stored-energy.py with the physics changed: G is still
`StellaratorTransport.StoredEnergy` and DESC still minimises 1/G, but the NN
ITG heat flux is transported (useITGProxy), the neoclassical flux is the
SI-units mock `dke_sol.physical_test_flux`, and the density is held fixed
(evolveDensity = False), so the one transported channel is the ion energy.

Starts from the first equilibrium of eq_itg_all_equilibria.h5 and writes its
own family, eq_itg_flux_all_equilibria.h5, so the stored-energy run's file is
left alone.
"""

import os
import sys

# physical test flux in place of yancc; must be set before stellarator_multichannel
os.environ["TEST_STELLARATOR"] = "true"
os.environ["STELLARATOR_TEST_FLUX"] = "physical"
# itg_wrapper reads it at import
os.environ.setdefault(
    "ITG_NN_DIR", "/home/eatocco/projects/itg-proxy/demo/neural_networks/"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sys.path.append("..")
from stellarator_multichannel import StellaratorTransport
from objective import make_objective, ObjectiveFromUserWithParams
from itg_wrapper import make_itg_builder, itg_at, with_itg
from config import rescale_config, field_norm_from_eq
from yancc_wrapper import yancc_data

import matplotlib.pyplot as plt
from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error
from desc.plotting import plot_comparison
from desc.objectives import (
    AspectRatio,
    FixCurrent,
    FixPsi,
    ForceBalance,
    LinearObjectiveFromUser,
    ObjectiveFunction,
    ObjectiveFromUser,
    Volume,
)
import desc
from desc.grid import Grid
from desc.profiles import SplineProfile
from desc.equilibrium import EquilibriaFamily
import yancc
import jax.numpy as jnp
import numpy as np
import jax
import manta as MaNTA


eq_in = "eq_itg"  # initial equilibrium: the first of this family
eq_name = "eq_itg_flux"  # what this run writes
fname = "stellarator_opt" + eq_name

st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": 0.02,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": 16.0,  # doubled from the stored-energy script's 8.0
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": 0.0,
    "ECHWidth": 0.1,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "ADCoefficient": 0.0,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": False,
    "useITGProxy": True,  # NN ITG heat flux (itg_wrapper.py); needs ITG_NN_DIR
    "ITGScale": 1.0,
    "useBatching": True,
    "nchunks": 3,
}

# st_config and the solver settings are written in the default normalisation.
# True converts them (rescale_config) so only the units change, not the physics;
# False reads them as numbers in the new units.
keep_physics = True

rho_upper = 1.0
rtol = 1e-2
atol = 1e-10
npoints = 9
degree = 4
base = 1.9
tau = 0.2

nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 4)
nodes = np.concatenate(([0, 0.05, 0.15, 0.3], nodes, [rho_upper]))
print(nodes)

# the initial run, from a cold start
initial_solver_config = {
    "OutputFilename": fname,
    "PolynomialDegree": degree,
    "GridPoints": nodes,
    "GridSize": len(nodes) - 1,
    "tau": tau,
    "LowerBoundary": 0.0,
    "UpperBoundary": rho_upper,
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
    "NewtonJacobianReuse": 5,
    "NewtonMaxIterations": 25,
    "MaxRejectedSteps": 10,
    "restart": False,
    "zeroFlux": True,
    "solveAdjoint": False,
    "PseudoTransientSERRate": 1.0,
    "PseudoTransientSERFloor": 2.0,
}

# every run inside the optimisation: a Newton solve from the last restart file
solver_config = {
    "OutputFilename": fname,
    "PolynomialDegree": degree,
    "GridPoints": nodes,
    "GridSize": len(nodes) - 1,
    "tau": tau,
    "LowerBoundary": 0.0,
    "UpperBoundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-2,
    "MinStepSize": 1e-10,
    "WriteDatFile": True,
    "NewtonJacobianReuse": 5,
    "NewtonMaxIterations": 20,
    "restart": True,
    "zeroFlux": True,
    "SteadyStateTolerance": 2e-4,
    "SteadyStateSolver": "Newton",
    "SteadyStateDiagnostics": True,
    "MaxRejectedSteps": 0,
    "SteadyStateStepDiagnostics": True,
    "PseudoTransientSERFloor": 2.0,
}

points = MaNTA.getNodes(nodes, degree)

yancc_rho = jnp.array(points)
yancc_ntheta = 17
yancc_nzeta = 25
yancc_res = {"na": 45, "nx": 7}

pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eqs = desc.io.load(eq_in + "_all_equilibria.h5")
eq = eqs[0].copy()
eq_init = eq.copy()
eqs = EquilibriaFamily(eq_init.copy())

V0 = eq_init.compute("V")["V"]

yancc_wrapper = yancc_data.from_eq(
    yancc_rho, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)

# Normalise lengths and fields to the initial equilibrium's a and <|B|>_rms,
# fixed from here on: every run below, and the objective, use these numbers
normalization = field_norm_from_eq(eq_init)
print(f"Normalization from the initial equilibrium: {normalization}")


def normalized(config):
    config = {**config, "Objective": "StoredEnergy"}
    if keep_physics:
        return rescale_config(config, normalization)
    return {**config, "Normalization": normalization}


# Built once, on the initial equilibrium; it serves every equilibrium after
itg_builder = make_itg_builder(st_config, eq_init, rho_upper)

config = normalized({"Stellarator": st_config, "Solver": initial_solver_config})
st = StellaratorTransport(
    config, yancc_wrapper=yancc_wrapper, itg=itg_at(itg_builder, eq_init.params_dict)
)
st.run()

config = normalized({"Stellarator": st_config, "Solver": solver_config})

# a failed solve returns make_objective's default 1e-10, so 1/G is huge there
manta_objective = make_objective(
    config, yancc_res=yancc_res, normalization=normalization
)


def objective_from_user_fun(grid, data, params=None):
    # keep (grid, data) first: ObjectiveFromUser calls fun(grid, data);
    # ObjectiveFromUserWithParams also passes params, which the ITG term
    # needs. The data keys must stay written out here: DESC reads them
    # from this function's source
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

    V = grid.compress(data["V(r)"])
    V_r = grid.compress(data["V_r(r)"])
    V_rr = grid.compress(data["V_rr(r)"])
    Vp = V_r / V[-1]
    Vpp = V_rr / V[-1]

    fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

    g, _ = manta_objective(with_itg((fields, Vp, Vpp), itg_builder, params), grid)

    print("------------ STORED ENERGY ----------------")
    print(g)
    print("-------------------------------------------")

    return 1 / g


yancc_desc_grid = desc.grid.LinearGrid(
    rho=yancc_rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
)

domainBoundary_rho = 1.0


def pressure_constraint_fun(params):
    # fix dp/dr=0 at axis and p at the edge
    p_l = params["p_l"]
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    p1 = desc_pressure(
        Grid(jnp.zeros((1, 3)).at[0, 0].set(domainBoundary_rho), jitable=True), p_l
    )
    return jnp.array([dp0, p1]).squeeze()


pi_edge = (
    st_config["EdgeTemperature"] * st_config["EdgeDensity"] * 1e20 * 1000 * 1.6e-19
)
pressure_constraint_target = jnp.array([0.0, pi_edge])


def fun_mirror_ratio(grid, data):
    return grid.compress(data["mirror ratio"])


# The geometry objectives build their Jacobians jac_chunk_size columns at a
# time. MaNTA's objective cannot be chunked: DESC chunks by mapping the JVP in
# a compiled loop, and the MaNTA solve calls back into Python (prepareEvaluation)
# with arrays that are tracers there. It is a scalar, so it takes reverse mode
# instead -- one backward pass, not a forward one per tangent. Forward mode over
# the optimiser's 126 tangents peaked at 18.6 GB; reverse, 7.0 GB, agreeing
# with forward-mode JVPs to 4e-8
jac_chunk_size = 8

obj_mirror_ratio = ObjectiveFromUser(
    fun=fun_mirror_ratio,
    thing=eq,
    grid=yancc_desc_grid,
    bounds=(0.0, 0.2),
    weight=10.0,
    name="my mirror ratio",
    jac_chunk_size=jac_chunk_size,
)

# as in desc-optimize-stored-energy.py
stored_energy_weight = 50.0
max_it = 8  # this machine runs out of memory past ~12

eqfam = EquilibriaFamily(eq)

objectives = [
    AspectRatio(eq=eq, bounds=(0, 8), weight=10.0, jac_chunk_size=jac_chunk_size),
    obj_mirror_ratio,
    Volume(eq=eq, target=V0, weight=100.0, jac_chunk_size=jac_chunk_size),
    ObjectiveFromUserWithParams(
        objective_from_user_fun,
        eqfam[-1],
        target=0.0,
        weight=stored_energy_weight,
        grid=yancc_desc_grid,
        deriv_mode="rev",  # a scalar: see jac_chunk_size above
    ),
]

# blocked: each objective's Jacobian uses its own jac_chunk_size
objective = ObjectiveFunction(objectives, deriv_mode="blocked")
objective.build(use_jit=False)

constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    FixCurrent(eq=eq),  # zero current; eventually a real bootstrap
    FixPsi(eq=eq),  # fix total magnetic flux
    LinearObjectiveFromUser(
        pressure_constraint_fun, eq, target=pressure_constraint_target
    ),
]

eq, info_out = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    x_scale="ess",
    maxiter=max_it,
    ftol=1e-4,
    xtol=1e-5,
    gtol=1e-6,
    verbose=3,
    copy=True,
)

eqfam.append(eq.copy())
eqs.append(eq.copy())
eqs.save(eq_name + "_all_equilibria.h5")

os.makedirs("figs", exist_ok=True)
fig, ax = plot_comparison(eqs=[eq_init, eq], labels=["Initial", "optimized"])
fig.savefig("figs/" + eq_name + "final_comparison.png")
fig, ax = plot_boundaries(eqs=[eq_init, eq], labels=["Initial", "optimized"])
fig.savefig("figs/" + eq_name + "final_comparison_boundary.png")
fig, ax = plot_boozer_surface(eqfam[-1], fieldlines=8)
fig.savefig("figs/" + eq_name + "final_boozer_surface.png")
fig, ax = plot_qs_error(eqfam[-1])
fig.savefig("figs/" + eq_name + "final_qs_error.png")
plt.show()
