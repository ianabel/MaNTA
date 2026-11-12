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


# the physical test flux has too large of an effect on the solution, so just use the linear term
# os.environ["STELLARATOR_TEST_FLUX"] = "physical"
# itg_wrapper reads it at import
# os.environ.setdefault(
#     "ITG_NN_DIR", "/home/eatocco/projects/itg-proxy/demo/neural_networks/"
# )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sys.path.append("..")
from stellarator_multichannel import StellaratorTransport
from objective import make_objective, ObjectiveFromUserWithParams, ObjectiveHistory
from itg_wrapper import make_itg_builder, itg_at, with_itg, itg_rho_at_nodes
from config import rescale_config, field_norm_from_eq
from yancc_wrapper import yancc_data

import matplotlib.pyplot as plt
from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error, plot_1d
from desc.plotting import plot_comparison
from desc.objectives import (
    AspectRatio,
    FixCurrent,
    FixPsi,
    FixBoundaryR,
    FixBoundaryZ,
    ForceBalance,
    LinearObjectiveFromUser,
    ObjectiveFunction,
    ObjectiveFromUser,
    Volume,
)
import desc
from desc.grid import Grid
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium, EquilibriaFamily
import yancc
import jax.numpy as jnp
import numpy as np
import jax
import manta as MaNTA


eq_in = "eq_itg"  # initial equilibrium: the first of this family
eq_name = "eq_itg_flux_no_nc"  # what this run writes
fname = "stellarator_opt" + eq_name
append = True
use_default = True

# --- pressure self-consistency with MaNTA, inside the one eq.optimize call ----
# DESC's pressure becomes an even power series to rho^16 (smooth -- a spline's
# slope rule switches branch inside a DESC step), with p(1) = 0, and a residual
# pulls it to MaNTA's pressure at MaNTA's nodes. Beyond MaNTA's edge it is
# pulled to a straight fall from MaNTA's last node to zero at rho = 1, which
# nothing else would pin: unconstrained, the profile dipped negative there and
# the optimiser deepened the dip. MaNTA's pressure carries its derivative
# (PressureObjectives in objective.py), so DESC's model sees how a boundary step
# moves it. See desc-optimize-stored-energy.py for the bootstrap current too.
pressure_consistency = False
# relative to stored_energy_weight, on the mismatch normalised by MaNTA's peak
# pressure: below a percent it barely registers beside the stored energy
pressure_weight = 1.0
# L = M = N for DESC. At 4 the interior was poorly determined: a boundary change
# of 7e-6 moved it by 1.4% and G by 4%, which stalled the trust region. None
# keeps the resolution of the loaded equilibrium.
desc_resolution = 8 if pressure_consistency else None
# every equilibrium solve, including each trial point's inside the proximal
# projection, converged to this: at DESC's default tolerance the interior
# depended on the path taken to a point, and G with it
tight_solve = dict(ftol=1e-8, xtol=1e-8, gtol=1e-8, maxiter=100)

if append:
    eq_in = eq_name

st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": 0.02,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": 20.0,  # doubled from the stored-energy script's 8.0
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": 0.0,
    "ECHWidth": 0.1,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "ADCoefficient": 0.0,
    "n0": 1.0,
    "T0": 1.0,
    "alpha": 4.0,
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

rho_upper = 0.9
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


def make_initial_eq():
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, 0.1],
        Z_lmn=[-0.125, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=4,
    )
    # # create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
    # # default is zero pressure and zero current
    eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
    eq = desc.compat.rescale(
        eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
    )
    # # #
    # # # # # # #
    # # # # # eq.pressure = desc_pressure
    return eq.solve(x_scale="ess")[0]


if not append:
    if use_default:
        eq_init = make_initial_eq()
    else:
        eq_init = desc.examples.get("precise_QH")
        eq_init = desc.compat.rescale(
            eq_init,
            L=("a", 1.7),
            B=("<B>", 5.86),
            scale_pressure=False,
            copy=True,
            verbose=1,
        )
        # #
        # # # # # #
        eq_init.pressure = desc_pressure
        eq_init = eq_init.solve(x_scale="ess")[0]

        # i_initial = -1 if append else 0

        # eq = eqs[i_initial].copy()
        # eq_init = eqs[0].copy()
    eqs = EquilibriaFamily(eq_init.copy())
    eq = eq_init.copy()
else:
    eq_init = eqs[0].copy()
    eq = eqs[-1].copy()

if desc_resolution is not None:
    eq.change_resolution(
        L=desc_resolution, M=desc_resolution, N=desc_resolution,
        L_grid=2 * desc_resolution, M_grid=2 * desc_resolution, N_grid=2 * desc_resolution,
    )
    eq.solve(objective="force", optimizer="lsq-exact", verbose=1, **{**tight_solve, "maxiter": 500})

V0 = eq_init.compute("V")["V"]

yancc_wrapper = yancc_data.from_eq(
    yancc_rho, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
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


# Built once, on the initial equilibrium; it serves every equilibrium after of
# the same resolution, so on the one being optimised when that has changed. A
# knot at every MaNTA node: between knots the monotone interpolation's slope
# rule switches branch inside a geometry step (see itg_rho_at_nodes).
itg_builder = make_itg_builder(
    st_config, eq if desc_resolution is not None else eq_init, rho_upper,
    rho=itg_rho_at_nodes(points),
)

config = normalized({"Stellarator": st_config, "Solver": initial_solver_config})
st = StellaratorTransport(
    config, yancc_wrapper=yancc_wrapper, itg=itg_at(itg_builder, eq.params_dict)
)
st.run()

points_np = np.asarray(points)
if pressure_consistency:
    # one tracking step: DESC's pressure set to MaNTA's at the starting point
    pm0 = np.asarray(st.getPressure())
    p_ref = float(pm0.max())
    P_MODES = np.arange(0, 17, 2)
    _A = np.stack([points_np**m for m in P_MODES], 1)
    _B = _A[:, 1:] - _A[:, :1]  # c_0 = -sum c_k: p(1) = 0
    _c = np.linalg.lstsq(_B, pm0, rcond=None)[0]
    _c = np.concatenate([[-_c.sum()], _c])
    print(f"pressure fit to rho^16: rms {np.sqrt(np.mean((_A @ _c - pm0) ** 2)) / p_ref:.2e} of peak")
    desc_pressure = PowerSeriesProfile(_c, modes=P_MODES)
    eq.pressure = desc_pressure
    eq.solve(objective="force", optimizer="lsq-exact", verbose=1, **{**tight_solve, "maxiter": 500})
    # between MaNTA's last node and rho = 1
    RHO_EDGE = np.linspace(points_np[-1], 1.0, 6)[1:-1]
    GRID_EDGE = Grid(np.stack([RHO_EDGE, 0 * RHO_EDGE, 0 * RHO_EDGE], 1), jitable=True)
    # the ITG proxy holds the pressure profile's layout from the equilibrium it
    # was built on, and it has just changed
    itg_builder = make_itg_builder(st_config, eq, rho_upper, rho=itg_rho_at_nodes(points))

config = normalized(
    {
        "Stellarator": st_config,
        "Solver": solver_config,
        "PressureObjectives": pressure_consistency,
    }
)

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

    desc_p = grid.compress(data["p"], surface_label="rho")

    g, manta_pressure = manta_objective(
        with_itg((fields, Vp, Vpp), itg_builder, params), grid
    )

    print("------------ STORED ENERGY ----------------")
    print(g)
    print("-------------------------------------------")

    if not pressure_consistency:
        return 1 / g
    p_edge_target = manta_pressure[-1] * (1.0 - RHO_EDGE) / (1.0 - points_np[-1])
    p_edge = desc_pressure(GRID_EDGE, params["p_l"])
    r_p = jnp.concatenate([manta_pressure - desc_p, p_edge_target - p_edge]) / p_ref
    if not isinstance(manta_pressure, jax.core.Tracer) and not isinstance(desc_p, jax.core.Tracer):
        mp, dp = np.asarray(manta_pressure), np.asarray(desc_p)
        print(f"consistency: L2 pressure error {np.linalg.norm(dp - mp) / np.linalg.norm(mp):.3e}")
    # 1/g first, so ObjectiveHistory's first column is still 1/g
    return jnp.concatenate([jnp.ravel(1 / g), pressure_weight * r_p])


yancc_desc_grid = desc.grid.LinearGrid(
    rho=yancc_rho, theta=yancc_ntheta, zeta=yancc_nzeta, NFP=eq.NFP
)

domainBoundary_rho = 1.0


def pressure_constraint_fun(params):
    # fix dp/dr=0 at axis and p at the edge
    p_l = params["p_l"]
    p1 = desc_pressure(
        Grid(jnp.zeros((1, 3)).at[0, 0].set(domainBoundary_rho), jitable=True), p_l
    )
    if pressure_consistency:
        # the even power series has dp/dr = 0 at the axis built in
        return jnp.atleast_1d(p1.squeeze())
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    return jnp.array([dp0, p1]).squeeze()


pi_edge = (
    st_config["EdgeTemperature"] * st_config["EdgeDensity"] * 1e20 * 1000 * 1.6e-19
)
if rho_upper == 1.0:
    pressure_constraint_target = jnp.array([0.0, pi_edge])
else:
    pressure_constraint_target = jnp.array([0.0, 0.0])
if pressure_consistency:
    pressure_constraint_target = pressure_constraint_target[1:]


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
    bounds=(0.0, 0.3),
    weight=100.0,
    name="my mirror ratio",
    jac_chunk_size=jac_chunk_size,
)

# as in desc-optimize-stored-energy.py
stored_energy_weight = 20.0
max_it = 40

eqfam = EquilibriaFamily(eq)


# for i in range(0, 10):
objectives = [
    # AspectRatio(eq=eq, target=8, weight=100.0, jac_chunk_size=jac_chunk_size),
    obj_mirror_ratio,
    Volume(eq=eq, target=V0, weight=100.0, jac_chunk_size=jac_chunk_size),
    ObjectiveFromUserWithParams(
        objective_from_user_fun,
        eqfam[-1],
        target=0.0,
        weight=stored_energy_weight,
        grid=yancc_desc_grid,
        deriv_mode="rev",  # a scalar: see jac_chunk_size above
        # with the pressure residual it is a vector: a row at a time keeps the
        # backward pass through DESC's geometry and the ITG proxy small
        **(dict(jac_chunk_size=1) if pressure_consistency else {}),
    ),
]

# blocked: each objective's Jacobian uses its own jac_chunk_size
objective = ObjectiveFunction(objectives, deriv_mode="blocked")
objective.build(use_jit=False)
# DESC returns the iterates but not the objective at each; record the MaNTA
# term (objective_from_user_fun's 1/g) as it is evaluated
stored_energy_history = ObjectiveHistory(objectives[-1])

k = 5
# #
R_modes = np.vstack(
    (
        [0, 0, 0],
        eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :],
    )
)
Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]


constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    FixCurrent(eq=eq),  # zero current; eventually a real bootstrap
    FixBoundaryR(eq=eqfam[-1], modes=R_modes),
    FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
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
    xtol=1e-6,
    gtol=1e-6,
    **({"options": {"solve_options": tight_solve}} if pressure_consistency else {}),
    verbose=3,
    copy=True,
)

stored_energy_history.attach(info_out, key="stored_energy_history")
# objective_from_user_fun returns 1 / g, first when it returns the pressure
# residual beside it
_h = np.asarray(info_out["stored_energy_history"])
info_out["stored_energy_history"] = 1.0 / (_h[:, 0] if _h.ndim > 1 else _h)
print("stored energy per iteration:", info_out["stored_energy_history"])
np.save(eq_name + "_stored_energy_history.npy", info_out["stored_energy_history"])

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

fig, ax = plot_1d(eq_init, "iota", color="r", label="initial")
plot_1d(eq, "iota", ax=ax, color="b", label="final")
fig.savefig("figs/" + eq_name + "iota.pdf", dpi=500)

fig, ax = plt.subplots()
W = info_out["stored_energy_history"]
ax.plot(np.arange(len(W)), W, "o-")
ax.set_xlabel("iteration")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylabel("stored energy $G$")
ax.set_title("MaNTA objective over the optimisation")
fig.savefig("figs/" + eq_name + "stored_energy_history.png", dpi=200)


plt.show()
