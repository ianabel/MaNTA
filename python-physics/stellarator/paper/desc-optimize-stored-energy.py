import os
import sys

sys.path.append("..")
# uncomment to run test flux
# os.environ["TEST_STELLARATOR"] = "true"
from stellarator_multichannel import StellaratorTransport
from objective import make_objective, ObjectiveFromUserWithParams, ObjectiveHistory
from itg_wrapper import make_itg_builder, itg_at, with_itg, itg_rho_at_nodes
from config import rescale_config, field_norm_from_eq

# os.environ.pop("TEST_STELLARATOR", None)
from yancc_wrapper import yancc_data
import matplotlib.pyplot as plt
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error, plot_1d
from desc.plotting import plot_comparison
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
import desc
from desc.grid import Grid, LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium, EquilibriaFamily
import yancc
import jax.numpy as jnp
import numpy as np
import jax
import manta as MaNTA


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

eq_name = "eq8"
fname = "stellarator_opt" + eq_name

#
# st_config = {
#     "ParticleSourceCenter": 0.0,
#     "ParticleSourceHeight": 1.25e-2,
#     "ParticleSourceWidth": 0.5,
#     "NBICenter": 0.0,
#     "NBIPower": 0.4,
#     "NBIWidth": 0.3,
#     "ECHCenter": 0.0,
#     "ECHPower": 0.04,
#     "ECHWidth": 0.26,
#     "EdgeTemperature": 0.2,
#     "EdgeDensity": 0.2,
#     "n0": 1.0,
#     "T0": 1.0,
#     "evolveDensity": True,
#     "useBatching": True,
# }
#


def make_config(Spi=300.0, Sn=1.2):
    return {
        "ParticleSourceCenter": 0.2,
        "ParticleSourceHeight": Sn,
        "ParticleSourceWidth": 0.6,
        "NBICenter": 0.0,
        "NBIPower": Spi,
        "NBIWidth": 0.15,
        "ECHCenter": 0.0,
        "ECHPower": 0.0,
        "ECHWidth": 0.1,
        "EdgeTemperature": 1.5,
        "EdgeDensity": 0.5,
        "ADCoefficient": 0.0,
        "n0": 3.0,
        "T0": 3.0,
        "FusionFactor": 1.0,
        "evolveDensity": True,
        # Er by current balance with a small diffusion rather than ambipolarity
        # node by node, so that one root is chosen where J() = 0 has several.
        # Sized from an algebraic run of this case (stellarator_opteq4.dat): with
        # J' ~ n D_i / T_i from its fluxes, D_E * |curvature of Er| / J' stays
        # under 5% of Er out to rho
        # grows four decades towards the edge while J' falls -- and the zero-flux
        # edge layer, sqrt(D_E / J') ~ 3e-3, then sits inside the last cell.
        # ErInertia keeps tau's stabilisation (an added diffusivity ~ ErInertia *
        # tau * h) below D_E on the widest cell; see DiffusiveErFlux.
        # "ErModel": "Diffusive",
        "reuseF1": False,
        "useShardMap": True,
        "ErDiffusion": 1e-6,
        "ErInertia": 5e-6,
        "useITGProxy": False,  # NN ITG heat flux (itg_wrapper.py); needs torch + ITG_MODEL_DIR
        "ITGScale": 1.0,  # gyro-Bohm is large in these units; scales the prediction
        "useBatching": True,
        "batch_size": 24,
        "nchunks": 2,
    }


# st_config and the solver settings are written in the default normalisation.
# True converts them (rescale_config) so only the units change, not the physics;
# False reads them as numbers in the new units. keep_physics = False

keep_physics = False
append = False
# the cold MaNTA solve before optimising. Appending usually skips it; then the
# consistency options need the loaded equilibrium to come from a consistency
# run, whose profiles they continue (see below)
initial_run = not append

# --- self-consistency with MaNTA, inside the one eq.optimize call -------------
# pressure_consistency: DESC's pressure becomes an even power series to rho^16
#   (smooth -- a spline's slope rule switches branch inside a DESC step), with
#   p(1) = 0, and a residual pulls it to MaNTA's total pressure at MaNTA's
#   nodes. Beyond MaNTA's edge it is pulled to a straight fall from MaNTA's last
#   node to zero at rho = 1, which nothing else would pin: unconstrained, the
#   profile dipped negative there and the optimiser deepened the dip.
# bootstrap_consistency: DESC's current becomes free -- a full power series with
#   c_0 = c_1 = 0, as DESC's bootstrap tutorial recommends -- started consistent
#   with MaNTA's <J.B> by Landreman & Catto (2012) eq. C3 steps, and a residual
#   pulls <J.B> to MaNTA's at its nodes. Needs evolveDensity.
# MaNTA's pressure and <J.B> carry their derivatives (PressureObjectives,
# BootstrapObjectives in objective.py), so DESC's model sees how a boundary
# step moves them.
pressure_consistency = False
bootstrap_consistency = False
consistency = pressure_consistency or bootstrap_consistency
# weights relative to stored_energy_weight, on mismatches normalised by MaNTA's
# peak pressure and peak |<J.B>|
pressure_weight = 1.0
bootstrap_weight = 1.0
# L = M = N for DESC. At 4 the interior was poorly determined: a boundary change
# of 7e-6 moved it by 1.4% and G by 4%, which stalled the trust region, and
# <J.B> at zero current was numerical noise the size of the bootstrap current.
# 8 fixes both; None keeps the M = N = 4 below.
desc_resolution = 8 if consistency else None
# every equilibrium solve, including each trial point's inside the proximal
# projection, converged to this: at DESC's default tolerance the interior
# depended on the path taken to a point, and G with it
tight_solve = dict(ftol=1e-8, xtol=1e-8, gtol=1e-8, maxiter=100)

rho_upper = 0.9
rtol = 1e-2
atol = 1e-5
# nodes = [0.0, 0.4, 0.6, 0.75, 0.95, 1.0]
npoints = 9
degree = 4
# base = 1.5
# tau = 0.5
#
base = 1.8
tau = 0.5
#
# nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 2)

# nodes = np.concatenate(
#     ([0, 0.1], nodes, [rho_upper])
# )  # nodes = np.concatenate(([0, 0.1], nodes, [0.98,1]))
nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 5)

# nodes = np.concatenate(([0, 0.1, 0.15, 0.2, 0.25], nodes, [rho_upper]))

nodes = np.linspace(0.05, 0.99 * rho_upper, npoints - 1)
nodes = np.concatenate(([0.0], nodes, [rho_upper]))
# nodes = np.linspace(0, rho_upper, npoints + 1)
print(nodes)
# # %%
solver_config = {
    "OutputFilename": fname + "_acc",
    "PolynomialDegree": degree,
    "GridPoints": nodes,
    "GridSize": len(nodes) - 1,
    "tau": tau,
    "LowerBoundary": 0.0,
    "UpperBoundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 0.1,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-5,
    # "SteadyStateTest": "Residual",  # its tolerance is in ||F||; unverified under NewtonStep
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "SteadyStateDiagnostics": True,
    "SteadyStateStepDiagnostics": True,
    # "tauScaling": "Diffusive",
    # "tauUpdate": "ContinuationStep",
    "NewtonJacobianReuse": 4,
    "NewtonMaxIterations": 25,
    "MaxRejectedSteps": 10,
    "restart": False,
    "zeroFlux": True,
    "solveAdjoint": False,
    "PseudoTransientSERRate": 0.5,
    "PseudoTransientSERFloor": 2.0,
}


st_config = make_config(250, 1.2)
config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}
points = MaNTA.getNodes(
    solver_config["GridPoints"],
    solver_config["PolynomialDegree"],
)


yancc_rho = jnp.array(points)
yancc_ntheta = 27
yancc_nzeta = 33

yancc_res = {"na": 65, "nx": 7}


# %%

pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)
#

# # create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# # default is zero pressure and zero current
# this is usually all you need to solve a fixed boundary equilibrium
#
if not append:
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
    eq = eq.solve(x_scale="ess")[0]
    eqs = EquilibriaFamily(eq)
# # desc_pressure = eq.get_profile('p')
#
# surf = FourierRZToroidalSurface.from_qp_model(
#     major_radius=1,
#     aspect_ratio=8,
#     elongation=3,
#     mirror_ratio=0.2,
#     torsion=0.1,
#     NFP=2,
#     sym=True,
# )
# # this value of Psi gives an average |B| on axis of about 1 T
# # the Equilibrium class defaults to vacuum pressure and current profiles
# eq = Equilibrium(Psi=3e-2, M=4, N=4, surface=surf, pressure=desc_pressure)
#
# # eq = desc.compat.rescale(eq, L=("R0", 10), B=("B0", 5.0))
# eq = desc.compat.rescale(
#     eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
# )
#
# eq = eq.solve(x_scale="ess")[0]
# eqs = EquilibriaFamily(eq)
#
else:
    eqs = desc.io.load(eq_name + "_all_equilibria.h5")
    eq = eqs[-1].copy()
# Appending onto a consistency run: its pressure (the even power series) and
# its current (the free one) are what DESC was tracking MaNTA's with, so they
# are kept -- replacing them would throw away what the run reached -- and the
# start needs no MaNTA solve.
continue_pressure = append and pressure_consistency and isinstance(eq.pressure, PowerSeriesProfile)
continue_current = (
    append and bootstrap_consistency and isinstance(eq.current, PowerSeriesProfile)
    and bool(np.any(np.asarray(eq.c_l) != 0))
)
if not continue_pressure:
    eq.pressure = desc_pressure
if desc_resolution is not None:
    eq.change_resolution(
        L=desc_resolution, M=desc_resolution, N=desc_resolution,
        L_grid=2 * desc_resolution, M_grid=2 * desc_resolution, N_grid=2 * desc_resolution,
    )
if bootstrap_consistency and not continue_current:
    # a current DESC can move: the full power series to rho^L, zero to start
    eq.current = PowerSeriesProfile(np.zeros(eq.L + 1), sym=False)
if desc_resolution is not None or bootstrap_consistency:
    eq.solve(objective="force", optimizer="lsq-exact", verbose=1, **{**tight_solve, "maxiter": 500})

eq_init = eq.copy()

V0 = eqs[0].compute("V")["V"]
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)


def cheb_nodes(nCells):
    nodes = np.ndarray((nCells + 1,))
    for i in range(0, len(nodes)):
        nodes[nCells - i] = 0.5 * (1 + np.cos(i * np.pi / nCells))
    return nodes


yancc_wrapper = yancc_data.from_eq(
    yancc_rho, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)

# Normalise lengths and fields to the initial equilibrium's a and <|B|>_rms,
# fixed from here on: every run below, and the objective, use these numbers
normalization = {"n0": 1e19, **field_norm_from_eq(eq_init)}
print(f"Normalization from the initial equilibrium: {normalization}")


def normalized(config):
    if keep_physics:
        return rescale_config(config, normalization)
    return {**config, "Normalization": normalization}


config = normalized(config)
## %%

#
# # # %%
# with jax.log_compiles(True):
#
itg_builder = make_itg_builder(config["Stellarator"], eq_init, rho_upper, rho=itg_rho_at_nodes(points))
st = StellaratorTransport(
    config, yancc_wrapper=yancc_wrapper, itg=itg_at(itg_builder, eq_init.params_dict)
)
if initial_run:
    st.run()


def current_from_JdotB(e, jb, rho):
    """Current profile coefficients [c_0, ..., c_L] (DESC's 2 pi / mu_0 I, A)
    whose parallel current is <J.B> = jb at rho: Landreman & Catto (2012) eq.
    C3, as DESC's "current Redl" uses it with the Redl <J.B>, fitted with
    c_0 = c_1 = 0. Its perpendicular term reads the current itself, so it is
    iterated with equilibrium solves."""
    g = LinearGrid(rho=rho, M=e.M_grid, N=e.N_grid, NFP=e.NFP, sym=e.sym)
    d = e.compute(["psi_r", "p_r", "current", "<|B|^2>"], grid=g)
    c = lambda k: np.asarray(g.compress(d[k]))
    mu_0 = 4e-7 * np.pi
    current_r = (
        -mu_0 * c("current") / c("<|B|^2>") * c("p_r")
        + 2 * np.pi * c("psi_r") * jb / c("<|B|^2>")
    )
    XX = np.vander(rho, e.L + 1)[:, :-1]
    c_r = np.pad(np.linalg.lstsq(XX, current_r, rcond=None)[0], (0, 1))
    current = np.polyval(np.polyint(c_r), rho)
    XX = np.fliplr(np.vander(rho, e.L + 1)[:, :-2])  # rho^2 .. rho^L
    return np.pad(np.linalg.lstsq(XX, current, rcond=None)[0], (2, 0))


def JdotB_at(e, rho):
    """DESC's <J.B> at rho: a surface average, so on a grid with angles."""
    g = LinearGrid(rho=rho, M=e.M_grid, N=e.N_grid, NFP=e.NFP, sym=e.sym)
    return np.asarray(g.compress(e.compute("<J*B>", grid=g)["<J*B>"]))


points_np = np.asarray(points)
if consistency:
    if (
        (pressure_consistency and not continue_pressure)
        or (bootstrap_consistency and not continue_current)
    ) and not initial_run:
        raise ValueError(
            "the consistency options start DESC's profiles from MaNTA's, which needs "
            "the initial MaNTA solve: set initial_run = True, or append onto an "
            "equilibrium from a consistency run, whose profiles are continued"
        )
    if pressure_consistency:
        # between MaNTA's last node and rho = 1
        RHO_EDGE = np.linspace(points_np[-1], 1.0, 6)[1:-1]
        GRID_EDGE = Grid(np.stack([RHO_EDGE, 0 * RHO_EDGE, 0 * RHO_EDGE], 1), jitable=True)
    if continue_pressure:
        desc_pressure = eq.pressure
        p_ref = float(np.max(np.asarray(desc_pressure(
            Grid(np.stack([points_np, 0 * points_np, 0 * points_np], 1))
        ))))
        print("pressure: continuing the loaded equilibrium's power series")
    elif pressure_consistency:
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
    if continue_current:
        j_ref = float(np.abs(JdotB_at(eq, points_np)).max())
        print("current: continuing the loaded equilibrium's current profile")
    elif bootstrap_consistency:
        # Landreman & Catto steps from MaNTA's <J.B> at the starting point
        jb0 = np.asarray(st.getBootstrapCurrent())
        j_ref = float(np.abs(jb0).max())
        for _k in range(3):
            eq.c_l = current_from_JdotB(eq, jb0, points_np)
            eq.solve(objective="force", optimizer="lsq-exact", verbose=1, **{**tight_solve, "maxiter": 500})
            _jd = JdotB_at(eq, points_np)
            print(f"current step {_k}: L2 <J.B> error {np.linalg.norm(_jd - jb0) / np.linalg.norm(jb0):.3e}")
    if itg_builder is not None:
        # the ITG proxy holds the profiles' layout from the equilibrium it was
        # built on, and they have just changed
        itg_builder = make_itg_builder(
            config["Stellarator"], eq, rho_upper, rho=itg_rho_at_nodes(points)
        )
# Sns = [1.1, 0.9]
# Spis = [350.0, 400.0]

# if not append:
#     st.run()

#     # need to get to lower collisionality like this


#     for Spi, Sn in zip(Spis, Sns):
#         _st_config = make_config(Spi, Sn)
#         _solver_config = solver_config.copy()
#         _solver_config["restart"] = True
#         _solver_config["initialTimestep"] = 1e6

#         _solver_config["SteadyStateSolver"] =  "Newton"

#         _config = {
#             "Stellarator": _st_config,
#             "Solver": _solver_config,
#         }
#         _config  = normalized(_config)
#         st = StellaratorTransport(
#             _config,
#             yancc_wrapper=yancc_wrapper,
#             itg=itg_at(itg_builder, eq_init.params_dict),
#         )
#         st.run()

# st_config = make_config()
# config = {
#     "Stellarator": st_config,
#     "Solver": solver_config,
# }
# config = normalized(config)
# st = StellaratorTransport(
#     config,
#     yancc_wrapper=yancc_wrapper,
#     itg=itg_at(itg_builder, eq_init.params_dict),
# )
# st.run()


#
# # %%
#
# plt.plot(points, 2.0 / 3.0 * st.InitialValue(0, points) / yancc_wrapper.Vp)  # %%
#
#
# solver_config = {
#     "OutputFilename": fname,
#     "RestartFile": "stellarator_opt_amb0.restart.nc",
#     "PolynomialDegree": degree,
#     "GridPoints": nodes,
#     "GridSize": len(nodes) - 1,
#     "tau": tau,
#     "LowerBoundary": 0.0,
#     "UpperBoundary": rho_upper,
#     "Relative_tolerance": rtol,
#     "Absolute_tolerance": [atol],
#     "delta_t": 1.0,
#     "initialTimestep": 1e-2,
#     "MinStepSize": 1e-10,
#     "restart": True,
#     "zeroFlux": True,
#     "SteadyStateTolerance": 1e-3,
#     "PseudoTransientSERRate": 2.0,
# }
#
# config = {
#     "Stellarator": st_config,
#     "Solver": solver_config,
# }
# pi = []
#
solver_config = {
    "OutputFilename": fname,
    "PolynomialDegree": degree,
    "RestartFile": fname + "_acc.restart.nc",
    "GridPoints": nodes,
    "GridSize": len(nodes) - 1,
    "tau": tau,
    "LowerBoundary": 0.0,
    "UpperBoundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 100.0,
    "initialTimestep": 1e4,
    "MinStepSize": 1e-10,
    "WriteDatFile": True,
    "NewtonJacobianReuse": 5,
    "NewtonMaxIterations": 15,
    "restart": True,
    "zeroFlux": True,
    # "tauScaling": "Constant",
    # "tauUpdate": "ContinuationStep",
    "SteadyStateTolerance": 2e-4,
    "SteadyStateStepTolerance": 0.33,
    # "SteadyStateTest": "Residual",  # its tolerance is in ||F||; unverified under NewtonStep
    "SteadyStateSolver": "Newton",
    "SteadyStateDiagnostics": True,
    "MaxRejectedSteps": 0,
    "MaxContinuationSteps": 30,
    "SteadyStateStepDiagnostics": True,
    # "ObjectiveDecreaseTolerance": 0.01,
    "PseudoTransientSERFloor": 2.0,
}
config = normalized(
    {
        "Stellarator": st_config,
        "Solver": solver_config,
        "PressureObjectives": pressure_consistency,
        "BootstrapObjectives": bootstrap_consistency,
    }
)


# fig, ax = plt.subplots()
# eq2 = eq.copy()
# fam2 = EquilibriaFamily(eq2)
# niters = 3
# for k in range(niters):
#     eq2 = eq2.copy()
#
#     fig, ax = plot_1d(eq2, "pressure", label="DESC " + str(k), ax=ax)
#
#     yancc_wrapper = yancc_data.from_eq(
#         points, eq=eq2, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
#     )
#     pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
#     st = StellaratorTransport(
#         config, yancc_wrapper=yancc_wrapper, itg=itg_at(itg_builder, eq2.params_dict)
#     )
#     st.run()
#
#     pi = st.getPressure()
#     pi_manta = jnp.concatenate([jnp.array([pi[0]]), pi, jnp.zeros(1)])
#     ax.plot(pressure_rho, pi_manta, label="MANTA" + str(k))
#     eq2.pressure = SplineProfile(pi_manta, pressure_rho)
#     # fit the current profile to a power series, with c_0=c_1=0
#     # XX = np.fliplr(np.vander(rho, eq2.L + 1)[:, :-2])
#     # eq2.c_l = np.pad(np.linalg.lstsq(XX, current, rcond=None)[0], (2, 0))
#     # re-solve the equilibrium
#     eq2, _ = eq2.solve(objective="force", optimizer="lsq-exact", verbose=3)
#     fam2.append(eq2)
#     eqs.append(eq2)
# eq_self_consistent = eq2.copy()
# # #
# ax.legend()
# # fig.savefig("initial_self_consistent_pressure.png")
# # # # %%
# # #
# # plot_comparison(eqs=[eq_init, eq2], labels=["Initial", "self-consistent"])
# #
# # plt.show()
# eq = eq2.copy()
# #
# # %%
#

manta_objective = make_objective(
    config, yancc_res=yancc_res, normalization=normalization
)

# def manta_yancc_fun(fields, grid, Vprime):

#     stored_energy, pressure = Objective(fields, grid, Vprime)

#     return stored_energy, pressure


def objective_from_user_fun(grid, data, params=None, desc_jb=None):
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

    out = manta_objective(with_itg((fields, Vp, Vpp), itg_builder, params), grid)
    g, manta_pressure = out[0], out[1]

    print("------------ STORED ENERGY ----------------")
    print(g)
    print("-------------------------------------------")

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    if not consistency:
        return 1 / g
    # 1/g first, so ObjectiveHistory's first column is still 1/g
    residuals = [jnp.ravel(1 / g)]
    errors = {}
    if pressure_consistency:
        p_edge_target = manta_pressure[-1] * (1.0 - RHO_EDGE) / (1.0 - points_np[-1])
        p_edge = desc_pressure(GRID_EDGE, params["p_l"])
        r_p = jnp.concatenate([manta_pressure - desc_p, p_edge_target - p_edge]) / p_ref
        residuals.append(pressure_weight * r_p)
        errors["pressure"] = (manta_pressure, desc_p)
    if bootstrap_consistency:
        manta_jb = out[2]
        residuals.append(bootstrap_weight * (manta_jb - desc_jb) / j_ref)
        errors["<J.B>"] = (manta_jb, desc_jb)
    concrete = not any(
        isinstance(v, jax.core.Tracer) for pair in errors.values() for v in pair
    )
    if concrete:
        print(
            "consistency: "
            + "  ".join(
                f"L2 {k} error {np.linalg.norm(np.asarray(d) - np.asarray(m)) / np.linalg.norm(np.asarray(m)):.3e}"
                for k, (m, d) in errors.items()
            )
        )
    return jnp.concatenate(residuals)


def objective_from_user_fun_bootstrap(grid, data, params=None):
    # objective_from_user_fun with DESC's <J.B> beside it. DESC takes the data
    # keys a user function needs from its source, so they are all named here --
    # objective_from_user_fun's, and <J*B> -- and <J*B> is computed only when
    # this is the function in use.
    (data["B^theta"], data["B^zeta"], data["B_theta"], data["B_zeta"], data["|B|"],
     data["|B|_t"], data["|B|_z"], data["sqrt(g)"], data["Psi"], data["a"], data["R0"],
     data["iota"], data["V(r)"], data["V_r(r)"], data["V_rr(r)"], data["p"])
    desc_jb = grid.compress(data["<J*B>"], surface_label="rho")
    return objective_from_user_fun(grid, data, params, desc_jb=desc_jb)


# The surface resolution yancc solves on during the optimisation, which has to
# be the one the standalone run uses (yancc_data.from_eq with yancc_ntheta and
# yancc_nzeta). Built from M_grid/N_grid instead, this was 17 x 17 for an
# M = N = 4 equilibrium against 27 x 33: a different, cheaper objective. With
# point counts, LinearGrid places its nodes exactly where Field.from_desc does
# (theta on [0, 2 pi), zeta on [0, 2 pi / NFP), no endpoint).
yancc_desc_grid = desc.grid.LinearGrid(
    rho=yancc_rho, theta=yancc_ntheta, zeta=yancc_nzeta, NFP=eq.NFP
)
# yancc_desc_grid = yancc_wrapper.grid

# domainBoundary_rho = rho_from_normalized_volume(0.9)
domainBoundary_rho = 1.0


def pressure_constraint_fun(params):
    # function to fix dp/dr=0 at axis and p=0 at edge
    # can modify this for other BC (eg fix p at rho=0.8)
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


# Mirror ratio, manually computed from "|B|"
def fun_mirror_ratio(grid, data):
    # alternatively, "mirror ratio" is something that can be computed in the data index
    # directly (see List of Variables docs), so can replace entirety of the above function code with this return statement
    return grid.compress(data["mirror ratio"])
    # or can just use GenericObjective(f="mirror ratio", thing=eq) or the already-existing objective, MirrorRatio(eq)


obj_mirror_ratio = ObjectiveFromUser(
    fun=fun_mirror_ratio,
    thing=eq,
    grid=yancc_desc_grid,
    bounds=(0.0, 0.2),
    weight=10.0,
    name="my mirror ratio",
)


# pressure_constraint_target = jnp.array([0.0, st.getPressure([0.9])[0]])

# %%

# other objectives are non-dimensionalized, so weights should account for that
# and handle relative weighting, this will likely need trial and error
# pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 1e-5)

# pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 2e-6)
stored_energy_weight = 20.0
# jnp.append(stored_energy_weight)
objective_from_user_weight = stored_energy_weight
fig, ax = plt.subplots()
max_it = 5

eqfam = EquilibriaFamily(eq)
# ks = [1, 2, eq.M + 1]
# for k in ks:
# print("\n==================================")
# print("Optimizing boundary modes M,N <= {}".format(k))
# print("====================================")
#
objectives = [
    # AspectRatio(eq=eq, bounds=(0, 8), weight=10.0),
    obj_mirror_ratio,
    Volume(eq=eq, target=V0, weight=100.0),
    # RotationalTransform(eq=eq, target=0.42, weight=10),
    (
        ObjectiveFromUserWithParams
        if itg_builder is not None or pressure_consistency
        else ObjectiveFromUser
    )(
        objective_from_user_fun_bootstrap if bootstrap_consistency else objective_from_user_fun,
        eqfam[-1],
        target=0.0,
        weight=objective_from_user_weight,
        grid=yancc_desc_grid,
        # need this assuming manta only has vjp, if using jvp switch to fwd.
        # With the consistency residuals the term is a vector: reverse mode a
        # row at a time keeps the backward pass through DESC's geometry small
        **(dict(deriv_mode="rev", jac_chunk_size=1) if consistency else dict(deriv_mode="fwd")),
    ),
]

# blocked: each objective's Jacobian uses its own jac_chunk_size
objective = ObjectiveFunction(objectives, deriv_mode="blocked" if consistency else "auto")
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

#
# def mirrorRatio(params):
#     """Custom linear function to constrain the mirror ratio on axis."""
#     B_lm = params["B_lm"]
#     f = jnp.array(
#         [
#             B_lm[0] -Bavg * (1 - delta),  # B_min on axis
#             Bavg * (1 + delta) - B_lm[-1],  # B_max on axis
#         ]
#     )
#     return f
# %%
constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    # fix zero current, or with bootstrap_consistency only c_0 = c_1 = 0
    FixCurrent(eq=eq, indices=[0, 1]) if bootstrap_consistency else FixCurrent(eq=eq),
    FixBoundaryR(eq=eqfam[-1], modes=R_modes),
    FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
    FixPsi(eq=eq),  # fix total magnetic flux
    LinearObjectiveFromUser(
        pressure_constraint_fun, eq, target=pressure_constraint_target
    ),
]
# print(objective.jac_scaled(objective.x(eq)))
eq, info_out = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    x_scale="ess",
    maxiter=max_it,
    ftol=1e-4,  # stopping tolerance on the function value
    xtol=1e-5,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    # options={
    #     # "max_trust_radius": 1.0,
    #     "initial_trust_radius": 0.01,
    #     # "perturb_options": {"order": 2, "verbose": 3},  # use 2nd-order perturbations
    #     #     # "solve_options": {
    #     #     #     "ftol": 5e-3,
    #     #     #     "xtol": 1e-6,
    #     #     #     "gtol": 1e-6,
    #     #     #     "verbose": 3,
    #     # },  # for equilibrium subproblem
    # },
    **({"options": {"solve_options": tight_solve}} if consistency else {}),
    verbose=3,
    copy=True,
)
stored_energy_history.attach(info_out, key="stored_energy_history")
# objective_from_user_fun returns 1 / g, first when it returns the consistency
# residuals beside it
_h = np.asarray(info_out["stored_energy_history"])
info_out["stored_energy_history"] = 1.0 / (_h[:, 0] if _h.ndim > 1 else _h)
print("stored energy per iteration:", info_out["stored_energy_history"])
np.save(eq_name + "_stored_energy_history.npy", info_out["stored_energy_history"])

# %%

# %%
eqfam.append(eq.copy())

eqs.append(eq.copy())
fig, ax = plot_comparison(eqs=[eq_init, eq], labels=["Initial", "optimized"])

fig.savefig("figs/" + eq_name + "comparison")

eq_sc = eqfam[-1].copy()
# final self consistency
# niters = 2
# fig, ax = plt.subplots()
# for k in range(niters):
#     fig, ax = plot_1d(eq_sc, "pressure", label="DESC " + str(k), ax=ax)

#     yancc_wrapper = yancc_data.from_eq(
#         points, eq=eq_sc, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
#     )

#     st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
#     st.run()

#     pi = st.getPressure()

#     pi_manta = jnp.concatenate([jnp.array([pi[0]]), pi, jnp.zeros(1)])
#     ax.plot(pressure_rho, pi_manta, label="MANTA" + str(k))
#     eq_sc.pressure = SplineProfile(pi_manta, pressure_rho)
#     # fit the current profile to a power series, with c_0=c_1=0

#     # XX = np.fliplr(np.vander(rho, eq2.L + 1)[:, :-2])ordering differs from Braginskii's mostly in permitting a mean flow ordered at the sound speed (e.g., allowing M~ or >1), which the standard Braginskii subsonic-flow ordering does not accommodate. The orderings otherwise differ in the collisional stress closures (e.g. the parallel/gyroviscous stress), which enter terms not retained in this study. The global, long-wavelength (k_\parallel= or ~0) KH and interchange modes examined here are MHD-scale and not governed by the fine-scale velocity-space resonances that demand a kinetic treatment, so a drift-reduced fluid description is appropriate for them. A fully kinetic description would be required for sub-ion-Larmor-scale turbulent transport in reactor-relevant, lower-collisionality regimes, which is outside the present scope.
#     # eq2.c_l = np.pad(np.linalg.lstsq(XX, current, rcond=None)[0], (2, 0))
#     # re-solve the equilibrium
#     eq_sc, _ = eq_sc.solve(
#         objective="force", x_scale="ess", optimizer="lsq-exact", verbose=3
#     )

# eqfam.append(eq_sc)
# ax.legend()
# fig.savefig("figs/" + eq_name + "final_pressure_comparison.png")
eqs.save(eq_name + "_all_equilibria.h5")
# %%

# %%

fig, ax = plot_comparison(eqs=[eqs[0], eq], labels=["Initial", "optimized"])
fig.savefig("figs/" + eq_name + "final_comparison.png")
fig, ax = plot_boundaries(eqs=[eqs[0], eq], labels=["Initial", "optimized"])
fig.savefig("figs/" + eq_name + "final_comparison_boundary.png")
# %%
# eq = desc.io.load("../python/eq2optimized_equilibrium.h5")#desc.examples.get("ESTELL")
# plot_boozer_surface(eq_init, fieldlines=8)
fig, ax = plot_boozer_surface(eqfam[-1], fieldlines=8)
fig.savefig("figs/" + eq_name + "final_boozer_surface.png")
fig, ax = plot_qs_error(eqfam[-1])
fig.savefig("figs/" + eq_name + "final_qs_error.png")

fig, ax = plt.subplots()
W = info_out["stored_energy_history"]
ax.plot(np.arange(len(W)), W, "o-")
ax.set_xlabel("iteration")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylabel("stored energy $G$")
ax.set_title("MaNTA objective over the optimisation")
fig.savefig("figs/" + eq_name + "stored_energy_history.png", dpi=200)
plt.show()

#
