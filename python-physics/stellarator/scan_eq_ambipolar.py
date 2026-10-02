from stellarator_multichannel import StellaratorTransport
from objective import make_objective
from yancc_wrapper import yancc_data
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
from scipy.constants import mu_0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

fac = 1.0
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": fac * 1.6e-2,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": fac * 1.25,
    "NBIWidth": 0.25,
    "ECHCenter": 0.0,
    "ECHPower": fac * 8.76e-2,
    "ECHWidth": 0.26,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.2,
    "n0": 1.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useBatching": True,
}


eq_name = "qh"

rho_upper = 1.0
rtol = 1e-2
atol = 1e-10
npoints = 8
degree = 3

base = 1.5
tau = 0.5
#
# nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 4)

# nodes = np.concatenate(([0, 0.05, 0.1], nodes, [0.99, rho_upper]))
nodes = np.concatenate(([0, 0.05, 0.1], jnp.linspace(0.4, 0.99, npoints - 2)))

# nodes = np
print(nodes)
# # %%c
solver_config = {
    "OutputFilename": "stellarator_" + eq_name,
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "Grid_size": len(nodes) - 1,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-3,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 5e-5,
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "restart": False,
    "zeroFlux": True,
    "solveAdjoint": False,
    "SteadyStateSolver": "PseudoTransient",
    "SteadyStateDiagnostics": True,
    "SteadyStateStepDiagnostics": True,
    "PseudoTransientSERRate": 1.0,
    "PseudoTransientSERFloor": 2.0,
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
yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}

## to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)
eq = desc.examples.get("precise_QH")
eq = desc.compat.rescale(
    eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
)
# eq = desc.examples.get("W7-X")
# # # # Reduce the number of modes (not sure if this is a good thing to do)
# # #
# eq = desc.compat.rescale(eq, L=("R0", 10), B=("B0", 5.0))
# #
#
# eq = desc.examples.get("reactor_QA")
eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
eq.solve(x_scale="ess")[0]
# eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
# eq = desc.io.load("eq_qa.h5")

# eq = desc.io.load("eq_omnigenity.h5")

eq_init = eq.copy()
yancc_wrapper = yancc_data.from_eq(
    points, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)
# with jax.log_compiles(True):
st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
st.run()
#


def make_tangent(params, idx, key="Rb_lmn"):
    def map_fn(path, val):
        keystr = (
            jax.tree_util.keystr((path[0],))
            .lstrip(".")
            .strip("[")
            .strip("]")
            .strip("'")
        )
        if keystr == key:
            print(keystr)
            z = jnp.zeros_like(val)
            return z.at[idx].set(1.0)
        else:
            return jnp.zeros_like(val)

    tangent_field = jax.tree.map_with_path(
        map_fn,
        params,
    )
    return tangent_field


solver_config = {
    "OutputFilename": "stellarator_" + eq_name,
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "Grid_size": len(nodes) - 1,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-2,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 5e-5,
    "AggressiveTimesteps": False,
    "solveAdjoint": False,
    "WriteDatFile": True,
    "restart": True,
    "zeroFlux": True,
    "SteadyStateSolver": "Newton",
    "SteadyStateDiagnostics": True,
    "SteadyStateStepDiagnostics": True,
    "MaxRejectedSteps": 4,
    "PseudoTransientSERRate": 2.0,
}


config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}


manta_objective = make_objective(config, yancc_res=yancc_res)


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

    V = grid.compress(data["V(r)"])
    V_r = grid.compress(data["V_r(r)"])
    V_rr = grid.compress(data["V_rr(r)"])
    Vp = V_r / V[-1]
    Vpp = V_rr / V[-1]

    fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

    desc_pressure = grid.compress(data["p"], surface_label="rho")

    stored_energy, manta_pressure = manta_objective((fields, Vp, Vpp), grid)
    print("------------ STORED ENERGY ----------------")
    print(stored_energy)
    print("-------------------------------------------")

    # not sure if the sign makes the difference here
    pressure_error = manta_pressure - desc_pressure

    print("------------TOTAL PRESSURE ERROR-----------")
    print(pressure_error)
    print("-------------------------------------------")

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return 1 / stored_energy


yancc_desc_grid = yancc_wrapper.grid

# domain_boundary_rho = rho_from_normalized_volume(0.9)
domain_boundary_rho = 1.0


def pressure_constraint_fun(params):
    # function to fix dp/dr=0 at axis and p=0 at edge
    # can modify this for other BC (eg fix p at rho=0.8)
    p_l = params["p_l"]
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    p1 = desc_pressure(
        Grid(jnp.zeros((1, 3)).at[0, 0].set(domain_boundary_rho), jitable=True), p_l
    )
    return jnp.array([dp0, p1]).squeeze()


pressure_constraint_target = jnp.array([0.0, 0.0])
# pressure_constraint_target = jnp.array([0.0, st.getPressure([0.9])[0]])
# other objectives are non-dimensionalized, so weights should account for that
# and handle relative weighting, this will likely need trial and error
# pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 1e-5)
stored_energy_weight = 1.0
# jnp.append(stored_energy_weight)
objective_from_user_weight = stored_energy_weight

objectives = [
    ObjectiveFromUser(
        objective_from_user_fun,
        eq,
        target=0,
        weight=objective_from_user_weight,
        grid=yancc_desc_grid,
        deriv_mode="fwd",
    ),
]
constraints = [
    ForceBalance(eq=eq),  # J x B - grad(p) = 0
    # FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
    # Volume(eq=eq, target=V0), # fix volume of outer flux surface
    # FixPsi(eq=eq),  # fix total magnetic flux
    # LinearObjectiveFromUser(
    #     pressure_constraint_fun, eq, target=pressure_constraint_target
    # ),
]

# Set up ProximalProjection object
o1 = ObjectiveFunction(objectives)
o1.build(use_jit=False)
obj = ProximalProjection(o1, ObjectiveFunction(constraints), eq)
obj.build()
N = 1
M = 1
# Get the index of a mode
idx = eq.surface.R_basis.get_idx(L=0, N=N, M=M)
v0 = eq.Rb_lmn[idx]
print(v0)
eqs = EquilibriaFamily(eq.copy())
grads = []
G = []

# Sweep in the proximity of initial value
f = 0.2
delta = f * jnp.abs(v0)
start = v0 - delta
end = v0 + delta
# start = -0.04
# end = 0.02
sweep = jnp.linspace(start, end, 10)
df = sweep[1] - sweep[0]


eq_init = eq.copy()
x_init = obj.x(eq_init)


for i in range(0, len(sweep)):
    print(f"--------------------------\n Iteration {i} \n--------------------------\n")
    lp = len(eq.p_l)
    lc = len(eq.i_l)
    # t = jax.flatten_util.ravel_pytree(make_tangent(eq.params_dict, idx))[0]
    eq_ = eq.copy()
    eqs.append(eq_)
    # ProximalProjection removes most of the fields so the index into the Rb_lmn field is this (I think?)
    x_in = x_init.at[lp + lc + 1 + idx].set(sweep[i])
    # Set the tangent to 1 at the same index
    t = jnp.zeros(obj.dim_x)
    t1 = t.at[lp + lc + 1 + idx].set(1.0)

    # Compute value of objective
    # Compute gradient
    G.append(obj.compute_scaled(x_in)[0])

    grads.append(obj.jvp_scaled(t1, x_in)[0])

plt.rcParams.update({"font.family": "serif", "font.size": 12})

fig, ax = plt.subplots(1, 2)

fd_grad = jnp.gradient(jnp.array(G)) / df

ax[1].plot(sweep, fd_grad, "ro", label="FD")
ax[1].plot(sweep, grads, "bx", label="Adjoints")
ax[1].set_xlabel(rf"$R_{{0, {M}, {N}}}$")
ax[1].set_ylabel(rf"$dG/dR_{{0, {M}, {N}}}$")
ax[1].axvline(v0, color="k", linestyle="--")
ax[1].legend()
ax[0].plot(sweep, G, marker="o", color="red")
ax[0].set_xlabel(rf"$R_{{0, {M}, {N}}}$")
ax[0].set_ylabel("G")
ax[0].axvline(v0, color="k", linestyle="--")

for a in ax:
    a.set_box_aspect(1)
fig.tight_layout()
fig.set_figwidth(5.0)
fig.set_figheight(3.0)
fig.savefig(f"figs/sweep_G_{eq_name}_{M}_{N}.eps", dpi=500)
eqs.save("sweep.h5")
plt.figure()
fig, ax = plot_comparison(eqs=eqs[0:-1:4])
fig.savefig(f"figs/eqs_{eq_name}_{M}_{N}.png")
print("done")
