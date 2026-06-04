import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"]= "FALSE"
    
from scipy.constants import mu_0
import MaNTA

from Objective2 import make_objective
from Stellarator2 import StellaratorTransport

import jax
import jax.numpy as jnp
import equinox as eqx 
import yancc

from yancc_wrapper2 import yancc_data

import desc
from desc import set_device
set_device("gpu")
import desc.io
from desc.equilibrium import Equilibrium, EquilibriaFamily
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
from desc.optimize._constraint_wrappers import ProximalProjection
from desc.profiles import SplineProfile
import matplotlib.pyplot as plt

# st_config = {
#     "SourceCenter": 0.1,
#     "SourceHeight": 40.0,
#     "SourceWidth": 0.8,
#     "EdgeTemperature":0.25,
#     "EdgeDensity": 0.0,
#     "n0": 1.0,
# }
st_config = {
    "sourcecenter": 0.2,
    "sourceheight": 6.0,
    "sourcewidth": 0.4,
    "edgetemperature":0.1,
    "edgedensity": 0.0,
    "n0": 1.0,
}
# st_config = {
#     "SourceCenter": 0.2,
#     "SourceHeight": 1000.0,
#     "SourceWidth": 0.8,
#     "EdgeTemperature":0.5,
#     "EdgeDensity": 0.0,
#     "n0": 0.5,
# }

rho_upper = 1.0
rtol = 1e-2
atol = 1e-3

# # %%
solver_config = {
    "OutputFilename": "stellarator_perturb_grad_test",
    "Polynomial_degree": 4,
    "Grid_size": 4,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1e-1,
    "initialTimestep": 1e-5,
    "MinStepSize": 1e-9, 
    "SteadyStateTolerance": 1e-4,
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
yancc_ntheta = 17
yancc_nzeta = 23

yancc_res = {"na":43,"nx":5}

# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired

pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)
print(pressure_rho)
#eq = desc.examples.get("ESTELL")
eq = desc.examples.get("W7-X")
# surf = eq.get_surface_at(rho=1)
eq.change_resolution(M=4, N=4,L_grid=len(points), M_grid=8, N_grid=8)# 
# eq = Equilibrium(M=4, N=4, Psi=0.5, surface=surf, pressure=desc_pressure)
eq = eq.solve(x_scale="ess")[0]

# eq = desc.io.load("eq_self_consistent_pressure.h5")
# desc_pressure = eq.get_profile('p')
eq_init = eq.copy()

V0 = eq.compute("V")["V"]
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
yancc_wrapper = yancc_data.from_eq(points, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta, **yancc_res)
# st = StellaratorTransport(config,yancc_wrapper=yancc_wrapper)
# st.run()
def make_tangent(params, idx, key='Rb_lmn'):
    def map_fn(path, val):
        keystr = jax.tree_util.keystr((path[0],)).lstrip(".").strip("[").strip("]").strip("'")
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
    
    pressure_error = manta_pressure - desc_pressure # not sure if the sign makes the difference here

    print("------------TOTAL PRESSURE ERROR-----------")
    print(pressure_error)
    print("-------------------------------------------")

    # optimization is easiest for least squares objectives, so instead of maximizing
    # stored energy we minimize 1/stored_energy^2 (the squaring happens later)
    return 1/stored_energy
yancc_desc_grid = yancc_wrapper.grid

# domain_boundary_rho = rho_from_normalized_volume(0.9)
domain_boundary_rho = 1.0
def pressure_constraint_fun(params):
    # function to fix dp/dr=0 at axis and p=0 at edge
    # can modify this for other BC (eg fix p at rho=0.8)
    p_l = params["p_l"]
    dp0 = desc_pressure(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
    p1 = desc_pressure(Grid(jnp.zeros((1, 3)).at[0, 0].set(domain_boundary_rho), jitable=True), p_l)
    return jnp.array([dp0, p1]).squeeze()


pressure_constraint_target = jnp.array([0.0, 0.0])
# pressure_constraint_target = jnp.array([0.0, st.getPressure([0.9])[0]])
# other objectives are non-dimensionalized, so weights should account for that
# and handle relative weighting, this will likely need trial and error
# pressure_error_weight = jnp.full(yancc_desc_grid.num_rho, 1e-5)
stored_energy_weight = 1.0
objective_from_user_weight = stored_energy_weight#jnp.append(stored_energy_weight)

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
    #FixCurrent(eq=eq),  # fix zero current, eventually should use real bootstrap
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
eqs = EquilibriaFamily(eq.copy())

# Sweep in the proximity of initial value
eq_init = eq.copy()
x_init = obj.x(eq_init)
G_init, grad_init = obj.value_and_grad(x_init)

G = [G_init[0]]
grads = [grad_init]
abs_step = 0.0
rel_step = 0.01
step_size = abs_step + rel_step * jnp.abs(x_init)

xi = x_init - step_size * grad_init / jnp.linalg.norm(grad_init)

niter = 8
for i in range(0, niter):

    eq_ = eq.copy()
    eqs.append(eq_)
    # ProximalProjection removes most of the fields so the index into the Rb_lmn field is this (I think?)
    g, gi = obj.value_and_grad(xi)
    G.append(g[0])
    if g[i + 1] > g[i]:
        print("decreasing step size")
        step_size *= 0.5
    else: 
        print("increasing step size")
        step_size *= 2.0
    xi = xi - step_size * gi/jnp.linalg.norm(gi)

    grads.append(jnp.linalg.norm(gi))
fd_grad = jnp.gradient(jnp.array(G))/jnp.linalg.norm(step_size)
fig,ax = plt.subplots() 
ax.plot(range(niter+1),grads,'ro', label="adjoints")
ax.plot(range(niter+1),fd_grad, 'bx', label="finite difference")
ax.legend()
ax.set_ylabel(fr"$dG/dR$")
ax.set_xlabel(fr"iteration")
fig.savefig("gradient_comparison.png", dpi=300)
fig, ax = plt.subplots()
ax.plot(range(niter+1), G)
ax.set_xlabel(fr"iteration")
ax.set_ylabel("G")
fig.savefig("G_sd.png", dpi=300)
plt.figure()
from desc.plotting import plot_comparison
fig, ax = plot_comparison(eqs=eqs)
fig.savefig("eq_sd.png", dpi=300)
plt.show()
