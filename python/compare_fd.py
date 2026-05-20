import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDG_USE_FILE_LOCKING"] = "FALSE"
from Stellarator2 import StellaratorTransport
from yancc_wrapper2 import yancc_data
import jax
import jax.numpy as jnp
import MaNTA
import desc
from desc.profiles import SplineProfile

st_config = {
    "SourceCenter": 0.2,
    "SourceHeight": 350.0,
    "SourceWidth": 0.4,
    "EdgeTemperature":0.2,
    "EdgeDensity": 0.0,
    "n0": 0.25,
}
# runner = MaNTA.Runner(st)

solver_config = {
    "OutputFilename": "stellarator_opt",
    "Polynomial_degree": 4,
    "Grid_size": 4,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-4,
    "MinStepSize": 1e-8, 
    "SteadyStateTolerance": 1e-2,
    "restart": True,
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
yancc_nzeta = 33

       
# to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

eq = desc.examples.get("W7-X")
eq.change_resolution(M=4, N=4,L_grid=len(points), M_grid=4, N_grid=4)
eq = eq.solve(x_scale="ess")[0]
# # eq = eq.solve(x_scale="ess")[0]
# # # store initial equilibrium for comparison later
# eq_init = eq.copy()
# yancc_grid = desc.grid.LinearGrid(rho=yancc_rho, M=eq_init.M_grid, N = eq_init.N_grid, NFP=eq_init.NFP)
# points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)
nx = 5
na = 43

yancc_wrapper = yancc_data.from_eq(points, eq=eq, nx=nx, na=na)
primals = yancc_wrapper.get_fields()
tangents = primals
grid = yancc_wrapper.grid
V0 = eq.compute("V")["V"]
def StellaratorFun(config, yin):
    st = StellaratorTransport(config, yancc_wrapper=yin)
    st.run()
    G, G_p = st.getAdjointGradients()

    pi = jnp.array(st.getPressure())

    return G[0], G_p, pi

abs_step = 1e-4
rel_step = 1e-2
(fields_in, vp, vpp) = primals
(v1, v2, v3)= tangents

yancc_wrapper = yancc_data.from_fields(fields_in, grid, vp, vpp)


x, unflatx = jax.flatten_util.ravel_pytree(fields_in)
v1, ______ = jax.flatten_util.ravel_pytree(v1)
# v2, ______ = jax.flatten_util.ravel_pytree(tree_dot[1])
# v3, ______ = jax.flatten_util.ravel_pytree(tree_dot[2])


# finite difference step size
fd_step = abs_step + rel_step * jnp.mean(jnp.abs(x))

# scale tangents to unit norm if nonzero
normv1 = jnp.linalg.norm(v1)

vh1 = jnp.pad(v1, (0, len(x)-len(v1)), mode='constant')
v1a = jnp.where(normv1 == 0, vh1, vh1 / normv1)
normv2 = jnp.linalg.norm(v2)
v2a = jnp.where(normv2 == 0, v2, v2 / normv2)
normv3 = jnp.linalg.norm(v3)
v3a = jnp.where(normv3 == 0, v3, v3 / normv3)
vcat = jnp.concatenate([vh1, v2a, v3a])
normv = jnp.linalg.norm(vcat)
# vh = jnp.where(normv == 0, vcat, vcat / normv)

steps = (abs_step + rel_step * jnp.mean(jnp.abs(x)) * v1a, 
            abs_step + rel_step * jnp.mean(jnp.abs(vp)) * v2a, 
            abs_step + rel_step * jnp.mean(jnp.abs(vpp)) * v3a)
yancc_wrapper_step = yancc_data.from_fields(unflatx(x + steps[0]), grid, vp + steps[1], vpp + steps[2])
G_step, _, _ = StellaratorFun(config, yancc_wrapper_step)



G, G_p, pi = StellaratorFun(config, yancc_wrapper)
primal_out = (G, pi)

G_p_field = G_p[:, :-2] # remove vprime component
G_p_vprime = G_p[:, -2] # extract vprime component
G_p_vpp = G_p[:, -1] # extract vpp component

# primal_out = _f_wrapped(*primals)
# flatten everything into 1D vectors for easier finite differences
# y, unflaty = jax.flatten_util.ravel_pytree(field_dot)


tangent_out = (G_step - G) / fd_step * normv
print(tangent_out)