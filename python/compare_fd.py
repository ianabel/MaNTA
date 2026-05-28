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
    "SourceHeight": 5.0,
    "SourceWidth": 0.4,
    "EdgeTemperature":0.2,
    "EdgeDensity": 0.0,
    "n0": 1.0,
}
# runner = MaNTA.Runner(st)

solver_config = {
    "OutputFilename": "stellarator_grad_test",
    "Polynomial_degree": 3,
    "Grid_size": 4,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-2,
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
yancc_ntheta = 17
yancc_nzeta = 23

       
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
na = 33

yancc_wrapper = yancc_data.from_eq(points, eq=eq, nx=nx, na=na, nz = yancc_nzeta, nt = yancc_ntheta)
primals = (yancc_wrapper.get_fields(), yancc_wrapper.grid)
# st = StellaratorTransport(config, yancc_wrapper = yancc_wrapper)
# st.run()

pert_keys = [
"B_sup_t", 
"B_sup_z", 
"B_sub_t", 
"B_sub_z", 
"Bmag" ,
"dBdt",
"dBdz",
"sqrtg",
]

rng_key = jax.random.PRNGKey(69)
def make_random_tangent(primal):
    global rng_key
    rng_key, key = jax.random.split(rng_key)
    if isinstance(primal, jax.Array):
        v = jax.random.normal(key, shape=primal.shape, dtype=primal.dtype)
        return v/jnp.linalg.norm(v)
    flat_primal, treedef = jax.tree_util.tree_flatten(primal)
    keys = jax.random.split(key, len(flat_primal))
    key_tree = jax.tree_util.tree_unflatten(treedef, keys)
    # key = jax.random.key(69)
    def map_fn(path, val, k):
        keystr = jax.tree_util.keystr((path[0],)).lstrip(".")
        if keystr in pert_keys:
            if jnp.isscalar(val) or jnp.isdtype(val, "integral"):
                return val
            else: 
                v = jax.random.normal(k, shape=val.shape, dtype=val.dtype)
                return v/jnp.linalg.norm(v) 
        else:
            return jnp.zeros_like(val)
    tangent_field = jax.tree.map_with_path(
        map_fn,
        primal, 
        key_tree
    )
    return tangent_field
tangents = primals
grid = yancc_wrapper.grid


V0 = eq.compute("V")["V"]

solver_config = {
    "OutputFilename": "stellarator_grad_test_t",
    "RestartFile": "stellarator_grad_test.restart.nc",
    "Polynomial_degree": 3,
    "Grid_size": 4,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-5,
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


def StellaratorFun(yin):
    st = StellaratorTransport(config, yancc_wrapper=yin)
    st.run()
    G, G_p = st.getAdjointGradients()

    pi = jnp.array(st.getPressure())

    return G[0], G_p, pi

G, G_p, p_i = StellaratorFun(yancc_wrapper)

def adj_jvp(primals, tangents):

    (fields, Vp, Vpp), grid = primals
    field_dot, Vp_dot, Vpp_dot= tangents
    
    yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, na = na, nx =nx)

    _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

    G_p_field = G_p[:, :-2] # remove field component
    G_p_vprime = G_p[:, -2] # extract vprime component
    G_p_vpp = G_p[:, -1] # extract vpp component
    G_p_padded = jnp.pad(G_p_field, pad_width=((0,0),(0,1)), mode='constant')

    G_p_unflattened = jax.vmap(unflatten_field)(jnp.float64(G_p_padded))
    def safe_mul(x, y):
        if x is None:
            return y
        if y is None:
            return x
        x_flat = jax.flatten_util.ravel_pytree(x)[0]
        y_flat = jax.flatten_util.ravel_pytree(y)[0]
        return jnp.dot(x_flat,y_flat) 

    # Apply tree_map
    # We need to treat None as a leaf
    result = jax.tree.map(safe_mul, G_p_unflattened, field_dot, is_leaf=lambda x: x is None)
    result_flattened, _ = jax.flatten_util.ravel_pytree(result)
    #now do vprime

    result_vprime = jnp.dot(G_p_vprime, Vp_dot)
    result_vpp = jnp.dot(G_p_vpp, Vpp_dot)

    return jnp.float32(jnp.sum(result_flattened)+result_vprime+result_vpp) 

# Finite difference objective for testing

abs_step = 1e-3
rel_step = 1e-3

solver_config = {
    "OutputFilename": "stellarator_grad_test_t",
    "RestartFile": "stellarator_grad_test.restart.nc",
    "Polynomial_degree": 3,
    "Grid_size": 4,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-3,
    "initialTimestep": 1e-5,
    "MinStepSize": 1e-8, 
    "SteadyStateTolerance": 1e-3,
    "restart": True,
    "solveAdjoint": True, 
    "zeroFlux": True,
}
config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}

def StellaratorFun(yin):
    st = StellaratorTransport(config, yancc_wrapper=yin)
    st.run()
    G, G_p = st.getAdjointGradients()

    pi = jnp.array(st.getPressure())

    return G[0], G_p, pi


def fd_jvp(primals, tangents):
    (fields_in, vp, vpp), grid = primals
    v1, v2, v3 = tangents

    x, unflatx = jax.flatten_util.ravel_pytree(fields_in)
    v1, ______ = jax.flatten_util.ravel_pytree(v1)
    # v2, ______ = jax.flatten_util.ravel_pytree(tree_dot[1])
    # v3, ______ = jax.flatten_util.ravel_pytree(tree_dot[2])
    
    # finite difference step size
    fd_step = abs_step + rel_step *(jnp.mean(jnp.abs(x)) + jnp.mean(jnp.abs(vp)) + jnp.mean(jnp.abs(vpp)))

    # scale tangents to unit norm if nonzero
    normv1 = jnp.linalg.norm(v1)
    
    vh1 = jnp.pad(v1, (0, len(x)-len(v1)), mode='constant')
    vcat = jnp.concatenate([vh1, v2, v3])
    normv = jnp.linalg.norm(vcat)
 
    # v1a = jnp.where(normv1 == 0, vh1, vh1 / normv)
    # normv2 = jnp.linalg.norm(v2)
    # v2a = jnp.where(normv2 == 0, v2, v2 / normv)
    # normv3 = jnp.linalg.norm(v3)
    # v3a = jnp.where(normv3 == 0, v3, v3 / normv)
   # vh = jnp.where(normv == 0, vcat, vcat / normv)

    steps = ((fd_step) * vh1, 
                (fd_step) * v2, 
                (fd_step) * v3)
    yancc_wrapper_step = yancc_data.from_fields(unflatx(x + steps[0]), grid, vp + steps[1], vpp + steps[2])
    G_step, _, _ = StellaratorFun(yancc_wrapper_step)

    # primal_out = _f_wrapped(*primals)
    # flatten everything into 1D vectors for easier finite differences
    # y, unflaty = jax.flatten_util.ravel_pytree(field_dot)

    tangent_out = (G_step - G) / fd_step * normv 
    #tangent_out = (, None)


    return jnp.float32(tangent_out)

nTangents = 3

for i in range(0,nTangents):
    t = tuple(make_random_tangent(p) for p in primals[0])
    fd = fd_jvp(primals, t)
    adj = adj_jvp(primals, t)

    print(f"Iteration {i} jvps:\n   adjoints={adj}, finite differences={fd}\n")
# fd2 = fd_jvp(primals, t2)

# adj2 = adj_jvp(primals, t2)
# print(f"Finite difference tangents {fd1}, {fd2}\n")