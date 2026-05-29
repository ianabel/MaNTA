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
    "Grid_size": 6,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-2,
    "MinStepSize": 1e-8, 
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

#pert_keys = [
#"B_sup_t", 
#"B_sup_z", 
#"B_sub_t", 
#"B_sub_z", 
#"Bmag" ,
#"dBdt",
#"dBdz",
#"sqrtg",
#]
#
#pert_keys = [
#    "Rb_lmn",
#    "Zb_lmn",
#    "Ra_n",
#    "Za_n",
#]
pert_keys = [
    "R_lmn",
    "Z_lmn",
    "L_lmn"
]
rng_key = jax.random.PRNGKey(10)
def make_random_tangent(primal):
    global rng_key
    rng_key, key = jax.random.split(rng_key)
    if isinstance(primal, jax.Array):
        v = jax.random.normal(key, shape=primal.shape, dtype=primal.dtype)
        return v/jnp.linalg.norm(v)
    flat_primal, treedef = jax.tree_util.tree_flatten(primal)
    keys = jax.random.split(key, len(flat_primal))
    key_tree = jax.tree_util.tree_unflatten(treedef, keys)
    # for perturbing pytrees
    def map_fn(path, val, k):
        keystr = jax.tree_util.keystr((path[0],)).lstrip(".").strip("[").strip("]").strip("'")
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
step_size = 1e-3
def perturb_eq(eq, step_size=step_size, pert_keys=pert_keys):
    params = eq.params_dict
    deltas = {}
    for key in pert_keys:
        deltas[key] = step_size * jnp.linalg.norm(params[key]) * make_random_tangent(params[key])
    # t = make_random_tangent(params)
    # params_flat, unflat = jax.flatten_util.ravel_pytree(params)
    # params_new = unflat( params_flat + step_size * jax.flatten_util.ravel_pytree(t)[0] )
    return eq.copy().perturb(deltas)
perturb_eq(eq)
tangents = primals
grid = yancc_wrapper.grid

V0 = eq.compute("V")["V"]
#
#solver_config = {
#    "OutputFilename": "stellarator_grad_test_t",
#    "RestartFile": "stellarator_grad_test.restart.nc",
#    "Polynomial_degree": 4,
#    "Grid_size": 4,
#    "tau": 100.0, 
#    "Lower_boundary": 0.0,
#    "Upper_boundary": 1.0,
#    "Relative_tolerance": 0.01,
#    "Absolute_tolerance": [1e-3],
#    "delta_t": 1e-2,
#    "initialTimestep": 1e-4,
#    "MinStepSize": 1e-8, 
#    "SteadyStateTolerance": 1e-3,
#    "restart": True,
#    "solveAdjoint": True, 
#    "zeroFlux": True,
#}
#config = {
#    "Stellarator": st_config,
#    "Solver": solver_config,
#}
#

def StellaratorFun(yin):
    st = StellaratorTransport(config, yancc_wrapper=yin)
    st.run()
    G, G_p = st.getAdjointGradients()

    pi = jnp.array(st.getPressure())

    return G[0], G_p, pi

G, G_p, p_i = StellaratorFun(yancc_wrapper)

def adj_jvp(primals, tangents):

    (fields, Vp, Vpp), grid = primals
    field_dot, Vp_dot, Vpp_dot = tangents
    
    _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

    G_p_field  = G_p[:, :-2] # remove field component
    G_p_vprime = G_p[:, -2] # extract vprime component
    G_p_vpp    = G_p[:, -1] # extract vpp component
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

abs_step = 1e-2
rel_step = 0.0

solver_config = {
    "OutputFilename": "stellarator_grad_test_t",
    "RestartFile": "stellarator_grad_test.restart.nc",
    "Polynomial_degree": 6,
    "Grid_size": 3,
    "tau": 100.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "Absolute_tolerance": [1e-3],
    "delta_t": 1e-2,
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
    x, grid = primals
    v = tangents

    x, unflatx = jax.flatten_util.ravel_pytree(x)
    v, _______ = jax.flatten_util.ravel_pytree(tangents)
    
    # finite difference step size
    fd_step = abs_step + rel_step *(jnp.mean(jnp.abs(x)))
    # scale tangents to unit norm if nonzero
 
    normv = jnp.linalg.norm(v)
    # vh = jnp.where(normv == 0, v, v / normv)
    x1 = x + fd_step * v

    f_step, vp_step, vpp_step = unflatx(x1)
    yancc_wrapper_step = yancc_data.from_fields(f_step, grid, vp_step, vpp_step)

    G_step = StellaratorFun(yancc_wrapper_step)[0]
    print(G_step)
    print(G)

    tangent_out = (G_step - G) / fd_step

    return jnp.float32(tangent_out)

nTangents = 2
import operator
from desc.plotting import plot_comparison
for i in range(0,nTangents):
    eq_pert = perturb_eq(eq)
    fig, ax = plot_comparison(eqs=[eq_pert, eq], labels=["perturbed", "initial"])
    fig.savefig(f"eq_pert{i}.png")
    fig.show()
    yancc_wrapper_pert = yancc_data.from_eq(points, eq=eq_pert, nx=nx, na=na, nz = yancc_nzeta, nt = yancc_ntheta)
    t = jax.tree.map(operator.sub, yancc_wrapper_pert.get_fields(), primals[0]) 
    fd = fd_jvp(primals, t)
    adj = adj_jvp(primals, t)

    print(f"Iteration {i} jvps:\n   adjoints={adj}, finite differences={fd}\n")
