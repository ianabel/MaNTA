from stellarator_multichannel import StellaratorTransport
from desc.backend import tree_unstack
from yancc_wrapper2 import yancc_data
import yancc
import functools
import equinox as eqx
import jax.numpy as jnp
import jax
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# from desc import set_device
# set_device('gpu')
# from desc.backend import pure_callback


def abstract_eval(yin):
    boundary_field = yin.fields_unstacked[-1]

    flat, _ = jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
    npoints = yin.grid.num_rho
    np = len(flat) - 1 + 1 + 1

    return (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((npoints, np), jnp.float32),
        jax.ShapeDtypeStruct((npoints,), jnp.float32),
    )


def make_objective(config, yancc_res=None):
    """Make an external (python) function work with JAX."""

    def StellaratorFun(config, yin):
        st = StellaratorTransport(config, yancc_wrapper=yin)
        st.run()
        G, G_p = st.getAdjointGradients()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi

    #    def wrap_callback(func):
    #
    #        @functools.wraps(func)
    #        def wrapper(*args, **kwargs):
    #            result_shape_dtype = abstract_eval(*args, **kwargs)
    #            return io_callback(
    #                func, result_shape_dtype, *args, ordered=False, **kwargs
    #            )
    #
    #        return wrapper
    solver_config = config["Solver"]
    grad_solver_config = solver_config.copy()
    grad_solver_config["delta_t"] = grad_solver_config["delta_t"] / 10000.0
    grad_solver_config["restart"] = True
    grad_solver_config["solveAdjoint"] = True
    grad_solver_config["SteadyStateSolver"] = "Newton"
    print("delta t=", grad_solver_config["delta_t"])

    grad_config = {"Stellarator": config["Stellarator"], "Solver": grad_solver_config}

    @eqx.filter_custom_jvp
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)

        st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
        st.run()
        G = st.G()[0]
        pi = jnp.array(st.getPressure())
        return G, pi

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        (fields, Vp, Vpp), grid = primals
        # (field_dot, Vp_dot, Vpp_dot), _= tangents
        v, _ = tangents

        # compute
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)
        # runs MaNTA and returns the adjoints + pressure profile
        G, G_p, pi = StellaratorFun(grad_config, yancc_wrapper)

        ntheta = fields.ntheta
        nzeta = fields.nzeta
        pad_width = 1 + 2 * (nzeta) + 2 * (ntheta)
        v_unstack = jax.vmap(
            lambda x: jnp.pad(
                jax.flatten_util.ravel_pytree(x)[0],
                pad_width=(pad_width, 0),
                mode="constant",
            )
        )(v)
        dp = jnp.float32(jnp.dot(G_p.flatten(), v_unstack.flatten()))
        # get unflattening function
        # _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

        # Separate out the different parts of the gradient
        #     G_p_field = G_p[:, :-2] # extract field component
        #     G_p_vprime = G_p[:, -2] # extract vprime component
        #     G_p_vpp = G_p[:, -1] # extract vpp component
        #     # need to pad the field portion because NFP gets removed by equinox during the gradient calculation
        #     G_p_padded = jnp.pad(G_p_field, pad_width=((0,0),(0,1)), mode='constant')

        #     # Create a field object from the padded G_p matrix
        #     G_p_unflattened = jax.vmap(unflatten_field)(jnp.float64(G_p_padded))

        #     # Function to compute the dot product between individual components of the field
        #     def safe_mul(x, y):
        #         if x is None:
        #             return y
        #         if y is None:
        #             return x
        #         x_flat = jax.flatten_util.ravel_pytree(x)[0]
        #         y_flat = jax.flatten_util.ravel_pytree(y)[0]
        #         return jnp.dot(x_flat,y_flat)

        #    # Apply tree_map to multiply G_p * tangents
        #     # We need to treat None as a leaf
        #     result = jax.tree.map(safe_mul, G_p_unflattened, field_dot, is_leaf=lambda x: x is None)
        #     result_flattened, _ = jax.flatten_util.ravel_pytree(result)

        #     #now do vprime
        #     result_vprime = jnp.dot(G_p_vprime, Vp_dot)
        #     result_vpp = jnp.dot(G_p_vpp, Vpp_dot)

        # Result is the sum of G_field * tangent_field + G_vp * tangent_vp + G_vpp * tangent_vpp
        # (jnp.float32(jnp.sum(result_flattened)+result_vprime+result_vpp), None)
        return (G, pi), (dp, None)

    return _objective_base
