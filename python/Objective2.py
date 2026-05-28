import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from Stellarator2 import StellaratorTransport
import yancc
from yancc_wrapper2 import yancc_data
from desc.backend import tree_unstack

from jax.experimental import io_callback

# from desc import set_device
# set_device('gpu')
# from desc.backend import pure_callback


def abstract_eval(yin):
    boundary_field = yin.fields_unstacked[-1]

    flat, _ = jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
    npoints = yin.grid.num_rho
    np = len(flat)-1+1+1

    return jax.ShapeDtypeStruct((),jnp.float32), jax.ShapeDtypeStruct((npoints, np), jnp.float32), jax.ShapeDtypeStruct((npoints,), jnp.float32)

def make_objective(config, yancc_res=None, vectorized=False):
    """Make an external (python) function work with JAX.

    callback syntax stolen from desc jaxify
    """

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
    
    _f_wrapped = functools.partial(StellaratorFun, config)

    @eqx.filter_custom_jvp
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)
     
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 


    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        (fields, Vp, Vpp), grid = primals
        (field_dot, Vp_dot, Vpp_dot), _= tangents

        # compute 
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)
        G, G_p, pi = _f_wrapped(yancc_wrapper) # runs MaNTA and returns the adjoints + pressure profile

        # get unflattening function
        _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

        # Separate out the different parts of the gradient 
        G_p_field = G_p[:, :-2] # extract field component
        G_p_vprime = G_p[:, -2] # extract vprime component
        G_p_vpp = G_p[:, -1] # extract vpp component
        # need to pad the field portion because NFP gets removed by equinox during the gradient calculation
        G_p_padded = jnp.pad(G_p_field, pad_width=((0,0),(0,1)), mode='constant')

        # Create a field object from the padded G_p matrix
        G_p_unflattened = jax.vmap(unflatten_field)(jnp.float64(G_p_padded))

        # Function to compute the dot product between individual components of the field
        def safe_mul(x, y):
            if x is None:
                return y
            if y is None:
                return x
            x_flat = jax.flatten_util.ravel_pytree(x)[0]
            y_flat = jax.flatten_util.ravel_pytree(y)[0]
            return jnp.dot(x_flat,y_flat) 

       # Apply tree_map to multiply G_p * tangents
        # We need to treat None as a leaf
        result = jax.tree.map(safe_mul, G_p_unflattened, field_dot, is_leaf=lambda x: x is None)
        result_flattened, _ = jax.flatten_util.ravel_pytree(result)
        
        #now do vprime
        result_vprime = jnp.dot(G_p_vprime, Vp_dot)
        result_vpp = jnp.dot(G_p_vpp, Vpp_dot)

        # Result is the sum of G_field * tangent_field + G_vp * tangent_vp + G_vpp * tangent_vpp
        return (G, pi), (jnp.float32(jnp.sum(result_flattened)+result_vprime+result_vpp), None)

    return _objective_base
"""
# Finite difference objective for testing
def make_objective_fd(config, abs_step=1e-4, rel_step=0):

    def StellaratorFun(config, yin):
        st = StellaratorTransport(config, yancc_wrapper=yin)
        st.run()
        G, G_p = st.getAdjointGradients()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi
    
    def wrap_callback(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_shape_dtype = abstract_eval(*args, **kwargs)
            return io_callback(
                func, result_shape_dtype, *args, ordered=False, **kwargs
            )

        return wrapper
    
    _f_wrapped = functools.partial(StellaratorFun, config)


    @eqx.filter_custom_jvp
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        (fields_in, vp, vpp), grid = primals
        (v1, v2, v3), _= tangents

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
        G_step, _, _ = _f_wrapped(yancc_wrapper_step)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        primal_out = (G, pi)
        # primal_out = _f_wrapped(*primals)
        # flatten everything into 1D vectors for easier finite differences
        # y, unflaty = jax.flatten_util.ravel_pytree(field_dot)

        tangent_out = (G_step - G) / fd_step * normv
        #tangent_out = (, None)

        return primal_out, (jnp.float32(tangent_out), None)

    return _objective_base

"""