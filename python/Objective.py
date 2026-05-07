import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from Stellarator import StellaratorTransport
import yancc
from yancc_wrapper import yancc_data


from jax.experimental import io_callback

# from desc import set_device
# set_device('gpu')
# from desc.backend import pure_callback


def abstract_eval(yin):
    boundary_field = yin.fields_unstacked[-1]

    flat, _ = jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
    npoints = yin.grid.num_rho
    np = len(flat)-1

    return jax.ShapeDtypeStruct((),jnp.float32), jax.ShapeDtypeStruct((npoints, np), jnp.float32), jax.ShapeDtypeStruct((npoints,), jnp.float32)

def make_objective(config, vectorized=False):
    """Make an external (python) function work with JAX.

    callback syntax stolen from desc jaxify
    """

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
    def _objective_base(fields, grid, Vprime):
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vprime)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 


    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        fields, grid, Vprime = primals
        field_dot,_,_ = tangents
    
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vprime)
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        # _, unflatten = jax.flatten_util.ravel_pytree(fields)

        _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

        G_p_padded = jnp.pad(G_p, pad_width=((0,0),(0,1)), mode='constant')

        G_p_unflattened = jax.vmap(unflatten_field)(jnp.float64(G_p_padded))

        def safe_mul(x, y):
            if x is None:
                return y
            if y is None:
                return x

            return jnp.multiply(x, y) 

        # Apply tree_map
        # We need to treat None as a leaf
        result = jax.tree.map(safe_mul, field_dot, G_p_unflattened, is_leaf=lambda x: x is None)
        result_flattened, _ = jax.flatten_util.ravel_pytree(result)

        return (G, pi), (jnp.float32(jnp.sum(result_flattened)), None)
        # field_dot_flatten, _ = jax.flatten_util.ravel_pytree(field_dot)
        
        # lg = len(G_p_flat)
        # lf = len(field_dot_flatten)
        # # print(lg)
        # # print(lf)
        # # print(field_dot)
        # # print(fields)

        # field_dot_pad = jnp.pad(field_dot_flatten, (lg-lf - len(pi),len(pi)), mode='constant') 
        # return (G, pi), (jnp.float32(jnp.dot(G_p_flat, field_dot_pad)), None)

    return _objective_base


# Finite difference objective for testing
def make_objective_fd(config, abs_step=1e-4, rel_step=0):

    def StellaratorFun(config, yin):
        st = StellaratorTransport(config, yancc_wrapper=yin)
        st.run()
        G, G_p = st.getAdjointGradients()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi
    
    _f_wrapped = functools.partial(StellaratorFun, config)

    @eqx.filter_custom_jvp
    def _objective_base(fields, grid, Vprime):
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vprime)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        fields, grid, Vprime = primals
        field_dot,_,_ = tangents
    
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vprime)
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        primal_out = (G, pi)
        # primal_out = _f_wrapped(*primals)

        # flatten everything into 1D vectors for easier finite differences
        # y, unflaty = jax.flatten_util.ravel_pytree(field_dot)
        x, unflatx = jax.flatten_util.ravel_pytree(primals)
        v, _______ = jax.flatten_util.ravel_pytree(tangents)

        # finite difference step size
        fd_step = abs_step + rel_step * jnp.mean(jnp.abs(x))

        # scale tangents to unit norm if nonzero
        normv = jnp.linalg.norm(v)
        vh = jnp.where(normv == 0, v, v / normv)

        def f(fields_in):
            yancc_wrapper = yancc_data.from_fields(fields_in, grid, Vprime)
            G, G_p, pi = _f_wrapped(yancc_wrapper)
            return G

        tangent_out = (f(x + fd_step * vh) - G) / fd_step * normv
        #tangent_out = (, None)

        return primal_out, (tangent_out, None)

    return _objective_base