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
        # 

        st = StellaratorTransport(config, yancc_wrapper=yin)
        G, G_p = st.run()
        # G, G_p = st.runAdjointSolve()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi


    def wrap_pure_callback(func):

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
 
        field_dot_flatten,_ = jax.flatten_util.ravel_pytree(field_dot)
        G_p_flat = G_p.flatten()
        lg = len(G_p_flat)
        lf = len(field_dot_flatten)

        field_dot_pad = jnp.pad(field_dot_flatten, (lg-lf,0), mode='constant') 
        return (G, pi), (jnp.float32(jnp.dot(G_p_flat, field_dot_pad)), None)

    return _objective_base



