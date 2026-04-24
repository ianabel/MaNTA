import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from Stellarator import StellaratorTransport

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
        st.run()
        G, G_p = st.runAdjointSolve()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi


    def wrap_pure_callback(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_shape_dtype = abstract_eval(*args, **kwargs)
            return io_callback(
                func, result_shape_dtype, *args, **kwargs
            )

        return wrapper
    
    _f_wrapped = wrap_pure_callback(functools.partial(StellaratorFun, config))
    
    @eqx.filter_custom_jvp
    def objective(fields, grid, yin):
        yancc_wrapper = yancc_data.from_other(fields, grid, yin)
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi

    @objective.def_jvp
    def objective_jvp(primals, tangents):
        fields, grid, yin = primals
        field_dot,_,_ = tangents
        yancc_wrapper = yancc_data.from_other(fields, grid, yin)
        G, G_p, pi = _f_wrapped(yancc_wrapper)

        field_dot_flatten,_ = jax.flatten_util.ravel_pytree(field_dot)

        return (G, pi), (jnp.float32(jnp.dot(G_p.flatten(), field_dot_flatten)), None)

    return objective



