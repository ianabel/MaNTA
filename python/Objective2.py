import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from Stellarator2 import StellaratorTransport
import yancc
from yancc_wrapper2 import yancc_data


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
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 


    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        (fields, Vp, Vpp), grid = primals
        (field_dot, Vp_dot, Vpp_dot), _= tangents
    
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp)
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        # _, unflatten = jax.flatten_util.ravel_pytree(fields)

        _, unflatten_field = jax.flatten_util.ravel_pytree(yancc_wrapper.fields_unstacked[0])

        G_p_field = G_p[:, :-2] # remove vprime component
        G_p_vprime = G_p[:, -2] # extract vprime component
        G_p_vpp = G_p[:, -1] # extract vpp component
        G_p_padded = jnp.pad(G_p_field, pad_width=((0,0),(0,1)), mode='constant')

        G_p_unflattened = jax.vmap(unflatten_field)((G_p_padded))

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

        #now do vprime

        result_vprime = jnp.dot(Vp_dot, G_p_vprime)
        result_vpp = jnp.dot(Vpp_dot, G_p_vpp)

        return (G, pi), ((jnp.sum(result_flattened)+result_vprime+result_vpp), None)

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
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp)

        G, G_p, pi = _f_wrapped(yancc_wrapper)
        return G, pi 

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        tree_in, grid = primals
        tree_dot, _= tangents
    
        yancc_wrapper = yancc_data.from_fields(tree_in[0], grid, tree_in[1], tree_in[2])
        G, G_p, pi = _f_wrapped(yancc_wrapper)
        primal_out = (G, pi)
        # primal_out = _f_wrapped(*primals)

        # flatten everything into 1D vectors for easier finite differences
        # y, unflaty = jax.flatten_util.ravel_pytree(field_dot)
        x, unflatx = jax.flatten_util.ravel_pytree(tree_in)
        v1, ______ = jax.flatten_util.ravel_pytree(tree_dot[0])
        v2, ______ = jax.flatten_util.ravel_pytree(tree_dot[1])
        v3, ______ = jax.flatten_util.ravel_pytree(tree_dot[2])

        # finite difference step size
        fd_step = abs_step + rel_step * jnp.mean(jnp.abs(x))

        # scale tangents to unit norm if nonzero
        normv1 = jnp.linalg.norm(v1)
        v1a = jnp.where(normv1 == 0, v1, v1 / normv1)
        vh1 = jnp.pad(v1a, (0, len(x)-len(v1a)), mode='constant')
        normv2 = jnp.linalg.norm(v2)
        v2a = jnp.where(normv2 == 0, v2, v2 / normv2)
        normv3 = jnp.linalg.norm(v3)
        v3a = jnp.where(normv3 == 0, v3, v3 / normv3)
        vcat = jnp.concatenate([vh1, v2a, v3a])
        normv = jnp.linalg.norm(vcat)
        vh = jnp.where(normv == 0, vcat, vcat / normv)
        def f(tree_in):
            tree_unflat = unflatx(tree_in)
            fields_in = tree_unflat[0]
            vp_in = tree_unflat[1]
            vpp_in = tree_unflat[2]
            yancc_wrapper = yancc_data.from_fields(fields_in, grid, vp_in, vpp_in)
            G, _, _ = _f_wrapped(yancc_wrapper)
            return G

        tangent_out = (f(x + fd_step * vh) - G) / fd_step * normv
        #tangent_out = (, None)

        return primal_out, (tangent_out, None)

    return _objective_base