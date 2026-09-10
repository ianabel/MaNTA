from stellarator_multichannel import StellaratorTransport
from desc.backend import tree_unstack
from yancc_wrapper2 import yancc_data
import yancc
import functools
import equinox as eqx
import jax.numpy as jnp
import jax
from jax.experimental import io_callback
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


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

    solver_config = config["Solver"]
    time_march_solver_config = solver_config.copy()
    time_march_solver_config["SteadyStateSolver"] = "PseudoTransient"
    time_march_solver_config["restart"] = False
    time_march_solver_config["MaxRejectedSteps"] = 4
    grad_solver_config = solver_config.copy()
    grad_solver_config["delta_t"] = grad_solver_config["delta_t"] / 10000.0
    grad_solver_config["restart"] = True
    grad_solver_config["solveAdjoint"] = True
    grad_solver_config["SteadyStateSolver"] = "Newton"

    grad_config = {"Stellarator": config["Stellarator"], "Solver": grad_solver_config}

    time_march_config = {
        "Stellarator": config["Stellarator"],
        "Solver": time_march_solver_config,
    }

    @eqx.filter_custom_jvp
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)

        st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
        with jax.default_device(jax.devices("cpu")[0]):
            ec = st.run()

            def true_fn():
                pass

            def false_fn():

                # put in callback to stop jax from trying to evaluate this during tracing
                conf_success = io_callback(
                    lambda: st.reconfigure(time_march_solver_config),
                    (jax.ShapeDtypeStruct((), jnp.bool),),
                    ordered=True,
                )
                jax.debug.print(
                    "Reconfigured solver with code {val}",
                    val=conf_success,
                    ordered=True,
                )
                st.run()

            jax.lax.cond(ec, true_fn, false_fn)

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

        return (G, pi), (dp, None)

    return _objective_base
