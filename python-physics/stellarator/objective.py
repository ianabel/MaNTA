from datetime import time

from stellarator_multichannel import StellaratorTransport
from desc.backend import tree_unstack
from yancc_wrapper import yancc_data
import yancc
import functools
import equinox as eqx
import jax.numpy as jnp
import jax
from jax.experimental import io_callback
from manta.jax.ffi_runner import cpu_fp_dtype
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def abstract_eval(yin):
    boundary_field = yin.fields_unstacked[-1]

    flat, _ = jax.flatten_util.ravel_pytree((eqx.filter(boundary_field, eqx.is_array)))
    npoints = yin.grid.num_rho
    np = len(flat) - 1 + 1 + 1

    return (
        jax.ShapeDtypeStruct((), cpu_fp_dtype),
        jax.ShapeDtypeStruct((npoints, np), cpu_fp_dtype),
        jax.ShapeDtypeStruct((npoints,), cpu_fp_dtype),
    )


def make_objective(config, yancc_res=None, normalization=None):
    """Make an external (python) function work with JAX.

    `normalization` replaces config["Normalization"] if given. Keep it fixed
    over the optimisation (config.field_norm_from_eq at the initial
    equilibrium, say): one that followed the geometry would move the physical
    sources with it, through a path the gradient does not carry.
    """
    if normalization is None:
        normalization = config.get("Normalization", {})
    # the primal, the time-march fallback and the gradient solve all carry it,
    # since the gradient solve restarts from the primal's restart file
    config = {**config, "Normalization": dict(normalization)}

    def StellaratorFun(config, yin):
        st = StellaratorTransport(config, yancc_wrapper=yin)
        st.run()
        G, G_p = st.getAdjointGradients()

        pi = jnp.array(st.getPressure())

        return G[0], G_p, pi, st

    solver_config = config["Solver"]
    time_march_solver_config = solver_config.copy()
    time_march_solver_config["SteadyStateSolver"] = "PseudoTransient"
    time_march_solver_config["restart"] = False
    time_march_solver_config["MaxRejectedSteps"] = 5
    grad_solver_config = solver_config.copy()
    grad_solver_config["delta_t"] = grad_solver_config["delta_t"] / 10000.0
    grad_solver_config["restart"] = True
    grad_solver_config["solveAdjoint"] = True
    grad_solver_config["SteadyStateSolver"] = "Newton"
    grad_solver_config["SteadyStateTest"] = "Residual"
    grad_solver_config["SteadyStateTolerance"] = 1.0

    grad_config = {**config, "Solver": grad_solver_config}

    time_march_config = {**config, "Solver": time_march_solver_config}

    @eqx.filter_custom_jvp
    def _objective_base(tree_in, grid):
        fields, Vp, Vpp = tree_in
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)

        st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
        # st2 = StellaratorTransport(time_march_config, yancc_wrapper=yancc_wrapper)
        with jax.default_device(jax.devices("cpu")[0]):
            ec = st.run()

            def true_fn():
                return st.G()[0]

            def false_fn():
                return jnp.array([1e-10], dtype=cpu_fp_dtype)
                # ec = st2.run()
                #
                # return st2.G()[0]

            G = jax.lax.cond(ec, true_fn, false_fn)

        pi = jnp.array(st.getPressure())
        return G, pi

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        (fields, Vp, Vpp), grid = primals

        v, _ = tangents

        # compute
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)
        # runs MaNTA and returns the adjoints + pressure profile
        G, G_p, pi, st = StellaratorFun(grad_config, yancc_wrapper)

        # the spatial columns, then one upper-boundary sensitivity per variable
        np_cell = st.adjointProblem.np_cell
        G_p_cell, G_p_boundary = G_p[:, :np_cell], G_p[:, np_cell:]

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
        dp = jnp.dot(G_p_cell.flatten(), v_unstack.flatten())

        # the Dirichlet edge data depend on V': chain dG/dg_D (carried by the
        # edge cell's rows, so summed over rows) through g_D(V')
        Vp_tangent = v[1]
        if Vp_tangent is not None:
            _, dgD = jax.jvp(st.upper_boundary_values, (Vp,), (Vp_tangent,))
            dp = dp + jnp.dot(G_p_boundary.sum(axis=0), dgD)
        dp = jnp.asarray(dp, dtype=cpu_fp_dtype)

        return (G, pi), (dp, None)

    return _objective_base
