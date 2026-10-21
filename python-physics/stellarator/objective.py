from datetime import time

from stellarator_multichannel import StellaratorTransport
from desc.backend import tree_unstack

# what ObjectiveFromUser.compute calls; the subclass below has to match it
from desc.compute.utils import _compute as compute_fun
from desc.objectives import ObjectiveFromUser
from yancc_wrapper import yancc_data
import yancc
import functools
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax
from jax.experimental import io_callback
from jax.flatten_util import ravel_pytree
from manta.jax.ffi_runner import cpu_fp_dtype
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# The last JVP's three contributions -- the field's (incl. V', V''), the ITG
# geometry's, the Dirichlet edge's -- recorded so a check can tell whether a
# gradient it verified had any ITG content at all. STELLARATOR_JVP_DEBUG=1
# also prints them.
_JVP_DEBUG = bool(os.environ.get("STELLARATOR_JVP_DEBUG"))
LAST_JVP_PARTS = {}


def _record_jvp_parts(field, itg, boundary):
    LAST_JVP_PARTS.update(
        field=np.asarray(field), itg=np.asarray(itg), boundary=np.asarray(boundary)
    )
    if _JVP_DEBUG:
        print(f"objective JVP: field {field}  itg {itg}  boundary {boundary}")


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


def _split_tree(tree_in):
    """(fields, Vp, Vpp) or (fields, Vp, Vpp, itg) -> the four, itg None if absent."""
    if len(tree_in) == 4:
        return tree_in
    fields, Vp, Vpp = tree_in
    return fields, Vp, Vpp, None


def _itg_node_tangent(itg, itg_tangent, points, n_itg):
    """The tangent of each node's ITG geometry, ravelled as the adjoint's
    columns are: (npoints, n_itg).

    `itg_tangent` is the ITGFlux-shaped tangent DESC hands the JVP; its leaves
    are None where nothing depends on the parameters (the weights, rho, a).
    Only the signals matter -- the transported flux is normalised to the
    constants' B0 and a, not the equilibrium's B_ref (`ITGFlux.local`).
    The interpolation is monotonic (PCHIP), whose slopes are nonlinear in the
    table, so the tangent is pushed through it by a jvp rather than by
    interpolating the tangent table.
    """
    points = jnp.asarray(points)
    sig_t = getattr(itg_tangent, "signals", None)
    if sig_t is None:
        return jnp.zeros((points.shape[0], n_itg))
    _, sig_nodes_t = jax.jvp(
        lambda s: itg.signals_at(points, signals=s), (itg.signals,), (sig_t,)
    )
    out = sig_nodes_t.reshape(points.shape[0], -1)
    assert out.shape[1] == n_itg, (out.shape, n_itg)
    return out


def make_objective(config, yancc_res=None, normalization=None, failure_value=1e-10):
    """Make an external (python) function work with JAX.

    The returned function takes `(tree_in, grid)`, where `tree_in` is
    `(fields, Vp, Vpp)`, or `(fields, Vp, Vpp, itg)` with `itg` the
    `itg_wrapper.ITGFlux` that `ITGFluxBuilder(params)` returns -- the second
    form needs `useITGProxy` in the config. Built from `params` outside this
    function, the ITGFlux carries its dependence on the equilibrium into the
    JVP as a tangent, which is contracted with the adjoint's ITG columns.

    `normalization` replaces config["Normalization"] if given. Keep it fixed
    over the optimisation (config.field_norm_from_eq at the initial
    equilibrium, say): one that followed the geometry would move the physical
    sources with it, through a path the gradient does not carry.

    `failure_value` is G when the steady solve fails. The default, 1e-10, is
    for a driver minimising 1/G (stored energy): a failure is then 1e10 and
    the step is rejected. A driver minimising G itself (the ITG heat flux)
    needs a large value instead.
    """
    if normalization is None:
        normalization = config.get("Normalization", {})
    # the primal, the time-march fallback and the gradient solve all carry it,
    # since the gradient solve restarts from the primal's restart file
    config = {**config, "Normalization": dict(normalization)}

    def StellaratorFun(config, yin, itg=None):
        st = StellaratorTransport(config, yancc_wrapper=yin, itg=itg)
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
        fields, Vp, Vpp, itg = _split_tree(tree_in)
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)

        st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper, itg=itg)
        # st2 = StellaratorTransport(time_march_config, yancc_wrapper=yancc_wrapper)
        with jax.default_device(jax.devices("cpu")[0]):
            ec = st.run()

            def true_fn():
                return st.G()[0]

            def false_fn():
                return jnp.array([failure_value], dtype=cpu_fp_dtype)
                # ec = st2.run()
                #
                # return st2.G()[0]

            G = jax.lax.cond(ec, true_fn, false_fn)

        pi = jnp.array(st.getPressure())
        return G, pi

    @_objective_base.def_jvp
    def _objective_base_jvp(primals, tangents):
        tree_in, grid = primals
        fields, Vp, Vpp, itg = _split_tree(tree_in)

        v, _ = tangents
        if v is None:
            v = (None,) * len(tree_in)
        v_fields, v_Vp, v_Vpp, v_itg = _split_tree(v)
        v = (v_fields, v_Vp, v_Vpp)

        # compute
        yancc_wrapper = yancc_data.from_fields(fields, grid, Vp, Vpp, **yancc_res)

        # runs MaNTA and returns the adjoints + pressure profile
        G, G_p, pi, st = StellaratorFun(grad_config, yancc_wrapper, itg)

        # columns: the field's (np_field, incl. V' and V''), the ITG geometry's
        # (n_itg, zero without the ITG term), then one upper-boundary
        # sensitivity per variable
        adj = st.adjointProblem
        np_field, np_cell = adj.np_field, adj.np_cell
        G_p_field = G_p[:, :np_field]
        G_p_itg = G_p[:, np_field:np_cell]
        G_p_boundary = G_p[:, np_cell:]

        ntheta = fields.ntheta
        nzeta = fields.nzeta
        pad_width = 1 + 2 * (nzeta) + 2 * (ntheta)
        # no field tangent at all when only the ITG geometry moves (a
        # perturbation of what the builder sees with the fields held fixed)
        dp_field = jnp.zeros(())
        if jax.tree.leaves(v):
            v_unstack = jax.vmap(
                lambda x: jnp.pad(
                    jax.flatten_util.ravel_pytree(x)[0],
                    pad_width=(pad_width, 0),
                    mode="constant",
                )
            )(v)
            dp_field = jnp.dot(G_p_field.flatten(), v_unstack.flatten())

        # the ITG geometry at each node, through the table it is interpolated from
        dp_itg = jnp.zeros(())
        if itg is not None and v_itg is not None:
            v_itg_nodes = _itg_node_tangent(itg, v_itg, st.points, adj.n_itg)
            dp_itg = jnp.dot(G_p_itg.flatten(), v_itg_nodes.flatten())

        # the Dirichlet edge data depend on V': chain dG/dg_D (carried by the
        # edge cell's rows, so summed over rows) through g_D(V')
        dp_boundary = jnp.zeros(())
        Vp_tangent = v[1]
        if Vp_tangent is not None:
            _, dgD = jax.jvp(st.upper_boundary_values, (Vp,), (Vp_tangent,))
            dp_boundary = jnp.dot(G_p_boundary.sum(axis=0), dgD)

        jax.debug.callback(_record_jvp_parts, dp_field, dp_itg, dp_boundary)
        dp = jnp.asarray(dp_field + dp_itg + dp_boundary, dtype=cpu_fp_dtype)

        return (G, pi), (dp, None)

    return _objective_base


class ObjectiveFromUserWithParams(ObjectiveFromUser):
    """`ObjectiveFromUser` whose function also receives `params`.

    `fun(grid, data, params)`: `data` as ObjectiveFromUser computes it, plus
    the parameter dictionary it was computed from -- which is what
    `ITGFluxBuilder` needs, and what ObjectiveFromUser never passes on. Works
    under ProximalProjection like its base: `compute(params)` is all DESC
    calls, and the gradient flows through `params` both ways.
    """

    def build(self, use_jit=True, verbose=1):
        # ObjectiveFromUser.build reads the data keys out of self._fun's source
        # and sizes the output by calling self._fun(grid, data); hand it a
        # two-argument shim bound to the equilibrium's own params, with
        # __wrapped__ so inspect.getsource reads the user's function
        user_fun = self._fun
        params0 = self.things[0].params_dict

        def shim(grid, data):
            return user_fun(grid, data, params0)

        shim.__wrapped__ = user_fun
        self._fun = shim
        try:
            super().build(use_jit=use_jit, verbose=verbose)
        finally:
            self._fun = user_fun

    def compute(self, params, constants=None):
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            self._p,
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            **self._compute_kwargs,
        )
        return self._fun(constants["transforms"]["grid"], data, params)
