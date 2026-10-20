"""ITG turbulent heat flux from DESC's NNITGProxy, in `compute_dke_sol`'s shape.

`desc.objectives.NNITGProxy` (Landreman et al. 2025) predicts the gyro-Bohm
normalised ion heat flux of a GX flux-tube simulation from two things that never
meet until the last layer of the network:

* the **geometry** -- seven GX coefficients (bmag, gbdrift, cvdrift,
  gbdrift0/shat, gds2, gds21/shat, gds22/shat^2) along a field line of fixed
  length, resampled to 96 points uniform in arclength. This depends on the
  equilibrium and on rho, and is what costs: a PEST -> DESC coordinate map and a
  compute on ~1000 points per field line.
* the **gradients** -- `[a/LT, a/Ln]`, concatenated onto the convolutional
  features before the fully connected layers.

So the expensive half is a function of `(eq, rho)` only and can be tabulated
once per equilibrium, exactly as `yancc_data` tabulates the yancc `Field`, while
the half that sees the plasma state is a small CNN ensemble (~2e5 weights) that
is cheap to evaluate and to differentiate at every MaNTA node.

Three layers, so the proxy can be declared once and re-evaluated at every
equilibrium an optimisation visits (`objective.py`):

    obj = build_itg_proxy(eq, rho, ensemble_dir=..., ensemble_csv=...)  # once
    builder = ITGFluxBuilder(obj)                                       # once
    itg = builder(params)        # per equilibrium; traceable in params
    flux, aux = itg(state, x, t, field, vp, vpp, pitchgrid, speedgrid, params, evolveDensity)

`make_itg_flux(eq, rho, ...)` is all three at `eq.params_dict`. Evaluating the
ensemble on the tabulated signals reproduces `NNITGProxy.compute`; see
`itg_check.py`.

Conventions, each of which is a choice the training data made for us:

* `a/LT = -d ln Ti / d rho` and `a/Ln = -d ln n / d rho`, with DESC's
  `rho = sqrt(psi / psi_b)`. GX's `tprim` is `-(a/T) dT/dx` with `x = a rho`,
  which is the same thing.
* The network's `Q` is `<q . grad x> / Q_gB` with `x = a rho` -- the training
  target is `Q_avgs_without_FSA_grad_x`, i.e. *not* divided by `<|grad x|>`.
  `Q_gB = n_i T_i v_ti rho_i^2 / a^2`, GX's normalisation:
  `v_ti = sqrt(T_i / m_i)` (no factor 2), `rho_i = v_ti / Omega_i`,
  `Omega_i = Z e B_ref / m_i`, `B_ref = 2 |psi_b| / a^2` (DESC's
  `gx_B_reference`). The signals are normalised with the *equilibrium's* `a`
  and `B_ref`, so a physical flux needs those too: `heat_flux_SI` returns
  `<q . grad(a rho)>` in W/m^2 that way, and the heat flow through the surface
  is `dV/drho * Q_SI / a`.
* What MaNTA transports is normalised exactly as `compute_dke_sol` normalises
  yancc's `<q . grad rho>` (W/m^3) -- `-<q . grad rho> V' / HeatNorm` -- times
  the config's `ITGScale`, but with `Q_gB` taken at `PlasmaConstants`' `B0` and
  `a`. `HeatEquationNormalization` is itself `n0 T0 sqrt(2) v_t rho*^2 / a` at
  those, so the transported flux is `ITGScale * Q * n T^(5/2) / sqrt(2) * V'`:
  large (Q is tens at a/LT of a few), the same on every equilibrium, and
  turned down with `ITGScale`.
* The models DESC selects (`pre_method = 2`) were trained on `log Q` clipped
  below at -2, so the ensemble cannot predict less than ~`e^-2 = 0.135` Q_gB: at
  zero gradient it still returns that. Left in, it is a gradient-independent
  outward flux that no steady state can balance. `floor="subtract"` removes it
  smoothly; `floor="keep"` returns the network's value as it stands.
* The training box is `a/LT` in (0, 14.4] and `a/Ln` in [-0.49, 8.9], at
  `Ti = Te` with adiabatic electrons. Gradients are clipped into that box
  (derivative zero outside it, i.e. constant extrapolation). `Te / Ti` is not an
  input, so a state with `Te != Ti` is outside what the model has seen; nothing
  here can check that. Adiabatic electrons also mean the model carries no
  particle flux and no electron heat flux, so in the multi-channel case those
  channels come back zero.

Differentiable with respect to the equilibrium end to end. The equilibrium
reaches the transported flux only through the geometry signals; `builder(params)`
is traceable, so the `ITGFlux` it returns carries d(signals)/d(params), and
`StellaratorAdjointProblem` takes each MaNTA node's signals as spatial
parameters, which `objective.py`'s custom JVP contracts with that tangent.
`fd_check_itg.py` checks it, directly and through ProximalProjection.

The network is smooth in a/LT and a/Ln but only piecewise smooth in the
signals (max pooling), so a finite difference in the geometry agrees with the
gradient only for steps small enough not to cross a pooling switch -- 1e-5
relative and below, measured; at 1e-3 they differ by ~10%. AD returns the
local derivative, which is what an optimiser wants; a check has to use small
steps.
"""

import csv
from typing import Callable

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
import os
from jaxtyping import Array, Float

from stellarator_state import StellaratorState

# Box the GX training set covers (20241210-01 random-gradient set, rounded
# outwards).
A_OVER_LT_RANGE = (0.0, 14.5)
A_OVER_LN_RANGE = (-0.5, 9.0)

# log Q was clipped at this value for pre_method = 2
_LOG_Q_CLIP = {2: -2.0}


def _detect_pre_method(ensemble_csv, n_ensemble):
    """The preprocessing method NNITGProxy will pick for this ensemble.

    Replicates `NNITGProxy._load_model_weights`: the most common `p:pre_method`
    among the `n_ensemble` best-scoring models.
    """
    with open(ensemble_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["objective"] != "F"]
    if not rows or "p:pre_method" not in rows[0]:
        return 0
    rows.sort(key=lambda r: float(r["objective"]), reverse=True)
    top = [int(r["p:pre_method"]) for r in rows[:n_ensemble]]
    return max(set(top), key=top.count)


class ITGFlux(eqx.Module):
    """NN ITG ion heat flux, callable with `compute_dke_sol`'s signature.

    An `eqx.Module` rather than a closure so that the weights and the geometry
    table are *traced* by `eqx.filter_jit` (which partitions the callable
    itself) rather than baked into the compiled kernel as constants: a new
    equilibrium is then new data for the same compiled function, not a
    recompile per DESC iteration.

    Build it with `make_itg_flux`.
    """

    rho: Float[Array, " nrho"]  # tabulation points, increasing
    signals: Float[Array, "nrho nalpha 7 npoints"]  # GX geometry on uniform arclength
    weights: list  # one CNN parameter pytree per ensemble member
    a_minor: Float[Array, ""]  # equilibrium minor radius [m]
    B_ref: Float[Array, ""]  # GX reference field 2|psi_b|/a^2 [T]
    Q_floor: Float[Array, ""]  # network's lower clip, in Q_gB
    floor_width: Float[Array, ""]  # softplus width for floor="subtract"
    floor: str = eqx.field(static=True)

    def signals_at(self, x, signals=None):
        """The tabulated signals interpolated to radius `x` (or to each of `x`).

        `signals` defaults to the module's own; passing a tangent of them
        interpolates the tangent, which is exact because interpolation is
        linear in the data.
        """
        signals = self.signals if signals is None else signals
        x = jnp.clip(x, self.rho[0], self.rho[-1])
        return interpax.interp1d(x, self.rho, signals, method="cubic")

    def geometry_at(self, points):
        """Per-point geometry, the signals at each of `points`, shape
        (n, nalpha, 7, npoints): the explicit form `local` takes, and what the
        adjoint differentiates with respect to. The equilibrium reaches the
        transported flux through nothing else (see `local`)."""
        return self.signals_at(jnp.asarray(points))

    def _Q(self, sig, a_over_LT, a_over_Ln):
        """Q / Q_gB from one point's signals, averaged over its field lines."""
        from desc.compute._turbulence import _cyclic_invariant_forward

        a_over_LT = jnp.clip(a_over_LT, *A_OVER_LT_RANGE)
        a_over_Ln = jnp.clip(a_over_Ln, *A_OVER_LN_RANGE)
        scalars = jnp.broadcast_to(jnp.stack([a_over_LT, a_over_Ln]), (sig.shape[0], 2))

        # NNITGProxy: mean of log Q over the ensemble, exp, then mean over alpha.
        # Inputs are left in the working precision rather than cast to float32
        # as DESC does; the weights were cast to match in ITGFluxBuilder.
        log_Q = jnp.mean(
            jnp.stack(
                [_cyclic_invariant_forward(sig, scalars, w)[:, 0] for w in self.weights]
            ),
            axis=0,
        )
        Q = jnp.mean(jnp.exp(log_Q))

        if self.floor == "subtract":
            w = self.floor_width
            Q = w * jax.nn.softplus((Q - self.Q_floor) / w)
        return Q

    def normalized_heat_flux(self, x, a_over_LT, a_over_Ln):
        """Q / Q_gB at radius `x`, averaged over the tabulated field lines."""
        return self._Q(self.signals_at(x), a_over_LT, a_over_Ln)

    def gyro_bohm(self, n, T, params, B_ref=None, a=None):
        """GX gyro-Bohm heat flux n T v_t rho^2 / a^2 [W/m^2].

        `n` in units of `n0`, `T` in units of `T0`, as StellaratorState holds
        them. `B_ref` and `a` default to the equilibrium's.
        """
        B_ref = self.B_ref if B_ref is None else B_ref
        a = self.a_minor if a is None else a
        c = params.constants
        species = c.IonSpecies
        T_J = T * c.T0
        vt = jnp.sqrt(T_J / species.mass)
        rho_i = species.mass * vt / (species.charge * B_ref)
        return n * c.n0 * T_J * vt * (rho_i / a) ** 2

    def heat_flux_SI(self, state: StellaratorState, x, params):
        """Ion heat flux <q . grad(a rho)> in W/m^2 at a local plasma state.

        Physical: gyro-Bohm at the equilibrium's own B_ref and a. A diagnostic
        -- what MaNTA transports is `local`'s, normalised to B0 and a instead.
        """
        sig = self.signals_at(x)
        Q = self._Q(sig, -state.dTidrho / state.Ti, -state.dndrho / state.n)
        return Q * self.gyro_bohm(state.n, state.Ti, params)

    def local(self, geometry, _state, x, t, vp, vpp, params, evolveDensity=False):
        """The flux at one point with its geometry (the signals) given.

        `__call__` is this with the geometry interpolated from the table; the
        adjoint calls it directly, to differentiate with respect to `geometry`.

        Normalised exactly as `compute_dke_sol` normalises yancc's flux,
        `-<q . grad rho> V' / HeatEquationNormalization`, times the
        config's `ITGScale`; with the gyro-Bohm flux taken at the normalisation's
        own B0 and a (`params.constants`), not the equilibrium's. That makes
        the result `ITGScale * Q * n T^(5/2) / sqrt(2) * V'` -- the heat
        normalisation is itself n0 T0 sqrt(2) v_t rho*^2 / a -- so `ITGScale`
        means the same thing on any equilibrium, and the equilibrium reaches
        this flux only through the geometry signals.
        """
        c = params.constants
        state = StellaratorState.from_state(_state, x, vp, vpp, params)
        Q = self._Q(geometry, -state.dTidrho / state.Ti, -state.dndrho / state.n)
        q_ref = Q * self.gyro_bohm(state.n, state.Ti, params, B_ref=c.B0, a=c.a)

        # <q . grad rho> = q_ref / a, then compute_dke_sol's normalisation; the
        # sign follows it too (yancc's flux enters negated)
        heat_flux = (
            -params.config.ITGScale * q_ref / c.a * vp / c.HeatEquationNormalization()
        )

        if evolveDensity:
            zero = 0.0 * vp
            # adiabatic electrons: no particle flux, no electron heat flux, and
            # so no radial current -- ambipolarity is left to the neoclassical
            return [zero, heat_flux, zero], [zero]
        return [heat_flux], []

    def __call__(
        self,
        _state,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params,
        evolveDensity=False,
    ):
        """Same arguments and return structure as `dke_sol.compute_dke_sol`.

        `field`, `pitchgrid` and `speedgrid` are accepted and ignored: the
        geometry this model needs is on the module.
        """
        return self.local(
            self.signals_at(x), _state, x, t, vp, vpp, params, evolveDensity
        )


class SumFlux(eqx.Module):
    """Neoclassical plus ITG, with `compute_dke_sol`'s signature.

    Fluxes add channel by channel; the auxiliary rows (the ambipolarity
    current) come from the neoclassical solve alone.
    """

    neoclassical: Callable = eqx.field(static=True)
    itg: ITGFlux

    def __call__(self, *args):
        nc_flux, nc_aux = self.neoclassical(*args)
        itg_flux, _ = self.itg(*args)
        return [a + b for a, b in zip(nc_flux, itg_flux)], nc_aux


def build_itg_proxy(
    eq,
    rho,
    ensemble_dir=None,
    ensemble_csv=None,
    model_path=None,
    alpha=None,
    n_ensemble=10,
    target_flux_tube_length=75.4,
    npoints=96,
    nz_internal=1001,
    solve_length_at_compute=False,
    use_jit=True,
    verbose=0,
):
    """Declare and build the NNITGProxy an `ITGFluxBuilder` reads from.

    Everything expensive and equilibrium-independent happens here, once: the
    weights are loaded, the transforms built, and the flux-tube extent solved
    for. `rho` are the tabulation points -- increasing, and off the axis, where
    the GX coefficients normalised by the shear are singular; the flux is
    multiplied by V', which vanishes there anyway, and `normalized_heat_flux`
    holds the innermost value inwards.

    Keep `solve_length_at_compute=False`, DESC's default and the recommended
    setting: the tube extent is solved at rho[0] here and its toroidal extent
    held fixed, so tube lengths drift from `target_flux_tube_length` with iota
    across the table -- on precise_QA that raises Q by up to 2.7% at the edge
    against re-solving per rho. True re-solves with a root solve per rho in
    every builder call; its derivative checks out as well as the default's,
    but it is slower and less exercised upstream.

    `use_jit=True` jits what DESC's `use_jit` covers (`compute_scaled`, the
    jacobians), which matters if the proxy also sits in an ObjectiveFunction;
    it does not cover the bare `compute` the builder calls, so the builder
    jits that itself, under the same flag.
    """
    from desc.objectives import NNITGProxy

    rho = np.asarray(rho, dtype=float)
    if np.any(np.diff(rho) <= 0):
        raise ValueError("rho must be strictly increasing")

    obj = NNITGProxy(
        eq=eq,
        rho=rho,
        alpha=alpha,
        model_path=model_path,
        ensemble_dir=ensemble_dir,
        ensemble_csv=ensemble_csv,
        n_ensemble=n_ensemble,
        target_flux_tube_length=target_flux_tube_length,
        npoints=npoints,
        nz_internal=nz_internal,
        solve_length_at_compute=solve_length_at_compute,
    )
    obj.build(use_jit=use_jit, verbose=verbose)
    return obj


class ITGFluxBuilder:
    """params -> ITGFlux, for a built NNITGProxy.

    The object to keep across an optimisation, e.g. inside `objective.py`:
    build the proxy once with `build_itg_proxy`, wrap it here, and call the
    builder with each new `params` dictionary. The call is traceable in
    `params` -- it is `NNITGProxy.compute` plus `B_ref` read off `params["Psi"]`
    -- so it can sit under DESC's jvp/jacobian, and the `ITGFlux` it returns
    carries the dependence on the equilibrium in its `signals` and `B_ref`
    leaves.

    Holds the ensemble weights, cast to float64 once here: the network's
    convolution refuses mixed precision, and a float32 network gives a Jacobian
    that is rounding noise at the 1e-7 level.
    """

    def __init__(self, obj, floor="subtract", floor_width=0.1):
        from desc.compute._turbulence import (
            _load_ensemble_weights_cached,
            _load_nn_weights,
        )

        if floor not in ("subtract", "keep"):
            raise ValueError(f"floor must be 'subtract' or 'keep', got {floor!r}")
        if not hasattr(obj, "_models"):
            raise ValueError("build the NNITGProxy before wrapping it")

        if obj._ensemble_dir is not None:
            # the models NNITGProxy loaded: same key, so DESC's cache hands them back
            pre_method = _detect_pre_method(obj._ensemble_csv, obj._n_ensemble)
            weights, _ = _load_ensemble_weights_cached(
                obj._ensemble_dir,
                obj._ensemble_csv,
                obj._n_ensemble,
                pre_method=pre_method,
            )
        else:
            # a single model carries no record of how it was trained; DESC
            # treats its output as log Q all the same
            pre_method = None
            weights = [_load_nn_weights(obj._model_path)[0]]

        if floor == "subtract" and pre_method not in _LOG_Q_CLIP:
            raise ValueError(
                f"no known log Q clip for pre_method={pre_method}; use floor='keep'"
            )
        Q_floor = np.exp(_LOG_Q_CLIP.get(pre_method, -np.inf))

        self.obj = obj

        def geometry(params):
            _, info = obj.compute(params, return_signals=True, return_per_alpha=True)
            # NNITGProxy normalises its signals with the minor radius frozen
            # at build (constants["a"]), so B_ref must use that same a
            a = obj.constants["a"]
            B_ref = 2.0 * jnp.abs(params["Psi"]) / (2.0 * jnp.pi) / a**2
            return jnp.asarray(info["signals"], dtype=float), jnp.reshape(B_ref, ())

        # one compiled kernel for every params an optimisation visits; the
        # Python flags above are closed over, so they are static
        self._geometry = jax.jit(geometry) if obj._use_jit else geometry
        self.weights = [
            jax.tree.map(lambda w: jnp.asarray(w, dtype=float), w) for w in weights
        ]
        self.floor = floor
        self.Q_floor = jnp.asarray(Q_floor)
        self.floor_width = jnp.asarray(floor_width * Q_floor)

    @property
    def rho(self):
        return jnp.asarray(self.obj.constants["rho"])

    def __call__(self, params) -> ITGFlux:
        signals, B_ref = self._geometry(params)
        return ITGFlux(
            rho=self.rho,
            signals=signals,
            weights=self.weights,
            a_minor=jnp.asarray(self.obj.constants["a"], dtype=float),
            B_ref=B_ref,
            Q_floor=self.Q_floor,
            floor_width=self.floor_width,
            floor=self.floor,
        )


def make_itg_flux(eq, rho, floor="subtract", floor_width=0.1, **proxy_kwargs):
    """One-shot: build the proxy and evaluate it at `eq`'s own parameters."""
    builder = ITGFluxBuilder(
        build_itg_proxy(eq, rho, **proxy_kwargs), floor, floor_width
    )
    return builder(eq.params_dict)


# --- what the drivers call ---------------------------------------------------

# read when the ITG term is built, so a run without it needs no models
DEFAULT_ITG_MODEL_DIR = os.environ.get("ITG_NN_DIR")


def make_itg_builder(
    st_config,
    eq,
    rho_upper=1.0,
    model_dir=None,
    n_rho=12,
    n_ensemble=10,
    **proxy_kwargs,
):
    """The ITGFluxBuilder a driver needs, or None when `useITGProxy` is unset.

    Builds the proxy once, on `eq` -- a driver's initial equilibrium; the
    builder then serves any equilibrium of the same resolution. The table
    starts off the axis, where the shear-normalised GX coefficients are
    singular; the flux is held at its innermost value inwards, and V' -> 0
    there. `model_dir` is the itg-proxy demo's `neural_networks/`, from the
    ITG_MODEL_DIR environment variable when not given.
    """
    if not st_config.get("useITGProxy", False):
        return None
    import os

    model_dir = model_dir or os.environ.get("ITG_MODEL_DIR", DEFAULT_ITG_MODEL_DIR)
    if model_dir is None:
        raise ValueError(
            "useITGProxy is set but no ITG model directory was given: pass "
            "model_dir, or set ITG_NN_DIR or ITG_MODEL_DIR"
        )
    proxy = build_itg_proxy(
        eq,
        np.linspace(0.1, rho_upper, n_rho),
        ensemble_dir=os.path.join(model_dir, "cyclic_invariant_models"),
        ensemble_csv=os.path.join(model_dir, "deephyper_exp", "results.csv"),
        n_ensemble=n_ensemble,
        **proxy_kwargs,
    )
    return ITGFluxBuilder(proxy)


def itg_at(builder, params):
    """`StellaratorTransport`'s `itg` argument: the ITGFlux at `params`, or
    None without the ITG term."""
    return None if builder is None else builder(params)


def with_itg(tree_in, builder, params):
    """`objective.py`'s `tree_in`, with the ITGFlux at `params` appended when
    there is one. `params` is None when the objective is a plain
    ObjectiveFromUser, which never passes it."""
    if builder is None or params is None:
        return tree_in
    return tuple(tree_in) + (builder(params),)
