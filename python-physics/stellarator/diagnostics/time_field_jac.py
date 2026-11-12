"""Time the adjoint's d(yancc)/d(field) against the forward d(yancc)/d(state).

At very low yancc resolution, on a few rho points, this compiles and runs:

* the flux itself (`compute_dke_sol`),
* `jacrev` over the state, as `StellaratorTransport.ComputePhysicsDerivatives`
  does without `reuseF1`,
* `dke_field_jac`, as `StellaratorAdjointProblem.ComputePhysicsDerivatives`
  does for the bare neoclassical flux,

each with and without the bootstrap row (`--no-bootstrap` drops the aux row
the flux returns after the radial current, which is the flux as it was before
BootstrapObjectives), and reports compile and run time separately.

    python diagnostics/time_field_jac.py [--eq FILE] [--desc-res L] [--nt 9 --nz 11 --na 17 --nx 4]

Uses the real yancc solve, so keep the resolution low on a small machine.
"""

import argparse
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
os.environ.pop("TEST_STELLARATOR", None)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import desc.io
from manta.jax import State

from config import StellaratorConfig, StellaratorParams, field_norm_from_eq, normalization_args
from dke_sol import compute_dke_sol, dke_field_jac
from stellarator_state import StellaratorState
from yancc_wrapper import yancc_data

p = argparse.ArgumentParser()
p.add_argument("--eq", default=os.path.join(here, "..", "paper", "eq5_all_equilibria.h5"))
p.add_argument("--desc-res", type=int, default=None, help="change_resolution(L=M=N) first")
p.add_argument("--rho", type=float, nargs="+", default=[0.4, 0.7])
p.add_argument("--nt", type=int, default=9)
p.add_argument("--nz", type=int, default=11)
p.add_argument("--na", type=int, default=17)
p.add_argument("--nx", type=int, default=4)
p.add_argument("--which", nargs="+", default=["flux", "state", "field"])
p.add_argument("--rule", choices=["jvp", "vjp"], default="jvp",
               help="with --which vjp: give the term a custom_jvp or a custom_vjp")
args = p.parse_args()

eq = desc.io.load(args.eq)
try:
    eq = eq[-1]
except TypeError:
    pass
if args.desc_res is not None:
    r = args.desc_res
    eq.change_resolution(L=r, M=r, N=r, L_grid=2 * r, M_grid=2 * r, N_grid=2 * r)
print(f"eq L, M, N = {eq.L}, {eq.M}, {eq.N}")

# desc-optimize-stored-energy.py's make_config(250, 1.2), algebraic Er
st_config = dict(
    ParticleSourceCenter=0.2, ParticleSourceHeight=1.2, ParticleSourceWidth=0.6,
    NBICenter=0.0, NBIPower=250.0, NBIWidth=0.15, ECHCenter=0.0, ECHPower=0.0,
    ECHWidth=0.1, EdgeTemperature=1.5, EdgeDensity=0.5, ADCoefficient=0.0,
    n0=3.0, T0=3.0, FusionFactor=1.0, evolveDensity=True, reuseF1=False,
)
params = StellaratorParams(
    StellaratorConfig(rhoUpper=0.9, **st_config),
    **normalization_args({"n0": 1e19, **field_norm_from_eq(eq)}),
)

rho = jnp.asarray(sorted(args.rho))
yw = yancc_data.from_eq(rho, eq=eq, nt=args.nt, nz=args.nz, na=args.na, nx=args.nx)
field, vp, vpp = yw.get_fields_at_rho(rho)


def profiles(x):
    n = StellaratorState.initial_profile(x, 0.5, 3.0, 0.9, 2.0)
    T = StellaratorState.initial_profile(x, 1.5, 3.0, 0.9, 2.0)
    return jnp.stack([n, 1.5 * n * T, 1.5 * n * T])


def vp_of(x):
    return jnp.interp(x, yw.rho, yw.Vp)


u = jax.vmap(lambda x: profiles(x) * vp_of(x))(rho)
q = jax.vmap(jax.jacfwd(lambda x: profiles(x) * vp_of(x)))(rho)
npts = len(rho)
states = State(
    Variable_=u,
    Derivative_=q,
    Flux_=jnp.zeros_like(u),
    Aux_=jnp.zeros((npts, 2)),  # Er, bootstrap
    Scalars_=None,
    VariableDot_=jnp.zeros_like(u),
)
axes = (State.vmap_axes(), 0, None, 0, 0, 0, None, None, None, None)
call = (states, rho, 0.0, field, vp, vpp, yw.pitchgrid, yw.speedgrid, params, True)


class NoBootstrap(eqx.Module):
    def __call__(self, *a):
        fluxes, aux = compute_dke_sol(*a)
        return fluxes, aux[:1]


def field_jac_of(flux):
    def jac(states, positions, t, field, vp, vpp, pg, sg, params, ev):
        return eqx.filter_jacrev(
            lambda tr: flux(states, positions, t, *tr, pg, sg, params, ev)
        )((field, vp, vpp))

    return jac


variants = {}
for bs, flux in (("with bootstrap", compute_dke_sol), ("no bootstrap", NoBootstrap())):
    if "flux" in args.which:
        variants[f"flux   {bs}"] = flux
    if "state" in args.which:
        variants[f"d/dstate {bs}"] = eqx.filter_jacrev(flux)
    if "field" in args.which:
        variants[f"d/dfield {bs}"] = (
            dke_field_jac if flux is compute_dke_sol else field_jac_of(flux)
        )


def block(tree):
    for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array)):
        leaf.block_until_ready()


for name, fn in variants.items():
    f = eqx.filter_jit(eqx.filter_vmap(fn, in_axes=axes))
    t0 = time.perf_counter()
    lowered = f.lower(*call)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    out = compiled(*call)
    block(out)
    t3 = time.perf_counter()
    out = compiled(*call)
    block(out)
    t4 = time.perf_counter()
    print(
        f"{name:28s} trace {t1 - t0:7.1f} s  compile {t2 - t1:7.1f} s  "
        f"run {t3 - t2:7.2f} s  rerun {t4 - t3:7.2f} s",
        flush=True,
    )


# --- the field Jacobian as objective.py reaches it in reverse mode ------------
# DESC's rev-mode Jacobian takes jax.vjp of the MaNTA term, so the custom JVP
# rule -- and the adjoint's dke_field_jac inside it -- runs while JAX is
# linearising the rule, not as a plain call. Same function, timed both ways.
if "vjp" in args.which:
    field_jac = eqx.filter_jit(eqx.filter_vmap(dke_field_jac, in_axes=axes))

    def run_field_jac(fld):
        out = field_jac(states, rho, 0.0, fld, vp, vpp, yw.pitchgrid, yw.speedgrid, params, True)
        block(out)
        return out

    def term_value(fld):
        return jnp.sum(vp) + 0.0 * jax.flatten_util.ravel_pytree(eqx.filter(fld, eqx.is_inexact_array))[0].sum()

    def jac_in_rule(fld):
        t0 = time.perf_counter()
        jac = run_field_jac(fld)
        print(f"  field jac inside the rule: {time.perf_counter() - t0:7.1f} s, "
              f"field leaves concrete: {not any(isinstance(l, jax.core.Tracer) for l in jax.tree.leaves(fld))}",
              flush=True)
        return jnp.sum(jax.flatten_util.ravel_pytree(jax.tree.map(jnp.sum, eqx.filter(jac, eqx.is_inexact_array)))[0])

    if args.rule == "jvp":
        @eqx.filter_custom_jvp
        def term(fld):
            return term_value(fld)

        @term.def_jvp
        def term_jvp(primals, tangents):
            (fld,), (v,) = primals, tangents
            g = jac_in_rule(fld)
            dv = jax.flatten_util.ravel_pytree(eqx.filter(v, eqx.is_inexact_array))[0].sum()
            return term(fld), g * dv
    else:
        @eqx.filter_custom_vjp
        def term(fld):
            return term_value(fld)

        @term.def_fwd
        def term_fwd(perturbed, fld):
            return term_value(fld), jac_in_rule(fld)

        @term.def_bwd
        def term_bwd(g, ct, perturbed, fld):
            return jax.tree.map(lambda l: g * ct * jnp.ones_like(l), eqx.filter(fld, eqx.is_inexact_array))

    t0 = time.perf_counter()
    run_field_jac(field)
    print(f"field jac plain (incl. compile): {time.perf_counter() - t0:7.1f} s", flush=True)
    t0 = time.perf_counter()
    run_field_jac(field)
    print(f"field jac plain:                 {time.perf_counter() - t0:7.1f} s", flush=True)
    if args.rule == "jvp":
        t0 = time.perf_counter()
        eqx.filter_jvp(term, (field,), (eqx.filter(field, eqx.is_inexact_array),))
        print(f"jax.jvp of the term:             {time.perf_counter() - t0:7.1f} s", flush=True)
    t0 = time.perf_counter()
    with jax.log_compiles(True):
        y, pullback = eqx.filter_vjp(term, field)
    print(f"jax.vjp of the term:             {time.perf_counter() - t0:7.1f} s", flush=True)

    # does each vjp compile again, or only the first?
    with jax.log_compiles(True):
        for k in range(2):
            t0 = time.perf_counter()
            eqx.filter_vjp(term, field)
            print(f"jax.vjp of the term, again ({k}): {time.perf_counter() - t0:7.1f} s", flush=True)
        t0 = time.perf_counter()
        run_field_jac(field)
        print(f"field jac plain, after:          {time.perf_counter() - t0:7.1f} s", flush=True)
