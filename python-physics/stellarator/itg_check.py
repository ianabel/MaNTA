"""Checks `itg_wrapper` against DESC's NNITGProxy and against itself.

Needs desc (the cj/itg-proxy branch), torch (to read the .pth weights), yancc,
the manta package, and the model data from the itg-proxy demo:

    ITG_MODEL_DIR=/path/to/itg-proxy/demo/neural_networks \
    TEST_STELLARATOR=true python itg_check.py

TEST_STELLARATOR only keeps the import of stellarator code from reaching for a
real yancc solve; nothing here calls one. Exits nonzero on any failed check.
"""

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.constants import elementary_charge

from desc.compat import rescale
from desc.examples import get

from manta.jax import State, vmap_batched
from config import StellaratorConfig, StellaratorParams
from stellarator_state import StellaratorState
from dke_sol import test_flux
from itg_wrapper import ITGFlux, ITGFluxBuilder, SumFlux, build_itg_proxy

MODEL_ROOT = os.environ.get(
    "ITG_MODEL_DIR", "/home/eatocco/projects/itg-proxy/demo/neural_networks"
)
MODEL_DIR = os.path.join(MODEL_ROOT, "cyclic_invariant_models")
MODEL_CSV = os.path.join(MODEL_ROOT, "deephyper_exp", "results.csv")
N_ENSEMBLE = 10

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


eq = rescale(get("precise_QA"), ("a", 1.0), verbose=False)
rho = np.linspace(0.1, 1.0, 10)
alpha = np.array([0.0, np.pi / 2])
kw = dict(ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=N_ENSEMBLE)

print("Building NNITGProxy outside the wrapper (precise_QA, 10 rho x 2 alpha)")
t0 = time.time()
obj = build_itg_proxy(eq, rho, alpha=alpha, **kw)
builder = ITGFluxBuilder(obj, floor="keep")
print(f"  build {time.time() - t0:.1f} s")
t0 = time.time()
itg = builder(eq.params_dict)
print(f"  {time.time() - t0:.1f} s, signals {itg.signals.shape}, B_ref {float(itg.B_ref):.4f} T")
p2 = dict(eq.params_dict, Psi=eq.params_dict["Psi"] * 1.001)
t0 = time.time()
jax.block_until_ready(builder(p2).signals)
print(f"  builder(params) again, new Psi: {time.time() - t0:.1f} s (compiled kernel reused)")

# --- 1. On the tabulation nodes, the bare ensemble is NNITGProxy ------------
print("\n1. Tabulation nodes against NNITGProxy.compute")
worst = 0.0
for aLT, aLn in [(0.5, 0.0), (3.0, 1.0), (7.0, 2.5), (12.0, -0.3)]:
    consts = dict(obj.constants, a_over_LT=aLT, a_over_Ln=aLn)
    Q_desc = np.asarray(obj.compute(eq.params_dict, constants=consts))
    Q_mine = np.array([itg.normalized_heat_flux(r, aLT, aLn) for r in rho])
    worst = max(worst, np.max(np.abs(Q_mine / Q_desc - 1)))
# DESC runs the network in float32, the wrapper in float64
check("matches NNITGProxy at every node", worst < 1e-5, f"max rel diff {worst:.2e}")

# --- 2. Between nodes, interpolating the signals is close to recomputing them
print("\n2. Off-node rho against a fresh NNITGProxy")
rho_off = np.array([0.33, 0.47, 0.81])
# DESC's default tube length holds the toroidal extent solved at a proxy's
# first rho; leading with the table's own rho[0] gives this proxy the same
# extent, i.e. the same tubes, so the comparison is of interpolation alone
obj_off = build_itg_proxy(eq, np.concatenate([rho[:1], rho_off]), alpha=alpha, **kw)
consts = dict(obj_off.constants, a_over_LT=4.0, a_over_Ln=1.0)
Q_desc = np.asarray(obj_off.compute(eq.params_dict, constants=consts))[1:]
Q_mine = np.array([itg.normalized_heat_flux(r, 4.0, 1.0) for r in rho_off])
err = np.max(np.abs(Q_mine / Q_desc - 1))
check("interpolated Q within 2%", err < 2e-2, f"max rel diff {err:.2e}")

# PlasmaConstants carries e rounded to 9 digits, hence rtol 1e-6 below
# --- 3. Units: the W/m^2 value is Q times GX's gyro-Bohm flux ----------------
print("\n3. Gyro-Bohm conversion")
config = StellaratorConfig(
    EdgeTemperature=1.0, EdgeDensity=1.0, NBICenter=0.0, NBIPower=0.0, NBIWidth=0.1
)
params = StellaratorParams(config, _B0=5.0, _a=1.0)
c = params.constants
x0, n_, T_, dT_, dn_ = 0.5, 1.3, 2.0, -6.0, -0.65  # a/LT = 3, a/Ln = 0.5
st = StellaratorState(
    *(jnp.asarray(v) for v in (n_, 0, 0, T_, T_, dn_, 0, 0, dT_, dT_, 0, 0, 0, 0, x0, 1, 0))
)
Q_SI = float(itg.heat_flux_SI(st, x0, params))
m, Z = float(c.IonSpecies.mass), float(c.IonSpecies.charge) / elementary_charge
T_J = T_ * 1000.0 * elementary_charge  # PlasmaConstants' default T0 = 1 keV
vt = np.sqrt(T_J / m)
rho_i = m * vt / (Z * elementary_charge * float(itg.B_ref))
Q_gB = n_ * 1e20 * T_J * vt * (rho_i / float(itg.a_minor)) ** 2
Q_hat = float(itg.normalized_heat_flux(x0, 3.0, 0.5))
check("Q_SI = Q * n T v_t rho_i^2 / a^2", np.isclose(Q_SI, Q_hat * Q_gB, rtol=1e-6),
      f"Q = {Q_hat:.3f} Q_gB, Q_gB = {Q_gB:.3e} W/m^2, Q_SI = {Q_SI:.3e} W/m^2")

# --- 4. Called the way stellarator_multichannel calls compute_dke_sol --------
print("\n4. compute_dke_sol signature, through vmap_batched")
npts = 12
x = jnp.linspace(0.05, 0.95, npts)
vp, vpp = 4 * jnp.pi**2 * x, 4 * jnp.pi**2 * jnp.ones_like(x)  # V' ~ rho
T = 2.0 - 1.5 * x**2
dT = -3.0 * x
n = config.EdgeDensity  # constant-density mode prescribes a flat n here
u = 1.5 * n * T * vp
du = 1.5 * n * (dT * vp + T * vpp)
col = lambda a: a[:, None]
# a real VariableDot column: Physics_Decorator strips the empty one before
# vmapping, and this test calls the flux without it
states = State(col(u), col(du), col(jnp.zeros(npts)), None, None, col(jnp.zeros(npts)))
vmap_axes_wfield = (State.vmap_axes(), 0, None, 0, 0, 0, None, None, None, None)
field = jnp.zeros(npts)  # ignored by ITGFlux; stands in for the yancc Field

def run(fn):
    return vmap_batched(fn, vmap_axes_wfield)(
        states, x, 0.0, field, vp, vpp, None, None, params, False
    )

out = run(itg)
ref = run(test_flux)
check("same output structure as test_flux",
      jax.tree.structure(out) == jax.tree.structure(ref))
flux = np.asarray(out[0][0])
check("flux finite and outward (negative, as compute_dke_sol's)",
      np.all(np.isfinite(flux)) and np.all(flux < 0))

# MaNTA flux = -ITGScale Q n T^(5/2) / sqrt(2) vp: compute_dke_sol's
# normalisation of a gyro-Bohm flux at PlasmaConstants' B0 and a, whose heat
# normalisation is n0 T0 sqrt(2) v_t rho*^2 / a -- computed here from the state
# alone, without the module's gyro-Bohm arithmetic
def closed_form(i):
    s = StellaratorState.from_state(
        State(u[i : i + 1], du[i : i + 1], jnp.zeros(1), None, None), x[i], vp[i], vpp[i], params
    )
    Q = itg.normalized_heat_flux(x[i], -s.dTidrho / s.Ti, -s.dndrho / s.n)
    return -params.config.ITGScale * Q * s.n * s.Ti**2.5 / np.sqrt(2) * vp[i]
man = np.array([np.ravel(closed_form(i))[0] for i in range(npts)])
check("normalisation: ITGScale Q n T^5/2 / sqrt(2) V'", np.allclose(flux, man, rtol=1e-7),  # e: species SI vs PlasmaConstants rounded
      f"max rel {np.max(np.abs(flux / man - 1)):.1e}")

jac = vmap_batched(eqx.filter_jacrev(itg), vmap_axes_wfield)(
    states, x, 0.0, field, vp, vpp, None, None, params, False
)
dflux_du = np.asarray(jac[0][0].Derivative[:, 0])
h = 1e-6 * np.abs(np.asarray(du))
up = vmap_batched(itg, vmap_axes_wfield)(
    State(col(u), col(du + h), col(jnp.zeros(npts)), None, None, col(jnp.zeros(npts))),
    x, 0.0, field, vp, vpp, None, None, params, False)
dn = vmap_batched(itg, vmap_axes_wfield)(
    State(col(u), col(du - h), col(jnp.zeros(npts)), None, None, col(jnp.zeros(npts))),
    x, 0.0, field, vp, vpp, None, None, params, False)
fd = (np.asarray(up[0][0]) - np.asarray(dn[0][0])) / (2 * h)
rel = np.max(np.abs(dflux_du - fd) / np.max(np.abs(fd)))
check("d flux / d u' agrees with finite differences", rel < 1e-6, f"rel err {rel:.1e}")

# --- 5. Floor handling ---------------------------------------------------------
print("\n5. Floor")
itg_sub = ITGFlux(**{f: getattr(itg, f) for f in itg.__dataclass_fields__ if f != "floor"},
                  floor="subtract")
q0_keep = float(itg.normalized_heat_flux(0.5, 0.0, 0.0))
q0_sub = float(itg_sub.normalized_heat_flux(0.5, 0.0, 0.0))
q_hi = [float(m.normalized_heat_flux(0.5, 6.0, 1.0)) for m in (itg, itg_sub)]
check("zero gradient: keep returns the clip, subtract removes it",
      abs(q0_keep - float(itg.Q_floor)) < 0.02 and q0_sub < 0.1 * q0_keep,
      f"keep {q0_keep:.4f}, subtract {q0_sub:.4f}")
check("strong gradient: subtract shifts Q by about the floor",
      abs((q_hi[0] - q_hi[1]) - float(itg.Q_floor)) < 1e-3,
      f"{q_hi[0]:.3f} -> {q_hi[1]:.3f}")

# --- 6. A new equilibrium is new data, not a new compile -----------------------
print("\n6. No retrace for new arrays")
traces = []

@eqx.filter_jit
def evaluate(model, states, x, vp, vpp):
    traces.append(1)
    return eqx.filter_vmap(model, in_axes=vmap_axes_wfield)(
        states, x, 0.0, field, vp, vpp, None, None, params, False
    )

evaluate(itg, states, x, vp, vpp)
itg2 = eqx.tree_at(lambda m: m.signals, itg, itg.signals * 1.01)
evaluate(itg2, states, x, vp, vpp)
check("one trace for two equilibria", len(traces) == 1, f"{len(traces)} traces")

# --- 7. Sum with the neoclassical model ---------------------------------------
print("\n7. SumFlux")
tot = run(SumFlux(test_flux, itg))
check("SumFlux = neoclassical + ITG",
      np.allclose(np.asarray(tot[0][0]), np.asarray(ref[0][0]) + flux))

# --- 8. builder(params) is differentiable in params ---------------------------
print("\n8. d(signals)/d(params) through the builder, against finite differences")
# What a gradient through the builder can get wrong is DESC's geometry chain
# (coordinate map, GX coefficients, arclength resampling), which is smooth and
# is what this differences. The network's own derivative is plain JAX AD with
# no custom rule, and cannot be checked at a fixed step: it is only piecewise
# smooth in the signals (max pooling), and a central difference of the summed
# flux at 1e-5 is off by ~1e-3 on that account alone, in either direction.
# End to end, through the solver, is fd_check_itg.py's job.
base_j = {k: jnp.asarray(val, dtype=float) for k, val in eq.params_dict.items()}
# a smooth, equilibrium-like change of shape: each mode scaled down with mode number
v = {k: jnp.zeros_like(val) for k, val in base_j.items()}
for k, basis in (("R_lmn", eq.R_basis), ("Z_lmn", eq.Z_basis)):
    mn = np.abs(basis.modes[:, 1]) + np.abs(basis.modes[:, 2])
    v[k] = base_j[k] * jnp.asarray(1.0 / (1.0 + mn))
geometry = lambda p: builder(p).signals
t0 = time.time()
_, jvp = jax.jvp(geometry, (base_j,), (v,))
print(f"  jvp {time.time() - t0:.1f} s")
# 1e-3: the geometry is smooth, and 1e-5 and below reach the round-off of
# DESC's 1e-10 coordinate map (9e-4 at 1e-5, measured)
eps = 1e-3
fd = (geometry({k: base_j[k] + eps * v[k] for k in base_j})
      - geometry({k: base_j[k] - eps * v[k] for k in base_j})) / (2 * eps)
rel = float(jnp.max(jnp.abs(jvp - fd)) / jnp.max(jnp.abs(fd)))
check("signal-table jvp agrees with central differences", rel < 1e-4, f"max rel {rel:.1e}")

# --- timing --------------------------------------------------------------------
f = jax.jit(lambda s: vmap_batched(eqx.filter_jacrev(itg), vmap_axes_wfield)(
    s, x, 0.0, field, vp, vpp, None, None, params, False))
jax.block_until_ready(f(states))
t0 = time.time()
jax.block_until_ready(f(states))
print(f"\n{npts}-point Jacobian, compiled: {1e3 * (time.time() - t0):.1f} ms")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
sys.exit(1 if failures else 0)
