"""How close the transported ion temperature gradient sits to the NN ITG
threshold, on the first and last equilibrium of desc-optimize-itg-flux.py's
saved family.

For each equilibrium: a cold-start MaNTA transport solve with the driver's own
configuration (heating, ITG term, normalisation, grid), then at each radius

  a/L_T       = -dTi/drho / Ti            the transported gradient
  a/L_T,crit  where the network's raw Q reaches Q_floor -- below it the
              floor="subtract" flux w softplus((Q - Q_floor)/w) is ~0
  Q/Q_gB      the normalised flux at the transported gradient

"Near marginal" is a/L_T just above a/L_T,crit.

CAVEAT: a/L_T,crit is the *first* a/L_T in the scan where the floored flux
reaches w ln 2. On the doubled-NBI run's final equilibrium the network's Q was
already above that at a/L_T = 0 inside rho ~ 0.9, yet only 0.01-0.04 at the
transported gradient: Q(a/L_T) is not monotonic at small gradients there, and
the first-crossing threshold reads as 0. Plot Q against a/L_T at a few radii
before trusting the critical-gradient column.

    python marginal_stability.py [driver.py] [family.h5]

defaulting to desc-optimize-itg-flux.py and the eq_itg_flux family beside this
script. MaNTA's output (marginal_*.nc) goes to this directory, the figure to
figs/marginal.png. Uses the driver's TEST_STELLARATOR settings.
"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
driver = os.path.abspath(
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(here, "desc-optimize-itg-flux.py")
)
family = os.path.abspath(
    sys.argv[2]
    if len(sys.argv) > 2
    else os.path.join(here, "eq_itg_flux_all_equilibria.h5")
)
outdir = here
os.chdir(os.path.dirname(driver))
# the driver appends a relative "..", which the chdir to outdir below would break
sys.path.insert(0, os.path.dirname(os.path.dirname(driver)))

src = open(driver).read()
cut = src.index("# Built once, on the initial equilibrium")
ns = {"__name__": "__setup__", "__file__": driver}
exec(compile(src[:cut], driver, "exec"), ns)

import numpy as np
import jax
import jax.numpy as jnp
import desc.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from manta.jax import State
from itg_wrapper import make_itg_builder, itg_at, A_OVER_LT_RANGE
from stellarator_multichannel import StellaratorTransport
from stellarator_state import StellaratorState

os.chdir(outdir)  # MaNTA writes its output to the cwd

fam = desc.io.load(family)
eq_init = fam[0]
eq_final = fam[-1]
st_config = ns["st_config"]
itg_builder = make_itg_builder(st_config, eq_init, ns["rho_upper"])

rho = jnp.linspace(0.05, 0.95, 37)
aLT_scan = jnp.linspace(*A_OVER_LT_RANGE, 291)


def solve(eq, name):
    yw = ns["yancc_data"].from_eq(
        ns["yancc_rho"],
        eq=eq,
        nt=ns["yancc_ntheta"],
        nz=ns["yancc_nzeta"],
        **ns["yancc_res"],
    )
    solver = {**ns["initial_solver_config"], "OutputFilename": "marginal_" + name}
    config = ns["normalized"]({"Stellarator": st_config, "Solver": solver})
    itg = itg_at(itg_builder, eq.params_dict)
    st = StellaratorTransport(config, yancc_wrapper=yw, itg=itg)
    ok = st.run()
    print(f"{name}: solve converged = {bool(ok)}", flush=True)

    data = Dataset("marginal_" + name + ".nc")
    x = np.asarray(data.variables["x"][:])
    grp = next(g for g in data.groups.values() if "u" in g.variables)
    u = np.asarray(grp.variables["u"][-1, :])
    q = np.asarray(grp.variables["q"][-1, :])
    data.close()
    ui = np.interp(np.asarray(rho), x, u)
    dui = np.interp(np.asarray(rho), x, q)

    _, vp, vpp = yw.get_fields_at_rho(rho)
    z = jnp.zeros((len(rho), 1))
    state = State.from_manta(
        {
            "Variable": jnp.asarray(ui)[:, None],
            "Derivative": jnp.asarray(dui)[:, None],
            "VariableDot": z,
            "Flux": z,
            "Aux": jnp.zeros((len(rho), 0)),
            "Scalars": [],
        }
    )
    s = jax.vmap(
        StellaratorState.from_state, in_axes=(State.vmap_axes(), 0, 0, 0, None)
    )(state, rho, vp, vpp, st.params)
    aLT = -s.dTidrho / s.Ti
    aLn = -s.dndrho / s.n

    Q = jax.vmap(itg.normalized_heat_flux)(rho, aLT, aLn)
    # threshold: the floored flux equals w ln 2 exactly where Q_nn = Q_floor
    q_thr = float(itg.floor_width) * np.log(2.0)
    Qscan = jax.vmap(
        lambda r, ln: jax.vmap(lambda lt: itg.normalized_heat_flux(r, lt, ln))(aLT_scan)
    )(rho, aLn)
    Qscan = np.asarray(Qscan)
    crit = np.full(len(rho), np.nan)
    for i in range(len(rho)):
        above = np.nonzero(Qscan[i] >= q_thr)[0]
        if len(above) and above[0] > 0:
            j = above[0]
            a0, a1 = float(aLT_scan[j - 1]), float(aLT_scan[j])
            q0, q1 = Qscan[i, j - 1], Qscan[i, j]
            crit[i] = a0 + (q_thr - q0) * (a1 - a0) / (q1 - q0)
        elif len(above):
            crit[i] = float(aLT_scan[0])
    return dict(Ti=np.asarray(s.Ti), aLT=np.asarray(aLT), crit=crit, Q=np.asarray(Q))


res = {"initial": solve(eq_init, "initial"), "final": solve(eq_final, "final")}

for name, r in res.items():
    print(f"\n=== {name} ===")
    print("  rho    Ti     a/L_T  a/L_T,crit  ratio   Q/Q_gB")
    for i in range(0, len(rho), 3):
        ratio = r["aLT"][i] / r["crit"][i] if np.isfinite(r["crit"][i]) else np.nan
        print(
            f"  {float(rho[i]):.3f}  {r['Ti'][i]:.3f}  {r['aLT'][i]:6.2f}  "
            f"{r['crit'][i]:9.2f}  {ratio:6.2f}  {r['Q'][i]:8.3f}"
        )

fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
for (name, r), c in zip(res.items(), ["r", "b"]):
    axs[0].plot(rho, r["Ti"], c, label=name)
    axs[1].semilogy(rho, r["Q"], c, label=name)
axs[0].set_ylabel(r"$T_i (keV)$")
axs[1].set_ylabel(r"$Q_i / Q_{GB}$")
for a in axs:
    a.set_xlabel("rho")
    a.legend(fontsize=7)
fig.tight_layout()
os.makedirs(os.path.join(outdir, "figs"), exist_ok=True)
fig.savefig(os.path.join(outdir, "figs", "compute_itg_gb.pdf"), dpi=500)
# print("\nwrote", os.path.join(outdir, "figs", "marginal.png"))
