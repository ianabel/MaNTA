"""Finite-difference checks of the ITG term's gradient through `objective.py`.

The ITG heat flux reaches the objective's JVP by a route of its own: DESC's
`params` -> `ITGFluxBuilder` -> the `ITGFlux` passed to `make_objective` as the
fourth element of `tree_in` -> its tangent, interpolated to MaNTA's nodes and
contracted with the adjoint's ITG columns (`StellaratorAdjointProblem`'s
`(signals, B_ref)` block). Two checks, each against a central difference:

* `params`: the yancc fields held fixed and only the `params` handed to the
  builder perturbed. Under TEST_STELLARATOR the neoclassical stand-in reads
  the field only through V' and V'', which do not move, so the whole
  derivative must come through the ITG columns -- the check fails if they
  are missing, misordered or mis-scaled.
* `proximal`: the arrangement `scan_eq_ambipolar.py` uses -- an
  ObjectiveFunction wrapped in ProximalProjection with ForceBalance -- with
  the objective as `ObjectiveFromUserWithParams`, so the fields *and* the ITG
  geometry follow the equilibrium. `jvp_scaled` along one boundary R mode
  against a central difference of `compute_scaled`, which re-solves force
  balance at each point. The ITG share of the JVP is reported, so a pass on
  a gradient with no ITG content in it is visible as such. It then sweeps the
  mode (`--sweep`, `--sweep-frac`) and saves dG/dR from the adjoint and from
  finite differences against the mode value to `--plot`, the picture
  scan_eq_ambipolar.py draws.

Run only with TEST_STELLARATOR=true, from a scratch directory (MaNTA writes
its output and restart files into the cwd):

    TEST_STELLARATOR=true PYTHONPATH=$PWD/build/python \\
        python python-physics/stellarator/diagnostics/fd_check_itg.py --mode both
"""

import os
import sys

# the stellarator modules (objective, itg_wrapper, config, ...) are one level up
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("TEST_STELLARATOR") != "true":
    sys.exit(
        "fd_check_itg.py: refusing to run without TEST_STELLARATOR=true "
        "(the real drift-kinetic solve is too expensive for this check)"
    )

import argparse
import tempfile
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import desc.examples
import desc.io
import yancc
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.optimize._constraint_wrappers import ProximalProjection

import manta as MaNTA
import objective as objective_module
from objective import ObjectiveFromUserWithParams, make_objective
from itg_wrapper import ITGFluxBuilder, build_itg_proxy
from fd_check_objective import DATA_NAMES, YANCC_RES, load_eq, make_config

MODEL_ROOT = os.environ.get(
    "ITG_MODEL_DIR", "/home/eatocco/projects/itg-proxy/demo/neural_networks"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=["params", "proximal", "both"], default="both")
    p.add_argument("--eq", default="precise_QA", help="DESC example name or .h5 path")
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--M", type=int, default=None, help="DESC grid M (default eq.M_grid)")
    p.add_argument("--N", type=int, default=None, help="DESC grid N (default eq.N_grid)")
    p.add_argument("--itg-rho", type=int, default=8, help="ITG table points in [0.1, 1]")
    p.add_argument("--n-ensemble", type=int, default=3)
    p.add_argument("--mode-MN", default="1,1", help="boundary R mode M,N to perturb")
    p.add_argument("--eps", type=float, default=1e-4, help="relative perturbation size")
    p.add_argument("--tol", type=float, default=1e-3, help="relative agreement required")
    p.add_argument("--ss-tol", type=float, default=1e-10, help="primal SteadyStateTolerance")
    p.add_argument("--evolve-density", action="store_true")
    p.add_argument("--itg-scale", type=float, default=1e-2, help="ITGScale")
    p.add_argument("--objective", default="StoredEnergy",
                   help="StellaratorTransport objective G, e.g. ITGHeatFlux")
    p.add_argument("--sweep", type=int, default=7,
                   help="proximal: points in the boundary-mode sweep plotted (0 to skip)")
    p.add_argument("--sweep-frac", type=float, default=0.1,
                   help="proximal: sweep the mode over +/- this fraction of its value")
    p.add_argument("--plot", default="fd_check_itg_proximal.png",
                   help="proximal: where to save the sweep plot (relative to the cwd at launch)")
    p.add_argument(
        "--rescale",
        default="none",
        help="a [m], <B> [T] to rescale the equilibrium to (e.g. the driver's "
        "1.7,5.86), or 'none'. Changes the geometry only: the transported ITG "
        "flux is normalised to PlasmaConstants' B0 and a, not the equilibrium's",
    )
    return p.parse_args()


def objective_fun_factory(manta_objective, builder):
    """`fun(grid, data, params)` for ObjectiveFromUserWithParams.

    The data keys are written out literally, as in the driver's
    `objective_from_user_fun`: ObjectiveFromUser finds what to compute by
    reading `data[...]` out of this function's source.
    """

    def fun(grid, data, params):
        yancc_dat = {
            "B_sup_t": data["B^theta"],
            "B_sup_z": data["B^zeta"],
            "B_sub_t": data["B_theta"],
            "B_sub_z": data["B_zeta"],
            "Bmag": data["|B|"],
            "dBdt": data["|B|_t"],
            "dBdz": data["|B|_z"],
            "sqrtg": data["sqrt(g)"],
        }
        yancc_dat = {
            key: grid.meshgrid_reshape(val, "rtz") for key, val in yancc_dat.items()
        }
        yancc_dat["Psi"] = grid.compress(
            data["Psi"] / grid.nodes[:, 0] ** 2, surface_label="rho"
        )
        yancc_dat["a_minor"] = jnp.full(grid.num_rho, data["a"])
        yancc_dat["R_major"] = jnp.full(grid.num_rho, data["R0"])
        yancc_dat["iota"] = grid.compress(data["iota"], surface_label="rho")
        yancc_dat["rho"] = grid.compress(grid.nodes[:, 0], surface_label="rho")

        V = grid.compress(data["V(r)"])
        Vp = grid.compress(data["V_r(r)"]) / V[-1]
        Vpp = grid.compress(data["V_rr(r)"]) / V[-1]
        fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)

        itg = builder(params)
        G, _ = manta_objective((fields, Vp, Vpp, itg), grid)
        return jnp.atleast_1d(jnp.ravel(G)[0])

    return fun


def verdict(jvp, fd, scale_floor, tol):
    scale = max(abs(jvp), abs(fd))
    if scale < scale_floor:
        return 0.0, "ok (both ~0)"
    rel = abs(jvp - fd) / scale
    return rel, ("ok" if rel < tol else "FAIL")


def check_params(args, fun, grid, data, eq):
    """Fields fixed, builder params perturbed: all of it is ITG."""
    print("\n=== params: builder input perturbed, fields fixed ===")

    def f(p):
        return fun(grid, data, p)[0]

    base = {k: jnp.asarray(v, dtype=float) for k, v in eq.params_dict.items()}
    # a smooth relative perturbation of the shape: every R and Z mode scaled
    # by a factor falling with mode number, so it is an equilibrium-like change
    v = {k: jnp.zeros_like(val) for k, val in base.items()}
    for k, basis in (("R_lmn", eq.R_basis), ("Z_lmn", eq.Z_basis)):
        mn = np.abs(basis.modes[:, 1]) + np.abs(basis.modes[:, 2])
        v[k] = base[k] * jnp.asarray(1.0 / (1.0 + mn))

    t0 = time.time()
    G0 = float(f(base))  # the JVP restarts from this solve's restart file
    print(f"base G = {G0:.12e}  ({time.time() - t0:.1f} s)")
    if G0 == 1e-10:
        sys.exit("base solve did not converge (objective returned its 1e-10 sentinel)")

    t0 = time.time()
    _, jvp = jax.jvp(f, (base,), (v,))
    jvp = float(jvp)
    parts = {k: float(np.ravel(x)[0]) for k, x in objective_module.LAST_JVP_PARTS.items()}
    print(f"jvp = {jvp:.6e}  ({time.time() - t0:.1f} s)  parts {parts}")

    eps = args.eps
    t0 = time.time()
    Gp = float(f({k: base[k] + eps * v[k] for k in base}))
    Gm = float(f({k: base[k] - eps * v[k] for k in base}))
    fd = (Gp - Gm) / (2 * eps)
    print(f"fd  = {fd:.6e}  ({time.time() - t0:.1f} s)")

    rel, ok = verdict(jvp, fd, 1e-8 * abs(G0), args.tol)
    if abs(parts["itg"]) < 1e-3 * max(abs(jvp), 1e-300):
        ok = "FAIL (no ITG content: the check would pass without the ITG columns)"
    print(f"params: jvp {jvp: .6e}  fd {fd: .6e}  rel {rel:.2e}  {ok}")
    return ok.startswith("ok")


def check_proximal(args, fun, grid, eq):
    """ProximalProjection over ForceBalance, as scan_eq_ambipolar.py sets it up."""
    print("\n=== proximal: boundary mode through ProximalProjection ===")
    obj = ObjectiveFromUserWithParams(
        fun, eq, target=0, weight=1.0, grid=grid, deriv_mode="fwd"
    )
    o1 = ObjectiveFunction([obj])
    o1.build(use_jit=False)
    prox = ProximalProjection(o1, ObjectiveFunction([ForceBalance(eq=eq)]), eq)
    prox.build()

    M, N = (int(s) for s in args.mode_MN.split(","))
    idx = int(eq.surface.R_basis.get_idx(L=0, M=M, N=N))
    x0 = prox.x(eq)
    # The reduced state is params[arg] for arg in prox._args, concatenated
    # (ProximalProjection.x). scan_eq_ambipolar.py's len(p_l) + len(i_l) + 1
    # offset assumes an argument order that does not hold in general, so the
    # offset is computed from that order and then confirmed.
    print(f"reduced state order: {[(arg, eq.dimensions[arg]) for arg in prox._args]}")
    offset = sum(eq.dimensions[arg] for arg in prox._args[: prox._args.index("Rb_lmn")])
    j = offset + idx
    j_scan = len(eq.p_l) + len(eq.i_l) + 1 + idx
    print(f"Rb_lmn[{idx}] is x[{j}]; scan_eq_ambipolar.py's formula gives x[{j_scan}]")
    if not np.isclose(float(x0[j]), float(eq.Rb_lmn[idx])):
        sys.exit(f"x[{j}] = {float(x0[j])} is not Rb_lmn[{idx}] = {float(eq.Rb_lmn[idx])}")
    t = jnp.zeros(prox.dim_x).at[j].set(1.0)

    t0 = time.time()
    G0 = float(np.ravel(prox.compute_scaled(x0))[0])
    print(f"base G = {G0:.12e}  ({time.time() - t0:.1f} s)  R_{{0,{M},{N}}} = {float(x0[j]):.6e}")

    t0 = time.time()
    jvp = float(np.ravel(prox.jvp_scaled(t, x0))[0])
    parts = {k: float(np.ravel(x)[0]) for k, x in objective_module.LAST_JVP_PARTS.items()}
    print(f"jvp = {jvp:.6e}  ({time.time() - t0:.1f} s)  parts {parts}")

    h = args.eps * max(abs(float(x0[j])), 1e-3)
    t0 = time.time()
    Gp = float(np.ravel(prox.compute_scaled(x0 + h * t))[0])
    Gm = float(np.ravel(prox.compute_scaled(x0 - h * t))[0])
    fd = (Gp - Gm) / (2 * h)
    print(f"fd  = {fd:.6e}  ({time.time() - t0:.1f} s)")

    rel, ok = verdict(jvp, fd, 1e-8 * abs(G0), args.tol)
    share = abs(parts["itg"]) / max(abs(jvp), 1e-300)
    print(
        f"proximal: jvp {jvp: .6e}  fd {fd: .6e}  rel {rel:.2e}  {ok}  "
        f"(ITG share of the jvp {share:.1%})"
    )
    if share < 10 * max(rel, args.tol):
        print(
            "  note: the ITG part is below what this tolerance can resolve, so "
            "this pass does not by itself confirm the ITG columns"
        )
    if args.sweep > 1:
        sweep_proximal(args, prox, t, x0, j, M, N)
    return ok.startswith("ok")


def sweep_proximal(args, prox, t, x0, j, M, N):
    """Sweep R_{0,M,N} and plot dG/dR from the adjoint against finite differences.

    The same picture scan_eq_ambipolar.py draws. At each point the objective
    comes first and the JVP second, since the JVP restarts from the restart
    file the objective's solve wrote. The finite difference is np.gradient
    over the sweep, as there: second order inside it, first order at its
    ends, so it is a picture of agreement rather than a tolerance; the tight
    comparison is the single central difference above.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    v0 = float(x0[j])
    sweep = np.linspace(v0 * (1 - args.sweep_frac), v0 * (1 + args.sweep_frac), args.sweep)
    G, adj, adj_itg = [], [], []
    print(f"\n--- sweep of R_{{0,{M},{N}}} over {sweep[0]:.6e} .. {sweep[-1]:.6e} ---")
    for i, v in enumerate(sweep):
        xi = x0.at[j].set(v)
        t0 = time.time()
        G.append(float(np.ravel(prox.compute_scaled(xi))[0]))
        adj.append(float(np.ravel(prox.jvp_scaled(t, xi))[0]))
        adj_itg.append(float(np.ravel(objective_module.LAST_JVP_PARTS["itg"])[0]))
        print(f"  {i}: R = {v: .6e}  G = {G[-1]:.10e}  dG/dR adjoint {adj[-1]: .6e} "
              f"(ITG {adj_itg[-1]: .6e})  ({time.time() - t0:.1f} s)")
    fd = np.gradient(np.array(G), sweep)
    for v, a, f in zip(sweep, adj, fd):
        print(f"  R = {v: .6e}: adjoint {a: .6e}  FD {f: .6e}  rel {abs(a - f) / max(abs(f), 1e-300):.1e}")

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8))
    label = rf"$R_{{0,{M},{N}}}$"
    ax[0].plot(sweep, G, "o-", color="C3")
    ax[0].set_ylabel("G")
    ax[1].plot(sweep, fd, "o", color="C3", label="finite differences")
    ax[1].plot(sweep, adj, "x", color="C0", markersize=8, label="adjoint")
    ax[1].plot(sweep, adj_itg, "--", color="C0", alpha=0.6, label="adjoint, ITG part")
    ax[1].set_ylabel(rf"$dG/d{label[1:-1]}$")
    ax[1].legend(fontsize=9)
    for a in ax:
        a.set_xlabel(label)
        a.axvline(v0, color="k", linestyle=":", linewidth=1)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    fig.suptitle(f"ITGScale = {args.itg_scale:g}, {args.eq}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=200)
    print(f"  saved {args.plot}")


def main():
    args = parse_args()
    config, nodes = make_config(args)
    config["Stellarator"]["useITGProxy"] = True
    config["Stellarator"]["ITGScale"] = args.itg_scale
    config["Objective"] = args.objective
    eq = load_eq(args.eq)
    if args.rescale != "none":
        from desc.compat import rescale

        a_new, B_new = (float(s) for s in args.rescale.split(","))
        eq = rescale(
            eq, L=("a", a_new), B=("<B>", B_new), scale_pressure=False, copy=True,
            verbose=0,
        )

    points = jnp.asarray(MaNTA.getNodes(nodes, args.degree))
    grid = LinearGrid(
        rho=np.asarray(points),
        M=args.M if args.M is not None else eq.M_grid,
        N=args.N if args.N is not None else eq.N_grid,
        NFP=eq.NFP,
    )
    computed = eq.compute(DATA_NAMES, grid=grid)
    data = {k: jnp.asarray(computed[k]) for k in DATA_NAMES}

    t0 = time.time()
    proxy = build_itg_proxy(
        eq,
        np.linspace(0.1, 1.0, args.itg_rho),
        ensemble_dir=os.path.join(MODEL_ROOT, "cyclic_invariant_models"),
        ensemble_csv=os.path.join(MODEL_ROOT, "deephyper_exp", "results.csv"),
        n_ensemble=args.n_ensemble,
    )
    builder = ITGFluxBuilder(proxy)
    print(f"ITG proxy built ({time.time() - t0:.1f} s)")

    manta_objective = make_objective(config, yancc_res=YANCC_RES)
    fun = objective_fun_factory(manta_objective, builder)

    args.plot = os.path.abspath(args.plot)  # before the chdir below
    workdir = tempfile.mkdtemp(prefix="manta-fd-itg-")
    print(f"run output in {workdir}")
    os.chdir(workdir)

    ok = True
    if args.mode in ("params", "both"):
        ok &= check_params(args, fun, grid, data, eq)
    if args.mode in ("proximal", "both"):
        ok &= check_proximal(args, fun, grid, eq)
    print("\nITG FD check:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
