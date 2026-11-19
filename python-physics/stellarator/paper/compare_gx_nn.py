#!/usr/bin/env python
"""Overlay GX ion heat fluxes on the NN ITG proxy's, along a MaNTA profile.

The GX runs are the ones `gx_inputs_from_manta.py` set up: a flux tube per rho
from the DESC equilibrium, with the gradients of a MaNTA run. This reads the
same equilibrium and MaNTA output, so the network is evaluated at exactly the
gradients the GX inputs were written with, and plots

* the network's Q/Q_gB along the MaNTA profile, as a curve in rho -- raw
  (`floor="keep"`, what the network predicts GX would give) and floored
  (`floor="subtract"`, what MaNTA transports);
* each GX run's time-averaged Q_i/Q_gB, with its standard deviation over the
  averaging window, at its rho;
* the network at each GX run's *own* tprim/fprim, read from the GX output, so a
  gap between the curve and a GX point can be told apart from a gap between
  the gradients (a GX run made from another MaNTA slice, say);
* the GX time traces with the averaging window, as `heat_flux.py` plots them.

    python compare_gx_nn.py eq_itg_flux_all_equilibria.h5 \\
        stellarator_opteq_itg_flux.nc gx_runs/rho* --out figs/gx_vs_nn.pdf

    python compare_gx_nn.py eq.h5 run.nc 0.85="~/sample gx/gx.out.nc"

Each GX argument is a directory holding one `*.out.nc` (as
`gx_inputs_from_manta.py` lays them out; rho is read from the `rhoX.XXX` name)
or a file, either given as `RHO=PATH` when the path does not carry rho.

Conventions, and what is and is not checked:

* GX's ion species is the one with `species_type == 0`. Its `HeatFlux_st` is
  averaged exactly as the training database was
  (`20241119-02-extract_heat_flux_from_gx_runs_and_tprim_fprim.py` in the zenodo
  archive's data_generation_and_analysis/): over `t > --t-start` (150), with
  samples of `|Q| >= 1e6` dropped, and the last value if the run ended before
  `t_start`. `--navgfac f` averages over the last `1 - f` of the run instead,
  as `heat_flux.py` does.
* The network's training target is `Q_avgs_without_FSA_grad_x`, which that
  script forms as `HeatFlux_st * <|grad x|>` -- "removing the division by
  fluxDenom" -- with `<|grad x|>` = mean(jacobian grho) / mean(jacobian) from
  the run's own Geometry group. That is what is plotted; `--raw-gx` plots
  `HeatFlux_st` as it is, and the table prints both. The factor is ~1.01 on
  the sample rho = 0.85 run.
* `--nn-tube per-rho` (the default) re-solves the network's tube at every rho
  for `--length`, as `gx_inputs_from_manta.py` does for GX. `fixed` is what the
  optimisation uses (solved at the innermost rho, toroidal extent held), whose
  tubes drift from the target length with iota -- a few percent in Q.
* The network's field line is `--alpha`, which should be the GX runs'; GX
  records it as `Geometry/alpha` and a mismatch is warned about.
* A run with the density held fixed has no density in its output: its profile
  is `--edge-density`, `--n0` and `--density-exponent`, which must be the run's
  st_config (defaults: `gx_inputs_from_manta.DENSITY_PROFILE`).
* The GX run's tprim/fprim are compared with the MaNTA profile at its rho and a
  difference above 0.5% is warned about: it means the GX input did not come
  from this MaNTA output (or slice, or density profile).

Needs DESC and the NN models: `--model-dir`, else ITG_MODEL_DIR / ITG_NN_DIR,
else the itg-proxy demo's `neural_networks/`. Runs no transport solve.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # the stellarator modules

from gx_inputs_from_manta import (  # noqa: E402
    A_OVER_LN_RANGE,
    DENSITY_PROFILE,
    A_OVER_LT_RANGE,
    load_equilibrium,
    manta_profiles,
)

PROJECTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECTS, "itg-proxy", "demo", "neural_networks")
RHO_IN_NAME = re.compile(r"rho(\d+(?:\.\d*)?)")


# --- GX output ----------------------------------------------------------------


def parse_gx_arg(arg):
    """`RHO=PATH` or `PATH` -> (rho or None, path to the .out.nc)."""
    rho = None
    m = re.match(r"^(\d*\.?\d+)=(.+)$", arg)
    if m:
        rho, arg = float(m.group(1)), m.group(2)
    path = os.path.expanduser(arg)
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.out.nc")))
        if len(found) != 1:
            raise ValueError(f"{path}: expected one *.out.nc, found {len(found)}")
        path = found[0]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if rho is None:
        m = RHO_IN_NAME.search(os.path.abspath(path))
        if m is None:
            raise ValueError(f"no rho in {path!r}: give it as RHO={arg}")
        rho = float(m.group(1))
    return rho, path


Q_CUTOFF = 1.0e6  # the database's: samples at or above are dropped


def read_gx(path, t_start=150.0, navgfac=None):
    """Time-averaged ion heat flux and the run's own inputs from a GX output."""
    from netCDF4 import Dataset

    with Dataset(path) as d:
        species = np.asarray(d["Inputs/Species/species_type"][:])
        ions = np.nonzero(species == 0)[0]
        if len(ions) == 0:
            raise ValueError(f"{path}: no ion species (species_type == 0)")
        i = int(ions[0])
        if "HeatFlux_st" not in d["Diagnostics"].variables:
            raise ValueError(f"{path}: no HeatFlux_st; run GX with fluxes = true")
        t = np.asarray(d["Grids/time"][:])
        Q = np.asarray(d["Diagnostics/HeatFlux_st"][:, i])
        geo = d["Geometry"]
        J = np.asarray(geo["jacobian"][:])
        grho = np.asarray(geo["grho"][:])
        out = {
            "path": path,
            "t": t,
            "Q_t": Q,
            "tprim": float(d["Inputs/Species/T0_prime"][i]),
            "fprim": float(d["Inputs/Species/n0_prime"][i]),
            "grad_x": float(np.sum(J * grho) / np.sum(J)),
            "alpha": float(geo["alpha"][...]),
            "shat": float(geo["shat"][...]),
            "q": float(geo["q"][...]),
        }
    keep = np.abs(Q) < Q_CUTOFF
    if not np.all(keep):
        print(f"{path}: dropping {np.sum(~keep)} samples with |Q| >= {Q_CUTOFF:g}")
    t_k, Q_k = t[keep], Q[keep]
    if navgfac is not None:
        window = np.arange(len(t_k)) >= int(len(t_k) * navgfac)
    else:
        window = t_k > t_start
    if not np.any(window):
        print(f"{path}: run ended before the averaging window; using its last Q")
        out["t_avg_start"] = t_k[-1]
        out["Q"], out["Q_std"] = float(Q_k[-1]), float(np.std(Q_k))
    else:
        out["t_avg_start"] = t_k[window][0]
        out["Q"] = float(np.mean(Q_k[window]))
        out["Q_std"] = float(np.std(Q_k[window]))
    return out


# --- NN -----------------------------------------------------------------------


def nn_fluxes(eq, rho_table, length, alpha, per_rho, model_dir, n_ensemble):
    """The proxy built once on `rho_table`, as (raw, floored) ITGFlux."""
    from itg_wrapper import ITGFluxBuilder, build_itg_proxy

    proxy = build_itg_proxy(
        eq,
        rho_table,
        ensemble_dir=os.path.join(model_dir, "cyclic_invariant_models"),
        ensemble_csv=os.path.join(model_dir, "deephyper_exp", "results.csv"),
        n_ensemble=n_ensemble,
        alpha=np.atleast_1d(alpha),
        target_flux_tube_length=length,
        solve_length_at_compute=per_rho,
    )
    params = eq.params_dict
    return ITGFluxBuilder(proxy, floor="keep")(params), ITGFluxBuilder(
        proxy, floor="subtract"
    )(params)


def evaluate(itg, rho, aLT, aLn):
    import jax
    import jax.numpy as jnp

    f = jax.jit(jax.vmap(itg.normalized_heat_flux))
    return np.asarray(f(jnp.asarray(rho), jnp.asarray(aLT), jnp.asarray(aLn)))


def in_box(aLT, aLn):
    return (A_OVER_LT_RANGE[0] < aLT <= A_OVER_LT_RANGE[1]) and (
        A_OVER_LN_RANGE[0] <= aLn <= A_OVER_LN_RANGE[1]
    )


# --- driver -------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("equilibrium", help="DESC .h5 (an Equilibrium or EquilibriaFamily)")
    p.add_argument("manta", help="MaNTA netCDF output the GX inputs were made from")
    p.add_argument("gx", nargs="+", help="GX run directory or .out.nc, or RHO=PATH")
    p.add_argument("--eq-index", type=int, default=-1, help="member of a family")
    p.add_argument("--time-index", type=int, default=-1, help="MaNTA output slice")
    p.add_argument("--edge-density", type=float, default=DENSITY_PROFILE["EdgeDensity"],
                   help="fixed-density runs: st_config's EdgeDensity")
    p.add_argument("--n0", type=float, default=DENSITY_PROFILE["n0"],
                   help="fixed-density runs: st_config's n0")
    p.add_argument("--density-exponent", type=float, default=DENSITY_PROFILE["alpha"],
                   help="fixed-density runs: st_config's alpha")
    p.add_argument("--t-start", type=float, default=150.0,
                   help="average GX over t > this, as the training database did")
    p.add_argument("--navgfac", type=float, default=None,
                   help="instead average over the run after this fraction of it")
    p.add_argument("--raw-gx", action="store_true",
                   help="plot HeatFlux_st itself, not times <|grad x|>")
    p.add_argument("--length", type=float, default=75.4, help="tube length / a")
    p.add_argument("--alpha", type=float, default=0.0, help="field line label")
    p.add_argument("--nn-tube", choices=["per-rho", "fixed"], default="per-rho")
    p.add_argument("--n-rho", type=int, default=25, help="NN curve points")
    p.add_argument("--rho-min", type=float, default=0.1)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--n-ensemble", type=int, default=10)
    p.add_argument("--out", default=os.path.join(HERE, "figs", "gx_vs_nn.pdf"))
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    model_dir = (
        args.model_dir
        or os.environ.get("ITG_MODEL_DIR")
        or os.environ.get("ITG_NN_DIR")
        or DEFAULT_MODEL_DIR
    )
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"no NN model directory {model_dir}; pass --model-dir")

    runs = []
    for a in args.gx:
        r, path = parse_gx_arg(a)
        run = read_gx(path, args.t_start, args.navgfac)
        run["rho"] = r
        run["Q_nn_target"] = run["Q"] if args.raw_gx else run["Q"] * run["grad_x"]
        run["Q_std_target"] = run["Q_std"] if args.raw_gx else run["Q_std"] * run["grad_x"]
        runs.append(run)
    runs.sort(key=lambda r: r["rho"])
    rho_gx = np.array([r["rho"] for r in runs])

    from netCDF4 import Dataset

    with Dataset(args.manta) as f:
        x = np.asarray(f.variables["x"][:])
    lo, hi = max(args.rho_min, x[0]), x[-1]
    if np.any(rho_gx < lo) or np.any(rho_gx > hi):
        raise ValueError(f"GX rho {rho_gx} outside the comparable range [{lo:g}, {hi:g}]")
    rho_curve = np.linspace(lo, hi, args.n_rho)
    rho_all = np.unique(np.concatenate([rho_curve, rho_gx]))

    eq = load_equilibrium(args.equilibrium, args.eq_index)
    density = {"EdgeDensity": args.edge_density, "n0": args.n0,
               "alpha": args.density_exponent}
    prof = manta_profiles(args.manta, rho_all, eq, time_index=args.time_index,
                          density=density)

    for run in runs:
        j = int(np.searchsorted(rho_all, run["rho"]))
        run["aLT_manta"], run["aLn_manta"] = prof["a/LTi"][j], prof["a/Ln"][j]
        for name, gx, mt in (
            ("tprim", run["tprim"], run["aLT_manta"]),
            ("fprim", run["fprim"], run["aLn_manta"]),
        ):
            if abs(gx - mt) > 5e-3 * max(abs(mt), 1e-3):
                print(f"WARNING rho = {run['rho']:g}: GX {name} = {gx:.5g} but MaNTA "
                      f"gives {mt:.5g} -- is this GX run from this MaNTA output?")
        if abs(run["alpha"] - args.alpha) > 1e-8:
            print(f"WARNING rho = {run['rho']:g}: GX alpha = {run['alpha']:g}, "
                  f"NN alpha = {args.alpha:g}")

    print(f"\nbuilding the NN proxy on {len(rho_all)} surfaces "
          f"(tube: {args.nn_tube}, length {args.length:g}, alpha {args.alpha:g})")
    itg_raw, itg_floor = nn_fluxes(
        eq, rho_all, args.length, args.alpha, args.nn_tube == "per-rho",
        model_dir, args.n_ensemble,
    )
    Q_raw = evaluate(itg_raw, rho_all, prof["a/LTi"], prof["a/Ln"])
    Q_floor = evaluate(itg_floor, rho_all, prof["a/LTi"], prof["a/Ln"])
    Q_nn_own = evaluate(
        itg_raw, rho_gx, [r["tprim"] for r in runs], [r["fprim"] for r in runs]
    )

    gx_label = "HeatFlux_st" if args.raw_gx else "HeatFlux_st*<|grad x|>"
    print(f"\n  rho    a/LT    a/Ln   <|gx|>   GX Q_st        GX {gx_label:<22s}"
          f"  NN(MaNTA)  NN(GX grads)  NN/GX")
    for run, qo in zip(runs, Q_nn_own):
        j = int(np.searchsorted(rho_all, run["rho"]))
        box = "" if in_box(run["tprim"], run["fprim"]) else "  [outside training box]"
        print(
            f"  {run['rho']:.3f}  {run['tprim']:6.3f}  {run['fprim']:6.3f}  "
            f"{run['grad_x']:6.4f}  {run['Q']:6.3f}+/-{run['Q_std']:<6.3f}  "
            f"{run['Q_nn_target']:8.3f}+/-{run['Q_std_target']:<13.3f}"
            f"  {Q_raw[j]:9.3f}  {qo:12.3f}  {qo / run['Q_nn_target']:5.2f}{box}"
        )

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(9, 3.5))

    on_curve = np.isin(rho_all, rho_curve)
    ax.semilogy(rho_all[on_curve], Q_raw[on_curve], "b-", label="NN (raw)")
    ax.semilogy(rho_all[on_curve], Q_floor[on_curve], "b--", lw=1,
                label="NN (floored, as transported)")
    ax.errorbar(rho_gx, [r["Q_nn_target"] for r in runs],
                yerr=[r["Q_std_target"] for r in runs],
                fmt="ko", capsize=3, label="GX")
    ax.semilogy(rho_gx, Q_nn_own, "rx", ms=7, label="NN at GX's gradients")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$Q_i / Q_\mathrm{GB}$")
    ax.legend(fontsize=7)

    for run in runs:
        line, = axt.plot(run["t"], run["Q_t"] * (1.0 if args.raw_gx else run["grad_x"]),
                         lw=0.8, label=rf"$\rho$ = {run['rho']:g}")
        axt.axvspan(run["t_avg_start"], run["t"][-1], color=line.get_color(), alpha=0.08)
        axt.axhline(run["Q_nn_target"], color=line.get_color(), ls=":", lw=0.8)
    axt.set_xlabel(r"$t\ (v_{ti}/a)$")
    axt.set_ylabel(r"GX $Q_i / Q_\mathrm{GB}$")
    axt.set_xlim(0)
    axt.set_ylim(0)
    axt.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"\nwrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
