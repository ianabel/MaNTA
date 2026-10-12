"""T3D's ReLU noise benchmark through MaNTA, over several noise realisations.

    python noise_benchmark.py                       # defaults below
    python noise_benchmark.py --seeds 3 --tol 5 --workers 4
    python noise_benchmark.py --csv runs.csv

Every run is a steady solve of `relu_ke.ReluKeCase` with T3D's noise (uniform,
+-`noise`, fresh every call) at one seed, one Jacobian and one stopping test:
either `SteadyStateTest = "Residual"` at a `SteadyStateTolerance` (`--tol`), or
`"NewtonStep"` at `Relative_tolerance = Absolute_tolerance` (`--ns-rtol`), with
`SuppressAlgebraicError` so that only u is tested -- the noisy flux puts its
noise straight into sigma, an algebraic unknown no step can damp. The
Jacobians are:

* "control" -- exact derivatives of the noiseless fluxes;
* "full"    -- forward differences in all six box inputs;
* "t3d"     -- T3D's own scheme: three kappa perturbations at T3D's step
               max(f |kappa|, f), for each factor f given (T3D's is 0.5).

Residual tolerances are absolute -- ||F|| has no natural scale here -- and are
best read against the residual floor, which one more run per seed measures: the
control at a tolerance (1e-10) that no noisy solve meets, so it stops on
MaxRejectedSteps. NewtonStep runs at the full Jacobian's arm are skipped: the
Residual rows show it costs twice T3D's scheme for the same iterations.

Errors are relative L_inf in n_e, p_H and p_e against the noiseless solution at
the same discretisation (`noise`) and at 64 cells, k = 4 (`converged`). Cost is
box calls, the unit a GX-like model charges in.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

FLOOR_TOL = 1.0e-10
SAMPLES = np.linspace(0.0, 0.8, 401)


def configuration(cells, degree, tolerance, test="Residual"):
    newton_step = test == "NewtonStep"
    return {
        "OutputFilename": "noise_benchmark",
        "PolynomialDegree": degree,
        "GridSize": cells,
        "LowerBoundary": 0.0,
        "UpperBoundary": 0.8,
        "Relative_tolerance": tolerance if newton_step else 1.0e-8,
        "Absolute_tolerance": tolerance if newton_step else 1.0e-8,
        "delta_t": 1.0e-2,
        "SteadyStateTest": test,
        "SuppressAlgebraicError": newton_step,
        # Arms the steady solve; read only by the Residual test.
        "SteadyStateTolerance": FLOOR_TOL if newton_step else tolerance,
        "WriteOutput": False,
    }


def _quiet():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def profiles(runner):
    return np.stack([np.asarray(runner.getSolution(i, list(SAMPLES))).reshape(-1)
                     for i in range(3)])


def solve(run):
    import manta
    from relu_ke import ReluKeCase, ReluFluxes
    box = ReluFluxes(noise=run["noise"], seed=run["seed"])
    f = run["step"] or 0.5
    case = ReluKeCase(box, jacobian=run["jacobian"], rel_step=f, abs_step=f)
    runner = manta.Runner(case)
    runner.configure(configuration(run["cells"], run["degree"], run["tolerance"], run["test"]))
    case.reset_counts()
    try:
        runner.run_ss()
    except RuntimeError:
        pass
    stats = runner.steadyStats()
    return dict(run, outcome=stats["outcome"].name, steps=stats["steps"],
                rejected=stats["rejected"], newton=stats["newton_iterations"],
                residual=stats["residual_norm"], calls=case.value_calls + case.fd_calls,
                guarded=case.guarded, u=profiles(runner))


def noiseless(cells, degree):
    import manta
    from relu_ke import ReluKeCase
    runner = manta.Runner(ReluKeCase(jacobian="control"))
    runner.configure(configuration(cells, degree, FLOOR_TOL, "Residual"))
    runner.run_ss()
    return profiles(runner)


def runs_for(args):
    runs = []
    for seed in range(args.seeds):
        common = dict(cells=args.cells, degree=args.degree, noise=args.noise, seed=seed,
                      test="Residual")
        runs.append(dict(common, kind="floor", jacobian="control", step=0.0,
                         tolerance=FLOOR_TOL))
        for tol in args.tol:
            at = dict(common, kind="solve", tolerance=tol)
            runs.append(dict(at, jacobian="control", step=0.0))
            runs.append(dict(at, jacobian="full", step=args.step[0]))
            for f in args.step:
                runs.append(dict(at, jacobian="t3d", step=f))
        for rtol in args.ns_rtol:
            at = dict(common, kind="solve", test="NewtonStep", tolerance=rtol)
            runs.append(dict(at, jacobian="control", step=0.0))
            runs.append(dict(at, jacobian="t3d", step=args.step[0]))
    return runs


def errors(u, ref):
    return [np.max(np.abs(u[i] - ref[i])) / np.max(np.abs(ref[i])) for i in range(3)]


def summarise(results):
    def fmt(e):
        return f"{np.median(e):.1e} ({np.max(e):.1e})"

    floor = [r for r in results if r["kind"] == "floor"]
    res = np.array([r["residual"] for r in floor])
    print(f"\nresidual floor ||F||: median {np.median(res):.2f}, range {res.min():.2f}-{res.max():.2f}; "
          f"reaching it cost a median {np.median([r['calls'] for r in floor]):.0f} box calls")
    en = np.array([r["err_noise"] for r in floor])
    print(f"noise error there (n, p_H, p_e), median (max): {fmt(en[:, 0])}, {fmt(en[:, 1])}, {fmt(en[:, 2])}\n")

    groups = {}
    for r in results:
        if r["kind"] == "solve":
            groups.setdefault((r["test"], r["tolerance"], r["jacobian"], r["step"]), []).append(r)
    print("| test | tol | Jacobian | step | converged | box calls, median (max) | Newton | guard | "
          "noise error n | p_H | p_e | error vs converged p_e |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(groups):
        rs = groups[k]
        calls = np.array([r["calls"] for r in rs])
        en = np.array([r["err_noise"] for r in rs])
        ec = np.array([r["err_conv"] for r in rs])
        conv = sum(r["outcome"] == "Converged" for r in rs)
        step = "--" if k[2] == "control" else (f"{k[3]:g}" if k[2] == "t3d" else f"{k[3]:g} (u, kappa)")
        print(f"| {k[0]} | {k[1]:g} | {k[2]} | {step} | {conv}/{len(rs)} | {np.median(calls):.0f} ({calls.max()}) | "
              f"{np.median([r['newton'] for r in rs]):.0f} | {np.median([r['guarded'] for r in rs]):.0f} | "
              f"{fmt(en[:, 0])} | {fmt(en[:, 1])} | {fmt(en[:, 2])} | {np.median(ec[:, 2]):.1e} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", type=int, default=16)
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--step", type=float, nargs="+", default=[0.5, 0.2],
                   help="T3D's step factor f: dkappa = max(f |kappa|, f); the first is also full's")
    p.add_argument("--tol", type=float, nargs="+", default=[3.0, 5.0, 10.0, 20.0],
                   help="SteadyStateTolerance for the Residual test, absolute")
    p.add_argument("--ns-rtol", type=float, nargs="*", default=[1e-2, 3e-3],
                   help="Relative_tolerance = Absolute_tolerance for the NewtonStep test")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--csv")
    args = p.parse_args()

    runs = runs_for(args)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_quiet) as pool:
        ref_same = pool.submit(noiseless, args.cells, args.degree)
        ref_conv = pool.submit(noiseless, 64, 4)
        ref_same, ref_conv = ref_same.result(), ref_conv.result()
        disc = errors(ref_same, ref_conv)
        print(f"{args.cells} cells, k = {args.degree}, noise +-{args.noise}: discretisation error "
              f"(n, p_H, p_e) {disc[0]:.1e} {disc[1]:.1e} {disc[2]:.1e}; {len(runs)} runs, "
              f"{args.seeds} seeds", flush=True)
        results = list(pool.map(solve, runs))

    for r in results:
        u = r.pop("u")
        r["err_noise"] = errors(u, ref_same)
        r["err_conv"] = errors(u, ref_conv)
    summarise(results)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
