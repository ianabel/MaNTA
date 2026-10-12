"""What a steady solve of the toy model costs, and how accurate it is, when its
flux is a stochastic black box -- over several noise realisations.

    python noise_study.py                      # the defaults below, ~minutes
    python noise_study.py --seeds 3 --noise 1e-3 --workers 4
    python noise_study.py --csv runs.csv       # every run, one row each

Every run is a steady solve of `blackbox.BlackBoxCase` at one noise level `eps`
(relative, fresh per point per call), one seed, one Jacobian and one
`SteadyStateTolerance`:

* "analytic"  -- the box's exact, noise-free derivatives: the control;
* "full"      -- forward differences in u and q;
* "gradients" -- forward differences in q only;

the two differenced ones at steps dz = f sqrt(eps) for each factor f given,
sqrt(eps) being the forward-difference optimum. Tolerances are multiples of
eps. One more run per seed and noise level measures the residual floor: the
control at a tolerance (1e-11) no noisy solve can meet, so it stops on
MaxRejectedSteps and its last ||F|| is the floor.

Two errors are reported, both relative L_inf with u sampled up to both faces
of every cell (see README.md): against the exact steady state, and against the
noiseless discrete solution, which isolates what the noise did from the
discretisation error printed in the header. Cost is box calls, the unit a
black box charges in -- every call evaluates every point of the batch.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

FLOOR_TOL = 1.0e-11


def samples(cells, inside=41):
    h = 1.0 / cells
    s = np.linspace(1e-9, 1.0 - 1e-9, inside)
    return np.concatenate([i * h + h * s for i in range(cells)])


def configuration(cells, degree, tolerance):
    return {
        "OutputFilename": "noise_study",
        "PolynomialDegree": degree,
        "GridSize": cells,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "Relative_tolerance": 1.0e-10,
        "Absolute_tolerance": 1.0e-10,
        "delta_t": 1.0,
        # This study measures the residual test -- its tolerances are multiples
        # of eps in ||F|| -- so it names it rather than taking the default.
        "SteadyStateTest": "Residual",
        "SteadyStateTolerance": tolerance,
        "WriteOutput": False,
    }


def _quiet():
    # The solver reports on stdout and SUNDIALS on stderr, both from C/C++, so
    # a worker's descriptors go nowhere; results come back by pickle.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def solve(run):
    """One steady solve. `run` is a dict; returns it with the results added."""
    import manta
    from blackbox import ToyFlux, BlackBoxCase

    box = ToyFlux(noise=run["noise"], seed=run["seed"])
    dz = run["dz_factor"] * np.sqrt(run["noise"]) if run["dz_factor"] else 1.0e-6
    case = BlackBoxCase(box, run["jacobian"], dz=dz, du=dz)
    runner = manta.Runner(case)
    runner.configure(configuration(run["cells"], run["degree"], run["tolerance"]))
    case.reset_counts()
    failure = ""
    try:
        runner.run_ss()
    except RuntimeError as e:
        failure = str(e).splitlines()[0]
    stats = runner.steadyStats()
    u = np.asarray(runner.getSolution(0, list(samples(run["cells"])))).reshape(-1)
    return dict(run, outcome=stats["outcome"].name, failure=failure,
                steps=stats["steps"], rejected=stats["rejected"],
                newton=stats["newton_iterations"], residual=stats["residual_norm"],
                calls=case.value_calls + case.fd_calls, fd_calls=case.fd_calls,
                floored=case.floored_points, u=u)


def reference(cells, degree):
    """The noiseless discrete solution, by the control."""
    import manta
    from blackbox import ToyFlux, BlackBoxCase
    runner = manta.Runner(BlackBoxCase(ToyFlux(), "analytic"))
    runner.configure(configuration(cells, degree, FLOOR_TOL))
    runner.run_ss()
    return np.asarray(runner.getSolution(0, list(samples(cells)))).reshape(-1)


def runs_for(args):
    runs = []
    base = dict(cells=args.cells, degree=args.degree)
    for noise in args.noise:
        for seed in range(args.seeds):
            common = dict(base, noise=noise, seed=seed)
            runs.append(dict(common, kind="floor", jacobian="analytic",
                             dz_factor=0.0, tol_factor=0.0, tolerance=FLOOR_TOL))
            for tf in args.tol:
                at = dict(common, kind="solve", tol_factor=tf, tolerance=tf * noise)
                runs.append(dict(at, jacobian="analytic", dz_factor=0.0))
                for jacobian in ("full", "gradients"):
                    for f in args.dz:
                        runs.append(dict(at, jacobian=jacobian, dz_factor=f))
    return runs


def summarise(results):
    def key(r):
        return (r["noise"], r["kind"], r["tol_factor"], r["jacobian"], r["dz_factor"])

    groups = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)

    for noise in sorted({r["noise"] for r in results}):
        print(f"\n## eps = {noise:.0e}\n")
        floor = groups.get((noise, "floor", 0.0, "analytic", 0.0), [])
        if floor:
            ratio = np.array([r["residual"] for r in floor]) / noise
            calls = np.array([r["calls"] for r in floor])
            err = np.array([r["err_noise"] for r in floor])
            print(f"residual floor ||F|| / eps: median {np.median(ratio):.2f}, "
                  f"range {ratio.min():.2f}-{ratio.max():.2f}; reaching it cost a median "
                  f"{np.median(calls):.0f} box calls and left a noise error of median "
                  f"{np.median(err):.1e} (max {err.max():.1e})\n")
        print("| tol / eps | Jacobian | dz / sqrt(eps) | converged | box calls, median (max) "
              "| Newton, median | guard trips, median | noise error, median (max) "
              "| error vs exact, median |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(k for k in groups if k[0] == noise and k[1] == "solve"):
            rs = groups[k]
            calls = np.array([r["calls"] for r in rs])
            newton = np.array([r["newton"] for r in rs])
            floored = np.array([r["floored"] for r in rs])
            en = np.array([r["err_noise"] for r in rs])
            ee = np.array([r["err_exact"] for r in rs])
            conv = sum(r["outcome"] == "Converged" for r in rs)
            dz = "--" if k[3] == "analytic" else f"{k[4]:g}"
            print(f"| {k[2]:g} | {k[3]} | {dz} | {conv}/{len(rs)} | "
                  f"{np.median(calls):.0f} ({calls.max()}) | {np.median(newton):.0f} | "
                  f"{np.median(floored):.0f} | {np.median(en):.1e} ({en.max():.1e}) | "
                  f"{np.median(ee):.1e} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", type=int, default=48)
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--noise", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--dz", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                   help="difference steps, as multiples of sqrt(eps)")
    p.add_argument("--tol", type=float, nargs="+", default=[10.0, 20.0, 50.0],
                   help="SteadyStateTolerance, as multiples of eps")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--csv", help="write every run to this file")
    args = p.parse_args()

    from toy_model import ExactSteadyState
    x = samples(args.cells)
    exact = ExactSteadyState(x)
    scale = np.max(np.abs(exact))
    runs = runs_for(args)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_quiet) as pool:
        ref = pool.submit(reference, args.cells, args.degree).result()
        disc = np.max(np.abs(ref - exact)) / scale
        print(f"{args.cells} cells, k = {args.degree}: discretisation error {disc:.2e} "
              f"(noiseless, relative L_inf); {len(runs)} runs, {args.seeds} seeds, "
              f"{args.workers} workers", flush=True)
        results = list(pool.map(solve, runs))

    for r in results:
        u = r.pop("u")
        r["err_noise"] = np.max(np.abs(u - ref)) / scale
        r["err_exact"] = np.max(np.abs(u - exact)) / scale

    summarise(results)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
