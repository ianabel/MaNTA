"""What a reported error bar is worth: the toy model's steady solve when the box
returns a mean and a standard error at every point, as GX through QUENDS does.

    python known_noise_study.py                  # the defaults below
    python known_noise_study.py --seeds 3 --workers 4 --csv runs.csv

Every run is a steady solve of `reporting.FilteredCase` under the NewtonStep
test. The solve is taken one continuation step at a time, and after each step
the run records the box calls so far and the noise error of the state. The
noise error is the relative L_inf distance from the noiseless discrete solution,
in units of eps, with u sampled up to both faces of every cell. A run ends when
the test passes or the budget of box calls is spent.

The arms are what the filter is told about each sample's variance (see
`reporting.py`):

* raw                 -- no filter, as today;
* uniform x f         -- one global relative level, f times the true median;
* weighted x f        -- the reported standard error, times f.

The scenarios are what the box does:

* flat       -- every run has the same number of effective samples;
* spread     -- 4 to 64 of them, log-uniform: the error varies 4x;
* irregular  -- spread, plus a fraction of runs with no steady segment, biased
                by up to +-50% and reported at 100%.

Cost is box calls; the error a run reaches is read at each budget given.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from noise_study import samples

SCENARIOS = {
    "flat": dict(n_min=16, n_max=16, p_irregular=0.0),
    "spread": dict(n_min=4, n_max=64, p_irregular=0.0),
    "irregular": dict(n_min=4, n_max=64, p_irregular=0.02),
}

ARMS = [("raw", 1.0), ("uniform", 1.0), ("uniform", 1 / 3), ("uniform", 3.0),
        ("weighted", 1.0), ("weighted", 0.5), ("weighted", 2.0)]


def configuration(cells, degree, rtol, steps):
    return {
        "OutputFilename": "known_noise_study",
        "PolynomialDegree": degree,
        "GridSize": cells,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "Relative_tolerance": rtol,
        "Absolute_tolerance": rtol,
        "delta_t": 1.0,
        "SteadyStateTest": "NewtonStep",
        "SuppressAlgebraicError": True,
        "SteadyStateTolerance": 1.0e-10,      # arms the steady solve; not read
        "MaxContinuationSteps": steps,
        "MaxRejectedSteps": 100000,
        "WriteOutput": False,
    }


def _quiet():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def reference(cells, degree):
    import manta
    from blackbox import ToyFlux, BlackBoxCase
    runner = manta.Runner(BlackBoxCase(ToyFlux(0.0), "analytic"))
    runner.configure(configuration(cells, degree, 1.0e-10, 200))
    runner.run_ss()
    return np.asarray(runner.getSolution(0, list(samples(cells)))).ravel()


def solve(run):
    import manta
    from reporting import QuendsFlux, FilteredCase

    box = QuendsFlux(run["noise"], seed=run["seed"], **SCENARIOS[run["scenario"]])
    if run["mode"] == "uniform":
        case = FilteredCase(box, "uniform", kappa=run["kappa"],
                            assumed_eps=run["factor"] * run["noise"])
    else:
        case = FilteredCase(box, run["mode"], kappa=run["kappa"], misreport=run["factor"])
    runner = manta.Runner(case)
    runner.configure(configuration(run["cells"], run["degree"], run["rtol"], 1))
    case.reset_counts()

    x = list(samples(run["cells"]))
    calls, states, outcome = [], [], "Budget"
    try:
        with manta.SteadySolve(runner, estimate=False) as steady:
            for result, _ in steady:
                calls.append(case.value_calls + case.fd_calls)
                states.append(np.asarray(runner.getSolution(0, x)).ravel())
                if result.name != "OutOfSteps":
                    outcome = result.name
                elif calls[-1] >= run["budget"]:
                    steady.stop()
    except RuntimeError:
        outcome = "Failed"
    return dict(run, outcome=outcome, calls=np.array(calls), states=np.array(states),
                irregular=box.irregular)


def at_budget(r, budget):
    """The noise error of the last state reached within `budget` calls."""
    i = np.searchsorted(r["calls"], budget, side="right") - 1
    return r["err"][max(i, 0)]


def summarise(results, budgets):
    groups = {}
    for r in results:
        groups.setdefault((r["scenario"], r["mode"], r["factor"], r["kappa"]), []).append(r)

    for scenario in SCENARIOS:
        rows = sorted(k for k in groups if k[0] == scenario)
        if not rows:
            continue
        irr = sum(r["irregular"] for k in rows for r in groups[k])
        print(f"\n## {scenario}" + (f" ({irr} irregular samples over every run)" if irr else "") + "\n")
        head = " | ".join(f"{b} calls" for b in budgets)
        print(f"| arm | kappa | {head} | test passed | calls when it did |")
        print("|---|---|" + "---|" * len(budgets) + "---|---|")
        for k in rows:
            rs = groups[k]
            cells = []
            for b in budgets:
                e = np.array([at_budget(r, b) for r in rs])
                cells.append(f"{np.median(e):.3f} ({e.max():.3f})")
            conv = [r for r in rs if r["outcome"] == "Converged"]
            when = (f"{np.median([r['calls'][-1] for r in conv]):.0f}" if conv else "--")
            arm = k[1] if k[1] == "raw" else f"{k[1]} x {k[2]:.2g}"
            kappa = "--" if k[1] == "raw" else f"{k[3]:g}"
            print(f"| {arm} | {kappa} | " + " | ".join(cells) + f" | {len(conv)}/{len(rs)} | {when} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", type=int, default=48)
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--noise", type=float, default=1.0e-2,
                   help="median relative standard error of a sample")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    p.add_argument("--kappa", type=float, nargs="+", default=[0.1])
    p.add_argument("--rtol", type=float, default=1.0e-6)
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--csv")
    args = p.parse_args()

    runs = []
    for scenario in args.scenario:
        for seed in range(args.seeds):
            for mode, factor in ARMS:
                for kappa in (args.kappa if mode != "raw" else [args.kappa[0]]):
                    runs.append(dict(cells=args.cells, degree=args.degree, noise=args.noise,
                                     rtol=args.rtol, budget=args.budget, scenario=scenario,
                                     seed=seed, mode=mode, factor=factor, kappa=kappa))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_quiet) as pool:
        ref = pool.submit(reference, args.cells, args.degree).result()
        print(f"{args.cells} cells, k = {args.degree}, eps = {args.noise:g}: {len(runs)} runs, "
              f"{args.seeds} seeds; noise error in units of eps, median (max)", flush=True)
        results = list(pool.map(solve, runs))

    scale = np.max(np.abs(ref))
    for r in results:
        r["err"] = np.max(np.abs(r.pop("states") - ref), axis=1) / scale / args.noise
    summarise(results, args.budgets)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fields = [k for k in results[0] if k not in ("calls", "err")]
            w = csv.writer(f)
            w.writerow(fields + ["calls", "err"])
            for r in results:
                for c, e in zip(r["calls"], r["err"]):
                    w.writerow([r[k] for k in fields] + [c, e])


if __name__ == "__main__":
    sys.exit(main())
