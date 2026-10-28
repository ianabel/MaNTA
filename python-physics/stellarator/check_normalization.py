"""Check that the stellarator model's answer does not depend on its normalisation.

`PlasmaConstants` normalises densities to n0, temperatures to T0 and time to
`NormalizingTime()`, which also reads a and B0. Those are choices of unit, not
physics: the same physical problem written in another set of them, by
`config.rescale_config`, must reach the same steady state once unnormalised.
That is what this checks. It solves one physical problem under several
normalisations and compares density, temperatures, Er and the bootstrap
current in SI units against the default normalisation's solution.

The drift-kinetic solve is replaced by `dke_sol.physical_test_flux`, a
diffusive mock written in SI units that goes through the same normalisation
code (`dke_sol._normalized_fluxes`) as yancc does. The existing `test_flux` will
not do for this: its diffusivity is a normalised number, so it describes
different physics in every normalisation by construction.

Each normalisation is solved in a subprocess of its own, writing into a scratch
directory, so no two FFIRunners share a process.

Run only with TEST_STELLARATOR=true:

    TEST_STELLARATOR=true PYTHONPATH=$PWD/build/python \\
        python python-physics/stellarator/check_normalization.py [--evolve-density]
        [--er-model Diffusive [--er-diffusion D]]
"""

import os
import sys

if os.environ.get("TEST_STELLARATOR") != "true":
    sys.exit(
        "check_normalization.py: refusing to run without TEST_STELLARATOR=true "
        "(the real drift-kinetic solve is too expensive for this check)"
    )
os.environ["STELLARATOR_TEST_FLUX"] = "physical"

import argparse
import json
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The default first: every other case is compared with it.
CASES = {
    "default": {},
    "n0": {"n0": 3e20},
    "T0": {"T0": 2500.0},
    "a": {"a": 1.7},
    "B0": {"B0": 2.5},
    "all": {"n0": 0.5e20, "T0": 400.0, "a": 0.6, "B0": 8.0},
    # a and B0 from the equilibrium (config.field_norm_from_eq); filled in by
    # the worker, which has the equilibrium
    "field": {},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eq", default="precise_QA", help="DESC example name")
    p.add_argument("--cells", type=int, default=6)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--evolve-density", action="store_true")
    p.add_argument(
        "--er-model",
        choices=("Algebraic", "Diffusive"),
        default="Algebraic",
        help="how Er is determined (with --evolve-density)",
    )
    p.add_argument(
        "--er-diffusion",
        type=float,
        default=1e-3,
        help="ErDiffusion under --er-model Diffusive, in the default normalisation",
    )
    p.add_argument(
        "--er-inertia",
        type=float,
        default=1.0,
        help="ErInertia under --er-model Diffusive, in the default normalisation",
    )
    p.add_argument("--cases", default=",".join(CASES), help="normalisations to run")
    p.add_argument("--tol", type=float, default=1e-6, help="relative agreement required")
    p.add_argument("--keep", action="store_true", help="keep the scratch directory")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    return p.parse_args()


def base_config(args):
    """The physical problem, written in the default normalisation."""
    st_config = {
        "ParticleSourceCenter": 0.0,
        "ParticleSourceHeight": 0.02,
        "ParticleSourceWidth": 0.4,
        "NBICenter": 0.0,
        "NBIPower": 0.5,
        "NBIWidth": 0.2,
        "ECHCenter": 0.0,
        "ECHPower": 0.5,
        "ECHWidth": 0.2,
        "EdgeTemperature": 0.5,
        "EdgeDensity": 0.4,
        "n0": 0.5,
        "T0": 1.0,
        # at full strength, so that alpha heating is a visible term
        "FusionFactor": 1.0,
        "ADCoefficient": 0.05,
        "evolveDensity": args.evolve_density,
        "useBatching": True,
        "adaptiveGrid": False,
    }
    solver_config = {
        "OutputFilename": "check_normalization",
        "PolynomialDegree": args.degree,
        "GridPoints": list(np.linspace(0.0, 1.0, args.cells + 1)),
        "GridSize": args.cells,
        "tau": 0.5,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "Relative_tolerance": 1e-10,
        "Absolute_tolerance": [1e-14],
        "delta_t": 1.0,
        "initialTimestep": 1e-2,
        "MinStepSize": 1e-14,
        "SteadyStateSolver": "PseudoTransient",
        "SteadyStateTolerance": 1e-12,
        "MaxRejectedSteps": 10,
        "restart": False,
        "zeroFlux": True,
        "solveAdjoint": False,
    }
    if args.er_model == "Diffusive":
        # ErInertia explicit, so that rescale_config converts it: it drops out
        # of the steady equation but not out of the HDG stabilisation, which
        # tau adds to the Er row undivided by it
        st_config.update(
            ErModel="Diffusive",
            ErDiffusion=args.er_diffusion,
            ErInertia=args.er_inertia,
        )
    return {"Stellarator": st_config, "Solver": solver_config, "Normalization": {}}


def worker(args):
    """Solve under one normalisation and save the SI profiles."""
    sys.path.insert(0, HERE)
    import jax

    jax.config.update("jax_enable_x64", True)
    import desc.examples
    import manta as MaNTA
    from netCDF4 import Dataset

    from config import (
        rescale_config,
        plasma_constants,
        field_norm_from_eq,
    )
    from stellarator_multichannel import StellaratorTransport
    from yancc_wrapper import yancc_data

    base = base_config(args)
    grid = np.asarray(base["Solver"]["GridPoints"])
    points = MaNTA.getNodes(grid, base["Solver"]["PolynomialDegree"])
    eq = desc.examples.get(args.eq)
    wrapper = yancc_data.from_eq(np.asarray(points), eq=eq, nt=9, nz=9, nx=3, na=9)

    if args.worker == "field":
        normalization = field_norm_from_eq(eq)
    else:
        normalization = dict(CASES[args.worker])
    print("normalisation:", normalization)
    config = rescale_config(base, normalization)
    solver = config["Solver"]
    solver["GridPoints"] = grid

    st = StellaratorTransport(config, yancc_wrapper=wrapper)
    converged = bool(st.run())

    c = plasma_constants(normalization)
    vp = np.asarray(st.vp)
    out = {"x": np.asarray(points), "converged": converged}
    if args.evolve_density:
        n = np.asarray(st.runner.Get_profile(0)) / vp
        pi = 2.0 / 3.0 * np.asarray(st.runner.Get_profile(1)) / vp
        pe = 2.0 / 3.0 * np.asarray(st.runner.Get_profile(2)) / vp
        if args.er_model == "Diffusive":
            # the fourth variable, V' Er, on the nodes like the others
            er = np.asarray(st.runner.Get_profile(3)) / vp
        else:
            with Dataset(solver["OutputFilename"] + ".nc") as nc:
                er = np.asarray(nc.variables["Er"][-1, :])
                er_x = np.asarray(nc.variables["x"][:])
            er = np.interp(points, er_x, er)
        # the bootstrap current, through the runner; checked against the aux
        # the netCDF output carries, a top-level variable like an algebraic Er
        jbs = np.asarray(st.getBootstrapCurrent(points))
        with Dataset(solver["OutputFilename"] + ".nc") as nc:
            jbs_nc = np.asarray(nc.variables["BootstrapCurrent"][-1, :])
            jbs_x = np.asarray(nc.variables["x"][:])
        # at the file's own points: interpolating it onto the nodes instead
        # costs 2.5e-4 here, which is the interpolation and not the runner
        jbs_nc = jbs_nc * c.BootstrapCurrentNormalization()
        jbs_at_nc = np.asarray(st.getBootstrapCurrent(jbs_x))
        gap = np.max(np.abs(jbs_at_nc - jbs_nc)) / np.max(np.abs(jbs_nc))
        print(f"getBootstrapCurrent against the netCDF output: {gap:.2e}")
        assert gap < 1e-6, f"getBootstrapCurrent disagrees with the output by {gap}"
        out.update(
            n=n * c.n0,
            Ti=pi / n * c.T0eV,
            Te=pe / n * c.T0eV,
            Er=er * c.T0eV,
            Jbs=jbs,
        )
    else:
        from stellarator_state import StellaratorState

        cfg = config["Stellarator"]
        n = StellaratorState.initial_profile(
            np.asarray(points), cfg["EdgeDensity"], cfg["n0"], 1.0, 2.0
        )
        pi = 2.0 / 3.0 * np.asarray(st.runner.Get_profile(0)) / vp
        out.update(Ti=pi / n * c.T0eV)
    np.savez(args.out, **out)


def main(args):
    names = [k.strip() for k in args.cases.split(",") if k.strip()]
    if names[0] != "default":
        names = ["default"] + [n for n in names if n != "default"]

    scratch = tempfile.mkdtemp(prefix="check_normalization_")
    results = {}
    for name in names:
        out = os.path.join(scratch, f"{name}.npz")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", name, "--out", out]
        cmd += ["--eq", args.eq, "--cells", str(args.cells), "--degree", str(args.degree)]
        cmd += ["--evolve-density"] if args.evolve_density else []
        cmd += ["--er-model", args.er_model, "--er-diffusion", str(args.er_diffusion)]
        cmd += ["--er-inertia", str(args.er_inertia)]
        log = os.path.join(scratch, f"{name}.log")
        print(f"solving under normalisation {name:8s} {json.dumps(CASES[name])}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(
                cmd, cwd=scratch, stdout=f, stderr=subprocess.STDOUT
            ).returncode
        if rc != 0 or not os.path.exists(out):
            print(f"  FAILED (exit {rc}); see {log}")
            results[name] = None
            continue
        results[name] = dict(np.load(out))

    base = results.get("default")
    if base is None:
        sys.exit(f"the default normalisation failed; logs in {scratch}")

    quantities = [q for q in ("n", "Ti", "Te", "Er", "Jbs") if q in base]
    print(f"\nmax |X - X_default| / max |X_default|, SI units ({len(base['x'])} nodes)")
    print(f"{'case':8s} " + " ".join(f"{q:>10s}" for q in quantities) + "  verdict")
    ok = bool(base["converged"])
    for name in names[1:]:
        r = results[name]
        if r is None:
            print(f"{name:8s} " + "solve failed".rjust(11 * len(quantities)) + "  FAIL")
            ok = False
            continue
        errs = [
            np.max(np.abs(r[q] - base[q])) / np.max(np.abs(base[q])) for q in quantities
        ]
        good = bool(r["converged"]) and max(errs) <= args.tol
        ok &= good
        note = "" if r["converged"] else " (not converged)"
        print(
            f"{name:8s} "
            + " ".join(f"{e:10.2e}" for e in errs)
            + ("  ok" if good else "  FAIL")
            + note
        )

    if args.keep or not ok:
        print(f"\nscratch directory kept: {scratch}")
    else:
        import shutil

        shutil.rmtree(scratch)
    print("normalisation invariance:", "OK" if ok else "VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        sys.exit(main(args))
