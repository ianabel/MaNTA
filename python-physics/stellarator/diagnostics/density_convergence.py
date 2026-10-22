"""How the steady solve converges when n0 goes from 1e20 to 1e19.

Three cases, each solved in a subprocess of its own:

* `baseline` -- the problem of `paper/desc-optimize-stored-energy.py`'s first
  solve, written in the default normalisation and converted
  (`rescale_config`) to a and B0 from the equilibrium, as that driver does
  with keep_physics = True.
* `rescaled` -- the same physical problem with n0 = 1e19 as well. Only the
  units change, so this isolates what the choice of reference density does to
  the solver itself: the profiles must agree with `baseline`.
* `low_density` -- the baseline's normalised numbers kept as they stand and
  n0 = 1e19. The plasma is 10x less dense and 10x less heated (the source
  normalisations scale with n0; the time unit does not), so the heating per
  particle is the same; where T settles depends on how transport scales with
  n, so read the collisionality off the reported profiles.

For each it reports the outcome, continuation steps, Newton iterations,
Jacobian builds, wall time and the core n and T, from `Runner.steadyStats()`.

With TEST_STELLARATOR=true the drift-kinetic solve is the SI-units mock
`dke_sol.physical_test_flux` (D ~ T^3.5 / (n B^2)), so that changing the
density changes its physics the way it would yancc's. Without it, this runs
yancc, which is far too expensive for a workstation:

    TEST_STELLARATOR=true PYTHONPATH=$PWD/build/python \\
        python python-physics/stellarator/density_convergence.py
"""

import os
import sys

if os.environ.get("TEST_STELLARATOR") == "true":
    os.environ["STELLARATOR_TEST_FLUX"] = "physical"

import argparse
import json
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ("baseline", "rescaled", "low_density")
LOW_N0 = 1e19


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n0", type=float, default=LOW_N0, help="the lowered n0 [m^-3]")
    p.add_argument(
        "--heat-factor",
        type=float,
        default=1.0,
        help="multiply low_density's NBIPower and ECHPower by this",
    )
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    return p.parse_args()


def base_config():
    """paper/desc-optimize-stored-energy.py's first solve, default normalisation."""
    st_config = {
        "ParticleSourceCenter": 0.0,
        "ParticleSourceHeight": 0.01,
        "ParticleSourceWidth": 0.4,
        "NBICenter": 0.0,
        "NBIPower": 1.0,
        "NBIWidth": 0.15,
        "ECHCenter": 0.0,
        "ECHPower": 8.0,
        "ECHWidth": 0.05,
        "EdgeTemperature": 0.5,
        "EdgeDensity": 0.2,
        "ADCoefficient": 0.0,
        "n0": 0.5,
        "T0": 1.0,
        "FusionFactor": 0.01,
        "evolveDensity": True,
        "useBatching": True,
        "nchunks": 3,
    }
    rho_upper, npoints, base = 1.0, 8, 1.8
    nodes = rho_upper - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 3)
    nodes = np.concatenate(([0, 0.05, 0.2], nodes, [rho_upper]))
    solver_config = {
        "OutputFilename": "density_convergence",
        "PolynomialDegree": 5,
        "GridPoints": list(nodes),
        "GridSize": len(nodes) - 1,
        "tau": 0.2,
        "LowerBoundary": 0.0,
        "UpperBoundary": rho_upper,
        "Relative_tolerance": 1e-2,
        "Absolute_tolerance": [1e-10],
        "delta_t": 1.0,
        "initialTimestep": 0.01,
        "MinStepSize": 1e-9,
        "SteadyStateTolerance": 1e-5,
        "AggressiveTimesteps": False,
        "NewtonJacobianReuse": 5,
        "NewtonMaxIterations": 25,
        "MaxRejectedSteps": 10,
        "restart": False,
        "zeroFlux": True,
        "solveAdjoint": False,
        "PseudoTransientSERRate": 1.0,
        "PseudoTransientSERFloor": 2.0,
    }
    return {"Stellarator": st_config, "Solver": solver_config}


def case_config(name, eq, low_n0, heat_factor=1.0):
    from config import field_norm_from_eq, rescale_config

    field = field_norm_from_eq(eq)
    baseline = rescale_config(base_config(), field)
    if name == "baseline":
        return baseline
    if name == "rescaled":
        return rescale_config(baseline, {**field, "n0": low_n0})
    if name == "low_density":
        st = dict(baseline["Stellarator"])
        for key in ("NBIPower", "ECHPower"):
            st[key] *= heat_factor
        return {**baseline, "Stellarator": st, "Normalization": {**field, "n0": low_n0}}
    raise ValueError(name)


def load_eq():
    import desc.examples
    import desc.compat

    eq = desc.examples.get("W7-X")
    # as runStellarator.py does
    return desc.compat.rescale(eq, L=("a", 1.7), B=("<B>", 5.86), copy=True)


def worker(args):
    sys.path.insert(0, HERE)
    import jax

    jax.config.update("jax_enable_x64", True)
    import manta as MaNTA

    from config import plasma_constants
    from stellarator_multichannel import StellaratorTransport
    from yancc_wrapper import yancc_data

    eq = load_eq()
    config = case_config(args.worker, eq, args.n0, args.heat_factor)
    solver = config["Solver"]
    solver["GridPoints"] = np.asarray(solver["GridPoints"])
    points = MaNTA.getNodes(solver["GridPoints"], solver["PolynomialDegree"])
    wrapper = yancc_data.from_eq(
        np.asarray(points), eq=eq, nt=17, nz=25, nx=7, na=45
    )

    st = StellaratorTransport(config, yancc_wrapper=wrapper)
    t0 = time.time()
    converged = bool(st.run())
    wall = time.time() - t0
    stats = {k: v for k, v in st.runner.steadyStats().items() if k != "outcome"}

    c = plasma_constants(config["Normalization"])
    vp = np.asarray(st.vp)
    n = np.asarray(st.runner.Get_profile(0)) / vp
    pi = 2.0 / 3.0 * np.asarray(st.runner.Get_profile(1)) / vp
    pe = 2.0 / 3.0 * np.asarray(st.runner.Get_profile(2)) / vp
    out = {
        "converged": converged,
        "wall": wall,
        "stats": {k: float(v) for k, v in stats.items()},
        "x": np.asarray(points).tolist(),
        "n": (n * c.n0).tolist(),
        "Ti": (pi / n * c.T0eV).tolist(),
        "Te": (pe / n * c.T0eV).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)


def main(args):
    names = [k.strip() for k in args.cases.split(",") if k.strip()]
    scratch = tempfile.mkdtemp(prefix="density_convergence_")
    results = {}
    for name in names:
        out = os.path.join(scratch, f"{name}.json")
        log = os.path.join(scratch, f"{name}.log")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", name]
        cmd += ["--out", out, "--n0", str(args.n0), "--heat-factor", str(args.heat_factor)]
        print(f"solving {name}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=scratch, stdout=f, stderr=subprocess.STDOUT)
        if rc.returncode != 0 or not os.path.exists(out):
            print(f"  FAILED (exit {rc.returncode}); see {log}")
            continue
        with open(out) as f:
            results[name] = json.load(f)

    cols = [
        ("steps", "steps"),
        ("rejected", "rej"),
        ("newton_iterations", "newton"),
        ("jacobian_builds", "jac"),
        ("residual_norm", "|F|"),
    ]
    print(
        f"\n{'case':12s} {'conv':>5s} "
        + " ".join(f"{h:>8s}" for _, h in cols)
        + f" {'wall s':>7s} {'n(0) m^-3':>10s} {'Ti(0) eV':>9s} {'Te(0) eV':>9s}"
    )
    for name, r in results.items():
        s = r["stats"]
        print(
            f"{name:12s} {str(r['converged']):>5s} "
            + " ".join(
                f"{s.get(k, float('nan')):8.3g}" if k == "residual_norm"
                else f"{s.get(k, float('nan')):8.0f}"
                for k, _ in cols
            )
            + f" {r['wall']:7.1f} {r['n'][0]:10.3e} {r['Ti'][0]:9.1f} {r['Te'][0]:9.1f}"
        )

    if "baseline" in results and "rescaled" in results:
        b, r = results["baseline"], results["rescaled"]
        d = max(
            np.max(np.abs(np.subtract(r[q], b[q]))) / np.max(np.abs(b[q]))
            for q in ("n", "Ti", "Te")
        )
        print(f"\nrescaled vs baseline profiles: max relative difference {d:.2e}")
    print(f"outputs and logs: {scratch}")


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        main(args)
