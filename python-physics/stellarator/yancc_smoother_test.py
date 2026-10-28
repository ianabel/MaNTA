"""Does yancc's solve_dke compile and run under each multigrid smooth_solver, and
how fast is it?

Mirrors how MaNTA calls yancc: a vmap over nodes, each with its own Field, of
the forward solve and of its jacrev (ComputePhysicsDerivatives). Each
smooth_solver runs in a subprocess of its own, because the failure this is
looking for -- an XLA RET_CHECK in algebraic_simplifier.cc, reached by "cr"
under vmap with jax 0.9.2 and 0.10.2 -- aborts the process rather than raising.

The plasma is the state stellarator_opteq5_acc stalled at, at the node
rho = 0.48678 (n0 = 1e19 m^-3, T0 = 1 keV); the second node reuses it with its
own field. Only yancc, DESC and jax are needed, not MaNTA.

    python yancc_smoother_test.py paper/eq5_all_equilibria.h5
    python yancc_smoother_test.py eq.h5 --smoothers dense,banded --rtol 1e-3 \
        --grids 3 --npoints 2 --repeat 3

A smoother that compiles reports the first call (compile included) and the
warm calls after it; one that crashes reports its exit code and the XLA check.
"""

import argparse
import os
import subprocess
import sys
import time

# The state at rho = 0.48678, in MaNTA's units: n in n0, T in T0, Er in T0/e
STATE = dict(n=14.01, dndrho=-13.14, Ti=1.919, dTidrho=-1.195, Te=1.918,
             dTedrho=-1.195, Er=-3.775)
N0, T0 = 1e19, 1000.0
NODES = [0.48678, 0.58619, 0.67228, 0.40069, 0.72199, 0.35099]


def worker(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import equinox as eqx
    import desc.io
    from yancc import Field
    from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
    from yancc.species import LocalMaxwellian, Electron, Species
    from yancc.solve import solve_dke

    smooth = None if args.worker == "None" else args.worker
    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)

    eq = desc.io.load(args.eq)
    if hasattr(eq, "__len__"):  # an EquilibriaFamily: the final equilibrium
        eq = eq[-1]
    rhos = jnp.array(NODES[: args.npoints])
    fields = jax.vmap(lambda r: Field.from_desc(eq, r, args.nt, args.nz))(rhos)
    pitch, speed = UniformPitchAngleGrid(args.na), MaxwellSpeedGrid(args.nx)
    ions = Species(2.5, 1)  # ion_species.DeuteriumTritium
    s = STATE

    def fluxes(fld, Erho, dTe):
        species = [
            LocalMaxwellian(ions, temperature=s["Ti"] * T0, density=s["n"] * N0,
                            dTdrho=s["dTidrho"] * T0, dndrho=s["dndrho"] * N0),
            LocalMaxwellian(Electron, temperature=s["Te"] * T0, density=s["n"] * N0,
                            dTdrho=dTe, dndrho=s["dndrho"] * N0),
        ]
        sol, _ = solve_dke(fld, pitch, speed, species, Erho=Erho, rtol=args.rtol,
                           throw=False, verbose=args.verbose,
                           multigrid_options={"smooth_solver": smooth,
                                              "max_grids": args.grids})
        return jnp.stack([sol.get("<particle_flux>")[1], sol.get("<heat_flux>")[1],
                          sol.get("J_rho")])

    Erhos = jnp.full(args.npoints, s["Er"] * T0)
    dTes = jnp.full(args.npoints, s["dTedrho"] * T0)
    for mode in args.modes.split(","):
        if mode == "forward":
            g = fluxes
        else:
            jac = eqx.filter_jacrev(lambda a, f: fluxes(f, a[0], a[1]))
            g = lambda f, e, d: jac((e, d), f)
        fn = eqx.filter_jit(eqx.filter_vmap(g, in_axes=(eqx.if_array(0), 0, 0)))
        t = time.time()
        out = jax.block_until_ready(fn(fields, Erhos, dTes))
        first = time.time() - t
        warm = []
        for r in range(args.repeat):
            t = time.time()
            jax.block_until_ready(fn(fields, Erhos * (1.0 + 1e-3 * (r + 1)), dTes))
            warm.append(time.time() - t)
        head = jax.tree.map(lambda v: float(jnp.ravel(v)[0]), out)
        print(f"RESULT smooth={args.worker} {mode}: first {first:.1f}s, warm "
              f"{', '.join(f'{w:.2f}' for w in warm)}s  (node 0, first entries: {head})",
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("eq", help="DESC equilibrium or EquilibriaFamily .h5")
    p.add_argument("--smoothers", default="dense,banded,cr,None")
    p.add_argument("--modes", default="forward,jacrev")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--grids", type=int, default=3)
    p.add_argument("--npoints", type=int, default=2, help=f"at most {len(NODES)}")
    p.add_argument("--repeat", type=int, default=2, help="warm calls to time")
    p.add_argument("--na", type=int, default=45)
    p.add_argument("--nx", type=int, default=7)
    p.add_argument("--nt", type=int, default=17)
    p.add_argument("--nz", type=int, default=25)
    p.add_argument("--verbose", type=int, default=0, help="yancc's verbose")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker is not None:
        return worker(args)

    passthrough = [args.eq] + [
        f"--{k}={getattr(args, k)}"
        for k in ("modes", "rtol", "grids", "npoints", "repeat", "na", "nx", "nt",
                  "nz", "verbose")
    ]
    summary = []
    for smooth in args.smoothers.split(","):
        print(f"\n=== smooth_solver={smooth} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, *passthrough, "--worker", smooth],
            capture_output=True, text=True,
        )
        lines = (proc.stdout + proc.stderr).splitlines()
        keep = [l for l in lines if l.startswith(("RESULT", "jax "))
                or "Check failed" in l or "Error" in l]
        print("\n".join(keep[:12]) if keep else "\n".join(lines[-20:]))
        summary.append((smooth, proc.returncode,
                        [l for l in lines if l.startswith("RESULT")]))

    print("\n=== summary ===")
    for smooth, code, results in summary:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"{smooth:>7}: {status}")
        for r in results:
            mode = r.split()[2].rstrip(":")
            print(f"         {mode:>8}: {r.split(':', 1)[1].split('(')[0].strip()}")


if __name__ == "__main__":
    main()
